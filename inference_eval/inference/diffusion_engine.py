"""Diffusion language model engine (LLaDA2, Dream, etc.).

Supports both ``generate_until`` (text generation) and ``loglikelihood``
(scoring) tasks.  Uses the `dllm <https://github.com/ZHZisZZ/dllm>`_
framework when available for optimised sampling, with a pure-transformers
fallback.

**Multi-GPU**: pass ``devices="cuda:0,cuda:1,cuda:2,cuda:3"`` to load
a model copy on each GPU and process prompts in parallel.  This gives
near-linear speedup with the number of GPUs.

Prompts from lm-eval-harness are raw text.  By default this engine
wraps each prompt in a chat template (``apply_chat_template=True``).
**Do NOT combine this with ``inference-eval extract --apply-chat-template``**
or the prompt will be double-wrapped.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


def _truncate_at_stop(text: str, stop_sequences: list[str]) -> str:
    """Truncate *text* at the first occurrence of any stop sequence."""
    earliest = len(text)
    for s in stop_sequences:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


@dataclass
class _Worker:
    """One model instance on one GPU."""

    model: Any
    tokenizer: Any
    sampler: Any  # dllm sampler or None
    sampler_config_cls: Any  # dllm config class or None
    device: str


class DiffusionEngine(InferenceEngine):
    """Inference engine for diffusion language models.

    Loads the model via ``transformers`` (with ``trust_remote_code``)
    and optionally wraps it with the **dllm** sampler for faster
    generation.  Also supports ``loglikelihood`` via the model's
    ``get_log_likelihood()`` method.

    **Multi-GPU parallelism**: pass ``devices="cuda:0,cuda:1,..."``
    to load a model copy on each GPU.  Prompts are distributed
    round-robin across GPUs and processed in parallel via threads
    (PyTorch releases the GIL during CUDA ops).

    Args:
        model: HuggingFace model name or local path.
        device: Single device string (``"auto"``, ``"cuda:0"``).
            Ignored if ``devices`` is set.
        devices: Comma-separated list of devices for multi-GPU
            parallelism (e.g. ``"cuda:0,cuda:1,cuda:2,cuda:3"``).
        dtype: Torch dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        gen_length: Default number of new tokens to generate.
        block_length: Block length for diffusion sampling.
        steps: Number of demasking steps per block.
        temperature: Sampling temperature (0 = greedy).
        apply_chat_template: Wrap each prompt as a chat user message.
        eos_early_stop: Stop generation early on EOS.
        mc_num: Monte-Carlo samples for ``get_log_likelihood``.
        use_dllm: Try to use the ``dllm`` framework for generation.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        devices: str | None = None,
        dtype: str = "bfloat16",
        gen_length: int = 512,
        block_length: int = 32,
        steps: int = 32,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
        eos_early_stop: bool = True,
        mc_num: int = 32,
        use_dllm: bool = True,
        **extra_kwargs: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for DiffusionEngine."
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self._torch = torch
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer

        self._model_path = model
        self._torch_dtype = torch_dtype
        self._gen_length = gen_length
        self._block_length = block_length
        self._steps = steps
        self._temperature = temperature
        self._apply_chat_template = apply_chat_template
        self._eos_early_stop = eos_early_stop
        self._mc_num = mc_num
        self._use_dllm = use_dllm

        device_list = [d.strip() for d in devices.split(",")] if devices else [device]

        self._workers: list[_Worker] = []
        for dev in device_list:
            w = self._create_worker(dev)
            self._workers.append(w)

        logger.info(
            "DiffusionEngine ready: %d GPU(s) %s, dllm=%s, "
            "gen_length=%d, steps=%d, block_length=%d",
            len(self._workers),
            [w.device for w in self._workers],
            self._workers[0].sampler is not None,
            gen_length,
            steps,
            block_length,
        )

    def _create_worker(self, device: str) -> _Worker:
        logger.info("Loading model on %s: %s", device, self._model_path)
        tokenizer = self._AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        model = (
            self._AutoModelForCausalLM.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                device_map=device,
            )
            .to(self._torch_dtype)
            .eval()
        )

        sampler = None
        sampler_config_cls = None
        if self._use_dllm:
            try:
                import dllm

                sampler = dllm.pipelines.llada2.LLaDA2Sampler(
                    model=model, tokenizer=tokenizer
                )
                sampler_config_cls = dllm.pipelines.llada2.LLaDA2SamplerConfig
            except Exception:
                pass

        return _Worker(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            sampler_config_cls=sampler_config_cls,
            device=device,
        )

    # ------------------------------------------------------------------
    # tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize_prompt(self, worker: _Worker, prompt: str) -> Any:
        if self._apply_chat_template:
            ids = worker.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            ids = worker.tokenizer(prompt, return_tensors="pt").input_ids
        return ids.to(worker.model.device)

    # ------------------------------------------------------------------
    # generate  (multi-GPU parallel via threads)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        n = len(self._workers)
        if n == 1:
            return self._generate_sequential(self._workers[0], prompts, gen_kwargs)

        results: list[str | None] = [None] * len(prompts)

        def _work(idx: int) -> tuple[int, str]:
            worker = self._workers[idx % n]
            return idx, self._generate_single(worker, prompts[idx], gen_kwargs[idx])

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_work, i): i for i in range(len(prompts))}
            for fut in as_completed(futures):
                idx, text = fut.result()
                results[idx] = text

        return [r if r is not None else "" for r in results]

    def _generate_sequential(
        self,
        worker: _Worker,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        return [
            self._generate_single(worker, p, kw) for p, kw in zip(prompts, gen_kwargs)
        ]

    def _generate_single(
        self,
        worker: _Worker,
        prompt: str,
        kw: dict[str, Any],
    ) -> str:
        input_ids = self._tokenize_prompt(worker, prompt)
        input_len = input_ids.shape[1]
        gen_len = kw.get("max_gen_toks", kw.get("max_tokens", self._gen_length))
        temp = kw.get("temperature", self._temperature)

        if worker.sampler is not None:
            text = self._gen_dllm(worker, input_ids, input_len, gen_len, temp)
        else:
            text = self._gen_transformers(worker, input_ids, input_len, gen_len, temp)

        stop = kw.get("until", kw.get("stop", []))
        if isinstance(stop, str):
            stop = [stop]
        if stop:
            text = _truncate_at_stop(text, stop)
        return text

    def _gen_dllm(
        self,
        worker: _Worker,
        input_ids: Any,
        input_len: int,
        gen_length: int,
        temperature: float,
    ) -> str:
        cfg = worker.sampler_config_cls(
            max_new_tokens=gen_length,
            block_size=self._block_length,
            steps_per_block=self._steps,
            temperature=temperature,
        )
        output = worker.sampler.sample(input_ids, cfg)
        if hasattr(output, "sequences"):
            out_ids = output.sequences[0]
        elif self._torch.is_tensor(output):
            out_ids = output[0]
        else:
            out_ids = output
        new_ids = out_ids[input_len:]
        return worker.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _gen_transformers(
        self,
        worker: _Worker,
        input_ids: Any,
        input_len: int,
        gen_length: int,
        temperature: float,
    ) -> str:
        with self._torch.no_grad():
            output_ids = worker.model.generate(
                inputs=input_ids,
                gen_length=gen_length,
                block_length=self._block_length,
                steps=self._steps,
                temperature=temperature,
                eos_early_stop=self._eos_early_stop,
            )
        new_ids = output_ids[0][input_len:]
        return worker.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # loglikelihood  (multi-GPU parallel)
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihoods using the model's diffusion-based scorer.

        Requires the model to expose ``get_log_likelihood()``
        (available in LLaDA / LLaDA2 via ``trust_remote_code``).
        """
        if not contexts:
            return []

        if not hasattr(self._workers[0].model, "get_log_likelihood"):
            raise NotImplementedError(
                "This model does not expose get_log_likelihood(). "
                "Make sure you are using a LLaDA / LLaDA2 model loaded "
                "with trust_remote_code=True, or install dllm."
            )

        n = len(self._workers)
        if n == 1:
            return [
                self._ll_single(self._workers[0], ctx, cont)
                for ctx, cont in zip(contexts, continuations)
            ]

        results: list[tuple[float, bool] | None] = [None] * len(contexts)

        def _work(idx: int) -> tuple[int, tuple[float, bool]]:
            worker = self._workers[idx % n]
            return idx, self._ll_single(worker, contexts[idx], continuations[idx])

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_work, i): i for i in range(len(contexts))}
            for fut in as_completed(futures):
                idx, result = fut.result()
                results[idx] = result

        return [r if r is not None else (0.0, True) for r in results]

    def _ll_single(
        self, worker: _Worker, context: str, continuation: str
    ) -> tuple[float, bool]:
        ctx_ids = worker.tokenizer(context, return_tensors="pt").input_ids
        full_ids = worker.tokenizer(
            context + continuation, return_tensors="pt"
        ).input_ids
        ctx_ids = ctx_ids.to(worker.model.device)
        full_ids = full_ids.to(worker.model.device)

        ctx_len = ctx_ids.shape[1]
        target_ids = full_ids.clone()
        target_ids[:, :ctx_len] = -100

        with self._torch.no_grad():
            ll = worker.model.get_log_likelihood(
                full_ids, target_ids, mc_num=self._mc_num
            )

        val = ll.item() if self._torch.is_tensor(ll) else float(ll)
        return (val, True)
