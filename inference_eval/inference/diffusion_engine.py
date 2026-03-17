"""Diffusion language model engine (LLaDA2, Dream, etc.).

Supports both ``generate_until`` (text generation) and ``loglikelihood``
(scoring) tasks.  Uses the `dllm <https://github.com/ZHZisZZ/dllm>`_
framework when available for optimised sampling, with a pure-transformers
fallback.

**Speed levers** (combine for maximum throughput):

- ``gpu_batch_size=4``  — process 4 prompts per GPU call (default 1)
- ``devices="cuda:0,cuda:1,cuda:2,cuda:3"`` — multi-GPU parallelism

Example: 4 GPUs × batch 4 = 16 prompts in flight simultaneously.

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
    sampler: Any
    sampler_config_cls: Any
    device: str


class DiffusionEngine(InferenceEngine):
    """Inference engine for diffusion language models.

    Args:
        model: HuggingFace model name or local path.
        device: Single device (``"auto"``, ``"cuda:0"``).
            Ignored if ``devices`` is set.
        devices: Comma-separated devices for multi-GPU parallelism
            (``"cuda:0,cuda:1,cuda:2,cuda:3"``).
        gpu_batch_size: Number of prompts to process per GPU call.
            Higher values increase GPU utilization.  Set to the largest
            value that fits in GPU memory (try 2, 4, 8).
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
        gpu_batch_size: int = 1,
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
        self._gpu_batch_size = gpu_batch_size
        self._gen_length = gen_length
        self._block_length = block_length
        self._steps = steps
        self._temperature = temperature
        self._apply_chat_template = apply_chat_template
        self._eos_early_stop = eos_early_stop
        self._mc_num = mc_num
        self._use_dllm = use_dllm

        device_list = [d.strip() for d in devices.split(",")] if devices else [device]

        self._batch_supported: bool | None = None if gpu_batch_size > 1 else False
        self._has_dllm = False
        try:
            import dllm.dllm  # noqa: F401

            self._has_dllm = True
        except ImportError:
            pass

        self._workers: list[_Worker] = []
        for dev in device_list:
            self._workers.append(self._create_worker(dev))

        logger.info(
            "DiffusionEngine ready: %d GPU(s) %s, gpu_batch_size=%d, "
            "dllm=%s, gen_length=%d, steps=%d, block_length=%d",
            len(self._workers),
            [w.device for w in self._workers],
            gpu_batch_size,
            self._workers[0].sampler is not None,
            gen_length,
            steps,
            block_length,
        )

    def _create_worker(self, device: str) -> _Worker:
        logger.info("Loading model on %s: %s", device, self._model_path)

        # Try dllm first — uses the same optimised loading as chat.py
        if self._use_dllm:
            try:
                worker = self._create_worker_dllm(device)
                logger.info("Loaded via dllm (optimised path)")
                return worker
            except Exception as exc:
                logger.info(
                    "dllm loading failed (%s), falling back to transformers",
                    exc,
                )

        # Fallback: raw transformers
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
        return _Worker(
            model=model,
            tokenizer=tokenizer,
            sampler=None,
            sampler_config_cls=None,
            device=device,
        )

    def _create_worker_dllm(self, device: str) -> _Worker:
        """Load model + sampler via dllm (same path as chat.py)."""
        from dataclasses import dataclass as _dc

        import dllm.dllm as _dllm

        @_dc
        class _Args:
            model_name_or_path: str = self._model_path

        args = _Args()
        model = _dllm.utils.get_model(model_args=args).eval()
        tokenizer = _dllm.utils.get_tokenizer(model_args=args)
        sampler = _dllm.pipelines.llada2.LLaDA2Sampler(model=model, tokenizer=tokenizer)
        sampler_config_cls = _dllm.pipelines.llada2.LLaDA2SamplerConfig

        actual_device = device
        if device != "auto":
            model = model.to(device)
            actual_device = device
        else:
            actual_device = str(next(model.parameters()).device)

        return _Worker(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            sampler_config_cls=sampler_config_cls,
            device=actual_device,
        )

    # ------------------------------------------------------------------
    # tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize_prompt(self, worker: _Worker, prompt: str) -> Any:
        """Tokenize a prompt.  Uses dllm.utils.build_chat_inputs when
        available (matches chat.py exactly), otherwise falls back to
        the tokenizer's apply_chat_template / raw encode.
        """
        if self._apply_chat_template and self._has_dllm:
            import dllm.dllm as _dllm

            messages = [{"role": "user", "content": prompt}]
            return _dllm.utils.build_chat_inputs(
                worker.tokenizer, [messages], add_generation_prompt=True
            )

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
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        n_workers = len(self._workers)
        if n_workers == 1:
            return self._generate_on_worker(self._workers[0], prompts, gen_kwargs)

        # Multi-GPU: distribute prompts round-robin across workers
        buckets: list[list[tuple[int, str, dict]]] = [[] for _ in range(n_workers)]
        for i, (p, kw) in enumerate(zip(prompts, gen_kwargs)):
            buckets[i % n_workers].append((i, p, kw))

        results: list[str | None] = [None] * len(prompts)

        def _run_bucket(worker_idx: int) -> list[tuple[int, str]]:
            worker = self._workers[worker_idx]
            bucket = buckets[worker_idx]
            idxs = [t[0] for t in bucket]
            ps = [t[1] for t in bucket]
            kws = [t[2] for t in bucket]
            texts = self._generate_on_worker(worker, ps, kws)
            return list(zip(idxs, texts))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_run_bucket, wi): wi
                for wi in range(n_workers)
                if buckets[wi]
            }
            for fut in as_completed(futs):
                for idx, text in fut.result():
                    results[idx] = text

        return [r if r is not None else "" for r in results]

    def _generate_on_worker(
        self,
        worker: _Worker,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        """Process prompts on a single worker in mini-batches."""
        bs = self._gpu_batch_size
        can_batch = (
            bs > 1
            and self._batch_supported is not False
            and worker.sampler is not None
            and self._has_dllm
        )
        results: list[str] = []
        for i in range(0, len(prompts), bs):
            batch_p = prompts[i : i + bs]
            batch_kw = gen_kwargs[i : i + bs]
            if can_batch and len(batch_p) > 1:
                results.extend(self._gen_dllm_batch(worker, batch_p, batch_kw))
            else:
                for p, kw in zip(batch_p, batch_kw):
                    results.append(self._generate_single(worker, p, kw))
        return results

    def _gen_dllm_batch(
        self,
        worker: _Worker,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        """Batch-generate via dllm sampler.

        Tokenizes each prompt individually, left-pads to the same
        length, then calls ``sampler.sample`` on the whole batch.
        """
        import dllm.dllm as _dllm

        torch = self._torch

        # Tokenize each prompt individually
        all_inputs = []
        for p in prompts:
            if self._apply_chat_template:
                inp = _dllm.utils.build_chat_inputs(
                    worker.tokenizer,
                    [[{"role": "user", "content": p}]],
                    add_generation_prompt=True,
                )
            else:
                inp = worker.tokenizer(p, return_tensors="pt").input_ids.to(
                    worker.model.device
                )
            all_inputs.append(inp)

        # Left-pad to the same length
        max_len = max(inp.shape[-1] for inp in all_inputs)
        pad_id = worker.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = worker.tokenizer.eos_token_id or 0

        device = worker.model.device
        padded = torch.full(
            (len(all_inputs), max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for i, inp in enumerate(all_inputs):
            t = inp[0] if inp.dim() > 1 else inp
            padded[i, max_len - t.shape[0] :] = t.to(device)

        kw0 = gen_kwargs[0]
        gen_len = kw0.get("max_gen_toks", kw0.get("max_tokens", self._gen_length))
        temp = kw0.get("temperature", self._temperature)

        cfg = worker.sampler_config_cls(
            max_new_tokens=gen_len,
            block_size=self._block_length,
            steps_per_block=self._steps,
            temperature=temp,
        )

        try:
            outputs = worker.sampler.sample(padded, cfg, return_dict=True)
        except Exception as exc:
            self._batch_supported = False
            logger.warning(
                "Batch generation failed (%s). Falling back to "
                "single-prompt mode for remaining requests.",
                exc,
            )
            return [
                self._generate_single(worker, p, kw)
                for p, kw in zip(prompts, gen_kwargs)
            ]

        self._batch_supported = True

        replies = _dllm.utils.sample_trim(
            worker.tokenizer,
            outputs.sequences.tolist(),
            padded,
        )

        results: list[str] = []
        for i, reply in enumerate(replies):
            stop = gen_kwargs[i].get("until", gen_kwargs[i].get("stop", []))
            if isinstance(stop, str):
                stop = [stop]
            if stop:
                reply = _truncate_at_stop(reply, stop)
            results.append(reply)
        return results

    def _generate_single(
        self,
        worker: _Worker,
        prompt: str,
        kw: dict[str, Any],
    ) -> str:
        gen_len = kw.get("max_gen_toks", kw.get("max_tokens", self._gen_length))
        temp = kw.get("temperature", self._temperature)

        if worker.sampler is not None and self._has_dllm:
            text = self._gen_dllm(worker, prompt, gen_len, temp)
        else:
            input_ids = self._tokenize_prompt(worker, prompt)
            input_len = input_ids.shape[1]
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
        prompt: str,
        gen_length: int,
        temperature: float,
    ) -> str:
        """Generate using dllm — matches chat.py flow exactly."""
        import dllm.dllm as _dllm

        if self._apply_chat_template:
            messages = [{"role": "user", "content": prompt}]
            inputs = _dllm.utils.build_chat_inputs(
                worker.tokenizer,
                [messages],
                add_generation_prompt=True,
            )
        else:
            inputs = worker.tokenizer(prompt, return_tensors="pt").input_ids.to(
                worker.model.device
            )

        cfg = worker.sampler_config_cls(
            max_new_tokens=gen_length,
            block_size=self._block_length,
            steps_per_block=self._steps,
            temperature=temperature,
        )
        outputs = worker.sampler.sample(inputs, cfg, return_dict=True)
        reply = _dllm.utils.sample_trim(
            worker.tokenizer,
            outputs.sequences.tolist(),
            inputs,
        )[0]
        return reply

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
    # loglikelihood
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihoods using the model's diffusion scorer."""
        if not contexts:
            return []

        if not hasattr(self._workers[0].model, "get_log_likelihood"):
            raise NotImplementedError(
                "This model does not expose get_log_likelihood(). "
                "Make sure you are using a LLaDA / LLaDA2 model "
                "loaded with trust_remote_code=True, or install dllm."
            )

        n_workers = len(self._workers)
        if n_workers == 1:
            return [
                self._ll_single(self._workers[0], ctx, cont)
                for ctx, cont in zip(contexts, continuations)
            ]

        results: list[tuple[float, bool] | None] = [None] * len(contexts)

        def _work(idx: int) -> tuple[int, tuple[float, bool]]:
            worker = self._workers[idx % n_workers]
            return idx, self._ll_single(worker, contexts[idx], continuations[idx])

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_work, i): i for i in range(len(contexts))}
            for fut in as_completed(futs):
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
