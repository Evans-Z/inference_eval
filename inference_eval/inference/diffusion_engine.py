"""Diffusion language model engine (LLaDA2, Dream, etc.).

Supports both ``generate_until`` (text generation) and ``loglikelihood``
(scoring) tasks.  Uses the `dllm <https://github.com/ZHZisZZ/dllm>`_
framework when available for optimised sampling, with a pure-transformers
fallback.

Prompts from lm-eval-harness are raw text.  By default this engine
wraps each prompt in a chat template (``apply_chat_template=True``).
**Do NOT combine this with ``inference-eval extract --apply-chat-template``**
or the prompt will be double-wrapped.
"""

from __future__ import annotations

import logging
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


class DiffusionEngine(InferenceEngine):
    """Inference engine for diffusion language models.

    Loads the model via ``transformers`` (with ``trust_remote_code``)
    and optionally wraps it with the **dllm** sampler for faster
    generation.  Also supports ``loglikelihood`` via the model's
    ``get_log_likelihood()`` method.

    Args:
        model: HuggingFace model name or local path.
        device: Device or device-map string (``"auto"``, ``"cuda:0"``, …).
        dtype: Torch dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        gen_length: Default number of new tokens to generate.
        block_length: Block length for diffusion sampling.
        steps: Number of demasking steps per block.
        temperature: Sampling temperature (0 = greedy).
        apply_chat_template: Wrap each prompt as a chat user message.
        eos_early_stop: Stop generation early on EOS.
        mc_num: Monte-Carlo samples for ``get_log_likelihood`` (higher
            is more accurate but slower; 32–128 is typical).
        use_dllm: Try to use the ``dllm`` framework for generation.
            Falls back to ``model.generate()`` if not installed.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
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

        logger.info("Loading model: %s (device=%s, dtype=%s)", model, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=True, device_map=device
            )
            .to(torch_dtype)
            .eval()
        )

        self._gen_length = gen_length
        self._block_length = block_length
        self._steps = steps
        self._temperature = temperature
        self._apply_chat_template = apply_chat_template
        self._eos_early_stop = eos_early_stop
        self._mc_num = mc_num

        self._sampler = None
        self._sampler_config_cls = None
        if use_dllm:
            self._init_dllm(model)

        has_ll = hasattr(self._model, "get_log_likelihood")
        logger.info(
            "DiffusionEngine ready: dllm=%s, loglikelihood=%s, gen_length=%d, "
            "steps=%d, block_length=%d",
            self._sampler is not None,
            has_ll,
            gen_length,
            steps,
            block_length,
        )

    def _init_dllm(self, model_path: str) -> None:
        """Try to initialise the dllm sampler for faster generation."""
        try:
            import dllm  # noqa: F401

            self._sampler = dllm.pipelines.llada2.LLaDA2Sampler(
                model=self._model, tokenizer=self.tokenizer
            )
            self._sampler_config_cls = dllm.pipelines.llada2.LLaDA2SamplerConfig
            logger.info("dllm sampler loaded")
        except Exception as exc:
            logger.info("dllm not available (%s), using model.generate()", exc)

    # ------------------------------------------------------------------
    # tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize_prompt(self, prompt: str) -> Any:
        if self._apply_chat_template:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return ids.to(self._model.device)

    # ------------------------------------------------------------------
    # generate  (sequential on GPU — each prompt uses full GPU memory)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        results: list[str] = []
        for prompt, kw in zip(prompts, gen_kwargs):
            input_ids = self._tokenize_prompt(prompt)
            input_len = input_ids.shape[1]

            gen_len = kw.get("max_gen_toks", kw.get("max_tokens", self._gen_length))
            temp = kw.get("temperature", self._temperature)

            text = self._generate_one(input_ids, input_len, gen_len, temp)

            stop = kw.get("until", kw.get("stop", []))
            if isinstance(stop, str):
                stop = [stop]
            if stop:
                text = _truncate_at_stop(text, stop)

            results.append(text)
        return results

    def _generate_one(
        self,
        input_ids: Any,
        input_len: int,
        gen_length: int,
        temperature: float,
    ) -> str:
        """Generate from a single tokenised prompt."""
        if self._sampler is not None:
            return self._generate_dllm(input_ids, input_len, gen_length, temperature)
        return self._generate_transformers(
            input_ids, input_len, gen_length, temperature
        )

    def _generate_dllm(
        self,
        input_ids: Any,
        input_len: int,
        gen_length: int,
        temperature: float,
    ) -> str:
        cfg = self._sampler_config_cls(
            max_new_tokens=gen_length,
            block_size=self._block_length,
            steps_per_block=self._steps,
            temperature=temperature,
        )
        output = self._sampler.sample(input_ids, cfg)
        if hasattr(output, "sequences"):
            out_ids = output.sequences[0]
        elif self._torch.is_tensor(output):
            out_ids = output[0]
        else:
            out_ids = output
        new_ids = out_ids[input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _generate_transformers(
        self,
        input_ids: Any,
        input_len: int,
        gen_length: int,
        temperature: float,
    ) -> str:
        with self._torch.no_grad():
            output_ids = self._model.generate(
                inputs=input_ids,
                gen_length=gen_length,
                block_length=self._block_length,
                steps=self._steps,
                temperature=temperature,
                eos_early_stop=self._eos_early_stop,
            )
        new_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # loglikelihood  (via model.get_log_likelihood)
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihoods using the model's diffusion-based scorer.

        Diffusion LMs estimate token likelihoods via Monte-Carlo
        sampling of masking patterns.  The ``mc_num`` parameter
        controls accuracy vs speed.

        Requires the model to expose ``get_log_likelihood()``
        (available in LLaDA / LLaDA2 via ``trust_remote_code``).
        """
        if not contexts:
            return []

        if not hasattr(self._model, "get_log_likelihood"):
            raise NotImplementedError(
                "This model does not expose get_log_likelihood(). "
                "Make sure you are using a LLaDA / LLaDA2 model loaded "
                "with trust_remote_code=True, or install dllm."
            )

        results: list[tuple[float, bool]] = []
        for ctx, cont in zip(contexts, continuations):
            ll = self._compute_ll_one(ctx, cont)
            results.append((ll, True))
        return results

    def _compute_ll_one(self, context: str, continuation: str) -> float:
        """Compute log-likelihood for a single (context, continuation) pair."""
        ctx_ids = self.tokenizer(context, return_tensors="pt").input_ids
        full_ids = self.tokenizer(context + continuation, return_tensors="pt").input_ids
        ctx_ids = ctx_ids.to(self._model.device)
        full_ids = full_ids.to(self._model.device)

        ctx_len = ctx_ids.shape[1]
        target_ids = full_ids.clone()
        target_ids[:, :ctx_len] = -100

        with self._torch.no_grad():
            ll = self._model.get_log_likelihood(
                full_ids, target_ids, mc_num=self._mc_num
            )

        if self._torch.is_tensor(ll):
            return ll.item()
        return float(ll)
