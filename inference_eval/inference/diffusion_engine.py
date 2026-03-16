"""Diffusion language model engine (e.g. LLaDA2).

Uses the ``transformers`` API with ``model.generate()`` for diffusion-based
text generation.  These models use iterative demasking rather than
autoregressive decoding, so loglikelihood is not supported — only
``generate_until`` tasks (gsm8k, etc.).

Prompts from lm-eval-harness are raw text.  By default this engine
wraps each prompt in a chat template before generation
(``apply_chat_template=True``).  **Do NOT combine this with
``inference-eval extract --apply-chat-template``** or the prompt
will be double-wrapped.
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
    """Inference engine for diffusion language models (LLaDA2, etc.).

    Loads the model via ``transformers.AutoModelForCausalLM`` with
    ``trust_remote_code=True`` and calls the model-specific
    ``model.generate()`` method.

    Args:
        model: HuggingFace model name or local path.
        device: Device or device-map string (``"auto"``, ``"cuda:0"``, …).
        dtype: Torch dtype string (``"bfloat16"``, ``"float16"``, ``"float32"``).
        gen_length: Default number of new tokens to generate.
        block_length: Block length for diffusion sampling.
        steps: Number of demasking steps per block.
        temperature: Sampling temperature (0 = greedy).
        apply_chat_template: Wrap each prompt as a chat-user message
            before generation.  Set ``False`` if prompts are already
            in chat format (e.g. extracted with ``--apply-chat-template``).
        eos_early_stop: Stop generation early when EOS is produced.
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
        **extra_kwargs: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for DiffusionEngine. "
                "Install them with: pip install torch transformers"
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info(
            "Loading diffusion model: %s (device=%s, dtype=%s)",
            model,
            device,
            dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, device_map=device
        )
        self._model = self._model.to(torch_dtype).eval()

        self._gen_length = gen_length
        self._block_length = block_length
        self._steps = steps
        self._temperature = temperature
        self._apply_chat_template = apply_chat_template
        self._eos_early_stop = eos_early_stop
        self._torch = torch

        logger.info(
            "DiffusionEngine ready: gen_length=%d, block_length=%d, "
            "steps=%d, temperature=%.2f, chat_template=%s",
            gen_length,
            block_length,
            steps,
            temperature,
            apply_chat_template,
        )

    def _tokenize(self, prompt: str) -> Any:
        """Tokenize a prompt, optionally applying a chat template."""
        if self._apply_chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self._model.device)

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        results: list[str] = []
        for prompt, kw in zip(prompts, gen_kwargs):
            input_ids = self._tokenize(prompt)
            input_len = input_ids.shape[1]

            gen_length = kw.get("max_gen_toks", kw.get("max_tokens", self._gen_length))
            temperature = kw.get("temperature", self._temperature)

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
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            stop = kw.get("until", kw.get("stop", []))
            if isinstance(stop, str):
                stop = [stop]
            if stop:
                text = _truncate_at_stop(text, stop)

            results.append(text)

        return results

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "Diffusion language models (LLaDA2, etc.) do not support "
            "loglikelihood computation.  Use generate_until tasks only "
            "(e.g. gsm8k, arc_challenge with generation config)."
        )
