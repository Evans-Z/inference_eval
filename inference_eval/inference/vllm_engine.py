"""vLLM inference engine backend."""

from __future__ import annotations

import logging
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class VLLMEngine(InferenceEngine):
    """Inference engine using vLLM for high-throughput inference.

    Supports both text generation and log-likelihood computation.

    Args:
        model: Model name or path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Data type (auto, float16, bfloat16, float32).
        max_model_len: Maximum model context length.
        gpu_memory_utilization: Fraction of GPU memory to use.
        extra_kwargs: Additional kwargs passed to vllm.LLM().
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        **extra_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vllm is required for VLLMEngine. "
                "Install it with: pip install 'inference-eval[vllm]'"
            ) from e

        self._SamplingParams = SamplingParams
        logger.info(
            "Initializing vLLM with model=%s, tp=%d", model, tensor_parallel_size
        )
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **extra_kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        results = []
        for prompt, kwargs in zip(prompts, gen_kwargs):
            stop = kwargs.get("until", kwargs.get("stop", []))
            if isinstance(stop, str):
                stop = [stop]
            max_tokens = kwargs.get("max_gen_toks", kwargs.get("max_tokens", 256))

            params = self._SamplingParams(
                max_tokens=max_tokens,
                stop=stop,
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", 1.0),
            )
            outputs = self.llm.generate([prompt], params)
            text = outputs[0].outputs[0].text
            results.append(text)
        return results

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in zip(contexts, continuations):
            full_text = context + continuation
            params = self._SamplingParams(
                max_tokens=1,
                prompt_logprobs=0,
                temperature=0.0,
            )
            outputs = self.llm.generate([full_text], params)
            prompt_logprobs = outputs[0].prompt_logprobs

            if prompt_logprobs is None:
                results.append((0.0, False))
                continue

            context_token_count = len(self.llm.get_tokenizer().encode(context))
            total_ll = 0.0
            is_greedy = True
            for i in range(context_token_count, len(prompt_logprobs)):
                if prompt_logprobs[i] is None:
                    continue
                token_logprobs = prompt_logprobs[i]
                if token_logprobs:
                    top_token_id = max(
                        token_logprobs, key=lambda k: token_logprobs[k].logprob
                    )
                    actual_token_ids = list(token_logprobs.keys())
                    if actual_token_ids:
                        actual_id = actual_token_ids[0]
                        total_ll += token_logprobs[actual_id].logprob
                        if actual_id != top_token_id:
                            is_greedy = False

            results.append((total_ll, is_greedy))
        return results
