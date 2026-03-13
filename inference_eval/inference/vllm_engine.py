"""vLLM inference engine backend (local, using vllm.LLM)."""

from __future__ import annotations

import json
import logging
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class VLLMEngine(InferenceEngine):
    """Inference engine using vLLM for high-throughput *local* inference.

    Uses ``vllm.LLM`` to load the model onto local GPUs and process
    all prompts in a single batched call, which is dramatically faster
    than one-by-one processing.

    Args:
        model: Model name or path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Data type (auto, float16, bfloat16, float32).
        max_model_len: Maximum model context length.
        gpu_memory_utilization: Fraction of GPU memory to use.
        extra_kwargs: Additional kwargs passed to ``vllm.LLM()``.
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
            "Initializing vLLM with model=%s, tp=%d",
            model,
            tensor_parallel_size,
        )
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **extra_kwargs,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kwargs_key(kwargs: dict[str, Any]) -> str:
        """Deterministic string key for a gen-kwargs dict."""
        return json.dumps(kwargs, sort_keys=True, default=str)

    def _make_sampling_params(
        self, kwargs: dict[str, Any], *, for_logprobs: bool = False
    ) -> Any:
        stop = kwargs.get("until", kwargs.get("stop", []))
        if isinstance(stop, str):
            stop = [stop]
        if for_logprobs:
            return self._SamplingParams(
                max_tokens=1,
                prompt_logprobs=0,
                temperature=0.0,
            )
        return self._SamplingParams(
            max_tokens=kwargs.get("max_gen_toks", kwargs.get("max_tokens", 256)),
            stop=stop if stop else None,
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
        )

    # ------------------------------------------------------------------
    # generate  (batched, quiet — progress is tracked by run_inference)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        groups: dict[str, tuple[list[int], list[str], dict]] = {}
        for idx, (prompt, kw) in enumerate(zip(prompts, gen_kwargs)):
            key = self._kwargs_key(kw)
            if key not in groups:
                groups[key] = ([], [], kw)
            groups[key][0].append(idx)
            groups[key][1].append(prompt)

        results: list[str | None] = [None] * len(prompts)

        for _key, (indices, group_prompts, kw) in groups.items():
            params = self._make_sampling_params(kw)
            outputs = self.llm.generate(group_prompts, params, use_tqdm=False)
            for orig_idx, out in zip(indices, outputs):
                results[orig_idx] = out.outputs[0].text

        return [r if r is not None else "" for r in results]

    # ------------------------------------------------------------------
    # loglikelihood  (batched, quiet)
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        if not contexts:
            return []

        tokenizer = self.llm.get_tokenizer()
        full_texts = [ctx + cont for ctx, cont in zip(contexts, continuations)]
        context_tok_lens = [len(tokenizer.encode(ctx)) for ctx in contexts]

        params = self._make_sampling_params({}, for_logprobs=True)
        outputs = self.llm.generate(full_texts, params, use_tqdm=False)

        results: list[tuple[float, bool]] = []
        for output, ctx_len in zip(outputs, context_tok_lens):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                results.append((0.0, False))
                continue

            total_ll = 0.0
            is_greedy = True
            for i in range(ctx_len, len(prompt_logprobs)):
                token_lps = prompt_logprobs[i]
                if token_lps is None:
                    continue
                if not token_lps:
                    continue
                top_id = max(token_lps, key=lambda k: token_lps[k].logprob)
                actual_id = next(iter(token_lps))
                total_ll += token_lps[actual_id].logprob
                if actual_id != top_id:
                    is_greedy = False

            results.append((total_ll, is_greedy))

        return results
