"""OpenAI-compatible API inference engine backend.

Uses the ``openai`` Python client.  For a lighter-weight alternative
that needs no extra dependency, see :class:`ServerEngine`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class OpenAIEngine(InferenceEngine):
    """Inference engine using the ``openai`` Python client.

    Works with OpenAI, Azure OpenAI, vLLM server, SGLang server,
    and other OpenAI-compatible endpoints.

    Args:
        model: Model name (e.g. "gpt-4", "meta-llama/Llama-3-8B-Instruct").
        base_url: API base URL. Defaults to OpenAI's endpoint.
        api_key: API key. Defaults to OPENAI_API_KEY env var.
        max_retries: Number of retries for failed requests.
        max_concurrent: Max parallel requests (default 32).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        max_concurrent: int = 32,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEngine. "
                "Install it with: pip install 'inference-eval[openai]'"
            ) from e

        self.model = model
        self.max_concurrent = max_concurrent
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            max_retries=max_retries,
        )
        logger.info(
            "OpenAI engine: model=%s, base_url=%s, concurrency=%d",
            model,
            base_url,
            max_concurrent,
        )

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        results: list[str | None] = [None] * len(prompts)

        def _one(idx: int) -> tuple[int, str]:
            kwargs = gen_kwargs[idx]
            stop = kwargs.get("until", kwargs.get("stop", None))
            if isinstance(stop, str):
                stop = [stop]
            max_tokens = kwargs.get("max_gen_toks", kwargs.get("max_tokens", 256))
            response = self.client.completions.create(
                model=self.model,
                prompt=prompts[idx],
                max_tokens=max_tokens,
                stop=stop,
                temperature=kwargs.get("temperature", 0.0),
            )
            return idx, response.choices[0].text

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(_one, i): i for i in range(len(prompts))}
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        return [r if r is not None else "" for r in results]

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        if not contexts:
            return []

        results: list[tuple[float, bool] | None] = [None] * len(contexts)

        def _one(idx: int) -> tuple[int, tuple[float, bool]]:
            context = contexts[idx]
            continuation = continuations[idx]
            full_text = context + continuation
            response = self.client.completions.create(
                model=self.model,
                prompt=full_text,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            choice = response.choices[0]

            if not (choice.logprobs and choice.logprobs.token_logprobs):
                return idx, (0.0, False)

            tokens = choice.logprobs.tokens or []
            ctx_len = len(context)
            ctx_tokens = 0
            accumulated = ""
            for t in tokens:
                accumulated += t
                ctx_tokens += 1
                if len(accumulated) >= ctx_len:
                    break

            cont_lps = choice.logprobs.token_logprobs[ctx_tokens:]
            total_ll = sum(lp for lp in cont_lps if lp is not None)

            top_lps = choice.logprobs.top_logprobs
            is_greedy = True
            if top_lps:
                for i in range(ctx_tokens, len(choice.logprobs.token_logprobs)):
                    if i < len(top_lps) and top_lps[i]:
                        best = max(top_lps[i].values())
                        actual = choice.logprobs.token_logprobs[i]
                        if actual is not None and abs(actual - best) > 1e-6:
                            is_greedy = False

            return idx, (total_ll, is_greedy)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(_one, i): i for i in range(len(contexts))}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return [r if r is not None else (0.0, False) for r in results]
