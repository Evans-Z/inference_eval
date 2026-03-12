"""OpenAI-compatible API inference engine backend."""

from __future__ import annotations

import logging
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class OpenAIEngine(InferenceEngine):
    """Inference engine using any OpenAI-compatible API.

    Works with OpenAI, Azure OpenAI, vLLM server, SGLang server,
    and other OpenAI-compatible endpoints.

    Args:
        model: Model name (e.g. "gpt-4", "meta-llama/Llama-3-8B-Instruct").
        base_url: API base URL. Defaults to OpenAI's endpoint.
        api_key: API key. Defaults to OPENAI_API_KEY env var.
        max_retries: Number of retries for failed requests.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEngine. "
                "Install it with: pip install 'inference-eval[openai]'"
            ) from e

        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            max_retries=max_retries,
        )
        logger.info("OpenAI engine: model=%s, base_url=%s", model, base_url)

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        results = []
        for prompt, kwargs in zip(prompts, gen_kwargs):
            stop = kwargs.get("until", kwargs.get("stop", None))
            if isinstance(stop, str):
                stop = [stop]
            max_tokens = kwargs.get("max_gen_toks", kwargs.get("max_tokens", 256))

            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                stop=stop,
                temperature=kwargs.get("temperature", 0.0),
            )
            results.append(response.choices[0].text)
        return results

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in zip(contexts, continuations):
            full_text = context + continuation
            response = self.client.completions.create(
                model=self.model,
                prompt=full_text,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            choice = response.choices[0]

            if choice.logprobs and choice.logprobs.token_logprobs:
                context_tokens = len(
                    self.client.completions.create(
                        model=self.model,
                        prompt=context,
                        max_tokens=0,
                        echo=True,
                    )
                    .choices[0]
                    .logprobs.tokens
                    or []
                )
                cont_logprobs = choice.logprobs.token_logprobs[context_tokens:]
                total_ll = sum(lp for lp in cont_logprobs if lp is not None)
                top_logprobs = choice.logprobs.top_logprobs
                is_greedy = all(
                    top_logprobs[i]
                    and max(top_logprobs[i].values())
                    == choice.logprobs.token_logprobs[i]
                    for i in range(context_tokens, len(choice.logprobs.token_logprobs))
                    if top_logprobs and i < len(top_logprobs)
                )
                results.append((total_ll, is_greedy))
            else:
                logger.warning("No logprobs returned for context, returning defaults")
                results.append((0.0, False))

        return results
