"""SGLang inference engine backend."""

from __future__ import annotations

import logging
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    """Inference engine using SGLang for fast inference.

    Can operate in two modes:
    - Local: launches SGLang runtime directly
    - API: connects to a running SGLang server

    Args:
        model: Model name or path (for local mode).
        base_url: SGLang server URL (for API mode, e.g. http://localhost:30000).
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        extra_kwargs: Additional kwargs for SGLang runtime.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        tensor_parallel_size: int = 1,
        **extra_kwargs: Any,
    ) -> None:
        self._use_api = base_url is not None
        self._base_url = base_url

        if self._use_api:
            try:
                import requests as _requests  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "requests library is required for SGLang API mode."
                ) from e
            self._requests = _requests
            logger.info("SGLang API mode: %s", base_url)
        else:
            if model is None:
                raise ValueError("Either model or base_url must be provided")
            try:
                import sglang as sgl  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "sglang is required for SGLangEngine local mode. "
                    "Install it with: pip install 'inference-eval[sglang]'"
                ) from e
            self._sgl = sgl
            logger.info(
                "SGLang local mode: model=%s, tp=%d", model, tensor_parallel_size
            )
            self.runtime = sgl.Runtime(
                model_path=model,
                tp_size=tensor_parallel_size,
                **extra_kwargs,
            )
            sgl.set_default_backend(self.runtime)

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if self._use_api:
            return self._generate_api(prompts, gen_kwargs)
        return self._generate_local(prompts, gen_kwargs)

    def _generate_api(
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

            resp = self._requests.post(
                f"{self._base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "stop": stop,
                        "temperature": kwargs.get("temperature", 0.0),
                    },
                },
            )
            resp.raise_for_status()
            results.append(resp.json()["text"])
        return results

    def _generate_local(
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

            output = self._sgl.gen(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
                temperature=kwargs.get("temperature", 0.0),
            )
            results.append(output)
        return results

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "Log-likelihood computation is not yet supported in SGLangEngine. "
            "Use VLLMEngine or a custom engine for log-likelihood tasks."
        )
