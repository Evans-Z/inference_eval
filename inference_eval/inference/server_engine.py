"""Server-based inference engine for remote OpenAI-compatible servers.

Connects to a running vLLM, SGLang, or any OpenAI-compatible server
and sends requests concurrently for high throughput.  No extra Python
dependencies beyond ``requests`` (already a transitive dep of lm-eval).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests as http_requests

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600  # seconds per request


class ServerEngine(InferenceEngine):
    """Inference engine that talks to an already-running server.

    Works with any server that exposes ``/v1/completions`` (OpenAI
    completions API), including:

    * ``vllm serve <model> --port 8068``
    * ``python -m sglang.launch_server --port 8068``
    * Any OpenAI-compatible endpoint

    Args:
        model: Model name the server is hosting (passed as ``model``
            field in every request).
        base_url: Root URL of the server's OpenAI-compatible API,
            e.g. ``http://localhost:8068/v1``.  A trailing ``/v1``
            is appended automatically if absent.
        api_key: Bearer token.  Defaults to ``"EMPTY"`` (vLLM default).
        max_concurrent: Maximum number of in-flight HTTP requests.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 64,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._session = http_requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        logger.info(
            "ServerEngine: model=%s  url=%s  concurrency=%d",
            model,
            self.base_url,
            max_concurrent,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        """POST to the server and return the parsed JSON response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _completions(self, payload: dict) -> dict:
        return self._post("completions", payload)

    # ------------------------------------------------------------------
    # generate  (concurrent)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []

        results: list[str | None] = [None] * len(prompts)
        t0 = time.perf_counter()

        def _one(idx: int) -> tuple[int, str]:
            kw = gen_kwargs[idx]
            stop = kw.get("until", kw.get("stop", []))
            if isinstance(stop, str):
                stop = [stop]
            payload = {
                "model": self.model,
                "prompt": prompts[idx],
                "max_tokens": kw.get("max_gen_toks", kw.get("max_tokens", 256)),
                "temperature": kw.get("temperature", 0.0),
            }
            if stop:
                payload["stop"] = stop
            data = self._completions(payload)
            return idx, data["choices"][0]["text"]

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(_one, i): i for i in range(len(prompts))}
            done = 0
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                done += 1
                if done % max(1, len(prompts) // 10) == 0 or done == len(prompts):
                    logger.info("  generate progress: %d/%d", done, len(prompts))

        elapsed = time.perf_counter() - t0
        logger.info(
            "ServerEngine.generate: %d prompts in %.1fs (%.1f req/s)",
            len(prompts),
            elapsed,
            len(prompts) / max(elapsed, 0.001),
        )
        return [r if r is not None else "" for r in results]

    # ------------------------------------------------------------------
    # loglikelihood  (concurrent, requires echo+logprobs support)
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihoods via the completions API.

        Requires the server to support ``echo=true`` and ``logprobs``
        (vLLM and some other servers do; vanilla OpenAI does not).
        """
        if not contexts:
            return []

        results: list[tuple[float, bool] | None] = [None] * len(contexts)
        t0 = time.perf_counter()

        def _one(idx: int) -> tuple[int, tuple[float, bool]]:
            ctx = contexts[idx]
            cont = continuations[idx]
            full = ctx + cont

            payload = {
                "model": self.model,
                "prompt": full,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0.0,
            }
            data = self._completions(payload)
            choice = data["choices"][0]
            lp_data = choice.get("logprobs")
            if not lp_data or not lp_data.get("token_logprobs"):
                return idx, (0.0, False)

            tokens = lp_data["tokens"]
            token_lps = lp_data["token_logprobs"]
            top_lps = lp_data.get("top_logprobs")

            ctx_token_count = self._find_context_boundary(tokens, token_lps, ctx, cont)

            total_ll = 0.0
            is_greedy = True
            for i in range(ctx_token_count, len(token_lps)):
                lp = token_lps[i]
                if lp is None:
                    continue
                total_ll += lp
                if top_lps and i < len(top_lps) and top_lps[i]:
                    best = max(top_lps[i].values())
                    if abs(lp - best) > 1e-6:
                        is_greedy = False

            return idx, (total_ll, is_greedy)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(_one, i): i for i in range(len(contexts))}
            done = 0
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                done += 1
                if done % max(1, len(contexts) // 10) == 0 or done == len(contexts):
                    logger.info(
                        "  loglikelihood progress: %d/%d",
                        done,
                        len(contexts),
                    )

        elapsed = time.perf_counter() - t0
        logger.info(
            "ServerEngine.loglikelihood: %d items in %.1fs (%.1f req/s)",
            len(contexts),
            elapsed,
            len(contexts) / max(elapsed, 0.001),
        )
        return [r if r is not None else (0.0, False) for r in results]

    # ------------------------------------------------------------------

    @staticmethod
    def _find_context_boundary(
        tokens: list[str],
        token_logprobs: list[float | None],
        context: str,
        continuation: str,
    ) -> int:
        """Approximate the boundary between context and continuation tokens.

        Walks through the token list and accumulates text until we reach
        approximately the context length.
        """
        accumulated = ""
        ctx_len = len(context)
        for i, tok in enumerate(tokens):
            accumulated += tok
            if len(accumulated) >= ctx_len:
                return i + 1
        return len(tokens)
