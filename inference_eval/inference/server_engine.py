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

    Works with any server that exposes ``/v1/completions`` **or**
    ``/v1/chat/completions`` (OpenAI-compatible API), including:

    * ``vllm serve <model> --port 8068``
    * ``python -m sglang.launch_server --port 8068``
    * Any OpenAI-compatible endpoint

    **Important**: ``model`` must be the *served* model name as the
    server knows it — not the local file path.  Run
    ``curl http://localhost:<port>/v1/models`` to see available names.

    Args:
        model: Served model name (e.g. ``"qwen3"``).  This is the name
            returned by ``/v1/models``, which may differ from the local
            path used to start the server.
        base_url: Root URL of the server's OpenAI-compatible API,
            e.g. ``http://localhost:8068/v1``.  A trailing ``/v1``
            is appended automatically if absent.
        api_key: Bearer token.  Defaults to ``"EMPTY"`` (vLLM default).
        max_concurrent: Maximum number of in-flight HTTP requests.
        timeout: Per-request timeout in seconds.
        api_type: Which endpoint to use.

            * ``"auto"`` (default) — tries ``/v1/completions`` first;
              if the server returns a *route-level* 404 falls back to
              ``/v1/chat/completions`` for the rest of the session.
              A *model-not-found* 404 is treated as an error (not a
              fallback trigger).
            * ``"completions"`` — always use ``/v1/completions``.
            * ``"chat"`` — always use ``/v1/chat/completions``.
              Prompts are sent as a single ``user`` message.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 64,
        timeout: int = _DEFAULT_TIMEOUT,
        api_type: str = "auto",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        if api_type not in ("auto", "completions", "chat"):
            raise ValueError(
                f"api_type must be 'auto', 'completions', or 'chat', got '{api_type}'"
            )
        self._api_type: str = api_type
        self._resolved_type: str | None = None if api_type == "auto" else api_type

        self._session = http_requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        # Validate model name against server early so the user gets a
        # clear error before processing thousands of requests.
        self._validate_model_name()

        logger.info(
            "ServerEngine: model=%s  url=%s  concurrency=%d  api_type=%s",
            self.model,
            self.base_url,
            max_concurrent,
            api_type,
        )

    # ------------------------------------------------------------------
    # model-name validation
    # ------------------------------------------------------------------

    def _list_models(self) -> list[str]:
        """Query ``GET /v1/models`` and return available model ids."""
        url = f"{self.base_url}/models"
        resp = self._session.get(url, timeout=min(self.timeout, 15))
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def _validate_model_name(self) -> None:
        """Check model name against /v1/models; fail fast if wrong."""
        try:
            available = self._list_models()
        except Exception:
            logger.debug(
                "Could not query /v1/models for validation (server "
                "may not support it) — skipping model-name check."
            )
            return

        if not available:
            return

        if self.model in available:
            logger.info("Model '%s' confirmed available on server.", self.model)
            return

        raise ValueError(
            f"Model '{self.model}' was not found on the server at "
            f"{self.base_url}.\n"
            f"Available model(s): {available}\n"
            f"Hint: --model should be the *served* model name "
            f"(as shown by /v1/models), not the local file path.\n"
            f"Try:  --model {available[0]}"
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

    @staticmethod
    def _is_model_not_found(resp: http_requests.Response) -> bool:
        """Return True if a 404 response is a model-name error.

        vLLM returns JSON like::

            {"message": "The model `X` does not exist.",
             "type": "NotFoundError", "param": "model", "code": 404}

        A *route-level* 404 typically returns HTML or an empty body.
        """
        try:
            body = resp.json()
        except Exception:
            return False

        # vLLM / OpenAI nested {"error": {...}} format
        err = body.get("error", body)
        if isinstance(err, dict):
            if err.get("param") == "model":
                return True
            msg = str(err.get("message", ""))
            if "does not exist" in msg.lower():
                return True

        msg_top = str(body.get("message", ""))
        if "does not exist" in msg_top.lower():
            return True

        return False

    def _resolve_api_type(self) -> str:
        """Auto-detect whether the server supports completions or chat.

        Sends a tiny probe to ``/v1/completions``.  If the 404 is due
        to the *endpoint* not existing we fall back to chat.  If it is
        due to a wrong *model name* we raise immediately — falling back
        would hit the same error.
        """
        if self._resolved_type is not None:
            return self._resolved_type

        probe = {
            "model": self.model,
            "prompt": "hi",
            "max_tokens": 1,
            "temperature": 0.0,
        }
        url = f"{self.base_url}/completions"
        try:
            resp = self._session.post(url, json=probe, timeout=min(self.timeout, 30))
            if resp.status_code != 404:
                self._resolved_type = "completions"
                logger.info(
                    "Auto-detected api_type='completions' "
                    "(server supports /v1/completions)"
                )
                return "completions"

            # Got 404 — is it a model-name error or a missing route?
            if self._is_model_not_found(resp):
                available = self._try_list_models_safe()
                raise ValueError(
                    f"Server returned 'model not found' for "
                    f"model='{self.model}'.  The /v1/completions "
                    f"endpoint exists but the model name is wrong.\n"
                    f"Available model(s): {available}\n"
                    f"Hint: --model should be the *served* model name, "
                    f"not the local file path."
                )

        except http_requests.RequestException:
            pass

        self._resolved_type = "chat"
        logger.info(
            "Auto-detected api_type='chat' "
            "(/v1/completions not available, using /v1/chat/completions)"
        )
        return "chat"

    def _try_list_models_safe(self) -> list[str]:
        try:
            return self._list_models()
        except Exception:
            return ["(could not query /v1/models)"]

    # ------------------------------------------------------------------
    # unified request helpers
    # ------------------------------------------------------------------

    def _generate_one_completions(self, prompt: str, kw: dict[str, Any]) -> str:
        """Generate via /v1/completions."""
        stop = kw.get("until", kw.get("stop", []))
        if isinstance(stop, str):
            stop = [stop]
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kw.get("max_gen_toks", kw.get("max_tokens", 256)),
            "temperature": kw.get("temperature", 0.0),
        }
        if stop:
            payload["stop"] = stop
        data = self._post("completions", payload)
        return data["choices"][0]["text"]

    def _generate_one_chat(self, prompt: str, kw: dict[str, Any]) -> str:
        """Generate via /v1/chat/completions."""
        stop = kw.get("until", kw.get("stop", []))
        if isinstance(stop, str):
            stop = [stop]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kw.get("max_gen_toks", kw.get("max_tokens", 256)),
            "temperature": kw.get("temperature", 0.0),
        }
        if stop:
            payload["stop"] = stop
        data = self._post("chat/completions", payload)
        return data["choices"][0]["message"]["content"]

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

        api = self._resolve_api_type()
        results: list[str | None] = [None] * len(prompts)
        t0 = time.perf_counter()

        def _one(idx: int) -> tuple[int, str]:
            kw = gen_kwargs[idx]
            if api == "chat":
                text = self._generate_one_chat(prompts[idx], kw)
            else:
                text = self._generate_one_completions(prompts[idx], kw)
            return idx, text

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

        Note: log-likelihood computation always uses ``/v1/completions``
        because ``/v1/chat/completions`` does not support ``echo``.
        If your server only exposes the chat endpoint, this will raise
        an error for loglikelihood-based tasks (e.g. hellaswag, mmlu).
        """
        if not contexts:
            return []

        results: list[tuple[float, bool] | None] = [None] * len(contexts)
        t0 = time.perf_counter()

        def _one(idx: int) -> tuple[int, tuple[float, bool]]:
            ctx = contexts[idx]
            cont = continuations[idx]
            full = ctx + cont

            payload: dict[str, Any] = {
                "model": self.model,
                "prompt": full,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0.0,
            }
            try:
                data = self._post("completions", payload)
            except http_requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    if self._is_model_not_found(exc.response):
                        available = self._try_list_models_safe()
                        raise ValueError(
                            f"Model '{self.model}' not found. "
                            f"Available: {available}. "
                            f"Use --model with the served model name."
                        ) from exc
                    raise RuntimeError(
                        "Log-likelihood computation requires the "
                        "/v1/completions endpoint with echo+logprobs "
                        "support.  Your server only exposes "
                        "/v1/chat/completions which does not support "
                        "this.  Use a server that exposes "
                        "/v1/completions for loglikelihood-based tasks "
                        "(hellaswag, mmlu, etc.), or switch to a "
                        "generate_until-only task (gsm8k, etc.)."
                    ) from exc
                raise

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
