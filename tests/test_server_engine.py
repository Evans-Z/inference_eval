"""Tests for ServerEngine using mock HTTP servers."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from inference_eval.inference.server_engine import ServerEngine

_SERVED_MODEL = "test-model"


# ------------------------------------------------------------------
# Handler helpers
# ------------------------------------------------------------------


class _BaseHandler(BaseHTTPRequestHandler):
    def _respond_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _check_model(self, body: dict) -> bool:
        """Return True if model name is valid, else send 404."""
        model = body.get("model", "")
        if model != _SERVED_MODEL:
            self._respond_json(
                {
                    "error": {
                        "message": f"The model `{model}` does not exist.",
                        "type": "NotFoundError",
                        "param": "model",
                        "code": 404,
                    }
                },
                status=404,
            )
            return False
        return True

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ------------------------------------------------------------------
# Mock: supports BOTH endpoints + /v1/models, validates model name
# ------------------------------------------------------------------


class _MockBothHandler(_BaseHandler):
    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._respond_json({"data": [{"id": _SERVED_MODEL, "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_body()
        if self.path.rstrip("/").endswith("/chat/completions"):
            if not self._check_model(body):
                return
            messages = body.get("messages", [])
            content = messages[-1]["content"] if messages else ""
            self._respond_json(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"chat_answer_{len(content)}",
                            }
                        }
                    ]
                }
            )
        elif self.path.rstrip("/").endswith("/completions"):
            if not self._check_model(body):
                return
            echo = body.get("echo", False)
            prompt = body.get("prompt", "")
            if echo:
                self._respond_json(
                    {
                        "choices": [
                            {
                                "text": prompt,
                                "logprobs": {
                                    "tokens": list(prompt),
                                    "token_logprobs": [
                                        None if i == 0 else -0.5
                                        for i in range(len(prompt))
                                    ],
                                    "top_logprobs": [
                                        None if i == 0 else {prompt[i]: -0.5}
                                        for i in range(len(prompt))
                                    ],
                                },
                            }
                        ]
                    }
                )
            else:
                self._respond_json(
                    {"choices": [{"text": f"completions_answer_{len(prompt)}"}]}
                )
        else:
            self.send_error(404)


# ------------------------------------------------------------------
# Mock: ONLY chat endpoint + /v1/models, no /v1/completions route
# ------------------------------------------------------------------


class _MockChatOnlyHandler(_BaseHandler):
    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._respond_json({"data": [{"id": _SERVED_MODEL, "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        body = self._read_body()
        if self.path.rstrip("/").endswith("/chat/completions"):
            if not self._check_model(body):
                return
            messages = body.get("messages", [])
            content = messages[-1]["content"] if messages else ""
            self._respond_json(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"chat_only_{len(content)}",
                            }
                        }
                    ]
                }
            )
        else:
            self.send_error(404)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _start_server(handler_cls: type) -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def mock_server():
    server, url = _start_server(_MockBothHandler)
    yield url
    server.shutdown()


@pytest.fixture(scope="module")
def chat_only_server():
    server, url = _start_server(_MockChatOnlyHandler)
    yield url
    server.shutdown()


# ------------------------------------------------------------------
# Tests — completions mode
# ------------------------------------------------------------------


class TestServerEngine:
    def test_generate_single(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            max_concurrent=4,
            api_type="completions",
        )
        results = engine.generate(
            ["What is 2+2?"],
            [{"until": ["\n"], "max_gen_toks": 32}],
        )
        assert len(results) == 1
        assert results[0].startswith("completions_answer_")

    def test_generate_batch(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            max_concurrent=8,
            api_type="completions",
        )
        prompts = [f"prompt_{i}" for i in range(20)]
        gen_kwargs = [{"until": ["\n"], "max_gen_toks": 32}] * 20
        results = engine.generate(prompts, gen_kwargs)
        assert len(results) == 20
        assert all(r.startswith("completions_answer_") for r in results)

    def test_generate_empty(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
        )
        assert engine.generate([], []) == []

    def test_loglikelihood(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            max_concurrent=4,
            api_type="completions",
        )
        results = engine.compute_loglikelihood(["The cat sat"], [" on the mat"])
        assert len(results) == 1
        ll, is_greedy = results[0]
        assert isinstance(ll, float)
        assert isinstance(is_greedy, bool)

    def test_loglikelihood_batch(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            max_concurrent=8,
            api_type="completions",
        )
        contexts = [f"context_{i}" for i in range(10)]
        continuations = [f" cont_{i}" for i in range(10)]
        results = engine.compute_loglikelihood(contexts, continuations)
        assert len(results) == 10
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_base_url_normalization(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server + "/v1",
        )
        assert engine.base_url == mock_server + "/v1"

        engine2 = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
        )
        assert engine2.base_url == mock_server + "/v1"

    def test_process_requests_integration(self, mock_server: str):
        from inference_eval.schema import InferenceRequest

        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            max_concurrent=4,
            api_type="completions",
        )
        reqs = [
            InferenceRequest(
                task_name="gsm8k",
                request_type="generate_until",
                doc_id=i,
                index=i,
                context=f"What is {i}+{i}?",
                generation_kwargs={"until": ["\n"], "max_gen_toks": 64},
            )
            for i in range(5)
        ]
        results = engine.process_requests(reqs)
        assert len(results) == 5
        assert all(r.generated_text is not None for r in results)


# ------------------------------------------------------------------
# Tests — chat mode
# ------------------------------------------------------------------


class TestServerEngineChatMode:
    def test_explicit_chat_mode(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            api_type="chat",
        )
        results = engine.generate(
            ["What is 2+2?"],
            [{"until": ["\n"], "max_gen_toks": 32}],
        )
        assert len(results) == 1
        assert results[0].startswith("chat_answer_")

    def test_chat_batch(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            max_concurrent=8,
            api_type="chat",
        )
        prompts = [f"prompt_{i}" for i in range(10)]
        gen_kwargs = [{"until": ["\n"]}] * 10
        results = engine.generate(prompts, gen_kwargs)
        assert len(results) == 10
        assert all(r.startswith("chat_answer_") for r in results)


# ------------------------------------------------------------------
# Tests — auto-detect
# ------------------------------------------------------------------


class TestServerEngineAutoDetect:
    def test_auto_detects_completions(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
            api_type="auto",
        )
        results = engine.generate(
            ["hello"],
            [{"max_gen_toks": 16}],
        )
        assert engine._resolved_type == "completions"
        assert results[0].startswith("completions_answer_")

    def test_auto_falls_back_to_chat(self, chat_only_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=chat_only_server,
            api_type="auto",
        )
        results = engine.generate(
            ["hello world"],
            [{"max_gen_toks": 16}],
        )
        assert engine._resolved_type == "chat"
        assert results[0].startswith("chat_only_")

    def test_auto_chat_batch(self, chat_only_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=chat_only_server,
            api_type="auto",
            max_concurrent=8,
        )
        prompts = [f"prompt_{i}" for i in range(10)]
        gen_kwargs = [{"until": ["\n"]}] * 10
        results = engine.generate(prompts, gen_kwargs)
        assert len(results) == 10
        assert all(r.startswith("chat_only_") for r in results)


# ------------------------------------------------------------------
# Tests — wrong model name (must fail fast)
# ------------------------------------------------------------------


class TestServerEngineModelValidation:
    def test_wrong_model_name_fails_at_init(self, mock_server: str):
        with pytest.raises(ValueError, match="not found on the server"):
            ServerEngine(
                model="/home/data/Qwen3-8B",
                base_url=mock_server,
            )

    def test_wrong_model_shows_available(self, mock_server: str):
        with pytest.raises(ValueError, match=_SERVED_MODEL):
            ServerEngine(
                model="wrong-name",
                base_url=mock_server,
            )

    def test_correct_model_works(self, mock_server: str):
        engine = ServerEngine(
            model=_SERVED_MODEL,
            base_url=mock_server,
        )
        assert engine.model == _SERVED_MODEL


class TestServerEngineInvalidApiType:
    def test_invalid_api_type(self, mock_server: str):
        with pytest.raises(ValueError, match="api_type must be"):
            ServerEngine(
                model=_SERVED_MODEL,
                base_url=mock_server,
                api_type="invalid",
            )
