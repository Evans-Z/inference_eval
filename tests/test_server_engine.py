"""Tests for ServerEngine using a mock HTTP server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from inference_eval.inference.server_engine import ServerEngine

# ------------------------------------------------------------------
# Mock server that supports BOTH /v1/completions and /v1/chat/completions
# ------------------------------------------------------------------


class _MockBothHandler(BaseHTTPRequestHandler):
    """Mock handler supporting both completions and chat completions."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body: dict[str, Any] = json.loads(self.rfile.read(length))

        if self.path.rstrip("/").endswith("/chat/completions"):
            self._handle_chat(body)
        elif self.path.rstrip("/").endswith("/completions"):
            self._handle_completions(body)
        else:
            self.send_error(404)

    def _handle_completions(self, body: dict) -> None:
        echo = body.get("echo", False)
        prompt = body.get("prompt", "")

        if echo:
            response = {
                "choices": [
                    {
                        "text": prompt,
                        "logprobs": {
                            "tokens": list(prompt),
                            "token_logprobs": [
                                None if i == 0 else -0.5 for i in range(len(prompt))
                            ],
                            "top_logprobs": [
                                None if i == 0 else {prompt[i]: -0.5}
                                for i in range(len(prompt))
                            ],
                        },
                    }
                ]
            }
        else:
            response = {"choices": [{"text": f"completions_answer_{len(prompt)}"}]}
        self._respond(response)

    def _handle_chat(self, body: dict) -> None:
        messages = body.get("messages", [])
        content = messages[-1]["content"] if messages else ""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"chat_answer_{len(content)}",
                    }
                }
            ]
        }
        self._respond(response)

    def _respond(self, data: dict) -> None:
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ------------------------------------------------------------------
# Mock server that supports ONLY /v1/chat/completions (no /v1/completions)
# ------------------------------------------------------------------


class _MockChatOnlyHandler(BaseHTTPRequestHandler):
    """Mock handler that only supports /v1/chat/completions."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body: dict[str, Any] = json.loads(self.rfile.read(length))

        if self.path.rstrip("/").endswith("/chat/completions"):
            messages = body.get("messages", [])
            content = messages[-1]["content"] if messages else ""
            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"chat_only_{len(content)}",
                        }
                    }
                ]
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_server():
    """Server supporting both /v1/completions and /v1/chat/completions."""
    server = HTTPServer(("127.0.0.1", 0), _MockBothHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="module")
def chat_only_server():
    """Server supporting ONLY /v1/chat/completions (returns 404 on /v1/completions)."""
    server = HTTPServer(("127.0.0.1", 0), _MockChatOnlyHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestServerEngine:
    def test_generate_single(self, mock_server: str):
        engine = ServerEngine(
            model="test-model",
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
            model="test-model",
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
            model="test-model",
            base_url=mock_server,
        )
        assert engine.generate([], []) == []

    def test_loglikelihood(self, mock_server: str):
        engine = ServerEngine(
            model="test-model",
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
            model="test-model",
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
            model="test-model",
            base_url=mock_server + "/v1",
        )
        assert engine.base_url == mock_server + "/v1"

        engine2 = ServerEngine(
            model="test-model",
            base_url=mock_server,
        )
        assert engine2.base_url == mock_server + "/v1"

    def test_process_requests_integration(self, mock_server: str):
        from inference_eval.schema import InferenceRequest

        engine = ServerEngine(
            model="test-model",
            base_url=mock_server,
            max_concurrent=4,
            api_type="completions",
        )
        requests = [
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
        results = engine.process_requests(requests)
        assert len(results) == 5
        assert all(r.generated_text is not None for r in results)
        assert all(r.task_name == "gsm8k" for r in results)


class TestServerEngineChatMode:
    """Tests for the chat completions endpoint."""

    def test_explicit_chat_mode(self, mock_server: str):
        engine = ServerEngine(
            model="test-model",
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
            model="test-model",
            base_url=mock_server,
            max_concurrent=8,
            api_type="chat",
        )
        prompts = [f"prompt_{i}" for i in range(10)]
        gen_kwargs = [{"until": ["\n"]}] * 10
        results = engine.generate(prompts, gen_kwargs)
        assert len(results) == 10
        assert all(r.startswith("chat_answer_") for r in results)


class TestServerEngineAutoDetect:
    """Tests for automatic api_type detection."""

    def test_auto_detects_completions(self, mock_server: str):
        engine = ServerEngine(
            model="test-model",
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
            model="test-model",
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
            model="test-model",
            base_url=chat_only_server,
            api_type="auto",
            max_concurrent=8,
        )
        prompts = [f"prompt_{i}" for i in range(10)]
        gen_kwargs = [{"until": ["\n"]}] * 10
        results = engine.generate(prompts, gen_kwargs)
        assert len(results) == 10
        assert all(r.startswith("chat_only_") for r in results)


class TestServerEngineInvalidApiType:
    def test_invalid_api_type(self):
        with pytest.raises(ValueError, match="api_type must be"):
            ServerEngine(model="x", api_type="invalid")
