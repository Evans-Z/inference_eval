"""Tests for ServerEngine using a mock HTTP server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from inference_eval.inference.server_engine import ServerEngine


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that mimics the OpenAI /v1/completions endpoint."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body: dict[str, Any] = json.loads(self.rfile.read(length))

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
            response = {"choices": [{"text": f"mock_answer_{len(prompt)}"}]}

        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock OpenAI-compatible server on a random port."""
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestServerEngine:
    def test_generate_single(self, mock_server: str):
        engine = ServerEngine(
            model="test-model",
            base_url=mock_server,
            max_concurrent=4,
        )
        results = engine.generate(
            ["What is 2+2?"],
            [{"until": ["\n"], "max_gen_toks": 32}],
        )
        assert len(results) == 1
        assert results[0].startswith("mock_answer_")

    def test_generate_batch(self, mock_server: str):
        engine = ServerEngine(
            model="test-model",
            base_url=mock_server,
            max_concurrent=8,
        )
        prompts = [f"prompt_{i}" for i in range(20)]
        gen_kwargs = [{"until": ["\n"], "max_gen_toks": 32}] * 20
        results = engine.generate(prompts, gen_kwargs)
        assert len(results) == 20
        assert all(r.startswith("mock_answer_") for r in results)

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
