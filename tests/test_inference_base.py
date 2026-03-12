"""Tests for inference engine base class."""

from typing import Any

from inference_eval.inference.base import InferenceEngine
from inference_eval.schema import InferenceRequest


class MockEngine(InferenceEngine):
    """A mock engine for testing the base class."""

    def generate(
        self, prompts: list[str], gen_kwargs: list[dict[str, Any]]
    ) -> list[str]:
        return [f"generated: {p[:20]}" for p in prompts]

    def compute_loglikelihood(
        self, contexts: list[str], continuations: list[str]
    ) -> list[tuple[float, bool]]:
        return [(-1.0, True) for _ in contexts]


class TestInferenceEngine:
    def test_process_generate_requests(self):
        engine = MockEngine()
        requests = [
            InferenceRequest(
                task_name="gsm8k",
                request_type="generate_until",
                doc_id=0,
                index=0,
                context="What is 2+2?",
                generation_kwargs={"until": ["\n"]},
            ),
        ]
        results = engine.process_requests(requests)
        assert len(results) == 1
        assert results[0].generated_text == "generated: What is 2+2?"
        assert results[0].task_name == "gsm8k"
        assert results[0].request_type == "generate_until"

    def test_process_loglikelihood_requests(self):
        engine = MockEngine()
        requests = [
            InferenceRequest(
                task_name="hellaswag",
                request_type="loglikelihood",
                doc_id=0,
                index=0,
                context="The cat",
                continuation=" sat",
            ),
        ]
        results = engine.process_requests(requests)
        assert len(results) == 1
        assert results[0].log_likelihood == -1.0
        assert results[0].is_greedy is True

    def test_process_mixed_requests(self):
        engine = MockEngine()
        requests = [
            InferenceRequest(
                task_name="gsm8k",
                request_type="generate_until",
                doc_id=0,
                index=0,
                context="Q: 2+2?",
                generation_kwargs={"until": ["\n"]},
            ),
            InferenceRequest(
                task_name="hellaswag",
                request_type="loglikelihood",
                doc_id=0,
                index=0,
                context="The cat",
                continuation=" sat",
            ),
        ]
        results = engine.process_requests(requests)
        assert len(results) == 2
        gen = [r for r in results if r.request_type == "generate_until"]
        ll = [r for r in results if r.request_type == "loglikelihood"]
        assert len(gen) == 1
        assert len(ll) == 1
