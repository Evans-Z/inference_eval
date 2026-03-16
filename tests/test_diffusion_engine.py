"""Tests for DiffusionEngine (no GPU required — tests logic only)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestTruncateAtStop:
    def test_single_stop(self):
        from inference_eval.inference.diffusion_engine import _truncate_at_stop

        assert _truncate_at_stop("hello\nworld", ["\n"]) == "hello"

    def test_multiple_stops(self):
        from inference_eval.inference.diffusion_engine import _truncate_at_stop

        text = "Answer: 42\nQuestion: next"
        assert _truncate_at_stop(text, ["Question:", "\n\n"]) == "Answer: 42\n"

    def test_no_stop_found(self):
        from inference_eval.inference.diffusion_engine import _truncate_at_stop

        assert _truncate_at_stop("hello world", ["XXX"]) == "hello world"

    def test_empty_stops(self):
        from inference_eval.inference.diffusion_engine import _truncate_at_stop

        assert _truncate_at_stop("hello", []) == "hello"

    def test_earliest_stop_wins(self):
        from inference_eval.inference.diffusion_engine import _truncate_at_stop

        text = "A Question: B\n\nC"
        assert _truncate_at_stop(text, ["\n\n", "Question:"]) == "A "


class TestDiffusionEngineNoModel:
    """Tests that work without a real model (mock-based)."""

    def test_loglikelihood_raises_without_method(self):
        from inference_eval.inference.diffusion_engine import DiffusionEngine

        with patch.object(DiffusionEngine, "__init__", lambda self, **kw: None):
            engine = DiffusionEngine.__new__(DiffusionEngine)
            engine._model = MagicMock(spec=[])
            with pytest.raises(NotImplementedError, match="get_log_likelihood"):
                engine.compute_loglikelihood(["ctx"], ["cont"])

    def test_process_requests_generate(self):
        from inference_eval.inference.diffusion_engine import DiffusionEngine
        from inference_eval.schema import InferenceRequest

        with patch.object(DiffusionEngine, "__init__", lambda self, **kw: None):
            engine = DiffusionEngine.__new__(DiffusionEngine)
            engine.generate = MagicMock(return_value=["answer"])
            engine.compute_loglikelihood = MagicMock()

            reqs = [
                InferenceRequest(
                    task_name="gsm8k",
                    request_type="generate_until",
                    doc_id=0,
                    index=0,
                    context="Q: 2+2?",
                    generation_kwargs={"until": ["\n"]},
                ),
            ]
            results = engine.process_requests(reqs)
            assert len(results) == 1
            assert results[0].generated_text == "answer"
            engine.generate.assert_called_once()
