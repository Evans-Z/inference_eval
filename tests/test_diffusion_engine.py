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


class TestDiffusionEngineLoglikelihood:
    def test_loglikelihood_raises(self):
        from inference_eval.inference.diffusion_engine import DiffusionEngine

        with patch.object(DiffusionEngine, "__init__", lambda self, **kw: None):
            engine = DiffusionEngine.__new__(DiffusionEngine)
            with pytest.raises(NotImplementedError, match="[Dd]iffusion"):
                engine.compute_loglikelihood(["ctx"], ["cont"])


class TestDiffusionEngineProcessRequests:
    def test_loglikelihood_requests_raise(self):
        from inference_eval.inference.diffusion_engine import DiffusionEngine
        from inference_eval.schema import InferenceRequest

        with patch.object(DiffusionEngine, "__init__", lambda self, **kw: None):
            engine = DiffusionEngine.__new__(DiffusionEngine)
            engine.generate = MagicMock(return_value=[])

            requests = [
                InferenceRequest(
                    task_name="hellaswag",
                    request_type="loglikelihood",
                    doc_id=0,
                    index=0,
                    context="ctx",
                    continuation=" cont",
                ),
            ]
            with pytest.raises(NotImplementedError):
                engine.process_requests(requests)
