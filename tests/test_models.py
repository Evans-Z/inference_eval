"""Tests for custom LM models (capture and offline)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from inference_eval.models.capture import RequestCaptureLM
from inference_eval.schema import InferenceResult, save_results


def _make_mock_instance(
    request_type: str,
    args: tuple,
    task_name: str = "test_task",
    doc_id: int = 0,
):
    """Create a mock lm-eval Instance for testing."""
    instance = MagicMock()
    instance.args = args
    instance.request_type = request_type
    instance.task_name = task_name
    instance.doc_id = doc_id
    instance.idx = 0
    instance.metadata = (task_name, 1, doc_id)
    return instance


class TestRequestCaptureLM:
    def test_loglikelihood_capture(self):
        lm = RequestCaptureLM()
        instances = [
            _make_mock_instance(
                "loglikelihood",
                ("The cat sat on the", " mat"),
                doc_id=0,
            ),
            _make_mock_instance(
                "loglikelihood",
                ("The dog ran in the", " park"),
                doc_id=1,
            ),
        ]
        results = lm.loglikelihood(instances)
        assert len(results) == 2
        assert all(r == (0.0, False) for r in results)
        assert len(lm.captured_requests) == 2
        assert lm.captured_requests[0].context == "The cat sat on the"
        assert lm.captured_requests[0].continuation == " mat"
        assert lm.captured_requests[1].context == "The dog ran in the"

    def test_generate_until_capture(self):
        lm = RequestCaptureLM()
        instances = [
            _make_mock_instance(
                "generate_until",
                ("What is 2+2?", {"until": ["\n"], "max_gen_toks": 256}),
                doc_id=0,
            ),
        ]
        results = lm.generate_until(instances)
        assert len(results) == 1
        assert results[0] == ""
        assert len(lm.captured_requests) == 1
        req = lm.captured_requests[0]
        assert req.context == "What is 2+2?"
        assert req.generation_kwargs == {"until": ["\n"], "max_gen_toks": 256}
        assert req.request_type == "generate_until"

    def test_loglikelihood_rolling_capture(self):
        lm = RequestCaptureLM()
        instances = [
            _make_mock_instance(
                "loglikelihood_rolling",
                ("Some long text here",),
                doc_id=0,
            ),
        ]
        results = lm.loglikelihood_rolling(instances)
        assert len(results) == 1
        assert lm.captured_requests[0].context == "Some long text here"

    def test_sequential_indexing(self):
        lm = RequestCaptureLM()
        instances = [
            _make_mock_instance("loglikelihood", ("ctx1", "cont1"), doc_id=0),
            _make_mock_instance("loglikelihood", ("ctx2", "cont2"), doc_id=1),
        ]
        lm.loglikelihood(instances)
        assert lm.captured_requests[0].index == 0
        assert lm.captured_requests[1].index == 1


class TestOfflineLM:
    def test_loglikelihood_lookup(self):
        from inference_eval.models.offline import OfflineLM

        results = [
            InferenceResult(
                task_name="test_task",
                request_type="loglikelihood",
                doc_id=0,
                index=0,
                log_likelihood=-1.5,
                is_greedy=True,
            ),
            InferenceResult(
                task_name="test_task",
                request_type="loglikelihood",
                doc_id=1,
                index=1,
                log_likelihood=-2.5,
                is_greedy=False,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_results(results, Path(tmpdir))
            offline_lm = OfflineLM(tmpdir)

            instances = [
                _make_mock_instance("loglikelihood", ("ctx1", "cont1"), doc_id=0),
                _make_mock_instance("loglikelihood", ("ctx2", "cont2"), doc_id=1),
            ]
            ll_results = offline_lm.loglikelihood(instances)
            assert ll_results[0] == (-1.5, True)
            assert ll_results[1] == (-2.5, False)

    def test_generate_until_lookup(self):
        from inference_eval.models.offline import OfflineLM

        results = [
            InferenceResult(
                task_name="test_task",
                request_type="generate_until",
                doc_id=0,
                index=0,
                generated_text="The answer is 4.",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_results(results, Path(tmpdir))
            offline_lm = OfflineLM(tmpdir)

            instances = [
                _make_mock_instance(
                    "generate_until",
                    ("What is 2+2?", {"until": ["\n"]}),
                    doc_id=0,
                ),
            ]
            gen_results = offline_lm.generate_until(instances)
            assert gen_results[0] == "The answer is 4."
