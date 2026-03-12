"""End-to-end test demonstrating the full extract → mock infer → evaluate pipeline."""

import json
import tempfile
from pathlib import Path

from inference_eval.schema import (
    InferenceResult,
    load_requests,
    save_results,
)


def test_extract_and_evaluate_gsm8k():
    """Test the full pipeline: extract, mock results, evaluate."""
    from inference_eval.evaluate import evaluate_results
    from inference_eval.extract import extract_requests

    with tempfile.TemporaryDirectory() as tmpdir:
        requests_dir = Path(tmpdir) / "requests"
        results_dir = Path(tmpdir) / "results"
        scores_file = Path(tmpdir) / "scores.json"

        counts = extract_requests(
            tasks=["gsm8k"],
            output_dir=requests_dir,
            limit=3,
        )
        assert "gsm8k/generate_until" in counts
        assert counts["gsm8k/generate_until"] == 3

        requests = load_requests(requests_dir)
        assert len(requests) == 3
        assert all(r.task_name == "gsm8k" for r in requests)
        assert all(r.request_type == "generate_until" for r in requests)

        mock_answers = [
            "#### 12",
            "#### 5",
            "#### 42",
        ]
        mock_results = []
        for req, answer in zip(requests, mock_answers):
            mock_results.append(
                InferenceResult(
                    task_name=req.task_name,
                    request_type=req.request_type,
                    doc_id=req.doc_id,
                    index=req.index,
                    generated_text=answer,
                    fingerprint=req.fingerprint,
                )
            )
        save_results(mock_results, results_dir)

        eval_results = evaluate_results(
            results_dir=results_dir,
            requests_dir=requests_dir,
            output_file=scores_file,
        )

        assert "results" in eval_results
        assert scores_file.exists()
        with open(scores_file) as f:
            scores = json.load(f)
        assert "results" in scores
