"""Tests for task group expansion (e.g. mmlu → mmlu_abstract_algebra, …)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from inference_eval.infer import _expand_task_names, _filter_by_tasks
from inference_eval.schema import (
    ExtractConfig,
    InferenceRequest,
    save_requests,
)


def _make_request(task_name: str, idx: int = 0) -> InferenceRequest:
    return InferenceRequest(
        task_name=task_name,
        request_type="loglikelihood",
        doc_id=idx,
        index=idx,
        context=f"ctx_{task_name}_{idx}",
        continuation=f" cont_{idx}",
    )


# Simulate mmlu-like extraction: user asked for "mmlu" but got sub-tasks
MMLU_SUBTASKS = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_college_biology",
]


class TestExpandTaskNames:
    def test_exact_match(self):
        available = {"gsm8k", "hellaswag"}
        expanded = _expand_task_names(["gsm8k"], available)
        assert expanded == {"gsm8k"}

    def test_prefix_match(self):
        available = set(MMLU_SUBTASKS)
        expanded = _expand_task_names(["mmlu"], available)
        assert expanded == set(MMLU_SUBTASKS)

    def test_prefix_does_not_match_partial(self):
        available = {"mmlu_pro_math", "mmlu_abstract_algebra"}
        expanded = _expand_task_names(["mmlu_pro"], available)
        assert expanded == {"mmlu_pro_math"}
        assert "mmlu_abstract_algebra" not in expanded

    def test_mixed_exact_and_group(self):
        available = {"gsm8k"} | set(MMLU_SUBTASKS)
        expanded = _expand_task_names(["gsm8k", "mmlu"], available)
        assert "gsm8k" in expanded
        assert set(MMLU_SUBTASKS).issubset(expanded)

    def test_with_config_group_map(self):
        available = set(MMLU_SUBTASKS)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExtractConfig(
                tasks=["mmlu"],
                task_group_map={"mmlu": MMLU_SUBTASKS},
                extracted_tasks=MMLU_SUBTASKS,
            )
            cfg.save(Path(tmpdir))

            expanded = _expand_task_names(["mmlu"], available, Path(tmpdir))
            assert expanded == set(MMLU_SUBTASKS)

    def test_unknown_task_warns(self, caplog):
        available = {"gsm8k"}
        expanded = _expand_task_names(["nonexistent"], available)
        assert expanded == set()
        assert "not found" in caplog.text.lower()


class TestFilterByTasks:
    def test_filter_with_group_name(self):
        requests = [_make_request(t, i) for i, t in enumerate(MMLU_SUBTASKS)]
        requests.append(_make_request("gsm8k", 99))

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExtractConfig(
                tasks=["mmlu", "gsm8k"],
                task_group_map={"mmlu": MMLU_SUBTASKS},
                extracted_tasks=MMLU_SUBTASKS + ["gsm8k"],
            )
            cfg.save(Path(tmpdir))
            save_requests(requests, Path(tmpdir))

            filtered = _filter_by_tasks(requests, ["mmlu"], Path(tmpdir))
            assert len(filtered) == len(MMLU_SUBTASKS)
            assert all(r.task_name.startswith("mmlu_") for r in filtered)

    def test_filter_exact_still_works(self):
        requests = [_make_request(t, i) for i, t in enumerate(MMLU_SUBTASKS)]
        requests.append(_make_request("gsm8k", 99))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_requests(requests, Path(tmpdir))
            filtered = _filter_by_tasks(requests, ["gsm8k"], Path(tmpdir))
            assert len(filtered) == 1
            assert filtered[0].task_name == "gsm8k"

    def test_no_tasks_filter_returns_empty(self):
        requests = [_make_request("gsm8k")]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_requests(requests, Path(tmpdir))
            filtered = _filter_by_tasks(requests, ["nonexistent"], Path(tmpdir))
            assert len(filtered) == 0

    def test_prefix_fallback_without_config(self):
        """Works even without config.json via prefix matching."""
        requests = [_make_request(t, i) for i, t in enumerate(MMLU_SUBTASKS)]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_requests(requests, Path(tmpdir))
            filtered = _filter_by_tasks(requests, ["mmlu"], Path(tmpdir))
            assert len(filtered) == len(MMLU_SUBTASKS)
