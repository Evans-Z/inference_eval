"""Tests for per-task num_fewshot / limit settings."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from inference_eval.evaluate import _group_tasks_by_settings
from inference_eval.schema import ExtractConfig


class TestGroupTasksBySettings:
    def test_same_settings_grouped(self):
        settings = {
            "gsm8k": {"num_fewshot": 5, "limit": None},
            "math": {"num_fewshot": 5, "limit": None},
        }
        groups = _group_tasks_by_settings(
            ["gsm8k", "math"],
            settings,
            default_fewshot=None,
            default_limit=None,
            override_fewshot=None,
            override_limit=None,
            cli_override_fewshot=False,
            cli_override_limit=False,
        )
        assert len(groups) == 1
        key = (5, None)
        assert set(groups[key]) == {"gsm8k", "math"}

    def test_different_fewshot_separated(self):
        settings = {
            "gsm8k": {"num_fewshot": 5, "limit": None},
            "mmlu": {"num_fewshot": 0, "limit": None},
        }
        groups = _group_tasks_by_settings(
            ["gsm8k", "mmlu"],
            settings,
            default_fewshot=None,
            default_limit=None,
            override_fewshot=None,
            override_limit=None,
            cli_override_fewshot=False,
            cli_override_limit=False,
        )
        assert len(groups) == 2
        assert groups[(5, None)] == ["gsm8k"]
        assert groups[(0, None)] == ["mmlu"]

    def test_cli_override_merges_all(self):
        """CLI --num-fewshot overrides per-task settings."""
        settings = {
            "gsm8k": {"num_fewshot": 5, "limit": None},
            "mmlu": {"num_fewshot": 0, "limit": None},
        }
        groups = _group_tasks_by_settings(
            ["gsm8k", "mmlu"],
            settings,
            default_fewshot=None,
            default_limit=None,
            override_fewshot=3,
            override_limit=None,
            cli_override_fewshot=True,
            cli_override_limit=False,
        )
        assert len(groups) == 1
        assert (3, None) in groups
        assert set(groups[(3, None)]) == {"gsm8k", "mmlu"}

    def test_unknown_task_uses_default(self):
        groups = _group_tasks_by_settings(
            ["gsm8k", "new_task"],
            {"gsm8k": {"num_fewshot": 5}},
            default_fewshot=0,
            default_limit=None,
            override_fewshot=None,
            override_limit=None,
            cli_override_fewshot=False,
            cli_override_limit=False,
        )
        assert groups[(5, None)] == ["gsm8k"]
        assert groups[(0, None)] == ["new_task"]

    def test_different_limits_separated(self):
        settings = {
            "gsm8k": {"num_fewshot": 5, "limit": 100},
            "math": {"num_fewshot": 5, "limit": 50},
        }
        groups = _group_tasks_by_settings(
            ["gsm8k", "math"],
            settings,
            default_fewshot=None,
            default_limit=None,
            override_fewshot=None,
            override_limit=None,
            cli_override_fewshot=False,
            cli_override_limit=False,
        )
        assert len(groups) == 2


class TestConfigPerTaskSettings:
    def test_incremental_preserves_per_task(self):
        """Two extractions with different fewshot are both preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            cfg1 = ExtractConfig(
                tasks=["gsm8k"],
                num_fewshot=5,
                extracted_tasks=["gsm8k"],
                task_settings={"gsm8k": {"num_fewshot": 5, "limit": None}},
            )
            cfg1.save(base)

            cfg2 = ExtractConfig(
                tasks=["mmlu"],
                num_fewshot=0,
                extracted_tasks=["mmlu_anatomy"],
                task_settings={"mmlu": {"num_fewshot": 0, "limit": None}},
                task_group_map={"mmlu": ["mmlu_anatomy"]},
            )
            cfg2.save(base)

            loaded = ExtractConfig.load(base)
            assert loaded.task_settings["gsm8k"]["num_fewshot"] == 5
            assert loaded.task_settings["mmlu"]["num_fewshot"] == 0

    def test_re_extract_updates_settings(self):
        """Re-extracting a task updates its settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            cfg1 = ExtractConfig(
                tasks=["gsm8k"],
                task_settings={"gsm8k": {"num_fewshot": 5, "limit": None}},
            )
            cfg1.save(base)

            cfg2 = ExtractConfig(
                tasks=["gsm8k"],
                task_settings={"gsm8k": {"num_fewshot": 8, "limit": 100}},
            )
            cfg2.save(base)

            loaded = ExtractConfig.load(base)
            assert loaded.task_settings["gsm8k"]["num_fewshot"] == 8
            assert loaded.task_settings["gsm8k"]["limit"] == 100


class TestEvaluatePerTaskSettings:
    @patch("inference_eval.evaluate.lm_eval.simple_evaluate")
    def test_calls_simple_evaluate_per_group(self, mock_eval):
        """Tasks with different fewshot trigger separate evaluate calls."""
        mock_eval.return_value = {"results": {}, "configs": {}}

        from inference_eval.evaluate import evaluate_results

        with tempfile.TemporaryDirectory() as tmpdir:
            req_dir = Path(tmpdir) / "requests"
            res_dir = Path(tmpdir) / "results"
            req_dir.mkdir()
            res_dir.mkdir()

            cfg = ExtractConfig(
                tasks=["gsm8k", "mmlu"],
                task_settings={
                    "gsm8k": {"num_fewshot": 5, "limit": None},
                    "mmlu": {"num_fewshot": 0, "limit": None},
                },
            )
            cfg.save(req_dir)

            evaluate_results(
                results_dir=res_dir,
                requests_dir=req_dir,
            )

            assert mock_eval.call_count == 2

            call_args_list = mock_eval.call_args_list
            fewshots = {c.kwargs["num_fewshot"] for c in call_args_list}
            assert fewshots == {5, 0}

            tasks_per_call = {
                tuple(c.kwargs["tasks"]): c.kwargs["num_fewshot"]
                for c in call_args_list
            }
            assert tasks_per_call[("gsm8k",)] == 5
            assert tasks_per_call[("mmlu",)] == 0

    @patch("inference_eval.evaluate.lm_eval.simple_evaluate")
    def test_cli_override_single_call(self, mock_eval):
        """CLI --num-fewshot overrides per-task → single evaluate call."""
        mock_eval.return_value = {"results": {}, "configs": {}}

        from inference_eval.evaluate import evaluate_results

        with tempfile.TemporaryDirectory() as tmpdir:
            req_dir = Path(tmpdir) / "requests"
            res_dir = Path(tmpdir) / "results"
            req_dir.mkdir()
            res_dir.mkdir()

            cfg = ExtractConfig(
                tasks=["gsm8k", "mmlu"],
                task_settings={
                    "gsm8k": {"num_fewshot": 5, "limit": None},
                    "mmlu": {"num_fewshot": 0, "limit": None},
                },
            )
            cfg.save(req_dir)

            evaluate_results(
                results_dir=res_dir,
                requests_dir=req_dir,
                num_fewshot=3,
            )

            assert mock_eval.call_count == 1
            assert mock_eval.call_args.kwargs["num_fewshot"] == 3
