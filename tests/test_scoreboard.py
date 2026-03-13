"""Tests for the scoreboard system."""

from __future__ import annotations

import tempfile
from pathlib import Path

from click.testing import CliRunner

from inference_eval.cli import cli
from inference_eval.scoreboard import (
    append_entry,
    load_entries,
    make_entry,
    render_summary,
)

_MOCK_EVAL_RESULTS = {
    "results": {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,flexible-extract": 0.75,
            "exact_match_stderr,flexible-extract": 0.01,
            "exact_match,strict-match": 0.72,
            "exact_match_stderr,strict-match": 0.01,
        },
        "hellaswag": {
            "alias": "hellaswag",
            "acc,none": 0.60,
            "acc_stderr,none": 0.01,
            "acc_norm,none": 0.65,
            "acc_norm_stderr,none": 0.01,
        },
    }
}


class TestMakeEntry:
    def test_basic(self):
        entry = make_entry("qwen3-8b", _MOCK_EVAL_RESULTS)
        assert entry["tag"] == "qwen3-8b"
        assert "timestamp" in entry
        assert "gsm8k" in entry["tasks"]
        assert "hellaswag" in entry["tasks"]

    def test_filters_stderr_and_alias(self):
        entry = make_entry("test", _MOCK_EVAL_RESULTS)
        gsm = entry["tasks"]["gsm8k"]
        assert "exact_match,flexible-extract" in gsm
        assert "exact_match_stderr,flexible-extract" not in gsm
        assert "alias" not in gsm

    def test_metadata(self):
        entry = make_entry("test", _MOCK_EVAL_RESULTS, metadata={"model": "qwen3"})
        assert entry["metadata"]["model"] == "qwen3"


class TestPersistence:
    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sb.jsonl"
            e1 = make_entry("model-a", _MOCK_EVAL_RESULTS)
            e2 = make_entry("model-b", _MOCK_EVAL_RESULTS)
            append_entry(e1, path)
            append_entry(e2, path)

            loaded = load_entries(path)
            assert len(loaded) == 2
            assert loaded[0]["tag"] == "model-a"
            assert loaded[1]["tag"] == "model-b"

    def test_load_nonexistent(self):
        entries = load_entries("/tmp/does_not_exist_12345.jsonl")
        assert entries == []

    def test_incremental_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sb.jsonl"
            for i in range(5):
                append_entry(make_entry(f"run-{i}", _MOCK_EVAL_RESULTS), path)
            assert len(load_entries(path)) == 5


class TestRenderSummary:
    def test_renders_table(self):
        entries = [
            make_entry("model-a", _MOCK_EVAL_RESULTS),
            make_entry("model-b", _MOCK_EVAL_RESULTS),
        ]
        table = render_summary(entries)
        assert "model-a" in table
        assert "model-b" in table
        assert "gsm8k" in table
        assert "hellaswag" in table

    def test_uses_short_metric_names(self):
        entries = [make_entry("m", _MOCK_EVAL_RESULTS)]
        table = render_summary(entries)
        assert "em_flex" in table
        assert "hellaswag (acc)" in table

    def test_filter_tasks(self):
        entries = [make_entry("m", _MOCK_EVAL_RESULTS)]
        table = render_summary(entries, tasks=["gsm8k"])
        assert "gsm8k" in table
        assert "hellaswag" not in table

    def test_empty_entries(self):
        table = render_summary([])
        assert "no scoreboard entries" in table.lower()

    def test_values_format(self):
        entries = [make_entry("m", _MOCK_EVAL_RESULTS)]
        table = render_summary(entries)
        assert "0.7500" in table


class TestSummaryCLI:
    def test_summary_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["summary", "--help"])
        assert result.exit_code == 0
        assert "--scoreboard" in result.output
        assert "--tag" in result.output
        assert "--csv" in result.output

    def test_summary_with_data(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path("sb.jsonl")
            append_entry(make_entry("model-a", _MOCK_EVAL_RESULTS), path)
            append_entry(make_entry("model-b", _MOCK_EVAL_RESULTS), path)

            result = runner.invoke(cli, ["summary", "-s", "sb.jsonl"])
            assert result.exit_code == 0
            assert "model-a" in result.output
            assert "model-b" in result.output

    def test_summary_empty(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["summary", "-s", "empty.jsonl"])
            assert result.exit_code == 0
            assert "no entries" in result.output.lower()

    def test_summary_filter_tag(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path("sb.jsonl")
            append_entry(make_entry("model-a", _MOCK_EVAL_RESULTS), path)
            append_entry(make_entry("model-b", _MOCK_EVAL_RESULTS), path)

            result = runner.invoke(
                cli, ["summary", "-s", "sb.jsonl", "--tag", "model-a"]
            )
            assert result.exit_code == 0
            assert "model-a" in result.output
            assert "model-b" not in result.output

    def test_summary_csv_export(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path("sb.jsonl")
            append_entry(make_entry("model-a", _MOCK_EVAL_RESULTS), path)

            result = runner.invoke(
                cli,
                ["summary", "-s", "sb.jsonl", "--csv", "out.csv"],
            )
            assert result.exit_code == 0
            assert Path("out.csv").exists()


class TestCSVAutoGeneration:
    def test_csv_created_on_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "sb.jsonl"
            csv_path = Path(tmpdir) / "sb.csv"
            append_entry(make_entry("model-a", _MOCK_EVAL_RESULTS), jsonl_path)
            assert csv_path.exists()
            content = csv_path.read_text()
            assert "model-a" in content
            assert "gsm8k" in content

    def test_csv_updated_on_second_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "sb.jsonl"
            csv_path = Path(tmpdir) / "sb.csv"
            append_entry(make_entry("model-a", _MOCK_EVAL_RESULTS), jsonl_path)
            append_entry(make_entry("model-b", _MOCK_EVAL_RESULTS), jsonl_path)
            content = csv_path.read_text()
            assert "model-a" in content
            assert "model-b" in content

    def test_csv_export_string(self):
        from inference_eval.scoreboard import export_csv_string

        entries = [make_entry("m", _MOCK_EVAL_RESULTS)]
        csv_str = export_csv_string(entries)
        assert "Tag" in csv_str
        assert "gsm8k" in csv_str
        assert "0.7500" in csv_str


class TestEvaluateCLITag:
    def test_evaluate_help_shows_tag(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert "--tag" in result.output
        assert "--scoreboard" in result.output
