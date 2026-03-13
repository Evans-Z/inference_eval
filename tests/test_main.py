"""Tests for inference_eval CLI and version."""

from click.testing import CliRunner

from inference_eval import __version__
from inference_eval.cli import cli


def test_version():
    assert __version__ == "0.1.0"


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output
    assert "infer" in result.output
    assert "evaluate" in result.output


def test_extract_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--tasks" in result.output
    assert "--output" in result.output


def test_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--results" in result.output


def test_infer_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["infer", "--help"])
    assert result.exit_code == 0
    assert "--base-url" in result.output
    assert "--max-concurrent" in result.output
    assert "--api-type" in result.output
    assert "--engine" in result.output
    assert "server" in result.output
