"""Tests for inference_eval.main."""

from inference_eval import __version__
from inference_eval.main import main


def test_version():
    assert __version__ == "0.1.0"


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "inference_eval v0.1.0" in captured.out
