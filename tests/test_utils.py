"""Tests for utility functions."""

from inference_eval.utils import make_content_key, make_result_key


def test_make_result_key():
    key = make_result_key("gsm8k", "generate_until", 0, 5)
    assert key == "gsm8k|generate_until|0|5"


def test_make_content_key():
    key = make_content_key("gsm8k", "generate_until", "What is 2+2?")
    assert "gsm8k" in key
    assert "What is 2+2?" in key


def test_make_content_key_with_continuation():
    key = make_content_key("hellaswag", "loglikelihood", "The cat", " sat")
    assert "hellaswag" in key
    assert " sat" in key
