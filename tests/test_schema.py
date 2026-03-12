"""Tests for schema module."""

import tempfile
from pathlib import Path

from inference_eval.schema import (
    ExtractConfig,
    InferenceRequest,
    InferenceResult,
    load_requests,
    load_results,
    save_requests,
    save_results,
)


def test_extract_config_roundtrip():
    config = ExtractConfig(
        tasks=["gsm8k", "hellaswag"],
        num_fewshot=5,
        limit=100,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save(Path(tmpdir))
        loaded = ExtractConfig.load(Path(tmpdir))
        assert loaded.tasks == ["gsm8k", "hellaswag"]
        assert loaded.num_fewshot == 5
        assert loaded.limit == 100
        assert loaded.random_seed == 0


def test_inference_request_fingerprint():
    req1 = InferenceRequest(
        task_name="gsm8k",
        request_type="generate_until",
        doc_id=0,
        index=0,
        context="What is 2+2?",
    )
    req2 = InferenceRequest(
        task_name="gsm8k",
        request_type="generate_until",
        doc_id=0,
        index=0,
        context="What is 2+2?",
    )
    req3 = InferenceRequest(
        task_name="gsm8k",
        request_type="generate_until",
        doc_id=1,
        index=1,
        context="What is 3+3?",
    )
    assert req1.fingerprint == req2.fingerprint
    assert req1.fingerprint != req3.fingerprint


def test_save_load_requests():
    requests = [
        InferenceRequest(
            task_name="gsm8k",
            request_type="generate_until",
            doc_id=0,
            index=0,
            context="What is 2+2?",
            generation_kwargs={"until": ["\n"], "max_gen_toks": 256},
        ),
        InferenceRequest(
            task_name="hellaswag",
            request_type="loglikelihood",
            doc_id=0,
            index=0,
            context="The cat sat on the",
            continuation=" mat",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = save_requests(requests, Path(tmpdir))
        assert counts["gsm8k/generate_until"] == 1
        assert counts["hellaswag/loglikelihood"] == 1

        loaded = load_requests(Path(tmpdir))
        assert len(loaded) == 2

        gen_req = [r for r in loaded if r.request_type == "generate_until"][0]
        assert gen_req.task_name == "gsm8k"
        assert gen_req.context == "What is 2+2?"
        assert gen_req.generation_kwargs == {"until": ["\n"], "max_gen_toks": 256}

        ll_req = [r for r in loaded if r.request_type == "loglikelihood"][0]
        assert ll_req.task_name == "hellaswag"
        assert ll_req.continuation == " mat"


def test_save_load_results():
    results = [
        InferenceResult(
            task_name="gsm8k",
            request_type="generate_until",
            doc_id=0,
            index=0,
            generated_text="4",
        ),
        InferenceResult(
            task_name="hellaswag",
            request_type="loglikelihood",
            doc_id=0,
            index=0,
            log_likelihood=-2.5,
            is_greedy=True,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        counts = save_results(results, Path(tmpdir))
        assert counts["gsm8k/generate_until"] == 1

        loaded = load_results(Path(tmpdir))
        assert len(loaded) == 2

        gen_res = [r for r in loaded if r.request_type == "generate_until"][0]
        assert gen_res.generated_text == "4"

        ll_res = [r for r in loaded if r.request_type == "loglikelihood"][0]
        assert ll_res.log_likelihood == -2.5
        assert ll_res.is_greedy is True


def test_result_to_lm_eval_response():
    gen_result = InferenceResult(
        task_name="gsm8k",
        request_type="generate_until",
        doc_id=0,
        index=0,
        generated_text="The answer is 4.",
    )
    assert gen_result.to_lm_eval_response() == "The answer is 4."

    ll_result = InferenceResult(
        task_name="hellaswag",
        request_type="loglikelihood",
        doc_id=0,
        index=0,
        log_likelihood=-1.5,
        is_greedy=True,
    )
    assert ll_result.to_lm_eval_response() == (-1.5, True)
