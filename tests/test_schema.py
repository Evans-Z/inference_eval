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


# ------------------------------------------------------------------
# Nested directory layout (task groups)
# ------------------------------------------------------------------

MMLU_SUBTASKS = ["mmlu_abstract_algebra", "mmlu_anatomy"]
GROUP_MAP = {"mmlu": MMLU_SUBTASKS}


def test_save_requests_nested_layout():
    """Sub-tasks should be nested under their group folder."""
    requests = [
        InferenceRequest(
            task_name="mmlu_abstract_algebra",
            request_type="loglikelihood",
            doc_id=0,
            index=0,
            context="ctx",
            continuation=" c",
        ),
        InferenceRequest(
            task_name="mmlu_anatomy",
            request_type="loglikelihood",
            doc_id=0,
            index=0,
            context="ctx2",
            continuation=" c2",
        ),
        InferenceRequest(
            task_name="gsm8k",
            request_type="generate_until",
            doc_id=0,
            index=0,
            context="q",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        save_requests(requests, base, task_group_map=GROUP_MAP)

        # mmlu sub-tasks should be under mmlu/
        assert (
            base / "mmlu" / "mmlu_abstract_algebra" / "loglikelihood.jsonl"
        ).exists()
        assert (base / "mmlu" / "mmlu_anatomy" / "loglikelihood.jsonl").exists()
        # gsm8k stays flat
        assert (base / "gsm8k" / "generate_until.jsonl").exists()
        # Flat mmlu folder should NOT exist
        assert not (base / "mmlu_abstract_algebra" / "loglikelihood.jsonl").exists()

        # load_requests still finds everything via rglob
        loaded = load_requests(base)
        assert len(loaded) == 3
        task_names = {r.task_name for r in loaded}
        assert task_names == {"mmlu_abstract_algebra", "mmlu_anatomy", "gsm8k"}


def test_save_results_nested_layout():
    results = [
        InferenceResult(
            task_name="mmlu_abstract_algebra",
            request_type="loglikelihood",
            doc_id=0,
            index=0,
            log_likelihood=-1.0,
            is_greedy=True,
        ),
        InferenceResult(
            task_name="gsm8k",
            request_type="generate_until",
            doc_id=0,
            index=0,
            generated_text="42",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        save_results(results, base, task_group_map=GROUP_MAP)
        assert (
            base / "mmlu" / "mmlu_abstract_algebra" / "loglikelihood.jsonl"
        ).exists()
        assert (base / "gsm8k" / "generate_until.jsonl").exists()

        loaded = load_results(base)
        assert len(loaded) == 2


def test_save_without_group_map_stays_flat():
    """Without a group map, layout stays flat (backward compat)."""
    requests = [
        InferenceRequest(
            task_name="mmlu_abstract_algebra",
            request_type="loglikelihood",
            doc_id=0,
            index=0,
            context="c",
            continuation=" c",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        save_requests(requests, base)
        assert (base / "mmlu_abstract_algebra" / "loglikelihood.jsonl").exists()


# ------------------------------------------------------------------
# Incremental config.json
# ------------------------------------------------------------------


def test_config_incremental_merge():
    """Second save merges tasks into existing config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # First extraction: gsm8k
        cfg1 = ExtractConfig(
            tasks=["gsm8k"],
            num_fewshot=5,
            extracted_tasks=["gsm8k"],
        )
        cfg1.save(base)

        # Second extraction: mmlu
        cfg2 = ExtractConfig(
            tasks=["mmlu"],
            num_fewshot=0,
            task_group_map={"mmlu": MMLU_SUBTASKS},
            extracted_tasks=MMLU_SUBTASKS,
        )
        cfg2.save(base)

        loaded = ExtractConfig.load(base)
        assert loaded.tasks == ["gsm8k", "mmlu"]
        assert "mmlu" in loaded.task_group_map
        assert loaded.task_group_map["mmlu"] == MMLU_SUBTASKS
        assert "gsm8k" in loaded.extracted_tasks
        assert "mmlu_abstract_algebra" in loaded.extracted_tasks


def test_config_incremental_dedup():
    """Re-extracting same task does not duplicate entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        cfg1 = ExtractConfig(tasks=["gsm8k"], extracted_tasks=["gsm8k"])
        cfg1.save(base)
        cfg2 = ExtractConfig(tasks=["gsm8k"], extracted_tasks=["gsm8k"])
        cfg2.save(base)
        loaded = ExtractConfig.load(base)
        assert loaded.tasks == ["gsm8k"]
        assert loaded.extracted_tasks == ["gsm8k"]


def test_config_incremental_three_steps():
    """Three incremental extractions accumulate correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        for task in ["gsm8k", "hellaswag", "mmlu"]:
            cfg = ExtractConfig(tasks=[task], extracted_tasks=[task])
            cfg.save(base)
        loaded = ExtractConfig.load(base)
        assert loaded.tasks == ["gsm8k", "hellaswag", "mmlu"]
