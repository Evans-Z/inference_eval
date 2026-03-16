"""Infer phase: run inference on extracted requests using a specified engine."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inference_eval.inference.base import InferenceEngine
from inference_eval.schema import (
    ExtractConfig,
    load_requests,
    save_results,
)

logger = logging.getLogger(__name__)

ENGINE_REGISTRY: dict[str, str] = {
    "vllm": "inference_eval.inference.vllm_engine.VLLMEngine",
    "sglang": "inference_eval.inference.sglang_engine.SGLangEngine",
    "openai": "inference_eval.inference.openai_engine.OpenAIEngine",
    "server": "inference_eval.inference.server_engine.ServerEngine",
    "diffusion": "inference_eval.inference.diffusion_engine.DiffusionEngine",
}


def get_engine(engine_name: str, engine_kwargs: dict[str, Any]) -> InferenceEngine:
    """Instantiate an inference engine by name."""
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Available: {list(ENGINE_REGISTRY.keys())}"
        )

    module_path, class_name = ENGINE_REGISTRY[engine_name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_class = getattr(module, class_name)
    return engine_class(**engine_kwargs)


def _expand_task_names(
    task_filters: list[str],
    available_tasks: set[str],
    requests_dir: Path | None = None,
) -> set[str]:
    """Expand user-level task names to actual sub-task names.

    Handles lm-eval-harness task groups (e.g. ``mmlu`` →
    ``mmlu_abstract_algebra``, ``mmlu_anatomy``, …) using:

    1. The ``task_group_map`` saved in ``config.json`` during extraction.
    2. Prefix matching (``X`` matches any task starting with ``X_``).
    """
    # Try to load the group map from config.json
    group_map: dict[str, list[str]] = {}
    if requests_dir is not None:
        config_path = requests_dir / "config.json"
        if config_path.exists():
            try:
                cfg = ExtractConfig.load(requests_dir)
                group_map = cfg.task_group_map
            except Exception:
                pass

    expanded: set[str] = set()
    for name in task_filters:
        if name in available_tasks:
            expanded.add(name)
            continue

        # Check group map first (precise)
        if name in group_map:
            expanded.update(group_map[name])
            logger.info(
                "Expanded task group '%s' → %d sub-tasks (from config)",
                name,
                len(group_map[name]),
            )
            continue

        # Prefix fallback (e.g. "mmlu" matches "mmlu_abstract_algebra")
        prefix_matches = {t for t in available_tasks if t.startswith(name + "_")}
        if prefix_matches:
            expanded.update(prefix_matches)
            logger.info(
                "Expanded task group '%s' → %d sub-tasks (prefix match)",
                name,
                len(prefix_matches),
            )
            continue

        logger.warning(
            "Task '%s' not found. Available tasks: %s",
            name,
            sorted(available_tasks)[:20],
        )

    return expanded


def _filter_by_tasks(
    all_requests: list,
    tasks: list[str],
    requests_dir: Path,
) -> list:
    """Filter requests by task names, expanding groups as needed."""
    available = {r.task_name for r in all_requests}
    expanded = _expand_task_names(tasks, available, requests_dir)
    filtered = [r for r in all_requests if r.task_name in expanded]

    if not filtered and all_requests:
        logger.warning(
            "No requests matched task filter %s. Available tasks in requests dir: %s",
            tasks,
            sorted(available),
        )

    return filtered


def run_inference(
    requests_dir: str | Path,
    output_dir: str | Path,
    engine: InferenceEngine | str,
    engine_kwargs: dict[str, Any] | None = None,
    batch_size: int = 32,
    tasks: list[str] | None = None,
) -> dict[str, int]:
    """Run inference on extracted requests and save results.

    Args:
        requests_dir: Directory containing extracted requests.
        output_dir: Directory to save inference results.
        engine: InferenceEngine instance or engine name string.
        engine_kwargs: Kwargs for engine construction (if engine is a string).
        batch_size: Number of requests to process at once.
        tasks: Optional list of task names to filter (run only these tasks).

    Returns:
        Dictionary mapping "task_name/request_type" to result counts.
    """
    requests_dir = Path(requests_dir)
    output_dir = Path(output_dir)

    if isinstance(engine, str):
        engine = get_engine(engine, engine_kwargs or {})

    # Load group map so results mirror the nested request directory layout
    group_map: dict[str, list[str]] = {}
    config_path = requests_dir / "config.json"
    if config_path.exists():
        try:
            cfg = ExtractConfig.load(requests_dir)
            group_map = cfg.task_group_map
        except Exception:
            pass

    all_requests = load_requests(requests_dir)
    if tasks:
        all_requests = _filter_by_tasks(all_requests, tasks, requests_dir)

    total = len(all_requests)
    logger.info("Running inference on %d requests", total)
    t0 = time.perf_counter()

    all_results = []
    with tqdm(total=total, desc="Inference", unit="req") as pbar:
        for i in range(0, total, batch_size):
            batch = all_requests[i : i + batch_size]
            batch_results = engine.process_requests(batch)
            all_results.extend(batch_results)
            pbar.update(len(batch))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: %d results in %.1fs (%.1f req/s). Saved to %s",
        len(all_results),
        elapsed,
        total / max(elapsed, 0.001),
        output_dir,
    )

    counts = save_results(all_results, output_dir, group_map)
    return counts


def convert_external_results(
    external_file: str | Path,
    requests_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    """Convert externally produced results to inference_eval format.

    Reads a JSON file with results and matches them to extracted requests.
    The external file should be a JSON array of objects with at minimum:
    - "context": the input prompt (for matching)
    - "result": the generated text or loglikelihood value

    Args:
        external_file: Path to external results JSON file.
        requests_dir: Directory containing extracted requests.
        output_dir: Directory to save converted results.

    Returns:
        Dictionary of result counts.
    """
    from inference_eval.schema import InferenceResult

    external_file = Path(external_file)
    with open(external_file) as f:
        external_data = json.load(f)

    all_requests = load_requests(Path(requests_dir))

    results = []
    for req, ext in zip(all_requests, external_data):
        if req.request_type == "generate_until":
            result_val = ext.get("result", ext.get("generated_text", ""))
            results.append(
                InferenceResult(
                    task_name=req.task_name,
                    request_type=req.request_type,
                    doc_id=req.doc_id,
                    index=req.index,
                    generated_text=str(result_val),
                    fingerprint=req.fingerprint,
                )
            )
        else:
            result_val = ext.get("result", ext.get("log_likelihood", 0.0))
            is_greedy = ext.get("is_greedy", False)
            if isinstance(result_val, list):
                ll_val, is_greedy = result_val[0], result_val[1]
            else:
                ll_val = float(result_val)
            results.append(
                InferenceResult(
                    task_name=req.task_name,
                    request_type=req.request_type,
                    doc_id=req.doc_id,
                    index=req.index,
                    log_likelihood=ll_val,
                    is_greedy=bool(is_greedy),
                    fingerprint=req.fingerprint,
                )
            )

    return save_results(results, Path(output_dir))
