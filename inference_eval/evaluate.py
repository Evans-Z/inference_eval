"""Evaluate phase: compute metrics using lm-eval-harness on pre-computed results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lm_eval

from inference_eval.models.offline import OfflineLM
from inference_eval.schema import ExtractConfig

logger = logging.getLogger(__name__)


def evaluate_results(
    results_dir: str | Path,
    requests_dir: str | Path | None = None,
    tasks: list[str] | None = None,
    output_file: str | Path | None = None,
    num_fewshot: int | None = None,
    limit: int | float | None = None,
    confirm_run_unsafe_code: bool = False,
    verbosity: str = "INFO",
    log_samples: bool = False,
    tag: str | None = None,
    scoreboard: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate pre-computed results using lm-eval-harness metrics.

    Supports per-task ``num_fewshot`` / ``limit`` that were saved
    during extraction.  Tasks with different settings are evaluated
    in separate ``simple_evaluate`` calls and the results are merged.

    When *tag* is provided the results are also appended to a
    persistent **scoreboard** JSONL file, enabling cross-model
    comparison via ``inference-eval summary``.

    Args:
        results_dir: Directory containing inference results.
        requests_dir: Directory with extracted requests (to load config).
            If None, tasks and other params must be provided explicitly.
        tasks: Task names to evaluate. Overrides config from requests_dir.
        output_file: Optional path to save evaluation scores as JSON.
        num_fewshot: Number of few-shot examples. Overrides ALL per-task
            settings when provided explicitly on the CLI.
        limit: Example limit per task. Overrides ALL per-task settings.
        confirm_run_unsafe_code: Allow unsafe task code execution.
        verbosity: Logging verbosity level.
        log_samples: Whether to log individual sample results.
        tag: Identifier for this run (e.g. model name + settings).
            When set, results are appended to the scoreboard.
        scoreboard: Path to the scoreboard JSONL file.
            Defaults to ``scoreboard.jsonl`` in the current directory.

    Returns:
        Dictionary containing evaluation results with task metrics.
    """
    results_dir = Path(results_dir)

    config = None
    if requests_dir is not None:
        config_path = Path(requests_dir) / "config.json"
        if config_path.exists():
            config = ExtractConfig.load(Path(requests_dir))
            logger.info("Loaded extraction config from %s", config_path)

    eval_tasks = tasks
    if eval_tasks is None and config is not None:
        eval_tasks = config.tasks
    if eval_tasks is None:
        raise ValueError(
            "Tasks must be specified either via --tasks flag or via "
            "--requests-dir containing config.json from extraction."
        )

    random_seed = config.random_seed if config else 0
    numpy_random_seed = config.numpy_random_seed if config else 1234
    torch_random_seed = config.torch_random_seed if config else 1234
    fewshot_random_seed = config.fewshot_random_seed if config else 1234

    offline_lm = OfflineLM(results_dir)

    # ------------------------------------------------------------------
    # Group tasks by their per-task settings so each group can be
    # evaluated with the correct num_fewshot / limit.
    # ------------------------------------------------------------------
    cli_override_fewshot = num_fewshot is not None
    cli_override_limit = limit is not None

    task_settings = config.task_settings if config else {}
    groups = _group_tasks_by_settings(
        eval_tasks,
        task_settings,
        default_fewshot=config.num_fewshot if config else None,
        default_limit=config.limit if config else None,
        override_fewshot=num_fewshot if cli_override_fewshot else None,
        override_limit=limit if cli_override_limit else None,
        cli_override_fewshot=cli_override_fewshot,
        cli_override_limit=cli_override_limit,
    )

    merged_results: dict[str, Any] = {}

    for (group_fewshot, group_limit), group_tasks in groups.items():
        logger.info(
            "Evaluating %d task(s) with num_fewshot=%s, limit=%s: %s",
            len(group_tasks),
            group_fewshot,
            group_limit,
            group_tasks,
        )
        group_eval = lm_eval.simple_evaluate(
            model=offline_lm,
            tasks=group_tasks,
            num_fewshot=group_fewshot,
            limit=group_limit,
            random_seed=random_seed,
            numpy_random_seed=numpy_random_seed,
            torch_random_seed=torch_random_seed,
            fewshot_random_seed=fewshot_random_seed,
            verbosity=verbosity,
            log_samples=log_samples,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
        )
        _merge_eval_results(merged_results, group_eval)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = _make_serializable(merged_results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info("Scores saved to %s", output_path)

    if tag is not None:
        from inference_eval.scoreboard import (
            DEFAULT_SCOREBOARD,
            append_entry,
            make_entry,
        )

        sb_path = Path(scoreboard) if scoreboard else Path(DEFAULT_SCOREBOARD)
        entry = make_entry(tag, merged_results)
        append_entry(entry, sb_path)

    _print_results_compact(merged_results, tag)
    return merged_results


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

_SettingsKey = tuple[int | None, int | float | None]


def _group_tasks_by_settings(
    eval_tasks: list[str],
    task_settings: dict[str, dict[str, Any]],
    *,
    default_fewshot: int | None,
    default_limit: int | float | None,
    override_fewshot: int | None,
    override_limit: int | float | None,
    cli_override_fewshot: bool,
    cli_override_limit: bool,
) -> dict[_SettingsKey, list[str]]:
    """Group tasks that share the same (num_fewshot, limit)."""
    groups: dict[_SettingsKey, list[str]] = {}
    for task in eval_tasks:
        ts = task_settings.get(task, {})
        nf = (
            override_fewshot
            if cli_override_fewshot
            else ts.get("num_fewshot", default_fewshot)
        )
        lim = override_limit if cli_override_limit else ts.get("limit", default_limit)
        key: _SettingsKey = (nf, lim)
        groups.setdefault(key, []).append(task)
    return groups


def _merge_eval_results(merged: dict[str, Any], new: dict[str, Any]) -> None:
    """Merge *new* eval results dict into *merged* in-place."""
    for top_key in ("results", "configs", "versions", "n-shot", "samples"):
        if top_key in new:
            merged.setdefault(top_key, {}).update(new[top_key])


def _make_serializable(obj: Any) -> Any:
    """Convert eval results to a JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, bool):
        return obj
    if obj is None:
        return obj
    return str(obj)


def _print_results_compact(eval_results: dict[str, Any], tag: str | None) -> None:
    """Print a compact one-line-per-task summary."""
    results = eval_results.get("results", {})
    if not results:
        logger.warning("No results to display")
        return

    from inference_eval.scoreboard import _pick_primary_metric, _short_metric

    print()
    if tag:
        print(f"[{tag}]")
    for task_name, metrics in sorted(results.items()):
        if not isinstance(metrics, dict):
            continue
        available = {
            k
            for k, v in metrics.items()
            if isinstance(v, (int, float))
            and not k.startswith("alias")
            and "stderr" not in k
        }
        pm = _pick_primary_metric(task_name, available)
        if pm and pm in metrics:
            short = _short_metric(pm)
            print(f"  {task_name:30s} {short}={metrics[pm]:.4f}")
        else:
            first = next(
                (
                    (k, v)
                    for k, v in sorted(metrics.items())
                    if isinstance(v, (int, float))
                    and not k.startswith("alias")
                    and "stderr" not in k
                ),
                None,
            )
            if first:
                print(f"  {task_name:30s} {first[0]}={first[1]:.4f}")
    print()
