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
) -> dict[str, Any]:
    """Evaluate pre-computed results using lm-eval-harness metrics.

    Loads results from disk, runs lm-eval-harness with an OfflineLM
    that returns the pre-computed results, and computes task metrics.

    Args:
        results_dir: Directory containing inference results.
        requests_dir: Directory with extracted requests (to load config).
            If None, tasks and other params must be provided explicitly.
        tasks: Task names to evaluate. Overrides config from requests_dir.
        output_file: Optional path to save evaluation scores as JSON.
        num_fewshot: Number of few-shot examples. Overrides config.
        limit: Example limit per task. Overrides config.
        confirm_run_unsafe_code: Allow unsafe task code execution.
        verbosity: Logging verbosity level.
        log_samples: Whether to log individual sample results.

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

    eval_num_fewshot = (
        num_fewshot
        if num_fewshot is not None
        else (config.num_fewshot if config else None)
    )
    eval_limit = limit if limit is not None else (config.limit if config else None)
    random_seed = config.random_seed if config else 0
    numpy_random_seed = config.numpy_random_seed if config else 1234
    torch_random_seed = config.torch_random_seed if config else 1234
    fewshot_random_seed = config.fewshot_random_seed if config else 1234

    offline_lm = OfflineLM(results_dir)

    logger.info("Evaluating tasks: %s", eval_tasks)
    eval_results = lm_eval.simple_evaluate(
        model=offline_lm,
        tasks=eval_tasks,
        num_fewshot=eval_num_fewshot,
        limit=eval_limit,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
        verbosity=verbosity,
        log_samples=log_samples,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
    )

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = _make_serializable(eval_results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info("Scores saved to %s", output_path)

    _print_results(eval_results)
    return eval_results


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


def _print_results(eval_results: dict[str, Any]) -> None:
    """Print evaluation results in a readable format."""
    results = eval_results.get("results", {})
    if not results:
        logger.warning("No results to display")
        return

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    for task_name, metrics in sorted(results.items()):
        print(f"\n{task_name}:")
        if isinstance(metrics, dict):
            for metric_name, value in sorted(metrics.items()):
                if metric_name.startswith("alias"):
                    continue
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")

    print("\n" + "=" * 70)
