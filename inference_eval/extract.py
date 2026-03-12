"""Extract phase: pull requests from lm-eval-harness tasks and save to disk."""

from __future__ import annotations

import logging
from pathlib import Path

import lm_eval

from inference_eval.models.capture import RequestCaptureLM
from inference_eval.schema import ExtractConfig, save_requests

logger = logging.getLogger(__name__)


def extract_requests(
    tasks: list[str],
    output_dir: str | Path,
    num_fewshot: int | None = None,
    limit: int | float | None = None,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    apply_chat_template: bool = False,
    system_instruction: str | None = None,
    confirm_run_unsafe_code: bool = False,
    verbosity: str = "INFO",
) -> dict[str, int]:
    """Extract all inference requests from the specified lm-eval-harness tasks.

    Runs lm-eval-harness with a dummy model that captures all requests
    without performing actual inference.

    Args:
        tasks: List of task names (e.g. ["gsm8k", "hellaswag", "mmlu"]).
        output_dir: Directory to save extracted requests.
        num_fewshot: Number of few-shot examples.
        limit: Limit number of examples per task.
        random_seed: Random seed for reproducibility.
        numpy_random_seed: Numpy random seed.
        torch_random_seed: Torch random seed.
        fewshot_random_seed: Fewshot random seed.
        apply_chat_template: Whether to apply chat template.
        system_instruction: System instruction for chat template.
        confirm_run_unsafe_code: Allow unsafe task code execution.
        verbosity: Logging verbosity level.

    Returns:
        Dictionary mapping "task_name/request_type" to request counts.
    """
    output_dir = Path(output_dir)

    config = ExtractConfig(
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
    )

    capture_lm = RequestCaptureLM()

    logger.info("Extracting requests for tasks: %s", tasks)
    lm_eval.simple_evaluate(
        model=capture_lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
        verbosity=verbosity,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
    )

    logger.info("Captured %d total requests", len(capture_lm.captured_requests))
    config.save(output_dir)
    counts = save_requests(capture_lm.captured_requests, output_dir)

    for key, count in sorted(counts.items()):
        logger.info("  %s: %d requests", key, count)

    return counts
