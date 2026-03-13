"""Command-line interface for inference_eval."""

from __future__ import annotations

import json
import logging

import click

from inference_eval import __version__


def _setup_logging(verbosity: str) -> None:
    level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers that spam during inference
    for name in ("urllib3", "requests", "httpx", "httpcore", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """inference_eval - Decouple inference from evaluation.

    Three-phase workflow:

    \b
    1. extract  - Pull task requests from lm-eval-harness
    2. infer    - Run inference with any framework
    3. evaluate - Compute metrics using lm-eval-harness
    """


@cli.command()
@click.option(
    "--tasks",
    "-t",
    required=True,
    help="Comma-separated task names (e.g. gsm8k,hellaswag,mmlu)",
)
@click.option(
    "--output", "-o", required=True, help="Output directory for extracted requests"
)
@click.option(
    "--num-fewshot", "-n", type=int, default=None, help="Number of few-shot examples"
)
@click.option(
    "--limit",
    "-l",
    type=float,
    default=None,
    help="Limit examples per task (int or float fraction)",
)
@click.option(
    "--apply-chat-template", is_flag=True, help="Apply chat template to prompts"
)
@click.option(
    "--system-instruction",
    type=str,
    default=None,
    help="System instruction for chat template",
)
@click.option(
    "--confirm-run-unsafe-code",
    is_flag=True,
    help="Allow execution of unsafe task code",
)
@click.option("--verbosity", type=str, default="INFO", help="Logging verbosity")
def extract(
    tasks: str,
    output: str,
    num_fewshot: int | None,
    limit: float | None,
    apply_chat_template: bool,
    system_instruction: str | None,
    confirm_run_unsafe_code: bool,
    verbosity: str,
) -> None:
    """Extract inference requests from lm-eval-harness tasks.

    Runs the specified tasks with a dummy model that captures all requests
    and saves them as JSONL files for external inference.
    """
    _setup_logging(verbosity)
    from inference_eval.extract import extract_requests

    task_list = [t.strip() for t in tasks.split(",")]

    int_limit = int(limit) if limit is not None and limit == int(limit) else limit

    counts = extract_requests(
        tasks=task_list,
        output_dir=output,
        num_fewshot=num_fewshot,
        limit=int_limit,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        verbosity=verbosity,
    )

    click.echo(f"\nExtracted requests to {output}:")
    for key, count in sorted(counts.items()):
        click.echo(f"  {key}: {count} requests")
    click.echo(f"\nTotal: {sum(counts.values())} requests")


@cli.command()
@click.option(
    "--requests",
    "-r",
    required=True,
    help="Directory containing extracted requests",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output directory for inference results",
)
@click.option(
    "--engine",
    "-e",
    required=True,
    help="Inference engine: vllm, sglang, openai, server",
)
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="Server URL (for server/openai engine, e.g. http://localhost:8068/v1)",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Max parallel requests (server/openai engine, default 64)",
)
@click.option(
    "--api-type",
    type=click.Choice(["auto", "completions", "chat"]),
    default=None,
    help=(
        "API endpoint type for server engine: "
        "'completions' (/v1/completions), "
        "'chat' (/v1/chat/completions), "
        "or 'auto' (try completions, fall back to chat)"
    ),
)
@click.option(
    "--engine-args",
    type=str,
    default="{}",
    help="JSON string of extra engine kwargs",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Batch size for process_requests calls",
)
@click.option(
    "--tasks",
    "-t",
    type=str,
    default=None,
    help="Comma-separated task names to filter",
)
@click.option(
    "--verbosity",
    type=str,
    default="INFO",
    help="Logging verbosity",
)
def infer(
    requests: str,
    output: str,
    engine: str,
    model: str,
    base_url: str | None,
    max_concurrent: int | None,
    api_type: str | None,
    engine_args: str,
    batch_size: int,
    tasks: str | None,
    verbosity: str,
) -> None:
    """Run inference on extracted requests using a specified engine.

    \b
    Examples:
      # Local vLLM (batched)
      inference-eval infer -r ./requests -o ./results \\
          -e vllm -m meta-llama/Llama-3-8B-Instruct
    \b
      # Running vLLM / SGLang server (auto-detect API)
      inference-eval infer -r ./requests -o ./results \\
          -e server -m Qwen3-8B \\
          --base-url http://localhost:8068/v1 --max-concurrent 64
    \b
      # Force chat completions endpoint
      inference-eval infer -r ./requests -o ./results \\
          -e server -m Qwen3-8B \\
          --base-url http://localhost:8068/v1 --api-type chat
    """
    _setup_logging(verbosity)
    from inference_eval.infer import run_inference

    kwargs = json.loads(engine_args)
    kwargs["model"] = model
    if base_url is not None:
        kwargs["base_url"] = base_url
    if max_concurrent is not None:
        kwargs["max_concurrent"] = max_concurrent
    if api_type is not None:
        kwargs["api_type"] = api_type

    task_list = [t.strip() for t in tasks.split(",")] if tasks else None

    counts = run_inference(
        requests_dir=requests,
        output_dir=output,
        engine=engine,
        engine_kwargs=kwargs,
        batch_size=batch_size,
        tasks=task_list,
    )

    click.echo(f"\nInference results saved to {output}:")
    for key, count in sorted(counts.items()):
        click.echo(f"  {key}: {count} results")


@cli.command()
@click.option(
    "--results", "-r", required=True, help="Directory containing inference results"
)
@click.option(
    "--requests",
    type=str,
    default=None,
    help="Directory with extracted requests (for config)",
)
@click.option(
    "--tasks",
    "-t",
    type=str,
    default=None,
    help="Comma-separated task names to evaluate",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output JSON file for evaluation scores",
)
@click.option(
    "--num-fewshot",
    "-n",
    type=int,
    default=None,
    help="Number of few-shot examples (overrides config)",
)
@click.option(
    "--limit", "-l", type=float, default=None, help="Limit examples (overrides config)"
)
@click.option(
    "--confirm-run-unsafe-code",
    is_flag=True,
    help="Allow execution of unsafe task code",
)
@click.option("--log-samples", is_flag=True, help="Log individual sample results")
@click.option("--verbosity", type=str, default="INFO", help="Logging verbosity")
def evaluate(
    results: str,
    requests: str | None,
    tasks: str | None,
    output: str | None,
    num_fewshot: int | None,
    limit: float | None,
    confirm_run_unsafe_code: bool,
    log_samples: bool,
    verbosity: str,
) -> None:
    """Evaluate inference results using lm-eval-harness metrics.

    Feeds pre-computed results back to lm-eval-harness for metric
    computation and displays scores.
    """
    _setup_logging(verbosity)
    from inference_eval.evaluate import evaluate_results

    task_list = [t.strip() for t in tasks.split(",")] if tasks else None
    int_limit = int(limit) if limit is not None and limit == int(limit) else limit

    evaluate_results(
        results_dir=results,
        requests_dir=requests,
        tasks=task_list,
        output_file=output,
        num_fewshot=num_fewshot,
        limit=int_limit,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        log_samples=log_samples,
        verbosity=verbosity,
    )


@cli.command()
@click.option("--external-file", "-f", required=True, help="External results JSON file")
@click.option(
    "--requests", "-r", required=True, help="Directory with extracted requests"
)
@click.option(
    "--output", "-o", required=True, help="Output directory for converted results"
)
@click.option("--verbosity", type=str, default="INFO", help="Logging verbosity")
def convert(
    external_file: str,
    requests: str,
    output: str,
    verbosity: str,
) -> None:
    """Convert externally produced results to inference_eval format.

    Use this when you run inference with a custom script and want to
    import results for evaluation.
    """
    _setup_logging(verbosity)
    from inference_eval.infer import convert_external_results

    counts = convert_external_results(
        external_file=external_file,
        requests_dir=requests,
        output_dir=output,
    )

    click.echo(f"\nConverted results saved to {output}:")
    for key, count in sorted(counts.items()):
        click.echo(f"  {key}: {count} results")


if __name__ == "__main__":
    cli()
