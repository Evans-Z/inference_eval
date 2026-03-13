"""Data models for requests and results."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractConfig:
    """Configuration saved during extraction for reproducible evaluation.

    Supports incremental extraction: calling :meth:`save` on a directory
    that already contains a ``config.json`` will **merge** the new tasks
    into the existing config rather than overwriting it.
    """

    tasks: list[str] = field(default_factory=list)
    num_fewshot: int | None = None
    limit: int | float | None = None
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234
    apply_chat_template: bool = False
    system_instruction: str | None = None
    task_group_map: dict[str, list[str]] = field(default_factory=dict)
    extracted_tasks: list[str] = field(default_factory=list)
    task_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, output_dir: Path) -> None:
        """Save config, merging with any existing config in the directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"

        if config_path.exists():
            try:
                existing = ExtractConfig.load(output_dir)
                self._merge_from(existing)
                logger.info(
                    "Merged with existing config (%d previous tasks)",
                    len(existing.tasks),
                )
            except Exception:
                logger.warning("Could not read existing config.json; overwriting.")

        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def _merge_from(self, existing: ExtractConfig) -> None:
        """Merge *existing* config into self (self = latest extraction)."""
        self.tasks = _dedup_ordered(existing.tasks + self.tasks)

        merged_map = dict(existing.task_group_map)
        merged_map.update(self.task_group_map)
        self.task_group_map = merged_map

        self.extracted_tasks = _dedup_ordered(
            existing.extracted_tasks + self.extracted_tasks
        )

        merged_settings = dict(existing.task_settings)
        merged_settings.update(self.task_settings)
        self.task_settings = merged_settings

    @classmethod
    def load(cls, input_dir: Path) -> ExtractConfig:
        with open(input_dir / "config.json") as f:
            data = json.load(f)
        return cls(**data)


def _dedup_ordered(seq: list[str]) -> list[str]:
    """Remove duplicates while preserving insertion order."""
    return list(dict.fromkeys(seq))


# ------------------------------------------------------------------
# Directory helpers — task groups nest under their parent folder
# ------------------------------------------------------------------


def _task_dir(
    base: Path,
    task_name: str,
    task_group_map: dict[str, list[str]],
) -> Path:
    """Return the on-disk directory for *task_name*.

    If the task belongs to a group (e.g. ``mmlu_anatomy`` is part of
    the ``mmlu`` group), the path is ``base / mmlu / mmlu_anatomy``.
    Otherwise it is simply ``base / task_name``.
    """
    for group, subtasks in task_group_map.items():
        if task_name in subtasks:
            return base / group / task_name
    return base / task_name


# ------------------------------------------------------------------
# Request / Result data models
# ------------------------------------------------------------------


@dataclass
class InferenceRequest:
    """A single inference request extracted from lm-eval-harness."""

    task_name: str
    request_type: str
    doc_id: int
    index: int
    context: str
    continuation: str | None = None
    generation_kwargs: dict[str, Any] | None = None

    @property
    def fingerprint(self) -> str:
        content = f"{self.task_name}|{self.request_type}|{self.doc_id}|"
        content += self.context[:200]
        if self.continuation is not None:
            content += f"|{self.continuation[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class InferenceResult:
    """Result of a single inference request."""

    task_name: str
    request_type: str
    doc_id: int
    index: int
    generated_text: str | None = None
    log_likelihood: float | None = None
    is_greedy: bool | None = None
    fingerprint: str = ""

    def to_lm_eval_response(self) -> Any:
        if self.request_type == "generate_until":
            return self.generated_text or ""
        else:
            return (self.log_likelihood or 0.0, self.is_greedy or False)


@dataclass
class TaskRequests:
    """All requests for a single task, organized by request type."""

    task_name: str
    requests: dict[str, list[InferenceRequest]] = field(default_factory=dict)


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


def save_requests(
    requests: list[InferenceRequest],
    output_dir: Path,
    task_group_map: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """Save requests to disk organized by task/request_type. Returns counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gmap = task_group_map or {}
    counts: dict[str, int] = {}

    groups: dict[tuple[str, str], list[InferenceRequest]] = {}
    for req in requests:
        key = (req.task_name, req.request_type)
        groups.setdefault(key, []).append(req)

    for (task_name, req_type), reqs in groups.items():
        tdir = _task_dir(output_dir, task_name, gmap)
        tdir.mkdir(parents=True, exist_ok=True)
        filepath = tdir / f"{req_type}.jsonl"
        with open(filepath, "w") as f:
            for req in reqs:
                f.write(json.dumps(asdict(req)) + "\n")
        counts[f"{task_name}/{req_type}"] = len(reqs)

    return counts


def load_requests(input_dir: Path) -> list[InferenceRequest]:
    """Load all requests from disk (works with flat or nested layout)."""
    requests = []
    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    requests.append(InferenceRequest(**data))
    return requests


def save_results(
    results: list[InferenceResult],
    output_dir: Path,
    task_group_map: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """Save results to disk organized by task/request_type. Returns counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gmap = task_group_map or {}
    counts: dict[str, int] = {}

    groups: dict[tuple[str, str], list[InferenceResult]] = {}
    for res in results:
        key = (res.task_name, res.request_type)
        groups.setdefault(key, []).append(res)

    for (task_name, req_type), res_list in groups.items():
        tdir = _task_dir(output_dir, task_name, gmap)
        tdir.mkdir(parents=True, exist_ok=True)
        filepath = tdir / f"{req_type}.jsonl"
        with open(filepath, "w") as f:
            for res in res_list:
                f.write(json.dumps(asdict(res)) + "\n")
        counts[f"{task_name}/{req_type}"] = len(res_list)

    return counts


def load_results(input_dir: Path) -> list[InferenceResult]:
    """Load all results from disk (works with flat or nested layout)."""
    results = []
    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(InferenceResult(**data))
    return results
