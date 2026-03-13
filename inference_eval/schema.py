"""Data models for requests and results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractConfig:
    """Configuration saved during extraction for reproducible evaluation."""

    tasks: list[str]
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

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> ExtractConfig:
        with open(input_dir / "config.json") as f:
            data = json.load(f)
        return cls(**data)


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


def save_requests(requests: list[InferenceRequest], output_dir: Path) -> dict[str, int]:
    """Save requests to disk organized by task/request_type. Returns counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    groups: dict[tuple[str, str], list[InferenceRequest]] = {}
    for req in requests:
        key = (req.task_name, req.request_type)
        groups.setdefault(key, []).append(req)

    for (task_name, req_type), reqs in groups.items():
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        filepath = task_dir / f"{req_type}.jsonl"
        with open(filepath, "w") as f:
            for req in reqs:
                f.write(json.dumps(asdict(req)) + "\n")
        counts[f"{task_name}/{req_type}"] = len(reqs)

    return counts


def load_requests(input_dir: Path) -> list[InferenceRequest]:
    """Load all requests from disk."""
    requests = []
    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    requests.append(InferenceRequest(**data))
    return requests


def save_results(results: list[InferenceResult], output_dir: Path) -> dict[str, int]:
    """Save results to disk organized by task/request_type. Returns counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    groups: dict[tuple[str, str], list[InferenceResult]] = {}
    for res in results:
        key = (res.task_name, res.request_type)
        groups.setdefault(key, []).append(res)

    for (task_name, req_type), res_list in groups.items():
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        filepath = task_dir / f"{req_type}.jsonl"
        with open(filepath, "w") as f:
            for res in res_list:
                f.write(json.dumps(asdict(res)) + "\n")
        counts[f"{task_name}/{req_type}"] = len(res_list)

    return counts


def load_results(input_dir: Path) -> list[InferenceResult]:
    """Load all results from disk."""
    results = []
    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(InferenceResult(**data))
    return results
