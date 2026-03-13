"""Scoreboard — persistent leaderboard across models and settings.

Each evaluation run appends a row to a JSONL file.  A companion CSV
file is auto-generated for easy viewing in Excel / Google Sheets.
The ``summary`` command renders a clean terminal table.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

DEFAULT_SCOREBOARD = "scoreboard.jsonl"

# Metrics to prefer when picking the "primary" metric for a task.
_PREFERRED_METRICS = [
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "acc_norm,none",
    "acc,none",
    "mc2",
    "f1,none",
    "em,none",
    "bleu,none",
    "word_perplexity",
]

# Short names for metrics (for compact terminal headers).
_METRIC_SHORT: dict[str, str] = {
    "exact_match,flexible-extract": "em_flex",
    "exact_match,strict-match": "em_strict",
    "acc_norm,none": "acc_norm",
    "acc,none": "acc",
    "mc2": "mc2",
    "f1,none": "f1",
    "em,none": "em",
    "bleu,none": "bleu",
    "word_perplexity": "ppl",
}


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


def make_entry(
    tag: str,
    eval_results: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a scoreboard entry from evaluation results."""
    raw = eval_results.get("results", {})
    tasks: dict[str, dict[str, float]] = {}
    for task_name, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if k.startswith("alias"):
                continue
            if isinstance(v, (int, float)) and "stderr" not in k:
                clean[k] = round(float(v), 6)
        if clean:
            tasks[task_name] = clean

    return {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": tasks,
        "metadata": metadata or {},
    }


# ------------------------------------------------------------------
# Persistence (JSONL — one line per run, easy to append)
# ------------------------------------------------------------------


def append_entry(entry: dict[str, Any], path: str | Path) -> None:
    """Append a scoreboard entry and auto-regenerate the CSV companion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    logger.info("Scoreboard entry '%s' appended to %s", entry["tag"], path)

    _rebuild_csv(path)


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load all scoreboard entries from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ------------------------------------------------------------------
# CSV auto-generation
# ------------------------------------------------------------------


def _rebuild_csv(jsonl_path: Path) -> None:
    """Regenerate the companion CSV from the JSONL scoreboard."""
    entries = load_entries(jsonl_path)
    if not entries:
        return
    csv_path = jsonl_path.with_suffix(".csv")
    _export_csv(entries, csv_path)
    logger.info("Scoreboard CSV updated: %s", csv_path)


def _export_csv(
    entries: list[dict[str, Any]],
    path: Path,
    tasks: list[str] | None = None,
    metric: str | None = None,
) -> None:
    """Write entries to a CSV file."""
    cols = _build_columns(entries, tasks, metric)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols["headers"])
        writer.writerows(cols["rows"])


def export_csv_string(
    entries: list[dict[str, Any]],
    tasks: list[str] | None = None,
    metric: str | None = None,
) -> str:
    """Return entries as a CSV string."""
    cols = _build_columns(entries, tasks, metric)
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols["headers"])
    writer.writerows(cols["rows"])
    return buf.getvalue()


# ------------------------------------------------------------------
# Primary metric selection
# ------------------------------------------------------------------


def _pick_primary_metric(
    task_name: str,
    available_metrics: set[str],
    user_metric: str | None = None,
) -> str | None:
    if user_metric and user_metric in available_metrics:
        return user_metric
    for pref in _PREFERRED_METRICS:
        if pref in available_metrics:
            return pref
    return next(iter(sorted(available_metrics)), None)


def _short_metric(metric: str) -> str:
    return _METRIC_SHORT.get(metric, metric)


# ------------------------------------------------------------------
# Column building (shared by terminal table and CSV)
# ------------------------------------------------------------------


def _build_columns(
    entries: list[dict[str, Any]],
    tasks: list[str] | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Build headers + rows for both terminal and CSV rendering."""
    all_tasks: dict[str, set[str]] = {}
    for entry in entries:
        for task_name, metrics in entry.get("tasks", {}).items():
            all_tasks.setdefault(task_name, set()).update(metrics.keys())

    show_tasks = (
        [t for t in tasks if t in all_tasks] if tasks else sorted(all_tasks.keys())
    )

    primary: dict[str, str] = {}
    for t in show_tasks:
        m = _pick_primary_metric(t, all_tasks[t], metric)
        if m:
            primary[t] = m

    headers = ["Tag", "Timestamp"] + [
        f"{t} ({_short_metric(primary[t])})" if t in primary else t for t in show_tasks
    ]

    rows = []
    for entry in entries:
        tag = entry.get("tag", "?")
        ts = entry.get("timestamp", "")[:19]
        row: list[str] = [tag, ts]
        for t in show_tasks:
            task_data = entry.get("tasks", {}).get(t, {})
            m = primary.get(t)
            val = task_data.get(m) if m else None
            row.append(f"{val:.4f}" if val is not None else "-")
        rows.append(row)

    return {"headers": headers, "rows": rows, "tasks": show_tasks}


# ------------------------------------------------------------------
# Terminal rendering
# ------------------------------------------------------------------


def render_summary(
    entries: list[dict[str, Any]],
    tasks: list[str] | None = None,
    metric: str | None = None,
    fmt: str = "simple_grid",
) -> str:
    """Render a comparison table from scoreboard entries.

    Args:
        entries: Loaded scoreboard entries.
        tasks: Filter to these tasks.  None = all tasks seen.
        metric: Force this metric for all tasks.  None = auto-pick.
        fmt: ``tabulate`` format string (default: simple_grid).

    Returns:
        Formatted table string.
    """
    if not entries:
        return "(no scoreboard entries)"

    cols = _build_columns(entries, tasks, metric)
    if not cols["tasks"]:
        return "(no matching tasks in scoreboard)"

    return tabulate(
        cols["rows"],
        headers=cols["headers"],
        tablefmt=fmt,
        disable_numparse=True,
    )


def render_detail(entry: dict[str, Any]) -> str:
    """Render a detailed view of a single scoreboard entry."""
    lines = [
        f"Tag:       {entry.get('tag', '?')}",
        f"Timestamp: {entry.get('timestamp', '?')}",
    ]
    meta = entry.get("metadata", {})
    if meta:
        lines.append(f"Metadata:  {json.dumps(meta)}")
    lines.append("")
    for task_name, metrics in sorted(entry.get("tasks", {}).items()):
        lines.append(f"  {task_name}:")
        for k, v in sorted(metrics.items()):
            lines.append(f"    {k}: {v:.4f}")
    return "\n".join(lines)
