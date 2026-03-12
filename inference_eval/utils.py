"""Utility functions for metadata extraction and request handling."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_task_name(instance: Any) -> str:
    """Extract task name from an lm-eval Instance, handling multiple formats."""
    if hasattr(instance, "task_name") and instance.task_name is not None:
        return str(instance.task_name)
    if hasattr(instance, "metadata"):
        meta = instance.metadata
        if isinstance(meta, dict) and "task" in meta:
            return str(meta["task"])
        if isinstance(meta, (tuple, list)) and len(meta) > 0 and meta[0] is not None:
            return str(meta[0])
    return "unknown"


def get_doc_id(instance: Any) -> int:
    """Extract doc_id from an lm-eval Instance, handling multiple formats."""
    if hasattr(instance, "doc_id") and instance.doc_id is not None:
        return int(instance.doc_id)
    if hasattr(instance, "metadata"):
        meta = instance.metadata
        if isinstance(meta, dict) and "doc_id" in meta:
            return int(meta["doc_id"])
        if isinstance(meta, (tuple, list)) and len(meta) > 2 and meta[2] is not None:
            return int(meta[2])
    if hasattr(instance, "idx"):
        return int(instance.idx)
    return 0


def make_result_key(task_name: str, request_type: str, doc_id: int, index: int) -> str:
    """Create a lookup key for matching requests to results."""
    return f"{task_name}|{request_type}|{doc_id}|{index}"


def make_content_key(
    task_name: str,
    request_type: str,
    context: str,
    continuation: str | None = None,
) -> str:
    """Create a content-based lookup key as a fallback matching strategy."""
    key = f"{task_name}|{request_type}|{context[:200]}"
    if continuation is not None:
        key += f"|{continuation[:200]}"
    return key
