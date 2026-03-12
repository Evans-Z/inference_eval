"""OfflineLM - returns pre-computed results for lm-eval-harness evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lm_eval.api.model import LM

from inference_eval.schema import InferenceResult, load_results
from inference_eval.utils import (
    get_doc_id,
    get_task_name,
    make_content_key,
    make_result_key,
)

logger = logging.getLogger(__name__)


class OfflineLM(LM):
    """An LM that returns pre-computed results from disk.

    Used in the 'evaluate' phase to compute metrics on externally
    generated inference results.
    """

    def __init__(self, results_dir: str | Path) -> None:
        super().__init__()
        self.results_dir = Path(results_dir)
        self._results = load_results(self.results_dir)
        self._index_lookup: dict[str, InferenceResult] = {}
        self._content_lookup: dict[str, InferenceResult] = {}
        self._build_lookups()

    def _build_lookups(self) -> None:
        for result in self._results:
            idx_key = make_result_key(
                result.task_name,
                result.request_type,
                result.doc_id,
                result.index,
            )
            self._index_lookup[idx_key] = result

            if result.fingerprint:
                self._content_lookup[result.fingerprint] = result

        logger.info(
            "Loaded %d results (%d unique index keys)",
            len(self._results),
            len(self._index_lookup),
        )

    def _find_result(
        self,
        task_name: str,
        request_type: str,
        doc_id: int,
        index: int,
        context: str,
        continuation: str | None = None,
    ) -> InferenceResult:
        idx_key = make_result_key(task_name, request_type, doc_id, index)
        result = self._index_lookup.get(idx_key)
        if result is not None:
            return result

        content_key = make_content_key(task_name, request_type, context, continuation)
        result = self._content_lookup.get(content_key)
        if result is not None:
            logger.debug("Fell back to content-based matching for %s", idx_key)
            return result

        raise KeyError(
            f"No result found for {idx_key}. "
            f"Available tasks: {sorted(set(r.task_name for r in self._results))}. "
            f"Make sure inference was run for all extracted requests."
        )

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        counters: dict[str, int] = {}
        results = []
        for instance in requests:
            task_name = get_task_name(instance)
            doc_id = get_doc_id(instance)
            context, continuation = instance.args[0], instance.args[1]

            counter_key = f"{task_name}|loglikelihood"
            index = counters.get(counter_key, 0)
            counters[counter_key] = index + 1

            result = self._find_result(
                task_name, "loglikelihood", doc_id, index, context, continuation
            )
            results.append((result.log_likelihood or 0.0, result.is_greedy or False))
        return results

    def loglikelihood_rolling(self, requests: list) -> list[tuple[float, bool]]:
        counters: dict[str, int] = {}
        results = []
        for instance in requests:
            task_name = get_task_name(instance)
            doc_id = get_doc_id(instance)
            context = instance.args[0]

            counter_key = f"{task_name}|loglikelihood_rolling"
            index = counters.get(counter_key, 0)
            counters[counter_key] = index + 1

            result = self._find_result(
                task_name, "loglikelihood_rolling", doc_id, index, context
            )
            results.append((result.log_likelihood or 0.0, result.is_greedy or False))
        return results

    def generate_until(self, requests: list) -> list[str]:
        counters: dict[str, int] = {}
        results = []
        for instance in requests:
            task_name = get_task_name(instance)
            doc_id = get_doc_id(instance)
            context = instance.args[0]

            counter_key = f"{task_name}|generate_until"
            index = counters.get(counter_key, 0)
            counters[counter_key] = index + 1

            result = self._find_result(
                task_name, "generate_until", doc_id, index, context
            )
            results.append(result.generated_text or "")
        return results

    @property
    def eot_token_id(self) -> int:
        return 0

    @property
    def max_length(self) -> int:
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def device(self) -> str:
        return "cpu"

    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        return list(range(len(string.split())))

    def tok_decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)
