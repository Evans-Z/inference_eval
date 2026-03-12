"""RequestCaptureLM - captures requests from lm-eval-harness and saves to disk."""

from __future__ import annotations

import logging
from typing import Any

from lm_eval.api.model import LM

from inference_eval.schema import InferenceRequest
from inference_eval.utils import get_doc_id, get_task_name

logger = logging.getLogger(__name__)


class RequestCaptureLM(LM):
    """A dummy LM that captures all requests instead of running inference.

    Used in the 'extract' phase to save task requests to disk without
    requiring an actual model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured_requests: list[InferenceRequest] = []
        self._counters: dict[tuple[str, str], int] = {}

    def _next_index(self, task_name: str, request_type: str) -> int:
        key = (task_name, request_type)
        idx = self._counters.get(key, 0)
        self._counters[key] = idx + 1
        return idx

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        for instance in requests:
            task_name = get_task_name(instance)
            doc_id = get_doc_id(instance)
            args = instance.args
            context, continuation = args[0], args[1]

            self.captured_requests.append(
                InferenceRequest(
                    task_name=task_name,
                    request_type="loglikelihood",
                    doc_id=doc_id,
                    index=self._next_index(task_name, "loglikelihood"),
                    context=context,
                    continuation=continuation,
                )
            )
        return [(0.0, False)] * len(requests)

    def loglikelihood_rolling(self, requests: list) -> list[tuple[float, bool]]:
        for instance in requests:
            task_name = get_task_name(instance)
            doc_id = get_doc_id(instance)
            args = instance.args
            context = args[0]

            self.captured_requests.append(
                InferenceRequest(
                    task_name=task_name,
                    request_type="loglikelihood_rolling",
                    doc_id=doc_id,
                    index=self._next_index(task_name, "loglikelihood_rolling"),
                    context=context,
                )
            )
        return [(0.0, False)] * len(requests)

    def generate_until(self, requests: list) -> list[str]:
        for instance in requests:
            task_name = get_task_name(instance)
            doc_id = get_doc_id(instance)
            args = instance.args
            context = args[0]
            gen_kwargs = args[1] if len(args) > 1 else {}
            if not isinstance(gen_kwargs, dict):
                gen_kwargs = {}

            self.captured_requests.append(
                InferenceRequest(
                    task_name=task_name,
                    request_type="generate_until",
                    doc_id=doc_id,
                    index=self._next_index(task_name, "generate_until"),
                    context=context,
                    generation_kwargs=gen_kwargs,
                )
            )
        return [""] * len(requests)

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
