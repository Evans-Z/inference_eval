"""Abstract base class for inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from inference_eval.schema import InferenceRequest, InferenceResult


class InferenceEngine(ABC):
    """Abstract interface for inference backends.

    Subclass this to integrate any inference framework (vLLM, SGLang,
    custom engines, etc.) with inference_eval.

    Each engine must implement two methods:
    - generate: for generate_until requests (text generation)
    - compute_loglikelihood: for loglikelihood requests (scoring)
    """

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        """Generate text completions for a batch of prompts.

        Args:
            prompts: List of input prompts/contexts.
            gen_kwargs: List of generation kwargs per prompt, containing
                keys like 'until' (stop sequences), 'max_gen_toks', etc.

        Returns:
            List of generated text strings (one per prompt).
        """

    @abstractmethod
    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts.

        Args:
            contexts: List of context/prefix strings.
            continuations: List of continuation/target strings.

        Returns:
            List of (log_likelihood, is_greedy) tuples.
        """

    def process_requests(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """Process a batch of InferenceRequests and return InferenceResults."""
        gen_requests = [r for r in requests if r.request_type == "generate_until"]
        ll_requests = [
            r
            for r in requests
            if r.request_type in ("loglikelihood", "loglikelihood_rolling")
        ]

        results: list[InferenceResult] = []

        if gen_requests:
            prompts = [r.context for r in gen_requests]
            gen_kwargs = [r.generation_kwargs or {} for r in gen_requests]
            generated = self.generate(prompts, gen_kwargs)
            for req, text in zip(gen_requests, generated):
                results.append(
                    InferenceResult(
                        task_name=req.task_name,
                        request_type=req.request_type,
                        doc_id=req.doc_id,
                        index=req.index,
                        generated_text=text,
                        fingerprint=req.fingerprint,
                    )
                )

        if ll_requests:
            contexts = [r.context for r in ll_requests]
            continuations = [r.continuation or "" for r in ll_requests]
            ll_results = self.compute_loglikelihood(contexts, continuations)
            for req, (ll, greedy) in zip(ll_requests, ll_results):
                results.append(
                    InferenceResult(
                        task_name=req.task_name,
                        request_type=req.request_type,
                        doc_id=req.doc_id,
                        index=req.index,
                        log_likelihood=ll,
                        is_greedy=greedy,
                        fingerprint=req.fingerprint,
                    )
                )

        return results
