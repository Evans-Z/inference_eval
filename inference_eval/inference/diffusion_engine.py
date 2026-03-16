"""Diffusion language model engine (LLaDA2, Dream, etc.).

Supports both ``generate_until`` and ``loglikelihood`` tasks.
Uses `dllm <https://github.com/ZHZisZZ/dllm>`_ when available.

**Speed levers** (combine for maximum throughput):

- ``gpu_batch_size=4``  — batch 4 prompts per GPU forward pass
- ``devices="cuda:0,cuda:1,cuda:2,cuda:3"`` — multi-GPU parallelism

The batched generation **reimplements** LLaDA2's block-diffusion
loop with full batch support.  ``model.forward()`` (the heavy part)
runs on the whole batch; per-sample confidence selection is a cheap
Python loop.

Prompts from lm-eval-harness are raw text.  By default this engine
wraps each prompt in a chat template (``apply_chat_template=True``).
**Do NOT combine with ``inference-eval extract --apply-chat-template``**.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from inference_eval.inference.base import InferenceEngine

logger = logging.getLogger(__name__)


def _truncate_at_stop(text: str, stop_sequences: list[str]) -> str:
    earliest = len(text)
    for s in stop_sequences:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


@dataclass
class _Worker:
    model: Any
    tokenizer: Any
    sampler: Any
    sampler_config_cls: Any
    device: str


class DiffusionEngine(InferenceEngine):
    """Inference engine for diffusion language models.

    Args:
        model: HuggingFace model name or local path.
        device: Single device (``"auto"``, ``"cuda:0"``).
        devices: Comma-separated devices for multi-GPU.
        gpu_batch_size: Prompts per GPU forward pass (try 2, 4, 8).
        dtype: ``"bfloat16"``, ``"float16"``, ``"float32"``.
        gen_length: New tokens to generate.
        block_length: Block size for diffusion sampling.
        steps: Demasking steps per block.
        temperature: 0 = greedy.
        apply_chat_template: Wrap prompt as chat user message.
        eos_early_stop: Stop early on EOS.
        mc_num: Monte-Carlo samples for ``get_log_likelihood``.
        threshold: Confidence threshold for token acceptance.
        use_dllm: Use dllm sampler for single-prompt fallback.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        devices: str | None = None,
        gpu_batch_size: int = 1,
        dtype: str = "bfloat16",
        gen_length: int = 512,
        block_length: int = 32,
        steps: int = 32,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
        eos_early_stop: bool = True,
        mc_num: int = 32,
        threshold: float = 0.95,
        use_dllm: bool = True,
        **extra_kwargs: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for DiffusionEngine."
            ) from e

        self._torch = torch
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer
        self._model_path = model
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self._gpu_batch_size = gpu_batch_size
        self._gen_length = gen_length
        self._block_length = block_length
        self._steps = steps
        self._temperature = temperature
        self._apply_chat_template = apply_chat_template
        self._eos_early_stop = eos_early_stop
        self._mc_num = mc_num
        self._threshold = threshold
        self._use_dllm = use_dllm

        device_list = [d.strip() for d in devices.split(",")] if devices else [device]
        self._workers: list[_Worker] = [self._create_worker(d) for d in device_list]
        logger.info(
            "DiffusionEngine: %d GPU(s) %s, batch=%d, steps=%d, block=%d, gen=%d",
            len(self._workers),
            [w.device for w in self._workers],
            gpu_batch_size,
            steps,
            block_length,
            gen_length,
        )

    def _create_worker(self, device: str) -> _Worker:
        logger.info("Loading model on %s: %s", device, self._model_path)
        tok = self._AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        mdl = (
            self._AutoModelForCausalLM.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                device_map=device,
            )
            .to(self._torch_dtype)
            .eval()
        )
        sampler = sampler_cfg = None
        if self._use_dllm:
            try:
                import dllm

                sampler = dllm.pipelines.llada2.LLaDA2Sampler(model=mdl, tokenizer=tok)
                sampler_cfg = dllm.pipelines.llada2.LLaDA2SamplerConfig
            except Exception:
                pass
        return _Worker(mdl, tok, sampler, sampler_cfg, device)

    # ------------------------------------------------------------------
    # tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, w: _Worker, prompt: str) -> Any:
        if self._apply_chat_template:
            return w.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(w.model.device)
        return w.tokenizer(prompt, return_tensors="pt").input_ids.to(w.model.device)

    # ------------------------------------------------------------------
    # generate (dispatch)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        if not prompts:
            return []
        nw = len(self._workers)
        if nw == 1:
            return self._gen_worker(self._workers[0], prompts, gen_kwargs)

        buckets: list[list[tuple[int, str, dict]]] = [[] for _ in range(nw)]
        for i, (p, k) in enumerate(zip(prompts, gen_kwargs)):
            buckets[i % nw].append((i, p, k))
        results: list[str | None] = [None] * len(prompts)

        def _run(wi: int) -> list[tuple[int, str]]:
            b = buckets[wi]
            ts = self._gen_worker(
                self._workers[wi], [x[1] for x in b], [x[2] for x in b]
            )
            return [(x[0], t) for x, t in zip(b, ts)]

        with ThreadPoolExecutor(max_workers=nw) as pool:
            for fut in as_completed(
                pool.submit(_run, i) for i in range(nw) if buckets[i]
            ):
                for idx, txt in fut.result():
                    results[idx] = txt
        return [r or "" for r in results]

    def _gen_worker(self, w: _Worker, prompts: list, kws: list) -> list:
        bs = self._gpu_batch_size
        out: list[str] = []
        for i in range(0, len(prompts), bs):
            bp = prompts[i : i + bs]
            bk = kws[i : i + bs]
            if len(bp) > 1 and bs > 1:
                out.extend(self._gen_batch(w, bp, bk))
            else:
                for p, k in zip(bp, bk):
                    out.append(self._gen_single(w, p, k))
        return out

    # ------------------------------------------------------------------
    # single-prompt generation (uses dllm or model.generate)
    # ------------------------------------------------------------------

    def _gen_single(self, w: _Worker, prompt: str, kw: dict) -> str:
        ids = self._tokenize(w, prompt)
        ilen = ids.shape[1]
        gl = kw.get("max_gen_toks", kw.get("max_tokens", self._gen_length))
        t = kw.get("temperature", self._temperature)

        if w.sampler is not None:
            cfg = w.sampler_config_cls(
                max_new_tokens=gl,
                block_size=self._block_length,
                steps_per_block=self._steps,
                temperature=t,
            )
            o = w.sampler.sample(ids, cfg)
            if hasattr(o, "sequences"):
                o = o.sequences
            out = o[0] if self._torch.is_tensor(o) else o
            text = w.tokenizer.decode(out[ilen:], skip_special_tokens=True)
        else:
            with self._torch.no_grad():
                o = w.model.generate(
                    inputs=ids,
                    gen_length=gl,
                    block_length=self._block_length,
                    steps=self._steps,
                    temperature=t,
                    eos_early_stop=self._eos_early_stop,
                )
            text = w.tokenizer.decode(o[0][ilen:], skip_special_tokens=True)

        stop = kw.get("until", kw.get("stop", []))
        if isinstance(stop, str):
            stop = [stop]
        return _truncate_at_stop(text, stop) if stop else text

    # ------------------------------------------------------------------
    # batched generation — reimplements LLaDA2's block-diffusion loop
    # ------------------------------------------------------------------

    def _gen_batch(
        self,
        w: _Worker,
        prompts: list[str],
        gen_kwargs: list[dict[str, Any]],
    ) -> list[str]:
        """Batched block-diffusion generation.

        ``model.forward()`` runs on the full batch (the expensive GPU
        part).  Per-sample confidence selection is a cheap Python loop
        on tensors of length ``block_length`` (typically 32).
        """
        torch = self._torch
        model = w.model
        device = model.device

        all_ids = [self._tokenize(w, p) for p in prompts]
        max_plen = max(ids.shape[1] for ids in all_ids)

        mask_id = getattr(model.config, "mask_token_id", 156895)
        eos_id = getattr(model.config, "eos_token_id", 156892)
        pad_id = w.tokenizer.pad_token_id or eos_id or 0

        # Left-pad prompts to the same length
        batch = len(all_ids)
        padded = torch.full((batch, max_plen), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(all_ids):
            padded[i, max_plen - ids.shape[1] :] = ids[0]

        kw0 = gen_kwargs[0]
        gen_length = kw0.get("max_gen_toks", kw0.get("max_tokens", self._gen_length))
        temperature = kw0.get("temperature", self._temperature)
        block_length = self._block_length
        steps = min(self._steps, gen_length)
        threshold = self._threshold

        prompt_length = max_plen
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        prefill_blocks = prompt_length // block_length

        # Block-diagonal causal mask — expanded to batch size
        # (LLaDA2 forward() requires exact batch dim match, no broadcast)
        bm = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
        attn = (
            bm.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .log()
            .to(torch.bfloat16)
        )

        pos = torch.arange(total_length, device=device).unsqueeze(0)

        x = torch.full((batch, total_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_length] = padded

        schedule = model._get_num_transfer_tokens(block_length, steps)
        done = torch.zeros(batch, dtype=torch.bool, device=device)

        for nb in range(prefill_blocks, num_blocks):
            if done.all():
                break
            we = (nb + 1) * block_length
            cx = x[:, :we]
            ca = attn[:, :, :we, :we]
            cp = pos[:, :we]

            for step in range(steps):
                amask = cx[:, -block_length:] == mask_id
                if amask.sum() == 0:
                    break

                with torch.no_grad():
                    logits = model.forward(
                        cx, attention_mask=ca, position_ids=cp
                    ).logits

                alog = logits[:, -block_length:, :]
                x0, x0_p = model._sample_with_temperature_topk_topp(
                    alog, temperature=temperature
                )

                ntrans = schedule[step].item()

                for b in range(batch):
                    if done[b]:
                        continue
                    conf = torch.where(
                        amask[b], x0_p[b], torch.tensor(-float("inf"), device=device)
                    )
                    high = conf > threshold
                    nh = high.sum().item()
                    na = amask[b].sum().item()
                    k = min(ntrans, na)

                    if k == 0:
                        continue

                    if nh >= ntrans:
                        idx = high.nonzero(as_tuple=True)[0][:k]
                    else:
                        _, idx = torch.topk(conf, k=k)

                    cx[b, -block_length + idx] = x0[b, idx]

                    if self._eos_early_stop and (x0[b, idx] == eos_id).any():
                        gr = cx[b, prompt_length:]
                        eh = (gr == eos_id).nonzero(as_tuple=True)[0]
                        if len(eh) > 0:
                            before = gr[: eh[0].item()]
                            if (before != mask_id).all():
                                done[b] = True

            x[:, :we] = cx

        # Extract generated text per sample
        results: list[str] = []
        for b in range(batch):
            gen = x[b, prompt_length : prompt_length + gen_length]
            eh = (gen == eos_id).nonzero(as_tuple=True)[0]
            end = eh[0].item() + 1 if len(eh) > 0 else gen_length
            text = w.tokenizer.decode(gen[:end], skip_special_tokens=True)
            stop = gen_kwargs[b].get("until", gen_kwargs[b].get("stop", []))
            if isinstance(stop, str):
                stop = [stop]
            if stop:
                text = _truncate_at_stop(text, stop)
            results.append(text)
        return results

    # ------------------------------------------------------------------
    # loglikelihood
    # ------------------------------------------------------------------

    def compute_loglikelihood(
        self,
        contexts: list[str],
        continuations: list[str],
    ) -> list[tuple[float, bool]]:
        if not contexts:
            return []
        if not hasattr(self._workers[0].model, "get_log_likelihood"):
            raise NotImplementedError(
                "Model does not expose get_log_likelihood(). "
                "Use a LLaDA/LLaDA2 model with trust_remote_code=True."
            )
        nw = len(self._workers)
        if nw == 1:
            return [
                self._ll_one(self._workers[0], c, t)
                for c, t in zip(contexts, continuations)
            ]
        results: list[tuple[float, bool] | None] = [None] * len(contexts)

        def _w(i: int) -> tuple[int, tuple[float, bool]]:
            return i, self._ll_one(self._workers[i % nw], contexts[i], continuations[i])

        with ThreadPoolExecutor(max_workers=nw) as pool:
            for f in as_completed(pool.submit(_w, i) for i in range(len(contexts))):
                i, r = f.result()
                results[i] = r
        return [r if r is not None else (0.0, True) for r in results]

    def _ll_one(
        self, w: _Worker, context: str, continuation: str
    ) -> tuple[float, bool]:
        ci = w.tokenizer(context, return_tensors="pt").input_ids.to(w.model.device)
        fi = w.tokenizer(context + continuation, return_tensors="pt").input_ids.to(
            w.model.device
        )
        ti = fi.clone()
        ti[:, : ci.shape[1]] = -100
        with self._torch.no_grad():
            ll = w.model.get_log_likelihood(fi, ti, mc_num=self._mc_num)
        v = ll.item() if self._torch.is_tensor(ll) else float(ll)
        return (v, True)
