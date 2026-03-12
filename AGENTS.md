# AGENTS.md

## Cursor Cloud specific instructions

This is a Python project managed with [uv](https://docs.astral.sh/uv/). It decouples inference from evaluation using lm-eval-harness.

### Quick reference

| Task | Command |
|---|---|
| Install / sync deps | `uv sync --dev` |
| Lint | `ruff check .` |
| Format check | `ruff format --check .` |
| Auto-fix lint + format | `ruff check --fix . && ruff format .` |
| Run tests | `pytest -v` |
| Run CLI | `inference-eval --help` |

All commands assume the virtualenv is activated (`. .venv/bin/activate`) or prefixed with `.venv/bin/`.

### Architecture

Three-phase workflow: **extract** (capture requests from lm-eval-harness tasks) → **infer** (run any inference engine) → **evaluate** (compute metrics via lm-eval-harness).

Key modules:
- `inference_eval/models/capture.py` — `RequestCaptureLM` captures requests without real inference
- `inference_eval/models/offline.py` — `OfflineLM` returns pre-computed results for evaluation
- `inference_eval/inference/base.py` — Abstract `InferenceEngine` interface for pluggable backends
- `inference_eval/schema.py` — Data models and JSONL serialization

### Notes

- Python 3.10+ required (`requires-python = ">=3.10"` in `pyproject.toml`).
- `lm-eval>=0.4.0` is a core dependency; torch is NOT required for the base extract/evaluate workflow.
- Inference engine backends (vllm, sglang, openai) are optional extras.
- The end-to-end test in `tests/test_e2e.py` downloads gsm8k data on first run (~5s).
- When running `inference-eval extract`, lm-eval-harness may download datasets on first use.
