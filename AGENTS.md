# AGENTS.md

## Cursor Cloud specific instructions

This is a Python project managed with [uv](https://docs.astral.sh/uv/) and uses `pyproject.toml` for configuration.

### Quick reference

| Task | Command |
|---|---|
| Install / sync deps | `uv sync --dev` |
| Lint | `ruff check .` |
| Format check | `ruff format --check .` |
| Run tests | `pytest -v` |
| Run application | `inference-eval` or `python -m inference_eval.main` |

All commands above assume the virtualenv is activated (`. .venv/bin/activate`) or prefixed with `.venv/bin/`.

### Notes

- Python 3.12+ is required (`requires-python = ">=3.12"` in `pyproject.toml`).
- Dev dependencies (ruff, pytest) are in the `[dependency-groups] dev` section of `pyproject.toml`.
- `uv sync --dev` creates `.venv/` and installs the project in editable mode along with dev tools.
