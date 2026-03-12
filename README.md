# inference_eval

Decouple inference from evaluation using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Motivation

lm-eval-harness couples inference and evaluation tightly. When you need to use a custom inference framework (e.g., for models like LLaDA that vLLM/SGLang don't support well), you can't easily plug it in.

**inference_eval** solves this by splitting the workflow into three phases:

1. **Extract** — Pull task requests (prompts) from lm-eval-harness
2. **Infer** — Run inference with *any* framework (vLLM, SGLang, custom, etc.)
3. **Evaluate** — Feed results back to lm-eval-harness for metric computation

## Installation

```bash
pip install -e .

# With inference engine extras:
pip install -e ".[vllm]"    # vLLM backend
pip install -e ".[sglang]"  # SGLang backend
pip install -e ".[openai]"  # OpenAI-compatible API
```

## Quick Start

### Step 1: Extract requests from tasks

```bash
inference-eval extract \
    --tasks gsm8k,hellaswag,mmlu \
    --output ./requests \
    --num-fewshot 5 \
    --limit 100
```

This saves task prompts as JSONL files in `./requests/<task_name>/<request_type>.jsonl`.

### Step 2: Run inference

**Using a built-in engine:**
```bash
inference-eval infer \
    --requests ./requests \
    --output ./results \
    --engine vllm \
    --model meta-llama/Llama-3-8B-Instruct \
    --engine-args '{"tensor_parallel_size": 4}'
```

**Using a custom script:**

Read requests from `./requests/<task>/generate_until.jsonl` (JSONL format),
run your inference, and write results to `./results/<task>/generate_until.jsonl`:

```jsonl
{"task_name": "gsm8k", "request_type": "generate_until", "doc_id": 0, "index": 0, "generated_text": "The answer is 4.\n#### 4"}
```

### Step 3: Evaluate

```bash
inference-eval evaluate \
    --results ./results \
    --requests ./requests \
    --output ./scores.json
```

## Request/Result Formats

### generate_until (generative tasks like gsm8k)
- **Request**: `context` (prompt), `generation_kwargs` (stop tokens, max tokens)
- **Result**: `generated_text` (model output)

### loglikelihood (multiple-choice tasks like hellaswag, mmlu)
- **Request**: `context` (prompt prefix), `continuation` (choice text)
- **Result**: `log_likelihood` (float), `is_greedy` (bool)

## Python API

```python
from inference_eval.extract import extract_requests
from inference_eval.evaluate import evaluate_results
from inference_eval.schema import InferenceResult, load_requests, save_results

# Extract
extract_requests(tasks=["gsm8k"], output_dir="./requests", limit=100)

# Load requests, run your inference, save results
requests = load_requests("./requests")
results = [
    InferenceResult(
        task_name=r.task_name,
        request_type=r.request_type,
        doc_id=r.doc_id,
        index=r.index,
        generated_text=your_model.generate(r.context),
        fingerprint=r.fingerprint,
    )
    for r in requests
]
save_results(results, "./results")

# Evaluate
scores = evaluate_results(results_dir="./results", requests_dir="./requests")
```

## Custom Inference Engine

Implement the `InferenceEngine` base class:

```python
from inference_eval.inference.base import InferenceEngine

class MyEngine(InferenceEngine):
    def generate(self, prompts, gen_kwargs):
        return [my_model.generate(p, **k) for p, k in zip(prompts, gen_kwargs)]

    def compute_loglikelihood(self, contexts, continuations):
        return [my_model.loglikelihood(c, t) for c, t in zip(contexts, continuations)]
```

## Development

```bash
uv sync --dev
ruff check .
ruff format --check .
pytest -v
```

## License

MIT
