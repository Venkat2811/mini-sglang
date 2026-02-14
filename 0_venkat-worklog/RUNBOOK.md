# Mini-SGLang Local Runbook (Sanitized)

Last updated: 2026-02-14

This runbook records how to set up and run `mini-sglang` locally for reproducible baseline measurement.

## 0. One-Command Harness (Preferred)

Use the built-in harness for stable profiles + JSON output:

```bash
cd mini-sglang
source .venv/bin/activate

# Offline baseline
python -m minisgl.benchmark.harness offline \
  --out 0_venkat-worklog/baselines/latest-offline.json \
  --gate 0_venkat-worklog/baselines/gates.offline.yaml

# Online baseline (server must already be running)
python -m minisgl.benchmark.harness online \
  --base-url http://127.0.0.1:1919 \
  --out 0_venkat-worklog/baselines/latest-online.json \
  --gate 0_venkat-worklog/baselines/gates.online.yaml
```

Output is machine-readable JSON and can be archived into a timestamped record.

## 1. Environment Setup

From repo root, use only relative paths:

```bash
cd mini-sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -e .
```

## 2. Sanity Check

```bash
source .venv/bin/activate
python - <<'PY'
import torch, transformers, flashinfer, sgl_kernel, tvm_ffi, minisgl
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("flashinfer", flashinfer.__version__)
print("sgl_kernel", sgl_kernel.__version__)
print("tvm_ffi", tvm_ffi.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

## 3. Offline Baseline (No HTTP)

Run a quick local baseline with `LLM` path:

```bash
source .venv/bin/activate
python - <<'PY'
import random, time, torch
from minisgl.llm import LLM
from minisgl.core import SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
random.seed(42)

t0 = time.time()
llm = LLM(
    MODEL,
    dtype=torch.float16,
    max_seq_len_override=2048,
    max_extend_tokens=4096,
    cuda_graph_max_bs=64,
)
print("init_time_s", round(time.time() - t0, 2))

t0 = time.time()
warm = llm.generate(["Hello"], SamplingParams(temperature=0.0, top_k=1, max_tokens=16))[0]
print("warmup_time_s", round(time.time() - t0, 2), "warmup_tokens", len(warm["token_ids"]))

t0 = time.time()
out = llm.generate(
    ["Write one short sentence explaining why batching helps GPU utilization."],
    SamplingParams(temperature=0.0, top_k=1, max_tokens=64),
)[0]
print("single_req_time_s", round(time.time() - t0, 2), "single_out_tokens", len(out["token_ids"]))

num_reqs = 16
prompts, params = [], []
for _ in range(num_reqs):
    in_len = random.randint(64, 256)
    prompts.append([random.randint(100, 20000) for _ in range(in_len)])
    params.append(SamplingParams(temperature=0.0, top_k=1, ignore_eos=True, max_tokens=64))
t0 = time.time()
res = llm.generate(prompts, params)
elapsed = time.time() - t0
sum_out = sum(len(r["token_ids"]) for r in res)
print("batch_num_reqs", num_reqs)
print("batch_total_out_tokens", sum_out)
print("batch_time_s", round(elapsed, 2))
print("batch_out_tok_per_s", round(sum_out / elapsed, 2))
PY
```

## 4. Online Baseline (Full Server/API Path)

Terminal 1: launch server

```bash
cd mini-sglang
source .venv/bin/activate
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 1919 \
  --cuda-graph-max-bs 16 \
  --memory-ratio 0.8 \
  --max-running-requests 64
```

Terminal 2: run benchmark client

```bash
cd mini-sglang
source .venv/bin/activate
python - <<'PY'
import asyncio, random, time
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from minisgl.benchmark.client import benchmark_one, benchmark_one_batch, generate_prompt, get_model_name, process_benchmark_results

async def main():
    random.seed(123)
    async with AsyncOpenAI(base_url="http://127.0.0.1:1919/v1", api_key="") as client:
        model = await get_model_name(client)
        tok = AutoTokenizer.from_pretrained(model)
        warm_prompt = generate_prompt(tok, 32)
        t0 = time.perf_counter()
        warm = await benchmark_one(client, warm_prompt, 16, model, pbar=False)
        print("warmup_elapsed_s", round(time.perf_counter() - t0, 4), "tokens", len(warm.tics) - 1)
        batch_size = 8
        prompts = [generate_prompt(tok, random.randint(64, 256)) for _ in range(batch_size)]
        out_lens = [64] * batch_size
        t0 = time.perf_counter()
        results = await benchmark_one_batch(client, prompts, out_lens, model, pbar=False)
        print("bench_batch_size", batch_size)
        print("bench_wall_time_s", round(time.perf_counter() - t0, 4))
        process_benchmark_results(results)

asyncio.run(main())
PY
```

Terminal 1: stop server with `Ctrl+C`.

## 5. Logging Rules for Public Repo

- Record only relative paths and sanitized host addresses.
- Never commit secrets, user/home paths, or personal identifiers.
- Keep benchmark logs in `0_venkat-worklog/baselines/` with timestamp + model name.

## 6. CPU Shadow Mode (Parity Diagnostics)

Run with one backend as primary and the other in shadow for metadata decision comparison.

Terminal 1: start server with shadow mode enabled

```bash
cd mini-sglang
source .venv/bin/activate
MINISGL_CPU_BACKEND=rust_hotpath \
MINISGL_CPU_BACKEND_SHADOW=1 \
MINISGL_CPU_BACKEND_SHADOW_REPORT=0_venkat-worklog/baselines/latest-shadow-divergence.jsonl \
MINISGL_CPU_BACKEND_SHADOW_MAX_DIFFS=256 \
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 1919 \
  --cuda-graph-max-bs 16 \
  --memory-ratio 0.8 \
  --max-running-requests 64
```

Terminal 2: drive traffic (example: online harness)

```bash
cd mini-sglang
source .venv/bin/activate
python -m minisgl.benchmark.harness online \
  --base-url http://127.0.0.1:1919 \
  --out 0_venkat-worklog/baselines/latest-online-shadow-check.json
```

Terminal 2: summarize divergence report

```bash
cd mini-sglang
source .venv/bin/activate
python -m minisgl.benchmark.shadow_report \
  --input 0_venkat-worklog/baselines/latest-shadow-divergence.jsonl \
  --top 20 \
  --allow-missing
```

## 7. Deterministic Token Parity (Python vs Rust CPU backend)

Run deterministic greedy decode on fixed text and synthetic token prompts, then compare exact output token IDs.

```bash
cd mini-sglang/python
../.venv/bin/python -m minisgl.benchmark.token_parity \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --backend-a python \
  --backend-b rust_hotpath \
  --max-tokens 16 \
  --token-prompt-count 4 \
  --min-input-len 32 \
  --max-input-len 64 \
  --cuda-graph-max-bs 1 \
  --master-port 2380 \
  --out ../0_venkat-worklog/baselines/latest-token-parity.json
```

Expected output:

- `parity_passed=True`
- `text_prompts` mismatch count: `0`
- `token_prompts` mismatch count: `0`
