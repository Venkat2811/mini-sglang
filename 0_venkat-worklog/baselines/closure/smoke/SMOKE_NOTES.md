# Closure Smoke Notes

Date: 2026-05-14
Host: local RTX 3060 12 GB
Model: `Qwen/Qwen2.5-0.5B-Instruct`

## Purpose

Validate the closure wrapper end-to-end **before any H100 spend**. Specifically:

- `scripts/closure/sharegpt_prep.py` accepts a hand-crafted ShareGPT-style fixture and emits a clean JSONL.
- `scripts/closure/run_closure_benchmark.py` boots the mini-sglang server with the requested backend env, waits for ready, drives chat completion requests, captures TTFT/TPOT/throughput, and tears the server down cleanly.
- `scripts/closure/compare_runs.py` reads the resulting artifacts and renders a side-by-side Markdown table without crashing.

## What This Smoke Does Not Show

The numbers in `SMOKE_SIDE_BY_SIDE.md` are **not** real evidence about Python vs Rust hotpath performance:

- **1 run per backend.** No stability CV, no multi-run averaging. Single-sample noise dominates.
- **8 prompts.** Far too few; first-request cold path skews TTFT badly.
- **`cuda-graph-max-bs=1`.** CUDA graph reuse is essentially disabled. The Python backend's per-step overhead is exaggerated at this setting.
- **`max-tokens=32`, `concurrency=4`.** Tiny decode loop where startup overhead per request dominates measured throughput.
- **No alternating order.** Python ran first cold, Rust ran second.
- **Tiny model on prosumer GPU.** Exactly the regime where the Feb 14 work already showed `+0.69%`, not the regime where the closure intends to take a real measurement.

The smoke produced an Outcome A signal of `+234.03%` online tok/s for the Rust hotpath. That number is **wrapper-validation noise**, not a closure result. Treat it accordingly. The real closure outcome will land in `0_venkat-worklog/baselines/closure/SIDE_BY_SIDE.md` after the H100 run.

## What This Smoke Does Show

- All three closure scripts run end-to-end.
- The Rust hotpath backend env (`MINISGL_CPU_BACKEND=rust_hotpath`, `MINISGL_TYPED_TRANSPORT=1`) wires through cleanly into the server.
- Subprocess + process-group shutdown works (no zombie processes, no port collisions on the second run after a different `MASTER_PORT`).
- Artifact JSON schema is consistent across backends and the compare script can consume it without modification.

## Files In This Directory

- `smoke_subset.jsonl` - 8 ShareGPT-style prompts from the hand-crafted fixture
- `smoke-python.json` - retained run artifact, Python backend
- `smoke-rust.json` - retained run artifact, Rust hotpath backend
- `smoke-python.server.log` - server stdout/stderr for the Python run
- `smoke-rust.server.log` - server stdout/stderr for the Rust run
- `SMOKE_SIDE_BY_SIDE.md` - compare script output
- `SMOKE_NOTES.md` - this file
