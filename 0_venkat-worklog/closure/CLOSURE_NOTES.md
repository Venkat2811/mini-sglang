# mini-sglang Rust CPU Engine Closure Notes

Date: 2026-05-14

## Status

Active closure of the `rust-engine-cpu-attempt-1` work shipped 2026-02-14.

Working branch: `rust-engine-cpu-attempt-1-closure` (forked from `rust-engine-cpu-attempt-1`).

## Driving Plan

The closure is governed by a single RFC kept outside this repo to keep `mini-sglang` itself slim:

- `<workspace>/ai-chat-exports/.0_agentic_engineering/8_minisgl/0_rust_cpu_closure/0_rfcs/0001_mini_sglang_rust_cpu_engine_closure_plan.md`

The corresponding kanban (PR 0 through PR 3) lives at:

- `<workspace>/ai-chat-exports/.0_agentic_engineering/8_minisgl/0_rust_cpu_closure/1_kanban/`

This file is the in-repo pointer. It does not duplicate the RFC content. Update only links and short status notes here.

## Scope Summary

- close the Feb 14 Rust CPU hot-path experiment at a defensible conclusion
- one bounded H100 weekend on `Qwen/Qwen3-4B` with ShareGPT-derived prompts
- compare existing `MINISGL_CPU_BACKEND=python` vs `MINISGL_CPU_BACKEND=rust_hotpath`
- retrospective grounded in `sglang-rs` reference standard
- move `P1-011` and `P1-012` to `deferred/` at closure
- no new Rust code on the hot path inside this closure

## Honest Concessions Adopted

- Myelon for tensor-parallel transport is treated as a lost cost; TP is GPU-collective territory.
- `cudarc` and cuda-oxide for GPU-side Rust IPC are a separate substrate, out of scope.
- `sglang-rs` plus Myelon evaluation is deferred until after this closure ships.

## Closure Directory Layout

```
0_venkat-worklog/closure/
  CLOSURE_NOTES.md                  this file
  H100_RUNBOOK.md                   provider-agnostic H100 bring-up runbook
  COST_ESTIMATE.md                  provider comparison and spend cap math
  2026-05-14-final-retrospective.md retrospective draft (placeholders for H100 numbers)

scripts/closure/
  sharegpt_prep.py                  bounded ShareGPT subset preparation
  run_closure_benchmark.py          single-backend run wrapper with full evidence capture
  compare_runs.py                   side-by-side stability and CV across runs
```

## PR Sequencing

- PR 0: scaffold and RFC (landed in `ai-chat-exports` repo)
- PR 1: H100 host bring-up, Python backend baseline captured
- PR 2: Rust hotpath run captured, side-by-side summary
- PR 3: retrospective filled in with H100 numbers, deferred-card moves, tag

## Privacy Guardrails

Per `0_venkat-worklog/RUNBOOK.md`:

- no absolute filesystem paths in committed files
- no usernames, hostnames, API keys, tokens, private endpoints
- generic loopback for local examples
- only non-sensitive hardware and software versions
