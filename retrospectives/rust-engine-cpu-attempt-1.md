# Retrospective: `rust-engine-cpu-attempt-1`

Head `48bb1b8`, 2026-02-14. 22 commits on top of upstream `82722ad6d` (2026-02-11), about +14,000 lines.
In-branch records: `0_venkat-worklog/` (kanban, baselines, research, RUNBOOK) and `rust/`.

## What was built

- Four crates: `minisgl-cpu-core` (radix cache, prefill admission, batch mapping, scheduler metadata; deps `serde`, `thiserror` only), `minisgl-cpu-py` (PyO3 bindings), `minisgl-cpu-tokenizer` (on `llm-tokenizer`, the crate `sgl-model-gateway` used), `minisgl-cpu-gateway` (axum skeleton, later reference-only).
- Runtime switch `MINISGL_CPU_BACKEND=python|rust_hotpath|rust_inprocess_ffi|rust_service` with fail-closed fallback to Python.
- Typed msgpack transport (schema v1, `MINISGL_TYPED_TRANSPORT=1` default) replacing ad hoc payloads between processes.
- Shadow parity comparator (`ShadowCpuBackend`, sampled with `MINISGL_CPU_BACKEND_SHADOW_EVERY_N`), a subprocess-isolated deterministic token-parity tool, and a composite release gate (perf + parity + shadow + stability) in `0_venkat-worklog/baselines/gates.*.yaml`.
- Ten kanban cards landed (`P0-001` to `P1-010`); `P1-011` (out-of-process cutover) in progress, `P1-012` (remove PyO3 path) todo.

## What was measured (RTX 3060 12 GB, Qwen2.5-0.5B-Instruct, 2026-02-14)

| Measurement | Result | Record |
| --- | --- | --- |
| Baseline offline / online | 2667 tok/s (16 req) / 1568 tok/s, TTFT 50.2 ms, TPOT 4.44 ms (8 req) | `baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md` |
| Rust hot path vs Python, end to end | offline +0.07%, online +0.69% | `kanban/BOARD.md` |
| CPU metadata microbench | +63.5% | `kanban/BOARD.md` |
| Radix `match_prefix` microbench | about 138x faster | `kanban/BOARD.md` |
| Typed transport encode/decode | +74% ops/s | `kanban/BOARD.md` |
| Rust tokenizer path, tokenizer-heavy profile | `rust_inprocess` -4.68%, `rust_tokenize_only` -4.35% throughput vs Python | `baselines/2026-02-14-tokenizer-backend-ab.md` |
| Parity | 0 token mismatches on short, long and shared-prefix profiles; 0 shadow divergences, deterministic and mixed sampling | `baselines/2026-02-14-shadow-parity-corpus.md` |

The CPU layer got faster by large factors at the microbenchmark level and the system did not move.

## What I got right

1. **Direction.** "Anything on the CPU side belongs in Rust", starting with tokenizer, detokenizer and the frontend. SGLang shipped exactly that: `sglang-server` runs the HTTP server, TokenizerManager, tokenizer and detokenizer as Rust threads inside the TP-rank-0 scheduler process (PR #29799, RFC #23206).
2. **Rust radix tree behind a selectable backend, with a parity comparator.** SGLang's Rust TreeCore (PR #32710) is registered next to the Python tree core and exercised by one shared parity suite; PR #39627 proposes making it the default.
3. **Typed msgpack at the process boundary.** SGLang replaced pickle with msgpack for IPC (RFC item 3, PR #28688) before the Rust server landed.
4. **Opt-in switch, default off, fail-closed.** `SGLANG_RUST_SERVER` is still opt-in and default off on SGLang main.
5. **Reusing an existing tokenizer stack instead of writing one.** The landscape note picked `llm-tokenizer` for parity with the gateway; SGLang's server uses `dynamo-tokenizers`. Same instinct, different crate.
6. **Gates before features.** Baseline record, deterministic parity, shadow sampling and a release gate came first (`P0-001`, `P0-006`, `P1-010`). Every later number on this fork is comparable because of that.

## What I got wrong

1. **The seam.** `rust_hotpath` calls into Rust synchronously inside every Python scheduler step. Rust work ran serially inside Python's step, so the ceiling was the fraction of step time that work occupied, and on this workload that fraction was small. SGLang's server is also in-process, but as autonomous threads with a coarse channel the scheduler drains once per loop (`recv_requests`), so the Rust work runs concurrently with the GPU step. The lesson is "coarse boundary plus concurrent work", not "in-process is wrong".
2. **The substrate.** Qwen2.5-0.5B on an RTX 3060 at batch 8 to 16 is GPU-bound. SGLang's own design deck expects the Rust frontend to pay off at concurrency 128 to 4096, or as a TTFT reduction that scales with sequence length and not with model size. Neither regime was measured here.
3. **The radix data structure.** `type NodeRef = Rc<RefCell<RadixNode>>` is not `Send`. A 138x faster `match_prefix` that cannot leave one thread cannot be used from a worker pool.
4. **The tokenizer path.** Hand-rolled, and 4 to 5% slower than Python on the profile that was meant to favor it. Detokenization per step was the diagnosed cost.
5. **The gateway.** `P1-007` built an axum gateway skeleton, then walked it back to reference-only. `sgl-model-gateway` already existed.
6. **Synthetic benchmarks.** Random token ids until the closure branch introduced ShareGPT-style prompts (`BK-002`).
7. **Shipping cards over answering the question.** Ten cards landed in one day. The one question that mattered, "does this seam move an end-to-end number on a workload where CPU time is visible", was not asked until May.

## What I would do now

- Start from the seam SGLang chose: Rust threads that own ingress and egress, one drain per scheduler loop, msgpack only at the boundary.
- Measure where the design deck says the gain lives: TTFT versus sequence length at concurrency 1, then throughput versus concurrency up to overload, with `--skip-tokenizer-init` to separate tokenizer cost from the rest.
- Keep the parity tooling. It is the part of this branch that still holds up.
