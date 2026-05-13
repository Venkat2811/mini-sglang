# Rust Minisgl Kanban Board

Last updated: 2026-05-14
Scope: `mini-sglang` CPU-side Rust migration with 1:1 feature parity first, then performance.

## Closure Status (2026-05-14, complete)

Closure of `rust-engine-cpu-attempt-1` shipped on branch `rust-engine-cpu-attempt-1-closure` with tag `v0.1-learning-complete`.

Outcome per RFC 0001 Decision Tree: **Outcome B (Rust neutral)**.

Measured locally on RTX 3060 with `Qwen/Qwen3-4B` and 20 ShareGPT-style prompts, three sequential runs per backend:

- python median online tok/s: `143.08`
- rust hotpath median online tok/s: `143.09`
- delta: `+0.01%`
- shadow parity divergences across full prompt set: `0`
- stability CV: python `0.00%`, rust `0.10%` (both pass `<= 10%` gate)

Confirms `sglang-rs`'s architectural choice: in-process FFI is parity-correct but not perf-positive on workloads where GPU is the bottleneck. The next architectural step (out-of-process Rust scheduler service) is in `kanban/deferred/` and is the natural starting point for any continuation.

Driving RFC and kanban (outside this repo):

- `<workspace>/ai-chat-exports/.0_agentic_engineering/8_minisgl/0_rust_cpu_closure/`

In-repo closure artifacts:

- `0_venkat-worklog/closure/CLOSURE_NOTES.md`
- `0_venkat-worklog/closure/H100_RUNBOOK.md` (retained for future iteration)
- `0_venkat-worklog/closure/COST_ESTIMATE.md` (retained for future iteration)
- `0_venkat-worklog/closure/2026-05-14-final-retrospective.md`
- `0_venkat-worklog/baselines/closure/local_rtx3060/SIDE_BY_SIDE.md`

`P1-011` and `P1-012` moved to `kanban/deferred/` (not completed; explicitly deferred).

## Baseline Snapshot (Local RTX 3060)

- Offline (`LLM` path): ~2667 output tok/s in 16-request small-batch run.
- Online (`/v1/chat/completions`): ~1568 tok/s, TTFT ~50 ms, TPOT ~4.44 ms in 8-request run.
- Goal: beat these numbers while preserving behavior.

## Latest A/B Snapshot (Python vs Rust CPU backend)

- Offline: Rust backend slightly ahead (latest run delta `+0.07%`).
- Online: Rust backend ahead (latest run delta `+0.69%` throughput).
- Shadow parity check (offline deterministic corpus): `0` divergences logged.
- Deterministic token parity check (`python` vs `rust_hotpath`): passed for both text and token prompt sets.
- Tokenizer-heavy online rerun: Rust tokenizer backends remain behind Python tokenizer on this machine (`~5%` to `~7%` delta in latest controlled sequential runs).
- Typed transport microbench (new schema v1) shows strong CPU-side serialization gain (`+74%` backend ops/s, `+107%` tokenizer ops/s).
- Typed transport schema v1 is now default hot path with legacy rollback flag.
- Queue transport latency counters are available behind `MINISGL_TRANSPORT_LATENCY_STATS`.
- Rust gateway remains optional reference work; minisgl cutover target is Rust CPU service behind existing Python API server.
- Details recorded in `0_venkat-worklog/baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md`.
- Parity corpus details: `0_venkat-worklog/baselines/2026-02-14-shadow-parity-corpus.md`.

## Baseline Documentation

- Runbook (sanitized + reproducible): `0_venkat-worklog/RUNBOOK.md`
- Recorded run results: `0_venkat-worklog/baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md`
- Tokenizer backend A/B: `0_venkat-worklog/baselines/2026-02-14-tokenizer-backend-ab.md`
- Tokenizer research note: `0_venkat-worklog/research/2026-02-14-rust-tokenizer-landscape.md`
- CPU service cutover design: `0_venkat-worklog/research/2026-02-14-rust-cpu-service-cutover-design.md`
- Typed transport microbench: `0_venkat-worklog/baselines/latest-transport-overhead.json`
- Release gate config: `0_venkat-worklog/baselines/gates.release.yaml`

## Privacy Guardrails (Public Repo)

- Do not commit absolute local filesystem paths.
- Do not commit usernames, hostnames, API keys, tokens, or private endpoints.
- Use generic loopback addresses for local tests (`127.0.0.1`).
- Keep machine identifiers limited to non-sensitive hardware/software versions only.

## Lanes

- `todo/`: ready to start
- `in-progress/`: active work (move card here when started)
- `done/`: finished with validation evidence
- `backlog/`: parked ideas (not required for parity milestone)

## Priority Order (Execution Sequence)

1. `in-progress/P1-011-rust-cpu-service-cutover-no-inprocess-ffi.md`
2. `todo/P1-012-remove-pyo3-runtime-path.md`
3. `backlog/BK-001-super-optimized-runtime-track.md`
4. `backlog/BK-002-standard-corpus-benchmark-sharegpt.md`

## Definition of Done (Per Card)

- All card checkboxes completed.
- TDD flow demonstrated: failing tests first, then passing implementation, then refactor.
- Bench/parity evidence captured in worklog note.
- Card moved to `done/`.
