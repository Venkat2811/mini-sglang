# Rust Minisgl Kanban Board

Last updated: 2026-02-14
Scope: `mini-sglang` CPU-side Rust migration with 1:1 feature parity first, then performance.

## Baseline Snapshot (Local RTX 3060)

- Offline (`LLM` path): ~2667 output tok/s in 16-request small-batch run.
- Online (`/v1/chat/completions`): ~1568 tok/s, TTFT ~50 ms, TPOT ~4.44 ms in 8-request run.
- Goal: beat these numbers while preserving behavior.

## Latest A/B Snapshot (Python vs Rust CPU backend)

- Offline: Rust backend slightly ahead (latest run delta `+0.07%`).
- Online: Rust backend ahead (latest run delta `+0.69%` throughput).
- Shadow parity check (offline deterministic corpus): `0` divergences logged.
- Deterministic token parity check (`python` vs `rust_hotpath`): passed for both text and token prompt sets.
- Details recorded in `0_venkat-worklog/baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md`.
- Parity corpus details: `0_venkat-worklog/baselines/2026-02-14-shadow-parity-corpus.md`.

## Baseline Documentation

- Runbook (sanitized + reproducible): `0_venkat-worklog/RUNBOOK.md`
- Recorded run results: `0_venkat-worklog/baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md`
- Tokenizer backend A/B: `0_venkat-worklog/baselines/2026-02-14-tokenizer-backend-ab.md`
- Tokenizer research note: `0_venkat-worklog/research/2026-02-14-rust-tokenizer-landscape.md`

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

1. `in-progress/P1-008-rust-tokenizer-detokenizer-service.md`
2. `todo/P1-009-typed-transport-migration.md`
3. `todo/P1-010-observability-and-release-gates.md`
4. `todo/P1-011-rust-cpu-service-cutover-no-inprocess-ffi.md`
5. `todo/P1-012-remove-pyo3-runtime-path.md`
6. `backlog/BK-001-super-optimized-runtime-track.md`

## Definition of Done (Per Card)

- All card checkboxes completed.
- TDD flow demonstrated: failing tests first, then passing implementation, then refactor.
- Bench/parity evidence captured in worklog note.
- Card moved to `done/`.
