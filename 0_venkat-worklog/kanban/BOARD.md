# Rust Minisgl Kanban Board

Last updated: 2026-02-14
Owner: venkat
Scope: `mini-sglang` CPU-side Rust migration with 1:1 feature parity first, then performance.

## Baseline Snapshot (Local RTX 3060)

- Offline (`LLM` path): ~2647 output tok/s in 16-request small-batch run.
- Online (`/v1/chat/completions`): ~420 tok/s, TTFT ~959 ms, TPOT ~4.37 ms in 8-request run.
- Goal: beat these numbers while preserving behavior.

## Lanes

- `todo/`: ready to start
- `in-progress/`: active work (move card here when started)
- `done/`: finished with validation evidence
- `backlog/`: parked ideas (not required for parity milestone)

## Priority Order (Execution Sequence)

1. `todo/P0-001-baseline-harness-and-gates.md`
2. `todo/P0-002-rust-workspace-and-ci-scaffold.md`
3. `todo/P0-003-rust-radix-cache-equivalence.md`
4. `todo/P0-004-rust-prefill-admission-and-batch-mapping.md`
5. `todo/P0-005-pyo3-bindings-and-python-integration.md`
6. `todo/P0-006-shadow-mode-and-parity-corpus.md`
7. `todo/P1-007-rust-gateway-skeleton-axum.md`
8. `todo/P1-008-rust-tokenizer-detokenizer-service.md`
9. `todo/P1-009-typed-transport-migration.md`
10. `todo/P1-010-observability-and-release-gates.md`
11. `backlog/BK-001-super-optimized-runtime-track.md`

## Definition of Done (Per Card)

- All card checkboxes completed.
- TDD flow demonstrated: failing tests first, then passing implementation, then refactor.
- Bench/parity evidence captured in worklog note.
- Card moved to `done/`.
