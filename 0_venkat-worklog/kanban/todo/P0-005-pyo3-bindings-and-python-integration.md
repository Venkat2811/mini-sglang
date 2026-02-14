# P0-005: PyO3 Bindings and Python Scheduler Integration

Priority: P0  
Status: todo  
Depends on: P0-004

## Objective

Integrate Rust hot path into current Python runtime with feature flags and safe rollback.

## Reuse Targets (from sgl-model-gateway)

- PyO3 class/exposed API style from `bindings/python/src/lib.rs`.
- Maturin packaging + Python src layout conventions.

## Checklist

- [ ] Expose Rust APIs for:
  - [ ] radix manager calls
  - [ ] prefill admission
  - [ ] mapping builders
- [ ] Add runtime flag:
  - [ ] `MINISGL_CPU_BACKEND=python|rust_hotpath`
- [ ] Add fallback path on Rust errors (fail closed to Python path).
- [ ] Add integration telemetry counters for backend path usage.

## TDD Subtasks

1. Red
- [ ] Add failing Python integration tests for both backend modes.
- [ ] Add failing tests for fallback behavior when Rust API raises error.

2. Green
- [ ] Implement bindings and scheduler plumbing to pass tests.

3. Refactor
- [ ] Minimize conversion overhead across Python/Rust boundary.
- [ ] Standardize type adapters.

## Acceptance Criteria

- [ ] Functional parity in both modes.
- [ ] Rust mode delivers measurable gain vs Python mode on baseline workload.
- [ ] Rollback via env var confirmed.
