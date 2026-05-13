# P1-011: Rust CPU Service Cutover (No In-Process FFI)

Priority: P1  
Status: in-progress  
Depends on: P1-008, P1-009, P1-010

## Objective

Cut over hot CPU path from in-process PyO3 calls to out-of-process Rust CPU service while keeping Python GPU worker and existing Python API server unchanged (no mandatory Rust gateway in minisgl).

## Checklist

- [x] Define cutover architecture and data flow:
  - [x] Python scheduler/tokenizer client to Rust CPU service transport path (behind current Python API server)
  - [x] Rust CPU service request router and worker model
  - [x] timeout/retry/error contracts
- [x] Add runtime mode flags:
  - [x] `python_cpu` (legacy)
  - [x] `rust_inprocess_ffi` (current)
  - [x] `rust_service` (new target)
- [ ] Implement dual-run verification mode (`rust_service` shadowing legacy path).
- [x] Add rollback switch and failure-mode handling.

## TDD Subtasks

1. Red
- [ ] Add failing parity tests for scheduler metadata and tokenizer flow between `rust_inprocess_ffi` and `rust_service`.
- [ ] Add failing integration tests for service unavailability and retry/timeout behavior.

2. Green
- [ ] Implement `rust_service` mode and pass parity/integration tests.

3. Refactor
- [ ] Remove hot-path dependence on Python object marshaling assumptions from service client code.
- [ ] Simplify mode-selection wiring after dual-run stabilizes.

## Acceptance Criteria

- [ ] Primary CPU hot path can run fully via `rust_service` without in-process PyO3 calls.
- [ ] A/B parity holds on deterministic corpus.
- [ ] Throughput is not worse than `rust_inprocess_ffi` baseline on local benchmark profile.

## Progress Notes (2026-02-14)

- Added cutover architecture doc:
  - `0_venkat-worklog/research/2026-02-14-rust-cpu-service-cutover-design.md`
  - covers transport path, request router/worker model, and timeout/retry/error contracts
- Added canonical CPU backend mode aliases in `create_cpu_backend(...)`:
  - `python_cpu` -> Python backend
  - `rust_inprocess_ffi` -> existing `rust_hotpath` backend
  - `rust_service` -> new transitional service mode
- Added transitional `RustServiceCpuBackend`:
  - stable mode surface for upcoming out-of-process service cutover
  - explicit fallback to in-process Rust backend when service path is unavailable
  - fallback is logged and exported via runtime metrics (`backend_fallback_counts`)
- Added service-mode fallback test coverage:
  - `tests/misc/test_cpu_backend.py::test_rust_service_mode_falls_back_to_inprocess_backend`
  - `tests/misc/test_cpu_backend.py::test_cpu_backend_mode_aliases_are_supported`
- Current gap to close:
  - replace transitional fallback with real Rust service client transport
  - add timeout/retry behavior and parity tests between `rust_inprocess_ffi` and true `rust_service`.
