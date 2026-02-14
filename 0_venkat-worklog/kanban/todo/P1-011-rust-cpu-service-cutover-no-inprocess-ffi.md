# P1-011: Rust CPU Service Cutover (No In-Process FFI)

Priority: P1  
Status: todo  
Depends on: P1-008, P1-009, P1-010

## Objective

Cut over hot CPU path from in-process PyO3 calls to out-of-process Rust CPU service while keeping Python GPU worker unchanged.

## Checklist

- [ ] Define cutover architecture and data flow:
  - [ ] Python scheduler/tokenizer client to Rust CPU service transport path
  - [ ] Rust CPU service request router and worker model
  - [ ] timeout/retry/error contracts
- [ ] Add runtime mode flags:
  - [ ] `python_cpu` (legacy)
  - [ ] `rust_inprocess_ffi` (current)
  - [ ] `rust_service` (new target)
- [ ] Implement dual-run verification mode (`rust_service` shadowing legacy path).
- [ ] Add rollback switch and failure-mode handling.

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
