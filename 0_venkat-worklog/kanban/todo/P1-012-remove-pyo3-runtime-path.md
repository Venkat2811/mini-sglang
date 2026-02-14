# P1-012: Remove PyO3 Runtime Path

Priority: P1  
Status: todo  
Depends on: P1-011

## Objective

Retire `minisgl-cpu-py` from runtime hot path after Rust CPU service cutover is stable.

## Checklist

- [ ] Remove runtime imports/usages of `minisgl_cpu.mini_sgl_cpu_rs` from production code paths.
- [ ] Delete or archive in-process FFI mode from scheduler CPU backend selection.
- [ ] Remove PyO3-only packaging/runtime requirements from default install path.
- [ ] Update CI matrix:
  - [ ] service-path parity tests
  - [ ] service-path perf gates
  - [ ] optional legacy compatibility job during rollback window
- [ ] Update docs/runbook with migration and rollback notes.

## TDD Subtasks

1. Red
- [ ] Add failing tests that assert service mode works with PyO3 module absent.
- [ ] Add failing startup tests that ensure clear error when legacy mode is selected after removal.

2. Green
- [ ] Remove in-process runtime path and pass updated test matrix.

3. Refactor
- [ ] Clean dead config/env flags tied to in-process FFI mode.
- [ ] Prune stale code/docs and simplify backend mode surface.

## Acceptance Criteria

- [ ] Runtime no longer depends on PyO3 bindings for CPU hot path.
- [ ] Service-only path passes parity/perf gates.
- [ ] Migration is documented with explicit rollback strategy.
