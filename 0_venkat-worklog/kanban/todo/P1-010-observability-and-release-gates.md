# P1-010: Observability and Release Gates

Priority: P1  
Status: todo  
Depends on: P1-009

## Objective

Add production-grade observability and strict rollout gates for Rust CPU backend.

## Reuse Targets (from sgl-model-gateway)

- `tracing` structured logs
- Prometheus metrics patterns
- health/readiness operational checks

## Checklist

- [ ] Add metrics:
  - [ ] scheduler step latency
  - [ ] queue length and inflight count
  - [ ] tokenizer latency
  - [ ] transport latency
  - [ ] backend selection/fallback counts
- [ ] Add release gate script:
  - [ ] parity gate
  - [ ] perf gate
  - [ ] stability gate
- [ ] Document go/no-go checklist.

## TDD Subtasks

1. Red
- [ ] Write failing tests for metric registration and endpoint exposure.
- [ ] Write failing tests for release gate checks on synthetic bad runs.

2. Green
- [ ] Implement metrics and gate evaluation logic.

3. Refactor
- [ ] Consolidate metric labels and avoid high-cardinality mistakes.

## Acceptance Criteria

- [ ] One-command gate run outputs clear pass/fail decision.
- [ ] Rollout criteria are objective and reproducible.
