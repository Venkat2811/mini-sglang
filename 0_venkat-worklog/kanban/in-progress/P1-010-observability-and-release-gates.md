# P1-010: Observability and Release Gates

Priority: P1  
Status: in-progress  
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
  - [x] transport latency
  - [ ] backend selection/fallback counts
- [x] Add release gate script:
  - [x] parity gate
  - [x] perf gate
  - [x] stability gate
- [ ] Document go/no-go checklist.

## TDD Subtasks

1. Red
- [ ] Write failing tests for metric registration and endpoint exposure.
- [x] Write failing tests for release gate checks on synthetic bad runs.

2. Green
- [x] Implement metrics and gate evaluation logic.

3. Refactor
- [ ] Consolidate metric labels and avoid high-cardinality mistakes.

## Acceptance Criteria

- [x] One-command gate run outputs clear pass/fail decision.
- [ ] Rollout criteria are objective and reproducible.

## Progress Notes (2026-02-14)

- Added release gate evaluator:
  - `python/minisgl/benchmark/release_gate.py`
  - supports perf/parity/shadow/stability checks
  - supports one-command config mode via `--config`
- Added release gate tests:
  - `tests/misc/test_release_gate.py`
  - includes perf fail, parity fail, shadow fail, stability-CV fail, and all-pass cases.
- Added release gate config template:
  - `0_venkat-worklog/baselines/gates.release.yaml`
- Transport latency observability now available behind:
  - `MINISGL_TRANSPORT_LATENCY_STATS`
  - API: `minisgl.utils.transport_stats_snapshot(reset=...)`
