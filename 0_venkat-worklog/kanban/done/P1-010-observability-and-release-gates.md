# P1-010: Observability and Release Gates

Priority: P1  
Status: done  
Depends on: P1-009

## Objective

Add production-grade observability and strict rollout gates for Rust CPU backend.

## Reuse Targets (from sgl-model-gateway)

- `tracing` structured logs
- Prometheus metrics patterns
- health/readiness operational checks

## Checklist

- [x] Add metrics:
  - [x] scheduler step latency
  - [x] queue length and inflight count
  - [x] tokenizer latency
  - [x] transport latency
  - [x] backend selection/fallback counts
- [x] Add release gate script:
  - [x] parity gate
  - [x] perf gate
  - [x] stability gate
- [x] Document go/no-go checklist.

## TDD Subtasks

1. Red
- [x] Write failing tests for metric registration and endpoint exposure.
- [x] Write failing tests for release gate checks on synthetic bad runs.

2. Green
- [x] Implement metrics and gate evaluation logic.

3. Refactor
- [x] Consolidate metric labels and avoid high-cardinality mistakes.

## Acceptance Criteria

- [x] One-command gate run outputs clear pass/fail decision.
- [x] Rollout criteria are objective and reproducible.

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

## Progress Notes (2026-02-14, later session)

- Added runtime metrics registry and snapshot API:
  - `python/minisgl/utils/runtime_metrics.py`
  - API: `minisgl.utils.runtime_metrics_snapshot(reset=...)`
  - controls: `MINISGL_RUNTIME_METRICS` (default enabled)
- Instrumented scheduler loop metrics:
  - scheduler step latency
  - queue depth (`prefill`, `decode`)
  - inflight decode tokens
- Instrumented tokenizer worker metrics:
  - tokenize/detokenize latency and item counts
  - backend selection and fallback counters
- Instrumented CPU backend fallback counters:
  - module load errors
  - make_positions/make_input/make_write fallback paths
  - unknown backend mode fallback
- Added runtime metrics tests:
  - `tests/misc/test_runtime_metrics.py`
  - includes disabled-mode no-op verification
- Updated runbook with:
  - runtime metrics snapshot command
  - release go/no-go checklist
