# P0-001: Baseline Harness and Performance Gates

Priority: P0  
Status: in-progress  
Depends on: none

## Objective

Create repeatable local baseline runs (offline + online) so every Rust change is measured against stable numbers.

## Checklist

- [x] Add `0_venkat-worklog/baselines/` folder with timestamped run logs.
- [x] Create one command for offline baseline and one for online baseline.
- [x] Lock benchmark input profiles (prompt lengths, output lengths, concurrency).
- [x] Define pass/fail thresholds for parity and perf.
- [x] Add a markdown template for benchmark result recording.

## TDD Subtasks

1. Red
- [x] Write failing tests for benchmark parser/aggregator utility (no regression if format changes).
- [x] Add failing test for threshold checker (should fail when throughput drops below gate).

2. Green
- [x] Implement parser and threshold checker.
- [x] Wire scripts to emit machine-readable summary JSON.

3. Refactor
- [x] Remove duplicated metric math from ad-hoc scripts.
- [x] Keep one shared `metrics.py` (or Rust equivalent later) path for calculations.

## Acceptance Criteria

- [ ] Same command on same machine produces comparable metrics (within agreed noise band). Deferred as low-priority follow-up.
- [x] Baseline numbers are saved and referenced by all later cards.

## Notes

- Use current local results as initial anchor:
  - offline ~2667 tok/s
  - online ~1568 tok/s
- Existing documented runbook and snapshot:
  - `0_venkat-worklog/RUNBOOK.md`
  - `0_venkat-worklog/baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md`
- Harness and gates:
  - `python -m minisgl.benchmark.harness offline ...`
  - `python -m minisgl.benchmark.harness online ...`
  - `0_venkat-worklog/baselines/gates.offline.yaml`
  - `0_venkat-worklog/baselines/gates.online.yaml`
