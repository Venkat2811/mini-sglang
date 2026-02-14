# P0-001: Baseline Harness and Performance Gates

Priority: P0  
Status: todo  
Depends on: none

## Objective

Create repeatable local baseline runs (offline + online) so every Rust change is measured against stable numbers.

## Checklist

- [ ] Add `0_venkat-worklog/baselines/` folder with timestamped run logs.
- [ ] Create one command for offline baseline and one for online baseline.
- [ ] Lock benchmark input profiles (prompt lengths, output lengths, concurrency).
- [ ] Define pass/fail thresholds for parity and perf.
- [ ] Add a markdown template for benchmark result recording.

## TDD Subtasks

1. Red
- [ ] Write failing tests for benchmark parser/aggregator utility (no regression if format changes).
- [ ] Add failing test for threshold checker (should fail when throughput drops below gate).

2. Green
- [ ] Implement parser and threshold checker.
- [ ] Wire scripts to emit machine-readable summary JSON.

3. Refactor
- [ ] Remove duplicated metric math from ad-hoc scripts.
- [ ] Keep one shared `metrics.py` (or Rust equivalent later) path for calculations.

## Acceptance Criteria

- [ ] Same command on same machine produces comparable metrics (within agreed noise band).
- [ ] Baseline numbers are saved and referenced by all later cards.

## Notes

- Use current local results as initial anchor:
  - offline ~2647 tok/s
  - online ~420 tok/s
- Existing documented runbook and snapshot:
  - `../RUNBOOK.md`
  - `../baselines/2026-02-14-rtx3060-qwen2.5-0.5b.md`
