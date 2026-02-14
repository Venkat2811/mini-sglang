# P0-006: Shadow Mode and Parity Corpus

Priority: P0  
Status: in-progress  
Depends on: P0-005

## Objective

Establish parity confidence before broader rollout by running Rust decisions in shadow against Python.

## Checklist

- [ ] Build parity corpus:
  - [ ] short prompts
  - [ ] long prompts
  - [ ] shared-prefix heavy workloads
  - [ ] mixed sampling params
- [ ] Implement shadow mode that:
  - [x] computes decisions in both paths
  - [x] serves production path from selected backend
  - [x] logs diffs with request IDs
- [x] Add divergence report command.

## TDD Subtasks

1. Red
- [x] Create failing tests with intentionally diverged fixtures to validate diff detection.
- [x] Add failing test for deterministic token equality under greedy decode.

2. Green
- [x] Implement shadow comparator and reporting.

3. Refactor
- [ ] Reduce shadow overhead for continuous CI runs.

## Acceptance Criteria

- [ ] Zero divergence on deterministic corpus.
- [ ] Documented known/allowed differences for stochastic settings.

## Progress Notes (2026-02-14)

- Added env toggles in `python/minisgl/env.py`:
  - `MINISGL_CPU_BACKEND_SHADOW`
  - `MINISGL_CPU_BACKEND_SHADOW_REPORT`
  - `MINISGL_CPU_BACKEND_SHADOW_MAX_DIFFS`
- Added shadow comparator backend in `python/minisgl/scheduler/cpu_backend.py`:
  - wraps primary backend (python/rust) and shadow backend (opposite path),
  - compares `positions`, `input_tuple`, `write_tuple` tensors,
  - logs divergence with request UIDs,
  - writes JSONL report entries with capped output volume.
- Added divergence report CLI:
  - `python/minisgl/benchmark/shadow_report.py`
  - summarizes top divergence kinds/reasons from JSONL logs.
  - supports `--allow-missing` for zero-divergence runs where no file is emitted.
- Added tests:
  - `tests/misc/test_cpu_backend_shadow.py`:
    - divergence detection/report entry,
    - shadow exception handling,
    - `create_cpu_backend` shadow wiring.
  - `tests/misc/test_shadow_report.py`:
    - JSONL summary command behavior.
- Validation:
  - `.venv/bin/python -m pytest tests/misc/test_cpu_backend.py tests/misc/test_cpu_backend_shadow.py tests/misc/test_shadow_report.py -q` passed.
  - offline shadow run:
    - `MINISGL_CPU_BACKEND=rust_hotpath MINISGL_CPU_BACKEND_SHADOW=1 ... python -m minisgl.benchmark.harness offline ...`
    - `python -m minisgl.benchmark.shadow_report --allow-missing ...`
    - observed: `divergence_entries=0`.
- Added deterministic token parity CLI + tests:
  - tool: `python/minisgl/benchmark/token_parity.py`
  - tests: `tests/misc/test_token_parity.py`
  - run artifact: `0_venkat-worklog/baselines/latest-token-parity.json`
  - observed: `parity_passed=True` (`text_prompts` and `token_prompts` both zero mismatches).
