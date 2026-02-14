# P0-006: Shadow Mode and Parity Corpus

Priority: P0  
Status: done  
Depends on: P0-005

## Objective

Establish parity confidence before broader rollout by running Rust decisions in shadow against Python.

## Checklist

- [x] Build parity corpus:
  - [x] short prompts
  - [x] long prompts
  - [x] shared-prefix heavy workloads
  - [x] mixed sampling params
- [x] Implement shadow mode that:
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
- [x] Reduce shadow overhead for continuous CI runs.

## Acceptance Criteria

- [x] Zero divergence on deterministic corpus.
- [x] Documented known/allowed differences for stochastic settings.

## Progress Notes (2026-02-14)

- Added env toggles in `python/minisgl/env.py`:
  - `MINISGL_CPU_BACKEND_SHADOW`
  - `MINISGL_CPU_BACKEND_SHADOW_REPORT`
  - `MINISGL_CPU_BACKEND_SHADOW_MAX_DIFFS`
  - `MINISGL_CPU_BACKEND_SHADOW_EVERY_N` (sample shadow checks every N metadata calls)
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
    - `create_cpu_backend` shadow wiring,
    - compare-sampling (`compare_every_n`) behavior.
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
  - run artifacts:
    - `0_venkat-worklog/baselines/latest-token-parity-short.json`
    - `0_venkat-worklog/baselines/latest-token-parity-long.json`
    - `0_venkat-worklog/baselines/latest-token-parity-shared-prefix.json`
  - observed: all `parity_passed=True` (`text_prompts` and `token_prompts` both zero mismatches).
- Added corpus note:
  - `0_venkat-worklog/baselines/2026-02-14-shadow-parity-corpus.md`
  - includes known/allowed differences guidance for stochastic decoding.
- Validation updates:
  - `.venv/bin/python -m pytest tests/misc/test_cpu_backend_shadow.py tests/misc/test_token_parity.py tests/misc/test_shadow_report.py -q` passed.
  - deterministic shadow run (`MINISGL_CPU_BACKEND_SHADOW=1`, `MINISGL_CPU_BACKEND_SHADOW_EVERY_N=1`) observed: `divergence_entries=0`.
  - mixed-sampling shadow run observed: `divergence_entries=0`.
