# P0-005: PyO3 Bindings and Python Scheduler Integration

Priority: P0  
Status: in-progress  
Depends on: P0-004

## Objective

Integrate Rust hot path into current Python runtime with feature flags and safe rollback.

## Reuse Targets (from sgl-model-gateway)

- PyO3 class/exposed API style from `bindings/python/src/lib.rs`.
- Maturin packaging + Python src layout conventions.

## Checklist

- [x] Expose Rust APIs for:
  - [x] radix manager calls
  - [x] prefill admission
  - [x] mapping builders
- [x] Add runtime flag:
  - [x] `MINISGL_CPU_BACKEND=python|rust_hotpath`
- [x] Add fallback path on Rust errors (fail closed to Python path).
- [x] Add integration telemetry counters for backend path usage.

## TDD Subtasks

1. Red
- [x] Add failing Python integration tests for both backend modes.
- [x] Add failing tests for fallback behavior when Rust API raises error.

2. Green
- [x] Implement bindings and scheduler plumbing to pass tests.

3. Refactor
- [x] Remove avoidable `positions` round-trip (`Tensor -> list -> Rust -> list -> Tensor`) in Rust path.
- [x] Minimize conversion overhead across Python/Rust boundary.
- [x] Standardize type adapters.

## Acceptance Criteria

- [x] Functional parity in both modes.
- [x] Rust mode delivers measurable gain vs Python mode on baseline workload.
- [x] Rollback via env var confirmed.

## Progress Notes (2026-02-14)

- Added scheduler CPU backend selector and telemetry:
  - `python/minisgl/scheduler/cpu_backend.py`
  - `python/minisgl/env.py` adds `MINISGL_CPU_BACKEND` handling.
  - `python/minisgl/scheduler/scheduler.py` now uses backend adapter for mapping builders.
- Implemented fallback behavior:
  - rust backend path falls back to python for missing module/import/runtime errors.
  - fallback counters are tracked in `CpuBackendStats`.
- Exposed PyO3 APIs in `rust/minisgl-cpu-py/src/lib.rs`:
  - `RadixCacheManager` class (`insert_prefix`, `match_prefix`, `lock_handle`, `evict`, `size_info`, `check_integrity`),
  - `prefill_admission_plan`,
  - `make_positions`, `make_input_mapping`, `make_write_mapping`.
- Updated Python package exports:
  - `rust/minisgl-cpu-py/src/minisgl_cpu/__init__.py`.
- Added tests:
  - `tests/misc/test_cpu_backend.py` (python/rust mode selection + fallback behavior),
  - `rust/minisgl-cpu-py/tests/test_import.py` extended smoke checks for new APIs.
- Refactor pass to reduce one major boundary inefficiency:
  - `rust/minisgl-cpu-core/src/prefill.rs` adds `make_input_mapping` (mapping-only helper).
  - `rust/minisgl-cpu-py/src/lib.rs` updates `make_input_mapping` API to avoid `positions` marshalling.
  - `python/minisgl/scheduler/cpu_backend.py` now reuses `batch.positions` directly in Rust mode.
- Fresh A/B evidence after refactor (same local harness profile):
  - offline delta: `-0.12%` (Rust vs Python),
  - online delta: `-2.31%` (Rust vs Python),
  - isolated metadata loop: Python backend still faster (`6547 ops/s` vs Rust backend `1789 ops/s`).
- Iteration update (packed buffers + single-call metadata path):
  - Added `make_metadata_buffers` in `rust/minisgl-cpu-py/src/lib.rs` (returns packed int32 bytearrays).
  - Rust backend now uses one Rust call per scheduler step and decodes via `torch.frombuffer` in `python/minisgl/scheduler/cpu_backend.py`.
  - Added bounded per-step cache for `positions/input/write` tensors (3-use lifecycle) to avoid repeated FFI calls in the same batch preparation.
  - Updated binding exports/tests:
    - `rust/minisgl-cpu-py/src/minisgl_cpu/__init__.py`
    - `rust/minisgl-cpu-py/tests/test_import.py`
  - New evidence on same machine/profile:
    - isolated metadata loop: Rust backend `10508 ops/s` vs Python backend `6428 ops/s`.
    - online harness: Rust backend `1614.35 tok/s` vs Python backend `1590.42 tok/s` (`+1.50%`).
    - offline harness: near parity (`-0.02%`).
- Practical takeaway from `sgl-model-gateway` reuse pattern:
  - efficient model-gateway usage keeps Python at coarse control plane boundary (`Router.from_args(...).start()`),
  - hot request path runs in Rust process, avoiding per-step Python↔Rust marshalling.
- Validation:
  - `cargo test --workspace` passed.
  - `cargo clippy --workspace --all-targets -- -D warnings` passed.
  - `.venv` + `uvx maturin develop --manifest-path rust/minisgl-cpu-py/Cargo.toml` passed.
  - `.venv/bin/python -m unittest discover -s rust/minisgl-cpu-py/tests -p 'test_*.py' -v` passed.
  - `.venv/bin/python -m pytest tests/misc/test_cpu_backend.py -q` passed.
