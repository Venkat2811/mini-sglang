# P0-002: Rust Workspace and CI Scaffold

Priority: P0  
Status: done  
Depends on: P0-001

## Objective

Initialize `mini-sglang/rust/` with a production-shaped workspace and Python binding path that matches proven structure from `sglang/sgl-model-gateway`.

## Reuse Targets (from sgl-model-gateway)

- `bindings/python` layout with `maturin` + `PyO3`.
- Split crates: core logic crate + binding crate.
- Build profiles (`release`, `ci`) and reproducible CLI packaging style.

## Checklist

- [x] Create Cargo workspace:
  - [x] `rust/minisgl-cpu-core`
  - [x] `rust/minisgl-cpu-py` (PyO3)
  - [x] placeholder `rust/minisgl-cpu-gateway`
- [x] Add `pyproject.toml` for Python bindings using maturin.
- [x] Add minimal CI script for `cargo test`, `cargo clippy`, `maturin develop`.
- [x] Add versioning strategy and crate/module naming conventions.

## TDD Subtasks

1. Red
- [x] Add failing tests for core crate skeleton APIs (scheduler types, cache trait stubs).
- [x] Add failing Python smoke test that imports extension module.

2. Green
- [x] Implement minimal structs/traits to satisfy tests.
- [x] Build extension and pass import smoke test.

3. Refactor
- [x] Normalize naming to match Python-side concepts (`Req`, `Batch`, `SamplingParams` analogs).

## Acceptance Criteria

- [x] `cargo test` passes for scaffold crates.
- [x] Python can import the Rust extension from local editable build.
- [x] Project structure is documented in `rust/README.md`.

## Evidence (2026-02-14)

- Red (expected failure): `python -m unittest discover -s rust/minisgl-cpu-py/tests -p 'test_*.py' -v` failed with `ModuleNotFoundError: No module named 'minisgl_cpu'` before building the extension.
- Green:
  - `cargo test --workspace` passed.
  - `cargo clippy --workspace --all-targets -- -D warnings` passed.
  - `source .venv/bin/activate && uvx maturin develop --manifest-path rust/minisgl-cpu-py/Cargo.toml` succeeded.
  - `python -m unittest discover -s rust/minisgl-cpu-py/tests -p 'test_*.py' -v` passed.
  - `./rust/scripts/ci_check.sh` passed.
