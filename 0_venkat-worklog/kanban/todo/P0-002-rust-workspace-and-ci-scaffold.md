# P0-002: Rust Workspace and CI Scaffold

Priority: P0  
Status: todo  
Depends on: P0-001

## Objective

Initialize `mini-sglang/rust/` with a production-shaped workspace and Python binding path that matches proven structure from `sglang/sgl-model-gateway`.

## Reuse Targets (from sgl-model-gateway)

- `bindings/python` layout with `maturin` + `PyO3`.
- Split crates: core logic crate + binding crate.
- Build profiles (`release`, `ci`) and reproducible CLI packaging style.

## Checklist

- [ ] Create Cargo workspace:
  - [ ] `rust/minisgl-cpu-core`
  - [ ] `rust/minisgl-cpu-py` (PyO3)
  - [ ] placeholder `rust/minisgl-cpu-gateway`
- [ ] Add `pyproject.toml` for Python bindings using maturin.
- [ ] Add minimal CI script for `cargo test`, `cargo clippy`, `maturin develop`.
- [ ] Add versioning strategy and crate/module naming conventions.

## TDD Subtasks

1. Red
- [ ] Add failing tests for core crate skeleton APIs (scheduler types, cache trait stubs).
- [ ] Add failing Python smoke test that imports extension module.

2. Green
- [ ] Implement minimal structs/traits to satisfy tests.
- [ ] Build extension and pass import smoke test.

3. Refactor
- [ ] Normalize naming to match Python-side concepts (`Req`, `Batch`, `SamplingParams` analogs).

## Acceptance Criteria

- [ ] `cargo test` passes for scaffold crates.
- [ ] Python can import the Rust extension from local editable build.
- [ ] Project structure is documented in `rust/README.md`.
