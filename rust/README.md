# mini-sglang Rust Workspace

This workspace contains the CPU-side Rust path for Mini-SGLang.

## Crates

- `minisgl-cpu-core`: scheduling/cache data model and shared CPU primitives.
- `minisgl-cpu-py`: PyO3 bindings for Python integration (`minisgl_cpu.mini_sgl_cpu_rs`).
- `minisgl-cpu-gateway`: placeholder OpenAI-compatible HTTP server scaffold with `axum`.

## Naming and Versioning

- Workspace version lives in `rust/Cargo.toml` under `[workspace.package]`.
- Crates use `minisgl-cpu-*` package names.
- Python extension module name is `minisgl_cpu.mini_sgl_cpu_rs`.
- Rust-side types mirror Python semantics for parity:
  - `Req`
  - `Batch`
  - `SamplingParams`

## Local Validation

From repo root:

```bash
cd rust
./scripts/ci_check.sh
```

The script runs:

- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `maturin develop --manifest-path minisgl-cpu-py/Cargo.toml`
- Python import smoke test (`unittest`)
