# P1-007: Rust Gateway Skeleton (Axum/Tokio)

Priority: P1  
Status: done  
Depends on: P0-006

## Objective

Create Rust control-plane skeleton for API ingress and lifecycle, while Python GPU worker remains backend executor.

## Reuse Targets (from sgl-model-gateway)

- `axum` routing pattern
- `tokio` runtime model
- structured `tracing` logger setup
- worker registry and health endpoint style

## Checklist

- [x] Create `minisgl-cpu-gateway` binary crate.
- [x] Add endpoints:
  - [x] `/liveness`
  - [x] `/readiness`
  - [x] `/v1/models`
  - [x] placeholder `/v1/chat/completions` pass-through
- [x] Add worker registration config (initial static mode).
- [x] Add graceful shutdown lifecycle.

## TDD Subtasks

1. Red
- [x] Add failing integration tests for all core endpoints and readiness behavior.

2. Green
- [x] Implement minimal endpoint behavior to satisfy tests.

3. Refactor
- [x] Consolidate config parsing and startup wiring.

## Acceptance Criteria

- [x] Gateway can front existing Python server without breaking client behavior.
- [x] Health checks correctly reflect worker availability.

## Progress Notes (2026-02-14)

- Implemented gateway routing and lifecycle in `rust/minisgl-cpu-gateway/src/main.rs`:
  - `/liveness`, `/readiness`, `/v1/models`, `/v1/chat/completions`
  - static worker registry via env:
    - `MINISGL_GATEWAY_WORKERS` (comma-separated worker base URLs)
    - `MINISGL_GATEWAY_ADDR`
    - `MINISGL_GATEWAY_MODEL_ID`
    - `MINISGL_GATEWAY_TIMEOUT_MS`
  - graceful shutdown using `tokio::signal` (`Ctrl+C` + SIGTERM on Unix).
- Added pass-through strategy:
  - `/v1/chat/completions` forwards JSON payload to configured workers in order and returns first successful upstream response.
  - returns structured `503/502` errors when no workers or all workers unreachable.
- Added endpoint integration tests (mock worker server):
  - liveness success
  - readiness unavailable with no workers
  - readiness success with healthy worker
  - models endpoint response
  - chat pass-through response
- Validation:
  - `cargo test -p minisgl-cpu-gateway` passed.
  - `cargo clippy -p minisgl-cpu-gateway --all-targets -- -D warnings` passed.
  - End-to-end check against real Python minisgl server:
    - backend launch: `.venv/bin/python -m minisgl ... --port 1919`
    - gateway launch: `MINISGL_GATEWAY_WORKERS=http://127.0.0.1:1919 cargo run -p minisgl-cpu-gateway`
    - readiness via gateway: `GET /readiness` -> `200` (`ready=true`, `healthy_workers=1`)
    - pass-through via gateway: `POST /v1/chat/completions` -> streamed `data: ... [DONE]` with `HTTP 200`.
