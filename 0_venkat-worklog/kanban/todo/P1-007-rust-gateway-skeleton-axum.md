# P1-007: Rust Gateway Skeleton (Axum/Tokio)

Priority: P1  
Status: todo  
Depends on: P0-006

## Objective

Create Rust control-plane skeleton for API ingress and lifecycle, while Python GPU worker remains backend executor.

## Reuse Targets (from sgl-model-gateway)

- `axum` routing pattern
- `tokio` runtime model
- structured `tracing` logger setup
- worker registry and health endpoint style

## Checklist

- [ ] Create `minisgl-cpu-gateway` binary crate.
- [ ] Add endpoints:
  - [ ] `/liveness`
  - [ ] `/readiness`
  - [ ] `/v1/models`
  - [ ] placeholder `/v1/chat/completions` pass-through
- [ ] Add worker registration config (initial static mode).
- [ ] Add graceful shutdown lifecycle.

## TDD Subtasks

1. Red
- [ ] Add failing integration tests for all core endpoints and readiness behavior.

2. Green
- [ ] Implement minimal endpoint behavior to satisfy tests.

3. Refactor
- [ ] Consolidate config parsing and startup wiring.

## Acceptance Criteria

- [ ] Gateway can front existing Python server without breaking client behavior.
- [ ] Health checks correctly reflect worker availability.
