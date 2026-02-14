# Rust CPU Service Cutover Design (2026-02-14)

Scope: `mini-sglang` CPU-side only. Python GPU worker and Python API server remain unchanged.

## North-Star Topology

- Python API server (existing)
- Python scheduler/tokenizer client stubs (thin transport adapters only)
- Rust CPU service (out-of-process) handling:
  - scheduler metadata ops (positions/input/write mapping)
  - tokenizer/detokenizer ops
  - optional shadow comparison hooks during rollout

## Transport Choice

- Primary: local loopback HTTP (`axum`) with explicit request timeout.
- Payload: typed schema v1 over `application/msgpack` (reuse existing typed transport shape).
- Rationale:
  - simple and debuggable during early cutover
  - can migrate later to UDS or gRPC without changing high-level mode surface

## Request Router + Worker Model

- Single Rust process with two logical handlers:
  - `/v1/cpu/metadata/*` for scheduler metadata kernels
  - `/v1/cpu/tokenizer/*` for tokenize/detokenize
- Internally:
  - bounded worker pool for metadata requests
  - stateful tokenizer workers keyed by tokenizer id/model
  - backpressure via bounded queues and fast reject on overload

## Timeout / Retry / Error Contracts

- Python client defaults:
  - connect timeout: `250 ms`
  - request timeout: `1000 ms`
  - retries: 1 retry only for transport-level transient failures
- Error contract:
  - `service_unavailable`: service down or health check fail
  - `timeout`: request deadline exceeded
  - `bad_payload`: schema/version mismatch
  - `internal_error`: service-side execution failure
- Rollback behavior:
  - `rust_service` mode falls back to `rust_inprocess_ffi` during rollout window
  - fallback counters are emitted in runtime metrics

## Rollout Phases

1. Mode-surface phase (done):
- add `python_cpu`, `rust_inprocess_ffi`, `rust_service` modes
- wire explicit fallback handling and counters

2. Service skeleton phase:
- stand up Rust service endpoints for metadata/tokenizer
- implement Python client adapters behind `rust_service`
- add parity tests and unavailability/timeout tests

3. Cutover phase:
- make `rust_service` primary for CPU hot path
- keep in-process fallback during stabilization
- gate by parity + perf + stability thresholds

4. Cleanup phase:
- remove runtime dependency on in-process PyO3 for hot path (`P1-012`)
