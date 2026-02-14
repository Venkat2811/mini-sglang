# P1-009: Typed Transport Migration

Priority: P1  
Status: in-progress  
Depends on: P1-008

## Objective

Replace Python dataclass recursive serialization path with typed schema and lower-overhead message transport.

## Checklist

- [x] Define stable request/reply schemas (versioned).
- [x] Implement serializer/deserializer and compatibility parser in Python.
- [x] Add dual-stack mode (old + new transport).
- [ ] Add transport latency counters.

## TDD Subtasks

1. Red
- [x] Add failing compatibility tests between schema versions.
- [x] Add failing roundtrip tests for all message types and edge payloads.

2. Green
- [x] Implement typed transport and pass compatibility suite.

3. Refactor
- [ ] Remove dead serialization code from hot path after cutover.
- [ ] Keep compatibility mode for rollback window.

## Acceptance Criteria

- [x] Transport roundtrip correctness at parity.
- [x] Lower per-message overhead vs current msgpack/dataclass path.

## Progress Notes (2026-02-14)

- Added typed schema v1 (feature-flagged by `MINISGL_TYPED_TRANSPORT`) for:
  - backend messages (`UserMsg`, `ExitMsg`, `BatchBackendMsg`)
  - tokenizer messages (`TokenizeMsg`, `DetokenizeMsg`, `AbortMsg`, `BatchTokenizerMsg`)
  - frontend messages (`UserReply`, `BatchFrontendMsg`)
- Added dual-stack decode behavior:
  - decoder accepts both typed schema payloads and legacy recursive dataclass payloads.
  - encoder switches via env flag (`false` default keeps legacy path).
- Added tests:
  - `tests/misc/test_typed_transport.py`
  - covers typed roundtrip, legacy compatibility, and unknown schema rejection.
- Added reproducible microbench CLI:
  - `python/minisgl/benchmark/transport_overhead.py`
  - output artifact: `0_venkat-worklog/baselines/latest-transport-overhead.json`
- Latest microbench result (encode + msgpack + decode):
  - backend ops/s: `38608.10` -> `67281.38` (`+74.27%`)
  - tokenizer ops/s: `183289.91` -> `379834.34` (`+107.23%`)
