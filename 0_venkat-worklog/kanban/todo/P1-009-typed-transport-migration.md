# P1-009: Typed Transport Migration

Priority: P1  
Status: todo  
Depends on: P1-008

## Objective

Replace Python dataclass recursive serialization path with typed schema and lower-overhead message transport.

## Checklist

- [ ] Define stable request/reply schemas (versioned).
- [ ] Implement serializer/deserializer in Rust and compatibility parser in Python.
- [ ] Add dual-stack mode (old + new transport).
- [ ] Add transport latency counters.

## TDD Subtasks

1. Red
- [ ] Add failing compatibility tests between schema versions.
- [ ] Add failing roundtrip tests for all message types and edge payloads.

2. Green
- [ ] Implement typed transport and pass compatibility suite.

3. Refactor
- [ ] Remove dead serialization code from hot path after cutover.
- [ ] Keep compatibility mode for rollback window.

## Acceptance Criteria

- [ ] Transport roundtrip correctness at parity.
- [ ] Lower per-message overhead vs current msgpack/dataclass path.
