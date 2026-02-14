# P0-006: Shadow Mode and Parity Corpus

Priority: P0  
Status: todo  
Depends on: P0-005

## Objective

Establish parity confidence before broader rollout by running Rust decisions in shadow against Python.

## Checklist

- [ ] Build parity corpus:
  - [ ] short prompts
  - [ ] long prompts
  - [ ] shared-prefix heavy workloads
  - [ ] mixed sampling params
- [ ] Implement shadow mode that:
  - [ ] computes decisions in both paths
  - [ ] serves production path from selected backend
  - [ ] logs diffs with request IDs
- [ ] Add divergence report command.

## TDD Subtasks

1. Red
- [ ] Create failing tests with intentionally diverged fixtures to validate diff detection.
- [ ] Add failing test for deterministic token equality under greedy decode.

2. Green
- [ ] Implement shadow comparator and reporting.

3. Refactor
- [ ] Reduce shadow overhead for continuous CI runs.

## Acceptance Criteria

- [ ] Zero divergence on deterministic corpus.
- [ ] Documented known/allowed differences for stochastic settings.
