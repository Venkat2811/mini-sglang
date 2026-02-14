# P0-004: Rust Prefill Admission and Batch Mapping

Priority: P0  
Status: todo  
Depends on: P0-003

## Objective

Move prefill admission logic and batch metadata mapping to Rust hot path while keeping Python scheduler orchestration flow intact.

## Checklist

- [ ] Port `PrefillAdder` logic and request chunking decisions.
- [ ] Port decode in-flight token budget interactions.
- [ ] Port mapping builders:
  - [ ] positions
  - [ ] input tuple
  - [ ] write tuple
- [ ] Keep output structures binary-compatible with Python expectations.

## TDD Subtasks

1. Red
- [ ] Write failing golden tests comparing Rust output to Python for mixed workloads:
  - [ ] only prefill
  - [ ] only decode
  - [ ] mixed with chunked prefill
  - [ ] near-capacity cache conditions

2. Green
- [ ] Implement Rust builders/admission until exact match with expected tensors/index arrays.

3. Refactor
- [ ] Remove redundant allocations and copies.
- [ ] Add benchmark for mapping generation latency.

## Acceptance Criteria

- [ ] Token-level output parity remains unchanged after integration.
- [ ] Scheduler CPU step time improves for mixed workload baseline.
