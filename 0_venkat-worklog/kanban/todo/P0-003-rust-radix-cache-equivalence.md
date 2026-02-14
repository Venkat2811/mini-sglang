# P0-003: Rust Radix Cache Equivalence

Priority: P0  
Status: todo  
Depends on: P0-002

## Objective

Port CPU radix cache manager behavior to Rust with strict behavioral parity vs `python/minisgl/kvcache/radix_manager.py`.

## Checklist

- [ ] Define Rust cache traits mirroring `BaseCacheManager` semantics:
  - [ ] `match_prefix`
  - [ ] `lock_handle`/`unlock`
  - [ ] `insert_prefix`
  - [ ] `evict`
- [ ] Implement node split behavior and ref-count semantics.
- [ ] Implement evictable/protected size accounting.
- [ ] Add invariants checker equivalent to Python integrity checks.

## TDD Subtasks

1. Red
- [ ] Build golden unit tests from Python edge cases:
  - [ ] exact prefix match
  - [ ] partial split match
  - [ ] lock/unlock nested nodes
  - [ ] eviction when multiple leaves share parent
- [ ] Add property tests for size accounting consistency.

2. Green
- [ ] Implement Rust radix manager until all tests pass.

3. Refactor
- [ ] Optimize hot loops (token compare and traversal) without changing API.
- [ ] Add microbench test for `match_prefix`.

## Acceptance Criteria

- [ ] Rust and Python managers produce identical outcomes for golden traces.
- [ ] No correctness regression under stress tests.
- [ ] At least neutral or better performance vs Python on cache operations.
