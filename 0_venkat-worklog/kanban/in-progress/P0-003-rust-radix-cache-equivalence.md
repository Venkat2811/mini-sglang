# P0-003: Rust Radix Cache Equivalence

Priority: P0  
Status: in-progress  
Depends on: P0-002

## Objective

Port CPU radix cache manager behavior to Rust with strict behavioral parity vs `python/minisgl/kvcache/radix_manager.py`.

## Checklist

- [x] Define Rust cache traits mirroring `BaseCacheManager` semantics:
  - [x] `match_prefix`
  - [x] `lock_handle`/`unlock`
  - [x] `insert_prefix`
  - [x] `evict`
- [x] Implement node split behavior and ref-count semantics.
- [x] Implement evictable/protected size accounting.
- [x] Add invariants checker equivalent to Python integrity checks.

## TDD Subtasks

1. Red
- [x] Build golden unit tests from Python edge cases:
  - [x] exact prefix match
  - [x] partial split match
  - [x] lock/unlock nested nodes
  - [x] eviction when multiple leaves share parent
- [x] Add property tests for size accounting consistency.

2. Green
- [x] Implement Rust radix manager until all tests pass.

3. Refactor
- [ ] Optimize hot loops (token compare and traversal) without changing API.
- [ ] Add microbench test for `match_prefix`.

## Acceptance Criteria

- [ ] Rust and Python managers produce identical outcomes for golden traces.
- [ ] No correctness regression under stress tests.
- [ ] At least neutral or better performance vs Python on cache operations.

## Progress Notes (2026-02-14)

- Added `PrefixCacheManager` trait and `SizeInfo` in `rust/minisgl-cpu-core/src/cache.rs`.
- Added `RadixCacheManager` in `rust/minisgl-cpu-core/src/radix.rs` with:
  - node split on partial-match walk,
  - lock/unlock refcount propagation,
  - evictable/protected size accounting,
  - integrity checker for parent/child and size invariants.
- Added parity-first tests in `rust/minisgl-cpu-core/tests/radix_cache_equivalence.rs`:
  - exact prefix match,
  - partial split behavior,
  - nested lock/unlock,
  - shared-parent leaf eviction,
  - property-style operation sequence for size/integrity checks.
- Validation:
  - `cargo test -p minisgl-cpu-core` passed.
  - `cargo clippy -p minisgl-cpu-core --all-targets -- -D warnings` passed.
  - `cargo test --workspace` passed.
