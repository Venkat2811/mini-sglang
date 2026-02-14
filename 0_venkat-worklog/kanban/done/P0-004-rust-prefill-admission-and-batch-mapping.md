# P0-004: Rust Prefill Admission and Batch Mapping

Priority: P0  
Status: done  
Depends on: P0-003

## Objective

Move prefill admission logic and batch metadata mapping to Rust hot path while keeping Python scheduler orchestration flow intact.

## Checklist

- [x] Port `PrefillAdder` logic and request chunking decisions.
- [x] Port decode in-flight token budget interactions.
- [x] Port mapping builders:
  - [x] positions
  - [x] input tuple
  - [x] write tuple
- [x] Keep output structures binary-compatible with Python expectations.

## TDD Subtasks

1. Red
- [x] Write failing golden tests comparing Rust output to Python for mixed workloads:
  - [x] only prefill
  - [x] only decode
  - [x] mixed with chunked prefill
  - [x] near-capacity cache conditions

2. Green
- [x] Implement Rust builders/admission until exact match with expected tensors/index arrays.

3. Refactor
- [x] Remove redundant allocations and copies.
- [x] Add benchmark for mapping generation latency.

## Acceptance Criteria

- [x] Token-level output parity remains unchanged after integration.
- [x] Scheduler CPU step time improves for mixed workload baseline.

## Progress Notes (2026-02-14)

- Added Rust scheduler hot-path primitives in `rust/minisgl-cpu-core/src/prefill.rs`:
  - `PrefillAdder`, `PrefillManager`, `PendingReq`, `ScheduledReq`,
  - mapping builders `make_positions`, `make_input_tuple`, `make_write_tuple`,
  - `decode_inflight_tokens` helper.
- Added unit tests in `rust/minisgl-cpu-core/tests/prefill_admission.rs` for:
  - near-capacity rejection,
  - chunked prefill admission,
  - decode in-flight budget interaction and chunk requeue behavior,
  - mapping output structure expectations.
- Added Python-golden parity exporter and replay:
  - exporter: `rust/scripts/export_prefill_trace.py`,
  - golden payload: `rust/minisgl-cpu-core/tests/data/prefill_golden_trace.yaml`,
  - replay test: `rust/minisgl-cpu-core/tests/prefill_python_trace_parity.rs`.
- Added mapping microbench:
  - `rust/minisgl-cpu-core/examples/prefill_mapping_bench.rs`.
  - local result: `prefill_mapping_ops_per_sec=249611.83`.
- Validation:
  - `cargo test -p minisgl-cpu-core` passed.
  - `cargo clippy -p minisgl-cpu-core --all-targets -- -D warnings` passed.
  - `cargo test --workspace` passed.
  - `.venv/bin/python -m pytest tests/misc/test_token_parity.py -q` passed.
  - Deterministic token parity run passed:
    - `.venv/bin/python -m minisgl.benchmark.token_parity --model-path Qwen/Qwen2.5-0.5B-Instruct --max-tokens 16 --token-prompt-count 4 --min-input-len 32 --max-input-len 64 --cuda-graph-max-bs 1 --master-port 2380 --out 0_venkat-worklog/baselines/latest-token-parity.json`
    - result: `parity_passed=True`, no mismatches in both `text_prompts` and `token_prompts`.
  - Mixed workload baseline A/B remains Rust-ahead in latest run:
    - offline throughput delta: `+0.07%`
    - online throughput delta: `+0.69%`
