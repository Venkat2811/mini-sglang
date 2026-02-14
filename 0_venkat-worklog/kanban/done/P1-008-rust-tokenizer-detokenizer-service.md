# P1-008: Rust Tokenizer/Detokenizer Service

Priority: P1  
Status: done  
Depends on: P1-007

## Objective

Move tokenizer/detokenizer worker path from Python process to Rust service implementation.

## Checklist

- [x] Define tokenizer interface compatible with current request/reply flow.
- [x] Implement batching support (fixes current Python TODO gap).
- [x] Implement incremental detokenization behavior parity.
- [x] Preserve streaming chunk semantics.

## TDD Subtasks

1. Red
- [x] Create failing golden tests for tokenize and detokenize parity vs Python implementation.
- [x] Add failing tests for multilingual and CJK streaming behavior.

2. Green
- [x] Implement tokenization and detokenization modules to pass parity tests.

3. Refactor
- [x] Optimize batching heuristics and memory usage.

## Acceptance Criteria

- [x] Output text chunks match Python behavior for parity corpus.
- [x] Tokenizer CPU time/request improves relative to Python worker baseline.

## Progress Notes (2026-02-14)

- Added new crate: `rust/minisgl-cpu-tokenizer`.
- Implemented interface types for tokenize/detokenize request and reply shapes:
  - `TokenizeRequest`, `TokenizeOutput`
  - `DetokenizeRequest`, `DetokenizeOutput`
  - chat prompt support via `PromptInput::{Text, Messages}`.
- Implemented `TokenizeManager`:
  - text-only batch path uses backend `encode_batch` for throughput.
  - mixed/text/chat path handles per-request encode and chat-template rendering.
- Implemented `DetokenizeManager` by porting Python incremental logic:
  - per-uid decode state map
  - EOS suppression parity
  - CJK/printable-text heuristics
  - Unicode-safe char-offset slicing for incremental chunks.
- Added `HfTokenizerBackend` using `llm-tokenizer` (same tokenizer stack reused by `sgl-model-gateway`).
- Added Python runtime integration path (feature-flagged) in `minisgl.tokenizer.server`:
  - `MINISGL_TOKENIZER_BACKEND=python` (default)
  - `MINISGL_TOKENIZER_BACKEND=rust_inprocess`
  - `MINISGL_TOKENIZER_BACKEND=rust_tokenize_only`
- Added PyO3 class `TokenizerWorker` in `rust/minisgl-cpu-py` and Python adapter `minisgl.tokenizer.rust_backend`.
- Added unit tests for Python adapter marshaling:
  - `tests/misc/test_tokenizer_rust_backend.py`
- TDD evidence in this cycle:
  - red: `cargo test -p minisgl-cpu-tokenizer` failed on unimplemented tokenize/detokenize managers.
  - green: same tests pass after implementation.
  - multilingual/CJK streaming tests added and passing.
- Validation:
  - `cargo test -p minisgl-cpu-tokenizer`
  - `cargo clippy -p minisgl-cpu-tokenizer --all-targets -- -D warnings`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `pytest tests/misc/test_tokenizer_rust_backend.py`
- Runtime A/B snapshot (tokenizer-heavy online profile):
  - python backend: `927.63 tok/s`
  - rust_inprocess backend: `884.22 tok/s` (`-4.68%`)
  - rust_tokenize_only backend: `887.31 tok/s` (`-4.35%`)
  - details: `0_venkat-worklog/baselines/2026-02-14-tokenizer-backend-ab.md`
- Controlled sequential rerun (one backend at a time, explicit `MASTER_PORT`, `MINISGL_CPU_BACKEND=rust_hotpath`) still showed Rust tokenizer behind on this profile:
  - python tokenizer: `938.12 tok/s`
  - rust_tokenize_only: `886.28 tok/s` (`-5.53%`)
  - rust_inprocess: `877.39 tok/s` (`-6.47%`)
- Follow-up variant (`torch.frombuffer` no-clone) remained below python tokenizer path:
  - rust_tokenize_only: `-7.63%`
  - rust_inprocess: `-5.13%`
- Decision: keep tokenizer work as active but non-blocking; prioritize typed transport migration (`P1-009`) for broader CPU-side overhead reduction.

## Progress Notes (2026-02-14, later session)

- Added Python-parity golden tests directly in Rust tokenizer crate:
  - `tokenize_matches_python_oracle_for_text_and_chat_prompts`
  - `detokenize_matches_python_oracle_for_interleaved_multilingual_streams`
- Refactored Rust tokenize hot path to reduce CPU overhead:
  - removed avoidable prompt string cloning in all-text batch path
  - switched batch encode API to borrowed slices (`&[&str]`)
  - reduced per-token cast overhead in `cast_u32_to_i32`
- Validation:
  - `cargo test -p minisgl-cpu-tokenizer`
  - `cargo clippy -p minisgl-cpu-tokenizer --all-targets -- -D warnings`
  - `pytest tests/misc/test_tokenizer_rust_backend.py`
- CPU-only tokenizer manager microbench (same machine, `Qwen/Qwen2.5-0.5B-Instruct`, batch size 16):
  - text-only batch latency: python `11.13 ms`, rust `1.49 ms` (`+86.57%` faster)
  - mixed text/chat batch latency: python `10.48 ms`, rust `6.25 ms` (`+40.43%` faster)
- Note: end-to-end online throughput still depends on additional components beyond tokenizer manager latency.
