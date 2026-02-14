# P1-008: Rust Tokenizer/Detokenizer Service

Priority: P1  
Status: in-progress  
Depends on: P1-007

## Objective

Move tokenizer/detokenizer worker path from Python process to Rust service implementation.

## Checklist

- [x] Define tokenizer interface compatible with current request/reply flow.
- [x] Implement batching support (fixes current Python TODO gap).
- [x] Implement incremental detokenization behavior parity.
- [ ] Preserve streaming chunk semantics.

## TDD Subtasks

1. Red
- [ ] Create failing golden tests for tokenize and detokenize parity vs Python implementation.
- [x] Add failing tests for multilingual and CJK streaming behavior.

2. Green
- [x] Implement tokenization and detokenization modules to pass parity tests.

3. Refactor
- [ ] Optimize batching heuristics and memory usage.

## Acceptance Criteria

- [ ] Output text chunks match Python behavior for parity corpus.
- [ ] Tokenizer CPU time/request improves relative to Python worker baseline.

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
