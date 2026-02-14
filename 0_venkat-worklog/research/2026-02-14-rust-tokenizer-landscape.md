# Rust Tokenizer Landscape (Mini-SGLang)

Date: 2026-02-14

## Scope

Research goal: pick a Rust tokenizer path for `mini-sglang` CPU-side migration that preserves current Python behavior and allows fast follow with minimal divergence.

## Local Reference Findings

### `sgl-model-gateway` reuse signal

- `sgl-model-gateway` already uses `llm-tokenizer` as its tokenizer abstraction.
- Code references:
  - `sglang/sgl-model-gateway/Cargo.toml` (`llm-tokenizer = "~1.0.0"`)
  - `sglang/sgl-model-gateway/src/lib.rs` (`pub use llm_tokenizer as tokenizer;`)
  - `sglang/sgl-model-gateway/src/routers/tokenize/handlers.rs`

Implication: reusing `llm-tokenizer` in mini-sgl keeps implementation direction aligned with the existing Rust gateway stack in the full project.

## Online/Primary Sources Reviewed

- `llm-tokenizer` crate metadata:
  - https://crates.io/crates/llm-tokenizer
  - https://docs.rs/llm-tokenizer
- `tokenizers` crate (Hugging Face Rust tokenizer core):
  - https://crates.io/crates/tokenizers
  - https://docs.rs/tokenizers
  - https://github.com/huggingface/tokenizers
- `sentencepiece` Rust bindings:
  - https://crates.io/crates/sentencepiece
  - https://docs.rs/sentencepiece
- HF chat-template reference behavior:
  - https://huggingface.co/docs/transformers/chat_templating

## Decision

Use `llm-tokenizer` as the backend abstraction for mini-sgl Rust tokenizer work.

Reasons:

1. Reuse parity with `sgl-model-gateway` tokenizer stack.
2. Built-in chat-template support and incremental decode utilities.
3. Backed by Hugging Face `tokenizers` Rust crate for fast tokenizer path.
4. Keeps design consistent with the long-term Rust CPU service direction.

## Implementation Notes Applied in `P1-008`

- New crate: `rust/minisgl-cpu-tokenizer`
- Added `HfTokenizerBackend` on top of `llm-tokenizer::HuggingFaceTokenizer`.
- Added manager interfaces to match mini-sgl tokenizer/detokenizer request/response semantics.
- Implemented text-batch encode path and incremental detokenize state machine parity logic.
