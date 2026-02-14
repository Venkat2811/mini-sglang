# Tokenizer Backend A/B (2026-02-14)

Scope: online benchmark profile on local RTX 3060, same server config, same model, same CPU backend (`MINISGL_CPU_BACKEND=rust_hotpath`).

Profile:

- `batch_size=16`
- `min_input_len=512`
- `max_input_len=1024`
- `batch_output_tokens=32`
- `warmup_input_tokens=256`
- `warmup_output_tokens=16`

## Results

### `MINISGL_TOKENIZER_BACKEND=python`

- Throughput: `927.63 tok/s`
- TTFT avg: `334.32 ms`
- TPOT avg: `7.4366 ms`
- E2E avg: `0.5649 s`

### `MINISGL_TOKENIZER_BACKEND=rust_inprocess`

- Throughput: `884.22 tok/s`
- TTFT avg: `331.26 ms`
- TPOT avg: `8.3441 ms`
- E2E avg: `0.5899 s`
- Delta vs python throughput: `-4.68%`

### `MINISGL_TOKENIZER_BACKEND=rust_tokenize_only`

- Throughput: `887.31 tok/s`
- TTFT avg: `324.20 ms`
- TPOT avg: `8.5263 ms`
- E2E avg: `0.5885 s`
- Delta vs python throughput: `-4.35%`

## Parity Spot Check

Spot parity check on mixed text/chat prompts and streaming detokenization:

- Text tokenization parity: pass
- Chat template tokenization parity: pass
- Incremental detokenization chunk parity: pass

Recorded JSON artifacts:

- `0_venkat-worklog/baselines/latest-online-tokenizer-python.json`
- `0_venkat-worklog/baselines/latest-online-tokenizer-rust-inprocess.json`
- `0_venkat-worklog/baselines/latest-online-tokenizer-rust-tokenize-only.json`
- `0_venkat-worklog/baselines/latest-tokenizer-parity-rust-inprocess.json`

## Diagnosis

Current Rust tokenizer path preserves behavior but does not improve end-to-end throughput on this profile yet. Main next optimization target is detokenization hot path (per-token/per-step decode costs), then transport/copy reduction between Rust and Python boundaries.
