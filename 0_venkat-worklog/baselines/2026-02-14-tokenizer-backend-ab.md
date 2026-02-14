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

## Controlled Sequential Re-Run (2026-02-14, later session)

Method:

- Strict one-at-a-time server runs (never simultaneous).
- Fresh server per mode with explicit `MASTER_PORT` values to avoid stale rendezvous port collisions.
- Same tokenizer-heavy profile as above.
- `MINISGL_CPU_BACKEND=rust_hotpath` held constant across modes.

Results:

### Rust CPU + Python tokenizer

- Throughput: `938.12 tok/s`
- TTFT avg: `318.54 ms`
- TPOT avg: `7.7626 ms`

### Rust CPU + Rust tokenizer (`rust_tokenize_only`)

- Throughput: `886.28 tok/s`
- TTFT avg: `329.19 ms`
- TPOT avg: `8.4810 ms`
- Delta vs Python tokenizer: `-5.53%`

### Rust CPU + Rust tokenizer (`rust_inprocess`)

- Throughput: `877.39 tok/s`
- TTFT avg: `331.86 ms`
- TPOT avg: `8.5434 ms`
- Delta vs Python tokenizer: `-6.47%`

Variant check (no-clone tensor conversion in Python adapter):

- Python tokenizer: `931.02 tok/s`
- Rust tokenize-only: `859.96 tok/s` (`-7.63%`)
- Rust inprocess: `883.29 tok/s` (`-5.13%`)

Takeaway:

- Rust tokenizer remains parity-correct but still below Python tokenizer path in end-to-end online throughput for this workload.
- Work will continue on lower-overhead transport path (`P1-009`) before further tokenizer micro-optimizations.

## CPU-Only Tokenizer Manager Microbench (2026-02-14, later session)

Purpose: isolate tokenizer manager CPU cost from GPU serving, scheduler, and transport effects.

Setup:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- batch size: `16`
- runs: `5` samples per mode, `200` iterations/sample after warmup
- compared managers:
  - Python: `minisgl.tokenizer.tokenize.TokenizeManager`
  - Rust: `minisgl.tokenizer.rust_backend.RustTokenizerManagers`

Results (mean batch latency):

- text-only prompts:
  - python: `11.13 ms`
  - rust: `1.49 ms`
  - rust faster by `86.57%`
- mixed text/chat prompts:
  - python: `10.48 ms`
  - rust: `6.25 ms`
  - rust faster by `40.43%`

Interpretation:

- Rust tokenizer manager now shows clear CPU-side latency advantage in isolation.
- Remaining end-to-end gap comes from non-tokenizer costs in the online serving path.
