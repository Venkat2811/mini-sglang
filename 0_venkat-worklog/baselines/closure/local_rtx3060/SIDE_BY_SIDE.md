# Closure A/B Side-By-Side

## Run Context

- model: `Qwen/Qwen3-4B`
- prompts: `0_venkat-worklog/baselines/closure/local_rtx3060/prompts.jsonl` (20 prompts)
- concurrency: `4`
- streaming: `True`
- typed transport: `MINISGL_TYPED_TRANSPORT=1`
- gpu: `NVIDIA GeForce RTX 3060`
- driver: `570.133.07`
- python: `3.12.12`

## Per-Backend Stability (3 runs each)

| Metric | Python mean | Python CV | Rust mean | Rust CV |
|---|---:|---:|---:|---:|
| tok/s | 143.08 | 0.00% | 143.02 | 0.10% |
| req/s | 2.27 | 0.00% | 2.27 | 0.10% |
| TTFT avg ms | 86.15 | 0.24% | 86.53 | 0.41% |
| TTFT p50 ms | 78.99 | 0.27% | 78.83 | 0.41% |
| TTFT p99 ms | 200.50 | 1.33% | 202.93 | 0.70% |
| E2E avg s | 1.75 | 0.02% | 1.75 | 0.10% |
| E2E p50 s | 1.74 | 0.04% | 1.74 | 0.11% |
| E2E p99 s | 1.86 | 0.15% | 1.86 | 0.11% |

## Rust vs Python (medians)

| Metric | Python median | Rust median | Delta % | Verdict |
|---|---:|---:|---:|:---|
| tok/s | 143.08 | 143.09 | +0.01% | BETTER |
| req/s | 2.27 | 2.27 | +0.01% | BETTER |
| TTFT avg ms | 86.19 | 86.53 | +0.40% | WORSE |
| TTFT p50 ms | 78.87 | 78.73 | -0.18% | BETTER |
| TTFT p99 ms | 199.78 | 203.76 | +1.99% | WORSE |
| E2E avg s | 1.75 | 1.75 | -0.02% | BETTER |
| E2E p50 s | 1.74 | 1.74 | -0.10% | BETTER |
| E2E p99 s | 1.86 | 1.86 | +0.21% | WORSE |

## Stability Gate

Per `release_gate.py` defaults, throughput CV above 10% indicates an unstable run series.
- python throughput CV: `0.00%` (PASS)
- rust throughput CV:   `0.10%` (PASS)

## Closure Outcome Per RFC 0001

- online tok/s delta: `+0.01%`
- Outcome B: Rust neutral (within +/- 2% online tok/s)

