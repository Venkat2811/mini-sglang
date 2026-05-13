# Closure A/B Side-By-Side

## Run Context

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- prompts: `0_venkat-worklog/baselines/closure/smoke/smoke_subset.jsonl` (8 prompts)
- concurrency: `4`
- streaming: `True`
- typed transport: `MINISGL_TYPED_TRANSPORT=1`
- gpu: `NVIDIA GeForce RTX 3060`
- driver: `570.133.07`
- python: `3.12.12`

## Per-Backend Stability (3 runs each)

| Metric | Python mean | Python CV | Rust mean | Rust CV |
|---|---:|---:|---:|---:|
| tok/s | 174.34 | 0.00% | 582.35 | 0.00% |
| req/s | 5.62 | 0.00% | 18.79 | 0.00% |
| TTFT avg ms | 530.04 | 0.00% | 44.50 | 0.00% |
| TTFT p50 ms | 969.47 | 0.00% | 50.01 | 0.00% |
| TTFT p99 ms | 970.85 | 0.00% | 147.38 | 0.00% |
| E2E avg s | 0.66 | 0.00% | 0.18 | 0.00% |
| E2E p50 s | 1.10 | 0.00% | 0.18 | 0.00% |
| E2E p99 s | 1.10 | 0.00% | 0.28 | 0.00% |

## Rust vs Python (medians)

| Metric | Python median | Rust median | Delta % | Verdict |
|---|---:|---:|---:|:---|
| tok/s | 174.34 | 582.35 | +234.03% | BETTER |
| req/s | 5.62 | 18.79 | +234.03% | BETTER |
| TTFT avg ms | 530.04 | 44.50 | -91.60% | BETTER |
| TTFT p50 ms | 969.47 | 50.01 | -94.84% | BETTER |
| TTFT p99 ms | 970.85 | 147.38 | -84.82% | BETTER |
| E2E avg s | 0.66 | 0.18 | -73.43% | BETTER |
| E2E p50 s | 1.10 | 0.18 | -83.45% | BETTER |
| E2E p99 s | 1.10 | 0.28 | -74.57% | BETTER |

## Stability Gate

Per `release_gate.py` defaults, throughput CV above 10% indicates an unstable run series.
- python throughput CV: `0.00%` (PASS)
- rust throughput CV:   `0.00%` (PASS)

## Closure Outcome Per RFC 0001

- online tok/s delta: `+234.03%`
- Outcome A: Rust positive (>= +2% online tok/s)

