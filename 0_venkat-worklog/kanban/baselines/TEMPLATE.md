# Baseline Record Template

Status: reference baseline  
Scope: local machine, single GPU, sanitized log

## Environment

- GPU:
- Driver:
- CUDA runtime:
- Python:
- Key packages:

## Model

- `<model-path>`

## Offline Baseline (`LLM` path)

- `init_time_s =`
- `warmup_time_s =` (`warmup_tokens =`)
- `single_req_time_s =` (`single_out_tokens =`)
- Batch run:
  - `batch_num_reqs =`
  - `batch_total_out_tokens =`
  - `batch_time_s =`
  - `batch_out_tok_per_s =`

## Online Baseline (server + API path)

- Warmup:
  - `warmup_elapsed_s =`
  - `tokens =`
- Benchmark:
  - `bench_wall_time_s =`
  - `Num requests =`
  - `Num tokens =`
  - `TTFT avg =`
  - `TPOT avg =`
  - `E2E avg =`
  - `Duration =`
  - `Throughput =`
  - `Req/s =`

## Gate Config Used

- Offline gate file:
- Online gate file:

## Notes

- Keep record sanitized for public repo:
  - no absolute paths
  - no user/host identifiers
  - no secrets
