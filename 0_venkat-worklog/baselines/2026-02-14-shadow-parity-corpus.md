# Shadow and Parity Corpus Record (2026-02-14)

Status: parity corpus validation snapshot  
Scope: local machine, single GPU, deterministic + shadow checks

## Deterministic Token Parity Profiles

All runs used:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- backend A: `python`
- backend B: `rust_hotpath`
- `--cuda-graph-max-bs 1`
- `parity_passed=True`

Profiles and artifacts:

1. Short prompts
- command artifact: `0_venkat-worklog/baselines/latest-token-parity-short.json`
- profile: input length `8..32`, `token_prompt_count=8`, `shared_prefix_len=0`
- result: `text_prompts` mismatches `0`, `token_prompts` mismatches `0`

2. Long prompts
- command artifact: `0_venkat-worklog/baselines/latest-token-parity-long.json`
- profile: input length `384..512`, `token_prompt_count=4`, `shared_prefix_len=0`
- result: `text_prompts` mismatches `0`, `token_prompts` mismatches `0`

3. Shared-prefix heavy prompts
- command artifact: `0_venkat-worklog/baselines/latest-token-parity-shared-prefix.json`
- profile: input length `96..128`, `token_prompt_count=8`, `shared_prefix_len=80`
- result: `text_prompts` mismatches `0`, `token_prompts` mismatches `0`

## Shadow Corpus Checks

1. Deterministic shadow run
- command artifact: `0_venkat-worklog/baselines/latest-offline-shadow-check.json`
- report path: `0_venkat-worklog/baselines/latest-shadow-divergence.jsonl`
- summary: `divergence_entries=0`

2. Mixed sampling params shadow run
- sampling mix included deterministic and stochastic settings:
  - `temperature=0.0, top_k=1, top_p=1.0`
  - `temperature=0.7, top_k=40, top_p=0.9`
  - `temperature=1.0, top_k=0, top_p=0.95`
  - `temperature=0.3, top_k=20, top_p=0.8`
- report path: `0_venkat-worklog/baselines/latest-shadow-mixed-divergence.jsonl`
- summary: `divergence_entries=0`

## Known/Allowed Differences for Stochastic Decoding

- Exact token-by-token output equality is only a hard gate for deterministic decoding (`temperature=0`, greedy settings).
- For stochastic decoding (`temperature>0`, or sampling-enabled `top_k/top_p`), token sequences may differ across runs due to RNG behavior and sampling order effects.
- In stochastic settings, this project tracks parity primarily via:
  - structural shadow metadata divergence (`positions`, `input_mapping`, `write_tuple`)
  - safety/quality metrics and throughput, not strict sequence identity.
