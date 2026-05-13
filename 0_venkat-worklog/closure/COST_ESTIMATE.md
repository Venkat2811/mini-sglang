# H100 Closure Spend Estimate

Date: 2026-05-14
Status: Draft

## Hard Cap

USD `100` total, per `RFC 0001` stop conditions.

## Workload Time Budget

| Phase | Wall-clock | Notes |
|---|---|---|
| Provider account ready, SSH in | 5-10 min | one-time |
| `uv` venv, deps install, `pip install -e .` | 8-12 min | one-time; CUDA wheels download is the main cost |
| `maturin develop --release` for `minisgl-cpu-py` | 4-6 min | one-time; release build of the PyO3 extension |
| `Qwen/Qwen3-4B` HF download | 4-8 min | one-time; about 8 GiB |
| ShareGPT subset prep | <1 min | one-time |
| Sanity smoke (one warmup request) | 2-3 min | per backend, two warmups |
| Six measured runs (3 python + 3 rust hotpath) | 30-45 min | core measurement |
| One shadow parity run | 6-10 min | optional but cheap |
| Retained artifact pull | 2-3 min | rsync off the host |
| Tear down | 1-2 min | provider dashboard |
| **Total wall-clock** | **~75-110 min** | one weekend session |

Add 30 min slack for first-time provider setup and one debug attempt before falling back to retro-only.

## Provider Compare (As Of 2026-05-14)

Numbers from public landing pages. Treat as rough; confirm at rental time.

| Provider | Instance | Per hour | 2-hour total | Notes |
|---|---|---|---|---|
| Lambda on-demand | `1x H100 PCIe 80GB` | about USD `2.49/hr` | about USD `5` | quick start, hourly billing, no commitment |
| RunPod community | `1x H100 80GB SXM5` | about USD `1.99-3.49/hr` | about USD `4-7` | spot risk varies; secure cloud pricier |
| Vast.ai | `1x H100 80GB` | about USD `1.50-2.50/hr` | about USD `3-5` | aggressive pricing, interruptible options exist |
| GCP A3 (1x H100) | `a3-highgpu-1g` | about USD `11/hr` | about USD `22` | only consider if other free credits offset |

## Recommended Plan

Pick Lambda on-demand or RunPod community for two reasons:

1. Hourly billing means a 90-minute session lands well under USD `10`, leaving plenty of cap headroom for one debug rerun.
2. Both expose pre-baked CUDA 12.4+ images so the install phase is not provider-dependent.

Spend cap math:

- target: USD `5-10` for the productive run
- buffer for one rerun: USD `5-10`
- total expected: USD `10-20`, well under the USD `100` cap

## What Stops The Spend Early

- USD `25` spent without a green sanity smoke: stop and fall back to retro-only closure with the existing Feb 14 numbers.
- More than one hour of single-blocker debugging: stop per RFC 0001 stop condition.
- Cap hit at USD `100`: hard stop regardless.

## What Is Not In This Estimate

- Storage (negligible at this scale)
- Egress (we only pull tens of MiB of JSON, negligible)
- Multi-node TP (not in scope; one H100 80GB is sufficient for `Qwen/Qwen3-4B`)
- Reserved or committed-use discounts (not relevant for a 2-hour ad hoc rental)
