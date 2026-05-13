# mini-sglang Rust CPU Engine - Final Retrospective

Date: 2026-05-14
Status: Final
Closure substrate: local RTX 3060 (12 GB VRAM); H100 weekend run not required (see "Why The H100 Spend Was Skipped" below).

## TL;DR

- workload: `Qwen/Qwen3-4B`, ShareGPT-style prompts (`20` per run), single RTX 3060 12 GB, concurrency `4`, streaming, `cuda-graph-max-bs=4`, `max_tokens=64`
- python backend median online tok/s: `143.08`
- rust hotpath median online tok/s: `143.09`
- delta: `+0.01%`
- outcome per RFC 0001 Decision Tree: **Outcome B (Rust neutral, within +/- 2%)**
- parity (shadow + stability): **pass** (`0` shadow divergences across 20 prompts, Python CV `0.00%`, Rust CV `0.10%`)

## Why This Project Existed

The Rust CPU engine work on `rust-engine-cpu-attempt-1` started 2026-02-14 after a Slack thread with the SGLang main contributor about CPU-side performance gains. The hypothesis was that the Python control plane in `mini-sglang` could give up 20-30% by moving to Rust without touching the GPU stack.

The hypothesis was reasonable. The execution made four mistakes against the standard later set by `sglang-rs`:

1. wrong seam (in-process PyO3, not coarse-grained command channel)
2. wrong workload (Qwen 0.5B on RTX 3060, GPU-bound the whole time)
3. wrong data structures in places (`Rc<RefCell<>>` radix, single-threaded)
4. wrong benchmark substrate (synthetic random tokens, not real prompts)

This retrospective measures the in-process FFI seam one last time on a substrate where CPU side matters more (a 4B model with realistic prompts on the same RTX 3060), then closes the project.

## What We Already Had Going In

Inherited from Feb 14 work, currently shipped on `rust-engine-cpu-attempt-1`:

- 10 cards landed: `P0-001` through `P0-006`, `P1-007` through `P1-010`
- `MINISGL_CPU_BACKEND=python|rust_hotpath` runtime switch with fail-closed fallback
- typed transport schema v1 default-on (`MINISGL_TYPED_TRANSPORT=1`)
- shadow-mode parity comparator (`ShadowCpuBackend` + `MINISGL_CPU_BACKEND_SHADOW_EVERY_N` sampling)
- subprocess-isolated deterministic token parity tool
- composite release gate (perf + parity + shadow + stability)
- queue-level transport latency counters
- runtime metrics snapshot API

Last recorded local A/B (RTX 3060, Qwen2.5-0.5B, Feb 14):

- offline delta: `+0.07%` (parity)
- online delta: `+0.69%`
- CPU metadata microbench: `+63.5%`
- radix `match_prefix` microbench: Rust about 138x faster
- backend transport encode/decode: typed schema v1 `+74%` ops/s

These say the Rust hot path is correct and the CPU layer is internally faster, but the in-process FFI seam compresses gains end-to-end on a workload that is overwhelmingly GPU-bound.

## What This Retrospective Resolves

Does the same in-process FFI seam hold up, regress, or win on:

- a real prompt distribution (ShareGPT-style, 20 prompts)
- a model size where CPU is a non-trivial fraction of the step (`Qwen3-4B`)
- the same RTX 3060 that produced the Feb 14 numbers

Three valid outcomes per RFC 0001:

- Outcome A: Rust positive (`> +2%` online tok/s)
- Outcome B: Rust neutral (within `+/- 2%`)
- Outcome C: Rust regression (`< -2%`)

All three close the project. The number is the evidence.

## Why The H100 Spend Was Skipped

The closure plan originally budgeted a USD `10-100` H100 weekend on `Qwen3-4B`. The local RTX 3060 run on the same model produced a closure-quality result without that spend:

- the closure scripts work end-to-end (`sharegpt_prep.py`, `run_closure_benchmark.py`, `compare_runs.py`)
- the chosen model fits and runs in `12 GB` VRAM with `cuda-graph-max-bs=4` and `memory-ratio=0.85`
- the result is decisively Outcome B with very low coefficient of variation across three sequential runs per backend
- the shadow parity gate passes (`0` divergences)

The H100 run would have produced a number on a different substrate at a different decode budget, but the architectural lesson is independent of GPU class: in-process FFI is parity-correct, not perf-positive, on a workload where the GPU step is non-trivial. The H100 plan and runbook remain on this branch (`0_venkat-worklog/closure/H100_RUNBOOK.md`) for any future iteration that wants to confirm at a different scale.

## Reference Standard: What sglang-rs Got Right

Used here as the bar for what a serious Rust CPU control plane looks like. Mini-sglang did not match these on this project. They are the things any successor effort should adopt on day one.

1. **Out of process at the right seam.** PyO3 used as a coarse command channel for `ForwardAndSample` / `AllocTokens` / `FreeTokens`. Hot path never crosses GIL per step.
2. **Real workload from day one.** Llama 3.1 8B FP8 and Llama 3 70B TP8 on H100.
3. **Lock-free throughout.** `rtrb` wait-free SPSC, `crossbeam-channel`, `crossbeam-epoch` for radix RCU, `slab` for request storage, `dashmap` for the tracker.
4. **cudarc async memcpy.** Pinned host buffers via `cuMemAllocHost`, fallback to overlap on cuda-copy init failure.
5. **Type alignment across crates.** `i64` tokens / KV indices / positions, `u32` output tokens, `i32` seq lens. Scheduler is an orchestrator, not an allocator.
6. **Comprehensive scope.** TP, DP, EP, chunked prefill, OOM retraction policies, decode fairness, mixed batching.
7. **Observability early.** Prometheus, Grafana, Loki, Alloy with per-rank labels, recording rules, alerts.
8. **Honest sequential A/B methodology.** Explicit `MASTER_PORT`, multi-run gates, machine-readable artifacts.

## Mirror: What We Did Wrong On This Project

1. **In-process PyO3 as the default hot path.** Every scheduler step crossed Python and Rust. The seam was the cap on gains, not the Rust code quality.
2. **Wrong substrate for the question.** Qwen 0.5B on RTX 3060 in Feb made the GPU step dominate so completely that the CPU-side gain was invisible at the system level, even when it was real at the microbench level. This closure used Qwen3-4B specifically to push past that regime.
3. **Single-threaded radix.** `Rc<RefCell<RadixNode>>` is not `Send`. Cannot be used from a worker pool. Toy compared to `sglang-rs`'s `crossbeam-epoch` tree.
4. **No async pipeline overlap in Rust.** Ran inside Python's overlap, not in parallel with it. No `cudarc` async memcpy.
5. **Synthetic benchmarks until this closure.** Random token IDs, no real-world prompt shape, until BK-002 was elevated for this closure.
6. **Built a Rust gateway and walked it back.** P1-007 effort that became reference-only after the scope clarification. Net wasted cycle.
7. **Defaulted to "ship a card" over "measure the right thing."** The Feb 14 cards advanced the seam without first proving the seam was correct on a workload where it mattered.

## What Sat Next Door And Was Not Used

`myelon-playground` was active in parallel during this same period. It ships a same-node shared-memory disruptor transport, exactly the substrate `BK-001` parked as "zero-copy transport path (shared memory / pinned memory aware flow)." It was never wired into mini-sglang.

Per the post-mortem decision summary, that was an oversight at the meta-workflow level: two projects with naturally complementary substrates ran in separate workspaces with separate kanbans and never met. The myelon-playground `myelon_mq` RFC dated 2026-04-08 names SGLang as the first target. mini-sglang is a SGLang fork. They are even closer than the RFC frames it.

This closure does not attempt to wire them. That work is deferred to a follow-up scoped under `9_sglang_rs_myelon_evaluation/` or similar.

## Methodology For This Retrospective Run

- substrate: local RTX 3060 12 GB, driver `570.133.07`, CUDA wheels from `torch 2.9.1+cu128`
- approximate spend: `USD 0` (no rental; H100 plan retained on branch for future use)
- model: `Qwen/Qwen3-4B` (cached locally; matches RFC 0001 target model)
- prompts: 20 ShareGPT-style first-user-turns from `scripts/closure/smoke_fixture.json`, prepared via `scripts/closure/sharegpt_prep.py` with seed `42`, length filter `[30, 500]` characters
- runs per backend: `3` sequential, alternating Python and Rust hotpath order with fresh `MASTER_PORT` each run
- streaming: `True` (chat-completions SSE)
- concurrency: `4`
- max tokens per request: `64`
- `cuda-graph-max-bs`: `4`
- `memory-ratio`: `0.85`
- `max-running-requests`: `8`
- typed transport: `MINISGL_TYPED_TRANSPORT=1` (default since Feb 14)
- shadow parity: one dedicated run with `MINISGL_CPU_BACKEND_SHADOW=1`, `EVERY_N=1`, JSONL report at `0_venkat-worklog/baselines/closure/local_rtx3060/shadow-divergence.jsonl`

## Results

Retained side-by-side: `0_venkat-worklog/baselines/closure/local_rtx3060/SIDE_BY_SIDE.md`.

### Per-Backend Stability (3 runs each)

| Metric | Python mean | Python CV | Rust mean | Rust CV |
|---|---:|---:|---:|---:|
| online tok/s | `143.08` | `0.00%` | `143.02` | `0.10%` |
| online req/s | `2.27` | `0.00%` | `2.27` | `0.10%` |
| TTFT avg ms | `86.15` | `0.24%` | `86.53` | `0.41%` |
| TTFT p50 ms | `78.99` | `0.27%` | `78.83` | `0.41%` |
| TTFT p99 ms | `200.50` | `1.33%` | `202.93` | `0.70%` |
| E2E avg s | `1.75` | `0.02%` | `1.75` | `0.10%` |
| E2E p50 s | `1.74` | `0.04%` | `1.74` | `0.11%` |
| E2E p99 s | `1.86` | `0.15%` | `1.86` | `0.11%` |

Both backends pass the `<= 10%` throughput CV stability gate trivially.

### Rust vs Python (Medians)

| Metric | Python median | Rust median | Delta % | Verdict |
|---|---:|---:|---:|:---|
| online tok/s | `143.08` | `143.09` | `+0.01%` | BETTER |
| online req/s | `2.27` | `2.27` | `+0.01%` | BETTER |
| TTFT avg ms | `86.19` | `86.53` | `+0.40%` | WORSE |
| TTFT p50 ms | `78.87` | `78.73` | `-0.18%` | BETTER |
| TTFT p99 ms | `199.78` | `203.76` | `+1.99%` | WORSE |
| E2E avg s | `1.75` | `1.75` | `-0.02%` | BETTER |
| E2E p50 s | `1.74` | `1.74` | `-0.10%` | BETTER |
| E2E p99 s | `1.86` | `1.86` | `+0.21%` | WORSE |

All deltas inside the noise floor for this hardware. Median throughput is indistinguishable.

### Parity Spot-Check (Shadow Run)

- shadow divergences logged across 20 prompts: `0`
- expected: `0`

## Verdict

**Outcome B: Rust neutral on real workload.**

In plain terms: on the same RTX 3060 used in February, with a real 4B model and real prompts, the in-process Rust hotpath produces statistically the same end-to-end throughput, TTFT, and E2E latency as the Python backend. The microbench-level CPU wins from Feb 14 (`+63.5%` on metadata builders, `~138x` on radix `match_prefix`, `+74-107%` on typed transport encode/decode) are absorbed by the PyO3 boundary and the dominant GPU step time. Shadow parity is bit-perfect across the full prompt set.

This confirms the architectural lesson that `sglang-rs` shipped on first: in-process FFI is parity-correct, not perf-positive on a workload where the GPU is the bottleneck. The next architectural step (out-of-process Rust scheduler service with coarse PyO3 command channel) is exactly what `RFC 0001` lists under "Deferred Tracks" and what would re-enable measurable CPU-side gains.

This outcome closes the mini-sglang Rust CPU engine project as a successful learning artifact. The branch ships with parity tooling that is upstream-contribution ready. The next time `mini-sglang` or another SGLang fork wants to push the seam further, the work starts at out-of-process service mode, not in-process FFI optimization.

## What Carries Forward

Reusable artifacts that survive the closure of the in-process Rust hot path. Each is a candidate upstream contribution to `sgl-project/sglang` per migration RFC `#23206` and PR `#23360`:

- `ShadowCpuBackend` metadata-diffing parity comparator with JSONL divergence log
- `python -m minisgl.benchmark.token_parity` subprocess-isolated deterministic-token A/B harness
- `MINISGL_CPU_BACKEND_SHADOW_EVERY_N` low-overhead sampling pattern
- typed transport schema v1 with `__schema__` discriminator and dual-stack decode
- composite release gate (perf + parity + shadow + stability)
- queue-level transport latency counters with phase timing
- runtime metrics snapshot API
- subprocess-isolated multi-LLM benchmark methodology (the `TP info has been set` workaround)
- `scripts/closure/` wrapper pattern (server lifecycle via subprocess + process group, streaming TTFT capture, side-by-side CV report with RFC Decision Tree classifier)

## What Is Deferred Out Of This Closure

These are not failures. They are intentionally not absorbed:

- **Out-of-process Rust CPU scheduler service** (`P1-011` moved to `kanban/deferred/`). Architecturally correct per sglang-rs evidence. Roughly three to four weeks of focused work. Not in this closure.
- **Runtime PyO3 path removal** (`P1-012` moved to `kanban/deferred/`). Depends on the above.
- **Myelon transport integration into mini-sglang.** Deferred. Requires the service first.
- **cudarc and cuda-oxide for GPU-side Rust IPC.** Separate substrate, separate repo, separate kanban.
- **sglang-rs plus Myelon evaluation.** Follow-up workspace.
- **Upstream PR of the carried-forward tooling.** Follow-up workspace.
- **H100 confirmation run.** The plan and runbook (`H100_RUNBOOK.md`, `COST_ESTIMATE.md`) are retained on this branch for any future iteration that wants the alternate-hardware data point.

## Honest Closing Assessment

The seam is parity-correct but not perf-positive on real workload at this hardware class. This confirms `sglang-rs`'s decision to skip in-process FFI and go directly to coarse PyO3 commands plus an out-of-process scheduler. The existing in-process implementation on this branch should be treated as a learning artifact, not as the base for further optimization.

The carry-forward tools remain useful regardless of whether the hot-path code is. Shadow mode, token parity, typed transport, the release gate, runtime metrics, and the closure benchmark scripts are exactly the migration-safety tooling Rain Jiang's upstream Rust migration (PR `#23360`) and Alex Nails' gRPC RFC (`#22558`) will need when their work crosses the pickle-to-msgpack boundary and starts validating Rust replacements for individual Python components.

The right next move for any continuation of this thread is in a separate workspace: either upstream-contribution evaluation of these tools, or the `sglang-rs` plus Myelon evaluation that the prior conversation deferred.

## Tag

Closure tagged at the final commit on `rust-engine-cpu-attempt-1-closure`: `v0.1-learning-complete`.
