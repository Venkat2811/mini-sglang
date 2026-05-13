# mini-sglang Rust CPU Engine - Final Retrospective

Date: 2026-05-14
Status: Draft (placeholders pending H100 numbers from PR 1/PR 2)

## TL;DR

(filled in after the side-by-side comparison lands at PR 2)

- workload: `Qwen/Qwen3-4B`, ShareGPT-derived prompts, single H100 80GB, concurrency `8`
- python backend median online tok/s: `__TOFILL__`
- rust hotpath median online tok/s: `__TOFILL__`
- delta: `__TOFILL__%`
- outcome per RFC 0001 Decision Tree: __Outcome A | B | C__
- parity (shadow + token equality): __pass | fail__

## Why This Project Existed

The Rust CPU engine work on `rust-engine-cpu-attempt-1` started 2026-02-14 after a Slack thread with the SGLang main contributor about CPU-side performance gains. The hypothesis was that the Python control plane in `mini-sglang` could give up 20-30% by moving to Rust without touching the GPU stack.

The hypothesis was reasonable. The execution made four mistakes against the standard later set by `sglang-rs`:

1. wrong seam (in-process PyO3, not coarse-grained command channel)
2. wrong workload (Qwen 0.5B on RTX 3060, GPU-bound the whole time)
3. wrong data structures in places (`Rc<RefCell<>>` radix, single-threaded)
4. wrong benchmark substrate (synthetic random tokens, not real prompts)

This retrospective measures the in-process FFI seam one last time on a substrate where CPU side matters, then closes the project.

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

Last recorded local A/B (RTX 3060, Qwen2.5-0.5B):

- offline delta: `+0.07%` (parity)
- online delta: `+0.69%`
- CPU metadata microbench: `+63.5%`
- radix `match_prefix` microbench: Rust about 138x faster
- backend transport encode/decode: typed schema v1 `+74%` ops/s

These say the Rust hot path is correct and the CPU layer is internally faster, but the in-process FFI seam compresses gains end-to-end on a workload that is overwhelmingly GPU-bound.

## What This Retrospective Resolves

Does the same in-process FFI seam hold up, regress, or win on:

- a real prompt distribution (ShareGPT)
- a model size where CPU is a non-trivial fraction of the step (`Qwen3-4B`)
- a real datacenter GPU (H100 80GB)

Three valid outcomes per RFC 0001:

- Outcome A: Rust positive (`> +2%` online tok/s)
- Outcome B: Rust neutral (within `+/- 2%`)
- Outcome C: Rust regression (`< -2%`)

All three close the project. The number is the evidence.

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
2. **Wrong substrate for the question.** Qwen 0.5B on RTX 3060 made the GPU step dominate so completely that the CPU-side gain was invisible at the system level, even when it was real at the microbench level.
3. **Single-threaded radix.** `Rc<RefCell<RadixNode>>` is not `Send`. Cannot be used from a worker pool. Toy compared to `sglang-rs`'s `crossbeam-epoch` tree.
4. **No async pipeline overlap in Rust.** Ran inside Python's overlap, not in parallel with it. No `cudarc` async memcpy.
5. **Synthetic benchmarks.** Random token IDs, no real-world prompt shape, until BK-002 was elevated for this closure.
6. **Built a Rust gateway and walked it back.** P1-007 effort that became reference-only after the user clarified scope. Net wasted cycle.
7. **Defaulted to "ship a card" over "measure the right thing."** The Feb 14 cards advanced the seam without first proving the seam was correct on a workload where it mattered.

## What Sat Next Door And Was Not Used

`myelon-playground` was active in parallel during this same period. It ships a same-node shared-memory disruptor transport, exactly the substrate `BK-001` parked as "zero-copy transport path (shared memory / pinned memory aware flow)." It was never wired into mini-sglang.

Per the post-mortem decision summary, that was an oversight at the meta-workflow level: two projects with naturally complementary substrates ran in separate workspaces with separate kanbans and never met. The myelon-playground `myelon_mq` RFC dated 2026-04-08 names SGLang as the first target. mini-sglang is a SGLang fork. They are even closer than the RFC frames it.

This closure does not attempt to wire them. That work is deferred to a follow-up scoped under `8_minisgl/1_*` or `9_sglang_rs_myelon_evaluation/`.

## Methodology For This Retrospective Run

(filled in after PR 1 lands)

- provider used: `__TOFILL__`
- approximate spend: USD `__TOFILL__`
- ShareGPT subset: `__TOFILL__` prompts after length filter `[__TOFILL__, __TOFILL__]` characters
- runs per backend: `3` (alternating order)
- streaming: `__TOFILL__`
- concurrency: `__TOFILL__`
- max tokens per request: `__TOFILL__`
- typed transport: `MINISGL_TYPED_TRANSPORT=1`
- shadow parity: one dedicated run with `MINISGL_CPU_BACKEND_SHADOW=1`, sampling `EVERY_N=1`

## Results

(filled in by `scripts/closure/compare_runs.py` output landing at `0_venkat-worklog/baselines/closure/SIDE_BY_SIDE.md`)

### Per-Backend Stability (CV)

| Metric | Python mean | Python CV | Rust mean | Rust CV |
|---|---:|---:|---:|---:|
| online tok/s | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |
| online req/s | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |
| TTFT avg ms | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |
| TTFT p99 ms | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |

### Rust vs Python (Medians)

| Metric | Python median | Rust median | Delta % | Verdict |
|---|---:|---:|---:|:---|
| online tok/s | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |
| online req/s | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |
| TTFT avg ms | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |
| TTFT p99 ms | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` | `__TOFILL__` |

### Parity Spot-Check (Shadow Run)

- shadow divergences logged: `__TOFILL__`
- expected: `0`

## Verdict

(filled in after results land)

Outcome __A | B | C__: `__TOFILL__`

What this means in plain terms: `__TOFILL__`.

## What Carries Forward

Reusable artifacts that survive even if the in-process Rust hot path itself does not. Each is a candidate upstream contribution to `sgl-project/sglang` per migration RFC `#23206` and PR `#23360`:

- `ShadowCpuBackend` metadata-diffing parity comparator with JSONL divergence log
- `python -m minisgl.benchmark.token_parity` subprocess-isolated deterministic-token A/B harness
- `MINISGL_CPU_BACKEND_SHADOW_EVERY_N` low-overhead sampling pattern
- typed transport schema v1 with `__schema__` discriminator and dual-stack decode
- composite release gate (perf + parity + shadow + stability)
- queue-level transport latency counters with phase timing
- runtime metrics snapshot API
- subprocess-isolated multi-LLM benchmark methodology (the `TP info has been set` workaround)

## What Is Deferred Out Of This Closure

These are not failures. They are intentionally not absorbed:

- **Out-of-process Rust CPU scheduler service** (`P1-011` moved to `kanban/deferred/`). Architecturally correct per sglang-rs evidence. Roughly three to four weeks of focused work. Not in this closure.
- **Runtime PyO3 path removal** (`P1-012` moved to `kanban/deferred/`). Depends on the above.
- **Myelon transport integration into mini-sglang.** Deferred. Requires the service first.
- **cudarc and cuda-oxide for GPU-side Rust IPC.** Separate substrate, separate repo, separate kanban.
- **sglang-rs plus Myelon evaluation.** Follow-up workspace.
- **Upstream PR of the carried-forward tooling.** Follow-up workspace.

## Honest Closing Assessment

(filled in after Outcome lands)

If Outcome A (Rust positive):
- the seam works on a workload that exercises CPU; this is the case for proceeding with the out-of-process Rust service track later; the in-process FFI version was not the wrong direction, it was the wrong starting workload.

If Outcome B (Rust neutral):
- the seam is parity-correct but not perf-positive on real workload; this confirms sglang-rs's choice to skip in-process FFI and go directly to coarse PyO3 commands plus out-of-process scheduler; the existing in-process implementation should be treated as a learning artifact, not as the base for further optimization.

If Outcome C (Rust regression):
- the seam introduces measurable overhead under load on real serving workloads; the implementation is honest evidence that in-process FFI is the wrong seam for hot paths in inference engine control planes; the carry-forward tools remain useful even though the hot-path code does not.

## Tag

Closure tagged at the final commit: `v0.1-learning-complete`.

Branch: `rust-engine-cpu-attempt-1-closure`.
