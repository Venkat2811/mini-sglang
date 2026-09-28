# Retrospective: `myelon-gloo-synthetic-bench`

Head `18b4120`, 2026-04-22. Five commits (2026-04-21 to 04-22): synthetic Gloo probes, an out-of-tree c10d
backend (`benchmark/synthetic/c10d_glooext/`), the retained results, and a TP=2 serving matrix
(`benchmark/online/run_tp2_backend_matrix.py`, `bench_tp2_serving.py`).
In-branch records: `benchmark/synthetic/README.md`, `2026-04-21_results.md`, `2026-04-21_c10d_results.md`.

## The question

mini-sglang's TP ranks coordinate over CPU Gloo: a rank-wide broadcast of the pending request count,
a NCCL unique-id `broadcast_object_list`, a barrier, and a tiny `ReduceOp.MIN` allreduce for free memory.
Would a shared-memory transport (Myelon) under Gloo speed those up, and would that show in serving?

## What was measured

Layer 1, transport proxy (macOS, external Gloo MPI benchmarks, Myelon vs `uv`):

| Proxy | uv p50 us | Myelon p50 us | Speedup |
| --- | ---: | ---: | ---: |
| scheduler broadcast 4 B / 16 B | 91.33 / 89.42 | 37.75 / 38.96 | 2.42x / 2.30x |
| NCCL UID bcast 256 B / 1 KiB | 81.08 / 95.08 | 46.21 / 33.79 | 1.75x / 2.81x |
| free-mem allreduce 16 B | 413.17 | 248.54 | 1.66x |
| small allreduce 1 KiB | 640.96 | 341.04 | 1.88x |

Layer 2, real `torch.distributed` through the custom c10d backend (Linux, world size 4, Myelon vs `glooext` on `uv`):
barrier 1.60 to 1.78x, scheduler count broadcast 1.88 to 2.06x, NCCL UID broadcast 1.92 to 1.98x,
free-memory MIN allreduce 1.70 to 1.83x; `init_process_group` too noisy to use.

Layer 3, serving (TP=2 on a two-GPU Blackwell host, 2026-04-22, `gloo` vs `glooext+uv` vs `glooext+myelon`, artifacts kept outside this repo):

| Model | Myelon vs builtin gloo, req/s | Myelon vs glooext+uv, req/s |
| --- | ---: | ---: |
| Qwen3-0.6B | +0.23% | -0.89% |
| Qwen3-4B | -0.66% | -1.32% |

`glooext+uv` was the best variant on Qwen3-4B (+0.67% req/s over builtin gloo). Myelon was never the serving winner.

## What I got right

1. **Three layers, in order.** Transport proxy, then the real c10d API, then serving. Each layer was allowed to kill the idea, and the last one did.
2. **A real out-of-tree c10d backend** instead of another proxy, so the numbers exercise the API mini-sglang actually calls.
3. **Finding and fixing the measurement bug** (a fresh Gloo tag per operation, which the transport partitions channels by) before publishing numbers.
4. **Honest caveats in the results files**: "still synthetic", "not a claim about end-to-end serving throughput". They turned out to be the headline.

## What I got wrong

1. **The transport was not on the critical path.** The collectives it accelerates are 100 to 300 us calls on gloo. Halving them is a small fraction of a multi-millisecond step, and at TP=2 there are few of them per step. The serving matrix showed exactly that.
2. **Synthetic wins were read as a forecast.** The 04-21 notes call the deltas "meaningful rather than marginal". They were, at the layer they were measured. Serving did not inherit them.
3. **Wrong lane for shared memory.** SGLang's Rust server design keeps "broadcast across other TP ranks" in Python and spends its transport effort on the frontend-to-scheduler queue ("thread-level message queue, implemented in Rust, faster than zmq"). The queue that mattered was that one, not the TP control plane. The closure branch later wrote Myelon-for-TP off as a lost cost; this branch is the evidence for that write-off.

## What still holds

The c10d backend and the matrix scripts are a reusable harness for any future CPU-transport question in mini-sglang, and the results files are a clean example of how to publish a negative result.
