# Retrospective: `rust-engine-cpu-attempt-1-closure`

Head `6a943d6`, 2026-05-14, tag `v0.1-learning-complete`. Two commits on top of `rust-engine-cpu-attempt-1`
(`0bea38a` scaffold and wrapper validation, `6a943d6` the A/B and the filled retrospective).
In-branch records: `0_venkat-worklog/closure/` (CLOSURE_NOTES, H100_RUNBOOK, COST_ESTIMATE, `2026-05-14-final-retrospective.md`),
`0_venkat-worklog/baselines/closure/local_rtx3060/` (SIDE_BY_SIDE and the seven server logs), `scripts/closure/`.

## What it did

Closed the February experiment with one more measurement, on a substrate where CPU time should matter more than in February:

| Setting | Value |
| --- | --- |
| GPU, driver, torch | RTX 3060 12 GB, 570.133.07, 2.9.1+cu128 |
| Model | `Qwen/Qwen3-4B` |
| Prompts | 20 ShareGPT-style first turns, seed 42, 30 to 500 chars |
| Runs | 3 per backend, alternating Python and Rust, fresh `MASTER_PORT` each |
| Serving | streaming, concurrency 4, `max_tokens` 64, `cuda-graph-max-bs` 4, `memory-ratio` 0.85, `max-running-requests` 8 |
| Parity | one shadow run, `EVERY_N=1` |

## Results

| Metric | Python median | Rust median | Delta |
| --- | ---: | ---: | ---: |
| online tok/s | 143.08 | 143.09 | +0.01% |
| req/s | 2.27 | 2.27 | +0.01% |
| TTFT p50 ms | 78.87 | 78.73 | -0.18% |
| TTFT p99 ms | 199.78 | 203.76 | +1.99% |
| E2E p50 s | 1.74 | 1.74 | -0.10% |

Stability CV 0.00% (Python) and 0.10% (Rust); 0 shadow divergences. Outcome B of the closure plan: Rust neutral, within plus or minus 2%.

Decisions taken: the budgeted H100 weekend was skipped (USD 0 spent) because the local result was decisive; `P1-011` and `P1-012` moved to `kanban/deferred/`; Myelon for tensor-parallel transport was written off ("TP is GPU-collective territory"); GPU-side Rust IPC via `cudarc` declared out of scope.

## What I got right

1. **Closing with a number.** Three alternating runs per backend, CV reported, artifacts kept, one of three predeclared outcomes. That is how the February work should have been judged in February.
2. **Naming the seam as the cap.** The retrospective's first finding is that every scheduler step crossed Python and Rust, and that the boundary, not Rust code quality, limited the gain. That matches how SGLang's server was then built.
3. **Deferring instead of deleting.** The cutover cards were parked with their checklists intact, so the design work is still readable.
4. **The reference standard.** Measuring the project against `sglang-rs` (out-of-process command channel, real workloads on H100, lock-free structures, `cudarc` async copies, observability first) produced a concrete list of gaps rather than a mood.

## What I got wrong

1. **The prescription.** The retrospective's fix is "out of process at the right seam". SGLang's shipped answer is in process: Rust threads inside the scheduler process, a bounded MPSC channel, the scheduler draining it once per loop. `sglang-rs` and SGLang converge on coarse granularity and concurrency, not on a process boundary. The retrospective conflated the two.
2. **Still the wrong regime.** Concurrency 4 with 64 output tokens on a 4B model is GPU-bound too. The closure confirmed the February result rather than testing the hypothesis where it could fail: high concurrency where tokenization, detokenization and IPC dominate, which is the regime SGLang's design deck targets.
3. **Skipping the H100 run cut both ways.** It saved money and the architectural lesson does not depend on GPU class, but it left the fork with no data point on hardware where the step is short enough for host time to show.
4. **The `Rc<RefCell>` radix stayed.** The closure measured around it instead of fixing it, so the branch never learned what a `Send` tree core buys in a multi-threaded seam.

## What still holds

The closure scripts, the shadow parity gate and the side-by-side methodology. Anyone reopening this should reuse them and change only the seam and the workload.
