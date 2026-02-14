# Rust Minisgl CPU Roadmap

Last updated: 2026-02-14

## 1. Scope and constraints

### In scope

- Build a `rust/minisgl` implementation for CPU-side inference orchestration.
- Keep GPU model execution in Python (`minisgl.engine`, attention backends, kernels) for fast research iteration.
- Preserve current `mini-sglang` capabilities and semantics first; optimize after parity.

### Out of scope

- No GPU kernel rewrite.
- No Candle migration.
- No dependency on SGLang DP/DPA/SMG features for v1 of this work.

## 2. Baseline architecture in mini-sglang

The current CPU path is split across multiple Python processes:

- Frontend API server: `python/minisgl/server/api_server.py`
- Tokenizer/detokenizer worker(s): `python/minisgl/tokenizer/server.py`
- Scheduler per TP rank: `python/minisgl/scheduler/scheduler.py`
- Message + serialization: `python/minisgl/message/*` + `python/minisgl/utils/mp.py`
- KV-cache manager (radix/naive): `python/minisgl/kvcache/*`

GPU execution remains in:

- `python/minisgl/engine/engine.py`
- `python/minisgl/attention/*`
- `python/minisgl/kernel/*`

Important boundary today:

- `Scheduler._forward(...)` calls `Engine.forward_batch(...)` and receives `next_tokens_gpu/next_tokens_cpu`.
- This is the narrowest practical seam for a CPU/GPU split without touching kernels.

## 3. CPU bottlenecks observed from docs and code

1. Per-step scheduling and metadata prep are Python-loop heavy:
   - `_make_positions`, `_make_input_tuple`, `_make_write_tuple` in `scheduler.py`
   - Prefill/decode admission logic in `prefill.py` and `decode.py`

2. Radix cache control path is Python object heavy:
   - `kvcache/radix_manager.py` does tree walk/split/evict in Python.
   - There is already a C++ fast path for key comparison (`kernel/csrc/src/radix.cpp`), which indicates this path is recognized as CPU-sensitive.

3. Message transport/serialization adds overhead:
   - Dataclass -> dict recursion -> msgpack -> ZMQ in Python (`message/utils.py`, `utils/mp.py`)
   - Tensor serialization currently copies buffers into bytes.

4. Tokenization worker still has TODO for batch tokenization:
   - `tokenizer/tokenize.py` currently tokenizes per request in a loop.

5. Overlap scheduling exists and is useful, but still runs within Python control flow:
   - `Scheduler.overlap_loop(...)`
   - Optional safety sync exists (`ENV.OVERLAP_EXTRA_SYNC`) due issue handling in comment.

## 4. Target architecture

Use a hybrid migration path, not a big-bang rewrite.

### Design principles

- Python remains the reference behavior.
- Rust is introduced first where CPU hotspots are deterministic and testable.
- Switches are runtime-selectable; rollback is instant.

### End-state component split (target)

- `minisgl-gpu-worker` (Python):
  - Owns CUDA context, model, KV tensors, attention kernels, sampling.
  - Exposes a narrow batch-step API.
- `rust/minisgl` (Rust):
  - Request ingress, queueing, scheduling policy, cache-aware request placement, tokenizer/detokenizer, lifecycle, observability.

### Integration path

- Phase A: Rust in-process accelerators (PyO3/maturin) called from Python scheduler.
- Phase B: Rust control-plane process (HTTP + routing + tokenizer), Python GPU workers as backends.
- Phase C: Optional Rust scheduler service if additional gains are still needed.

## 5. Proposed repository layout

```text
mini-sglang/
  rust/
    minisgl-cpu-core/        # pure Rust scheduling/cache/messaging logic
    minisgl-cpu-py/          # PyO3 bindings for in-process acceleration
    minisgl-cpu-gateway/     # standalone Rust server/router binary
  python/minisgl/
    ... existing code ...
```

## 6. Phased execution plan

## Phase 0: Baseline + observability hardening (1 week)

Deliverables:

- Add step-level timers for:
  - scheduler decision time
  - metadata prep time
  - serialize/deserialize time
  - tokenizer and detokenizer latency
- Capture baseline with existing scripts:
  - `benchmark/offline/bench.py`
  - `benchmark/online/bench_simple.py`
  - `benchmark/online/bench_qwen.py`

Acceptance gate:

- Reproducible baseline report with:
  - decode token/s
  - TTFT p50/p95/p99
  - scheduler step latency p50/p95/p99
  - CPU utilization by process

## Phase 1: Rust hot-path library in Python runtime (2-3 weeks)

Move the highest-frequency deterministic CPU routines into Rust:

- Radix cache operations (walk/split/evict/lock bookkeeping)
- Prefill admission logic (`PrefillAdder` equivalent)
- Batch mapping builders (positions, input mapping, write mapping)

Keep unchanged:

- `Engine.forward_batch(...)`
- CUDA graph capture/replay
- attention backend and kernels

Acceptance gate:

- Full token-level parity on deterministic settings (`temperature=0`, fixed seeds).
- >=10% decode throughput gain in at least one representative workload.
- No regression in correctness tests.

## Phase 2: Rust frontend/tokenizer/router control plane (3-4 weeks)

Implement standalone Rust ingress:

- OpenAI-compatible `/v1/chat/completions` streaming
- Request lifecycle and backpressure
- Tokenizer/detokenizer service in Rust
- Worker health/readiness endpoints

Bridge to existing Python backend first with compatibility transport (can start with current message shape).

Acceptance gate:

- API compatibility for existing clients.
- Lower frontend CPU per token than Python FastAPI + tokenizer process baseline.
- Stable streaming behavior under concurrency.

## Phase 3: Transport modernization (2-3 weeks)

Replace Python-centric message conversion path:

- Remove recursive dataclass serialization as default fast path.
- Introduce typed schema protocol for CPU<->GPU-worker control messages.
- Add zero-copy or reduced-copy path for token/id payloads where possible.

Acceptance gate:

- Measurable drop in per-token control-plane overhead.
- No ordering regressions in streamed outputs.

## Phase 4: Optional Rust scheduler service (4-6 weeks, only if needed)

If Phases 1-3 do not hit targets, move scheduler state machine out-of-process:

- Rust scheduler manages request queues and cache metadata.
- Python GPU worker executes only batch-step commands.

Acceptance gate:

- Net improvement justifies complexity (target total +20-30% throughput vs original baseline).

## 7. Parity and fast-follow strategy

Python remains reference spec. Rust is validated continuously.

Mechanisms:

- Golden trace replay:
  - identical prompts/sampling params
  - compare emitted token ids and finish states
- Shadow mode:
  - run Rust scheduler decisions in parallel with Python decisions
  - log divergences without serving from shadow path
- Feature flag rollout:
  - `MINISGL_CPU_BACKEND=python|rust_hotpath|rust_gateway`

## 8. KPI targets

Minimum targets for initial launch:

- Throughput: +20% on at least one online benchmark profile.
- Scheduler CPU time per decode step: -30%.
- No correctness regressions in deterministic decode traces.
- Startup and shutdown reliability equal or better than current launcher behavior.

Stretch targets:

- Throughput: +30% on mixed prefill/decode online trace.
- Reduced p99 TTFT under bursty traffic.

## 9. Risks and mitigations

1. Risk: behavior drift between Python and Rust schedulers
   - Mitigation: shadow mode + golden replay before cutover.

2. Risk: operational complexity from mixed runtimes
   - Mitigation: phase-in with strict feature flags and reversible deployment.

3. Risk: transport rewrite introduces streaming bugs
   - Mitigation: keep compatibility mode until typed path is proven under load.

4. Risk: unclear gain from frontend rewrite alone
   - Mitigation: prioritize Phase 1 hot-path wins before larger process-level rewrite.

## 10. First 30-day implementation sequence

Week 1:

- Add instrumentation and baseline report.
- Define parity corpus and deterministic test harness.

Week 2:

- Build `minisgl-cpu-core` crate with radix + prefill admission primitives.
- Add PyO3 bindings and optional runtime flag.

Week 3:

- Integrate Rust hot-path into scheduler loop.
- Validate parity and benchmark gains.

Week 4:

- Start Rust gateway skeleton with health endpoints and basic request proxying.
- Prepare transport schema draft and migration plan.

## 11. References used for this plan

- Mini-SGLang docs and codebase:
  - `README.md`
  - `docs/features.md`
  - `docs/structures.md`
  - `python/minisgl/scheduler/scheduler.py`
  - `python/minisgl/tokenizer/server.py`
  - `python/minisgl/kvcache/radix_manager.py`
  - `python/minisgl/message/utils.py`
  - `python/minisgl/utils/mp.py`
- SGLang model gateway reference (Rust + Python bindings pattern):
  - `sglang/sgl-model-gateway/README.md`
  - `sglang/sgl-model-gateway/bindings/python/README.md`
- Linked external references from Mini-SGLang docs:
  - Radix attention background: https://lmsys.org/blog/2024-01-17-sglang/
  - Overlap scheduling background: https://lmsys.org/blog/2024-12-04-sglang-v0-4/
  - Chunked prefill paper: https://arxiv.org/abs/2403.02310
  - NanoFlow paper: https://arxiv.org/abs/2408.12757
