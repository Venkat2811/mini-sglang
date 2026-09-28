# Branch retrospectives

Written 2026-09-28, on one branch so the experimental branches stay as they were.
Each file covers one branch of this fork: what was built, what was measured, what
turned out right, what turned out wrong, judged against what SGLang main later shipped
in Rust (`sgl-project/sglang` at `2e7e0802f4`, and the Rust server RFC
https://github.com/sgl-project/sglang/issues/23206 with its linked design decks).

| Branch | Head | Date | One-line outcome | Retrospective |
| --- | --- | --- | --- | --- |
| `rust-engine-cpu-attempt-1` | `48bb1b8` | 2026-02-14 | Rust CPU hot path behind PyO3, parity-correct, +0.07% offline / +0.69% online on RTX 3060 | [rust-engine-cpu-attempt-1.md](rust-engine-cpu-attempt-1.md) |
| `rust-engine-cpu-attempt-1-closure` | `6a943d6` | 2026-05-14 | Closed as Outcome B (neutral): 143.08 vs 143.09 tok/s on Qwen3-4B, tag `v0.1-learning-complete` | [rust-engine-cpu-attempt-1-closure.md](rust-engine-cpu-attempt-1-closure.md) |
| `myelon-gloo-synthetic-bench` | `18b4120` | 2026-04-22 | Faster CPU collectives (1.6x to 2.8x synthetic), no serving win at TP=2 | [myelon-gloo-synthetic-bench.md](myelon-gloo-synthetic-bench.md) |

Upstream `sgl-project/mini-sglang` never adopted Rust (last upstream commit 2026-05-17).
SGLang itself did, at two seams: a Rust frontend embedded as threads in the scheduler
process (PR #29799, merged 2026-07-31) and a Rust radix tree core (PR #32710, merged
2026-08-31). Those two facts are the yardstick used below.
