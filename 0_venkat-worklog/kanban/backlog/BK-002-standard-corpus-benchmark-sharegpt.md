# BK-002: Standard Corpus Benchmark (ShareGPT-Style)

Priority: Backlog  
Status: backlog  
Depends on: baseline harness stable + data sanitization guardrails

## Objective

Add a standard, reproducible real-prompt benchmark track using ShareGPT-style conversations (plus length buckets) so performance claims are not based only on synthetic/random prompts.

## Scope

- Keep existing synthetic harness for fast regression checks.
- Add real-corpus benchmark profile as primary reporting artifact.
- Keep dataset artifacts sanitized and redistributable (no private/user data).

## Checklist

- [ ] Define corpus source and licensing/compliance note.
- [ ] Build prompt pack with deterministic seed and fixed split:
  - [ ] small set (`~500` prompts)
  - [ ] medium set (`~2k` prompts)
  - [ ] long-context set (`~100-300` prompts, high input length)
- [ ] Add corpus preprocessing script:
  - [ ] normalize to OpenAI chat message format
  - [ ] trim/clip rules documented
  - [ ] token-length metadata per sample
- [ ] Add harness mode for corpus-driven online benchmark.
- [ ] Add baseline report template for corpus runs.
- [ ] Add parity/perf gates for corpus profile.

## Acceptance Criteria

- [ ] Reproducible corpus benchmark run from one command.
- [ ] Results include both synthetic and corpus profiles.
- [ ] Rust-vs-Python decisions use corpus results as final signal.

## TDD Expectations

- [ ] Failing tests for corpus loader determinism and split consistency.
- [ ] Failing tests for malformed sample handling.
- [ ] Green tests + benchmark artifact generation.
