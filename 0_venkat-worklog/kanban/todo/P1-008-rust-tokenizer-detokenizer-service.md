# P1-008: Rust Tokenizer/Detokenizer Service

Priority: P1  
Status: todo  
Depends on: P1-007

## Objective

Move tokenizer/detokenizer worker path from Python process to Rust service implementation.

## Checklist

- [ ] Define tokenizer interface compatible with current request/reply flow.
- [ ] Implement batching support (fixes current Python TODO gap).
- [ ] Implement incremental detokenization behavior parity.
- [ ] Preserve streaming chunk semantics.

## TDD Subtasks

1. Red
- [ ] Create failing golden tests for tokenize and detokenize parity vs Python implementation.
- [ ] Add failing tests for multilingual and CJK streaming behavior.

2. Green
- [ ] Implement tokenization and detokenization modules to pass parity tests.

3. Refactor
- [ ] Optimize batching heuristics and memory usage.

## Acceptance Criteria

- [ ] Output text chunks match Python behavior for parity corpus.
- [ ] Tokenizer CPU time/request improves relative to Python worker baseline.
