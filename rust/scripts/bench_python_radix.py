#!/usr/bin/env python3
"""Microbench for Python radix match_prefix performance."""

from __future__ import annotations

import time

import torch

from minisgl.kvcache.radix_manager import RadixCacheManager


def make_ids(base: int, seq_len: int) -> list[int]:
    out = [1, 2]
    for i in range(seq_len - 2):
        out.append(((base + i) % 1024) + 3)
    return out


def main() -> None:
    manager = RadixCacheManager(torch.device("cpu"))
    corpus = 8192
    seq_len = 16

    for i in range(corpus):
        input_ids = torch.tensor(make_ids(i, seq_len), dtype=torch.int32)
        indices = torch.tensor([(i * 100) + x for x in range(seq_len)], dtype=torch.int32)
        manager.insert_prefix(input_ids, indices)

    queries = []
    for i in range(corpus):
        query = make_ids(i, seq_len)
        query.append(2048 + (i % 17))
        queries.append(torch.tensor(query, dtype=torch.int32))

    warmup = 20_000
    for i in range(warmup):
        manager.match_prefix(queries[i % len(queries)])

    iters = 300_000
    total_cached = 0
    start = time.perf_counter()
    for i in range(iters):
        handle, _ = manager.match_prefix(queries[i % len(queries)])
        total_cached += handle.cached_len
    elapsed = time.perf_counter() - start

    ops_per_sec = iters / elapsed
    avg_cached_len = total_cached / iters

    print(f"python_match_prefix_ops_per_sec={ops_per_sec:.2f}")
    print(f"python_match_prefix_avg_cached_len={avg_cached_len:.2f}")


if __name__ == "__main__":
    main()
