from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import msgpack
import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchBackendMsg, DetokenizeMsg, TokenizeMsg, UserMsg


@contextmanager
def _typed_transport(enabled: bool):
    old = ENV.TYPED_TRANSPORT.value
    ENV.TYPED_TRANSPORT.value = enabled
    try:
        yield
    finally:
        ENV.TYPED_TRANSPORT.value = old


def _bench_backend(enabled: bool, iterations: int, token_len: int) -> dict[str, Any]:
    msg = BatchBackendMsg(
        [
            UserMsg(
                uid=1,
                input_ids=torch.arange(token_len, dtype=torch.int32),
                sampling_params=SamplingParams(max_tokens=64),
            ),
            UserMsg(
                uid=2,
                input_ids=torch.arange(token_len, dtype=torch.int32),
                sampling_params=SamplingParams(max_tokens=64),
            ),
        ]
    )
    with _typed_transport(enabled):
        t0 = time.perf_counter()
        for _ in range(iterations):
            payload = msg.encoder()
            packed = msgpack.packb(payload, use_bin_type=True)
            unpacked = msgpack.unpackb(packed, raw=False)
            BaseBackendMsg.decoder(unpacked)
        elapsed = time.perf_counter() - t0
    return {
        "iterations": iterations,
        "elapsed_s": elapsed,
        "ops_per_s": iterations / elapsed,
        "avg_us": elapsed / iterations * 1e6,
    }


def _bench_tokenizer(enabled: bool, iterations: int) -> dict[str, Any]:
    msgs = [
        TokenizeMsg(
            uid=1,
            text=[{"role": "user", "content": "hello world"}],
            sampling_params=SamplingParams(max_tokens=16),
        ),
        DetokenizeMsg(uid=1, next_token=123, finished=False),
    ]
    with _typed_transport(enabled):
        t0 = time.perf_counter()
        for _ in range(iterations):
            for msg in msgs:
                payload = BaseTokenizerMsg.encoder(msg)
                packed = msgpack.packb(payload, use_bin_type=True)
                unpacked = msgpack.unpackb(packed, raw=False)
                BaseTokenizerMsg.decoder(unpacked)
        elapsed = time.perf_counter() - t0
    total = iterations * len(msgs)
    return {
        "iterations": total,
        "elapsed_s": elapsed,
        "ops_per_s": total / elapsed,
        "avg_us": elapsed / total * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transport serialization overhead benchmark")
    parser.add_argument("--backend-iters", type=int, default=20000)
    parser.add_argument("--tokenizer-iters", type=int, default=30000)
    parser.add_argument("--token-len", type=int, default=1024)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("0_venkat-worklog/baselines/latest-transport-overhead.json"),
    )
    args = parser.parse_args()

    out = {
        "profile": {
            "backend_iters": args.backend_iters,
            "tokenizer_iters": args.tokenizer_iters,
            "token_len": args.token_len,
        },
        "backend_legacy": _bench_backend(False, args.backend_iters, args.token_len),
        "backend_typed": _bench_backend(True, args.backend_iters, args.token_len),
        "tokenizer_legacy": _bench_tokenizer(False, args.tokenizer_iters),
        "tokenizer_typed": _bench_tokenizer(True, args.tokenizer_iters),
    }
    out["delta_pct"] = {
        "backend_ops_per_s": (
            (out["backend_typed"]["ops_per_s"] - out["backend_legacy"]["ops_per_s"])
            / out["backend_legacy"]["ops_per_s"]
            * 100.0
        ),
        "tokenizer_ops_per_s": (
            (out["tokenizer_typed"]["ops_per_s"] - out["tokenizer_legacy"]["ops_per_s"])
            / out["tokenizer_legacy"]["ops_per_s"]
            * 100.0
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
