#!/usr/bin/env python3
"""Export Python scheduler prefill/mapping golden traces for Rust parity tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder
from minisgl.scheduler.scheduler import _make_input_tuple, _make_positions, _make_write_tuple
from minisgl.scheduler.utils import PendingReq


@dataclass
class FakeHandle:
    cached_len: int
    handle_id: int


class FakeCacheManager:
    def __init__(
        self,
        available_size: int,
        cached_len: int,
        match_indices: list[int],
        lock_impact: int = 0,
    ) -> None:
        self._available_size = available_size
        self.cached_len = cached_len
        self.match_indices = torch.tensor(match_indices, dtype=torch.int32)
        self.lock_impact = lock_impact

    def match_req(self, req: PendingReq):
        return FakeHandle(cached_len=self.cached_len, handle_id=1), self.match_indices

    @property
    def available_size(self) -> int:
        return self._available_size

    def lock(self, handle: FakeHandle) -> None:
        _ = handle
        self._available_size -= self.lock_impact

    def unlock(self, handle: FakeHandle) -> None:
        _ = handle
        self._available_size += self.lock_impact


class FakeTableManager:
    def __init__(self, slots: list[int], page_width: int = 32) -> None:
        self._slots = slots[:]
        max_slot = max(slots) if slots else 0
        shape = (max_slot + 1, page_width)
        self.token_pool = torch.zeros(shape, dtype=torch.int32)
        self.page_table = torch.zeros(shape, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._slots)

    def allocate(self) -> int:
        return self._slots.pop()


def make_pending(uid: int, ids: list[int], max_tokens: int) -> PendingReq:
    return PendingReq(
        uid=uid,
        input_ids=torch.tensor(ids, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def run_adder_case(
    *,
    name: str,
    token_budget: int,
    reserved_size: int,
    cache_available_size: int,
    table_slots: list[int],
    cached_len: int,
    match_indices: list[int],
    input_ids: list[int],
    output_len: int,
    lock_impact: int = 0,
) -> dict[str, Any]:
    cache = FakeCacheManager(
        available_size=cache_available_size,
        cached_len=cached_len,
        match_indices=match_indices,
        lock_impact=lock_impact,
    )
    table = FakeTableManager(table_slots)
    adder = PrefillAdder(
        token_budget=token_budget,
        reserved_size=reserved_size,
        cache_manager=cache,
        table_manager=table,
    )
    pending = make_pending(uid=1000, ids=input_ids, max_tokens=output_len)
    req = adder.try_add_one(pending)

    expected: dict[str, Any] = {
        "admitted": req is not None,
        "token_budget_after": int(adder.token_budget),
        "reserved_size_after": int(adder.reserved_size),
    }
    if req is not None:
        expected.update(
            {
                "is_chunked": bool(isinstance(req, ChunkedReq)),
                "cached_len": int(req.cached_len),
                "device_len": int(req.device_len),
                "table_idx": int(req.table_idx),
                "extend_len": int(req.extend_len),
                "remain_len": int(req.remain_len),
                "can_decode": bool(req.can_decode),
            }
        )

    return {
        "name": name,
        "token_budget": token_budget,
        "reserved_size": reserved_size,
        "cache_available_size": cache_available_size,
        "table_slots": table_slots,
        "cached_len": cached_len,
        "match_indices": match_indices,
        "lock_impact": lock_impact,
        "input_ids": input_ids,
        "output_len": output_len,
        "expected": expected,
    }


def run_mapping_case() -> dict[str, Any]:
    req_a = Req(
        input_ids=torch.tensor([11, 12, 13, 14, 15], dtype=torch.int32),
        table_idx=7,
        cached_len=2,
        output_len=4,
        uid=1,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=FakeHandle(cached_len=2, handle_id=11),
    )
    req_b = ChunkedReq(
        input_ids=torch.tensor([21, 22, 23], dtype=torch.int32),
        table_idx=9,
        cached_len=1,
        output_len=7,
        uid=2,
        sampling_params=SamplingParams(max_tokens=7),
        cache_handle=FakeHandle(cached_len=1, handle_id=22),
    )
    req_c = Req(
        input_ids=torch.tensor([31, 32, 33, 34, 35], dtype=torch.int32),
        table_idx=11,
        cached_len=4,
        output_len=1,
        uid=3,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=FakeHandle(cached_len=4, handle_id=33),
    )

    batch = Batch(reqs=[req_a, req_b, req_c], phase="prefill")
    batch.padded_reqs = [req_a, req_b, req_c]
    batch.out_loc = torch.arange(sum(r.extend_len for r in batch.padded_reqs), dtype=torch.int32)
    positions = _make_positions(batch, torch.device("cpu"))
    batch.positions = positions
    input_tuple = _make_input_tuple(batch, torch.device("cpu"))
    write_tuple = _make_write_tuple(batch, torch.device("cpu"))

    return {
        "name": "mixed_prefill_decode_mappings",
        "positions": [int(x) for x in positions.tolist()],
        "input_mapping": [int(x) for x in input_tuple[0].tolist()],
        "input_positions": [int(x) for x in input_tuple[1].tolist()],
        "write_req_mapping": [int(x) for x in write_tuple[0].tolist()],
        "write_pos": [int(x) for x in write_tuple[1].tolist()],
    }


def run_decode_only_write_case() -> dict[str, Any]:
    req_a = Req(
        input_ids=torch.tensor([101, 102, 103, 104, 105, 106], dtype=torch.int32),
        table_idx=13,
        cached_len=5,
        output_len=1,
        uid=10,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=FakeHandle(cached_len=5, handle_id=101),
    )
    req_b = Req(
        input_ids=torch.tensor([201, 202, 203, 204], dtype=torch.int32),
        table_idx=14,
        cached_len=3,
        output_len=2,
        uid=11,
        sampling_params=SamplingParams(max_tokens=2),
        cache_handle=FakeHandle(cached_len=3, handle_id=102),
    )
    batch = Batch(reqs=[req_a, req_b], phase="decode")
    write_tuple = _make_write_tuple(batch, torch.device("cpu"))
    return {
        "name": "only_decode_write_mapping",
        "write_req_mapping": [int(x) for x in write_tuple[0].tolist()],
        "write_pos": [int(x) for x in write_tuple[1].tolist()],
    }


def build_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "generator": "rust/scripts/export_prefill_trace.py",
        "adder_cases": [
            run_adder_case(
                name="only_prefill",
                token_budget=16,
                reserved_size=0,
                cache_available_size=64,
                table_slots=[2],
                cached_len=1,
                match_indices=[99],
                input_ids=[1, 2, 3, 4, 5],
                output_len=3,
            ),
            run_adder_case(
                name="mixed_chunked_prefill",
                token_budget=2,
                reserved_size=0,
                cache_available_size=64,
                table_slots=[3],
                cached_len=1,
                match_indices=[99],
                input_ids=[1, 2, 3, 4, 5],
                output_len=3,
            ),
            run_adder_case(
                name="near_capacity_reject",
                token_budget=16,
                reserved_size=2,
                cache_available_size=10,
                table_slots=[4],
                cached_len=1,
                match_indices=[99],
                input_ids=[1, 2, 3, 4, 5],
                output_len=5,
            ),
            run_adder_case(
                name="decode_inflight_recheck_reject",
                token_budget=16,
                reserved_size=2,
                cache_available_size=12,
                table_slots=[5],
                cached_len=1,
                match_indices=[99],
                input_ids=[1, 2, 3, 4, 5],
                output_len=5,
                lock_impact=2,
            ),
        ],
        "mapping_case": run_mapping_case(),
        "decode_only_case": run_decode_only_write_case(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="rust/minisgl-cpu-core/tests/data/prefill_golden_trace.yaml",
        help="Output path (JSON payload with .yaml extension for repo gitignore compatibility).",
    )
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
