#!/usr/bin/env python3
"""Export deterministic Python radix-cache traces for Rust parity replay tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from minisgl.kvcache.radix_manager import RadixCacheHandle, RadixCacheManager


@dataclass(frozen=True)
class ScenarioStep:
    op: str
    input_ids: list[int] | None = None
    indices: list[int] | None = None
    slot: str | None = None
    unlock: bool | None = None
    size: int | None = None


def _size_info(manager: RadixCacheManager) -> dict[str, int]:
    size_info = manager.size_info
    return {
        "evictable_size": int(size_info.evictable_size),
        "protected_size": int(size_info.protected_size),
    }


def run_case(name: str, steps: list[ScenarioStep]) -> dict[str, Any]:
    manager = RadixCacheManager(torch.device("cpu"))
    handles: dict[str, RadixCacheHandle] = {}
    ops: list[dict[str, Any]] = []

    for step in steps:
        record: dict[str, Any] = {"op": step.op}
        if step.op == "insert":
            assert step.input_ids is not None and step.indices is not None
            input_ids = torch.tensor(step.input_ids, dtype=torch.int32)
            indices = torch.tensor(step.indices, dtype=torch.int32)
            prefix_len = int(manager.insert_prefix(input_ids, indices))
            record["input_ids"] = step.input_ids
            record["indices"] = step.indices
            record["expect_prefix_len"] = prefix_len
        elif step.op == "match":
            assert step.input_ids is not None and step.slot is not None
            input_ids = torch.tensor(step.input_ids, dtype=torch.int32)
            handle, matched = manager.match_prefix(input_ids)
            handles[step.slot] = handle
            record["slot"] = step.slot
            record["input_ids"] = step.input_ids
            record["expect_cached_len"] = int(handle.cached_len)
            record["expect_indices"] = [int(x) for x in matched.tolist()]
        elif step.op == "lock":
            assert step.slot is not None and step.unlock is not None
            manager.lock_handle(handles[step.slot], unlock=step.unlock)
            record["slot"] = step.slot
            record["unlock"] = bool(step.unlock)
        elif step.op == "evict":
            assert step.size is not None
            evicted = manager.evict(step.size)
            record["size"] = int(step.size)
            record["expect_evicted"] = [int(x) for x in evicted.tolist()]
        else:
            raise ValueError(f"unsupported op: {step.op}")

        record["expect_size"] = _size_info(manager)
        ops.append(record)

    return {"name": name, "ops": ops}


def build_trace_payload() -> dict[str, Any]:
    cases = [
        run_case(
            "split_lock_evict",
            [
                ScenarioStep("insert", input_ids=[1, 2, 3, 4], indices=[10, 11, 12, 13]),
                ScenarioStep("insert", input_ids=[1, 2, 9], indices=[20, 21, 22]),
                ScenarioStep("match", input_ids=[1, 2, 3, 8], slot="a"),
                ScenarioStep("lock", slot="a", unlock=False),
                ScenarioStep("match", input_ids=[1, 2, 9, 7], slot="b"),
                ScenarioStep("lock", slot="b", unlock=False),
                ScenarioStep("lock", slot="a", unlock=True),
                ScenarioStep("evict", size=1),
                ScenarioStep("lock", slot="b", unlock=True),
                ScenarioStep("evict", size=2),
            ],
        ),
        run_case(
            "shared_parent_and_reinsert",
            [
                ScenarioStep("insert", input_ids=[5, 6, 7], indices=[50, 60, 70]),
                ScenarioStep("insert", input_ids=[5, 6, 8], indices=[50, 60, 80]),
                ScenarioStep("match", input_ids=[5, 6, 7, 9], slot="c"),
                ScenarioStep("lock", slot="c", unlock=False),
                ScenarioStep("evict", size=1),
                ScenarioStep("lock", slot="c", unlock=True),
                ScenarioStep("evict", size=2),
                ScenarioStep("insert", input_ids=[5, 6, 9], indices=[50, 60, 90]),
                ScenarioStep("match", input_ids=[5, 6, 9, 1], slot="d"),
            ],
        ),
    ]
    return {
        "version": 1,
        "generator": "rust/scripts/export_radix_trace.py",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="rust/minisgl-cpu-core/tests/data/radix_golden_trace.yaml",
        help="Output path (JSON payload with .yaml extension for repo gitignore compatibility).",
    )
    args = parser.parse_args()

    payload = build_trace_payload()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
