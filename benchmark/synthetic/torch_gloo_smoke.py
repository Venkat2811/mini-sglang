from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List


@dataclass
class WorkerResult:
    rank: int
    all_reduce_value: int
    broadcast_value: int


def _worker(rank: int, world_size: int, init_method: str, queue: mp.Queue) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
        init_method=init_method,
    )

    reduced = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    broadcasted = torch.tensor([-1], dtype=torch.int64)
    if rank == 0:
        broadcasted[0] = 42
    dist.broadcast(broadcasted, src=0)

    queue.put(
        WorkerResult(
            rank=rank,
            all_reduce_value=int(reduced.item()),
            broadcast_value=int(broadcasted.item()),
        )
    )
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the wheel-backed torch.distributed gloo path on CPU."
    )
    parser.add_argument("--world-size", type=int, default=2)
    return parser.parse_args()


def torch_lib_root() -> Path:
    import torch

    return Path(torch.__file__).resolve().parent / "lib"


def find_matching_names(root: Path, pattern: str) -> List[str]:
    return sorted(path.name for path in root.glob(pattern))


def main() -> int:
    args = parse_args()

    import torch
    import torch.distributed as dist

    print(f"torch={torch.__version__}")
    print(f"torch_file={torch.__file__}")
    print(f"distributed_available={dist.is_available()}")
    print(f"gloo_available={dist.is_gloo_available()}")
    print(f"nccl_available={dist.is_nccl_available()}")

    lib_root = torch_lib_root()
    print(f"torch_lib_root={lib_root}")
    print(f"libgloo_matches={find_matching_names(lib_root, '*gloo*')}")
    print(f"libtorch_cpu_matches={find_matching_names(lib_root, 'libtorch_cpu*.dylib')}")
    print(f"libtorch_python_matches={find_matching_names(lib_root, 'libtorch_python*.dylib')}")

    mp.set_start_method("spawn", force=True)
    queue: mp.Queue = mp.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        rendezvous_file = handle.name

    init_method = f"file://{rendezvous_file}"
    processes = [
        mp.Process(target=_worker, args=(rank, args.world_size, init_method, queue))
        for rank in range(args.world_size)
    ]

    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=30) for _ in range(args.world_size)]
        for process in processes:
            process.join(timeout=30)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        Path(rendezvous_file).unlink(missing_ok=True)

    results.sort(key=lambda item: item.rank)
    expected_reduce = sum(rank + 1 for rank in range(args.world_size))
    expected_broadcast = 42

    print("\nresults")
    for result in results:
        print(
            f"  rank={result.rank} all_reduce={result.all_reduce_value} "
            f"broadcast={result.broadcast_value}"
        )

    exit_codes = [process.exitcode for process in processes]
    print(f"exit_codes={exit_codes}")

    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"one or more worker processes failed: {exit_codes}")

    for result in results:
        if result.all_reduce_value != expected_reduce:
            raise RuntimeError(
                f"unexpected all_reduce result on rank {result.rank}: "
                f"{result.all_reduce_value} != {expected_reduce}"
            )
        if result.broadcast_value != expected_broadcast:
            raise RuntimeError(
                f"unexpected broadcast result on rank {result.rank}: "
                f"{result.broadcast_value} != {expected_broadcast}"
            )

    print("\nsummary")
    print("  wheel-backed torch.distributed gloo works on CPU in this uv venv")
    print("  the wheel does not ship a standalone libgloo dylib to swap at runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
