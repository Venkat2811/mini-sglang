from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
EXTENSION_DIR = ROOT / "benchmark" / "synthetic" / "c10d_glooext"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    kind: str


SCENARIOS = [
    Scenario(
        name="barrier",
        description="Exact proxy for SchedulerIOMixin.sync_all_ranks().",
        kind="barrier",
    ),
    Scenario(
        name="scheduler_count_broadcast",
        description="Exact proxy for scheduler rank0 broadcasting the pending raw-message count as an int64 CPU tensor.",
        kind="broadcast_int64",
    ),
    Scenario(
        name="pynccl_uid_broadcast_object",
        description="PyTorch-level proxy for init_pynccl broadcast_object_list on a NCCL-like unique ID payload.",
        kind="broadcast_object_128b",
    ),
    Scenario(
        name="free_mem_min_allreduce",
        description="Exact proxy for Engine._sync_get_memory() reducing [free, -free] as two int64 CPU values with ReduceOp.MIN.",
        kind="allreduce_min_2xint64",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Mini-SGLang-relevant CPU control-plane collectives through "
            "builtin torch.distributed gloo and the custom glooext c10d backend."
        )
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=EXTENSION_DIR,
        help="Path containing the custom c10d extension setup.py",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--results-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--backend-name", help=argparse.SUPPRESS)
    parser.add_argument("--backend-label", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--master-port", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def build_extension(extension_dir: Path) -> None:
    setup_py = extension_dir / "setup.py"
    if not setup_py.exists():
        raise FileNotFoundError(f"missing extension setup.py: {setup_py}")
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    subprocess.run(cmd, cwd=extension_dir, check=True)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def percentile(sorted_values: List[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = round(pct * (len(sorted_values) - 1))
    return float(sorted_values[index])


def build_row(samples_ns: List[int]) -> Dict[str, float]:
    ordered = sorted(samples_ns)
    return {
        "min_us": ordered[0] / 1_000.0,
        "p50_us": percentile(ordered, 0.50) / 1_000.0,
        "p95_us": percentile(ordered, 0.95) / 1_000.0,
        "p99_us": percentile(ordered, 0.99) / 1_000.0,
        "max_us": ordered[-1] / 1_000.0,
        "mean_us": mean(ordered) / 1_000.0,
    }


def run_backend(
    args: argparse.Namespace,
    extension_dir: Path,
    backend_name: str,
    backend_label: str,
    extra_env: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    port = find_free_port()
    with tempfile.TemporaryDirectory(prefix=f"{backend_label}_") as tmpdir:
        results_dir = Path(tmpdir)
        processes = []
        for rank in range(args.world_size):
            env = os.environ.copy()
            env.update(
                {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(port),
                    "WORLD_SIZE": str(args.world_size),
                    "RANK": str(rank),
                    "PYTHONPATH": str(ROOT / "python"),
                }
            )
            if extra_env:
                env.update(extra_env)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                f"--results-dir={results_dir}",
                f"--extension-dir={extension_dir}",
                f"--backend-name={backend_name}",
                f"--backend-label={backend_label}",
                f"--rank={rank}",
                f"--world-size={args.world_size}",
                f"--warmup={args.warmup}",
                f"--iterations={args.iterations}",
                f"--master-port={port}",
            ]
            processes.append(
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=ROOT,
                )
            )

        stdout_chunks = []
        stderr_chunks = []
        failed = []
        for rank, proc in enumerate(processes):
            try:
                stdout, stderr = proc.communicate(timeout=90)
            except subprocess.TimeoutExpired:
                for child in processes:
                    child.kill()
                stdout, stderr = proc.communicate()
                raise RuntimeError(
                    f"{backend_label} timed out while waiting for rank {rank}\n"
                    f"partial stdout:\n{stdout}\npartial stderr:\n{stderr}"
                )
            if stdout:
                stdout_chunks.append(f"[rank{rank}]\n{stdout}")
            if stderr:
                stderr_chunks.append(f"[rank{rank}]\n{stderr}")
            if proc.returncode != 0:
                failed.append(rank)
        if failed:
            raise RuntimeError(
                f"{backend_label} worker failure on ranks {failed}\n"
                f"stdout:\n{''.join(stdout_chunks)}\n"
                f"stderr:\n{''.join(stderr_chunks)}"
            )

        per_rank = []
        for rank in range(args.world_size):
            rank_file = results_dir / f"{backend_label}_rank{rank}.json"
            per_rank.append(json.loads(rank_file.read_text()))

    scenario_rows: Dict[str, Dict[str, float]] = {}
    for scenario in SCENARIOS:
        sample_matrix = [rank_data["scenarios"][scenario.name]["samples_ns"] for rank_data in per_rank]
        reduced = [max(row[i] for row in sample_matrix) for i in range(args.iterations)]
        all_verified = all(rank_data["scenarios"][scenario.name]["verified"] for rank_data in per_rank)
        if not all_verified:
            raise RuntimeError(f"verification failed for {backend_label}:{scenario.name}")
        scenario_rows[scenario.name] = build_row(reduced)

    init_samples = [int(rank_data["init_ns"]) for rank_data in per_rank]
    init_row = build_row(init_samples)
    return {"init": init_row, "scenarios": scenario_rows}


def print_table(
    args: argparse.Namespace,
    builtin: Dict[str, Any],
    ext_uv: Dict[str, Any],
    ext_myelon: Dict[str, Any],
) -> None:
    print(f"world_size={args.world_size} warmup={args.warmup} iterations={args.iterations}")
    print()
    print("+---------------------------+---------------+-----------------+-----------------+-----------------+------------------+")
    print("| Scenario                  | Builtin Gloo  | glooext UV      | glooext Myelon  | Myelon vs UV    | Myelon vs Builtin|")
    print("+---------------------------+---------------+-----------------+-----------------+-----------------+------------------+")
    print(
        f"| {'init_process_group':<25} | "
        f"{builtin['init']['p50_us']:>11.2f} | "
        f"{ext_uv['init']['p50_us']:>15.2f} | "
        f"{ext_myelon['init']['p50_us']:>15.2f} | "
        f"{ext_uv['init']['p50_us'] / ext_myelon['init']['p50_us']:>15.2f}x | "
        f"{builtin['init']['p50_us'] / ext_myelon['init']['p50_us']:>16.2f}x |"
    )
    for scenario in SCENARIOS:
        b = builtin["scenarios"][scenario.name]["p50_us"]
        u = ext_uv["scenarios"][scenario.name]["p50_us"]
        m = ext_myelon["scenarios"][scenario.name]["p50_us"]
        print(
            f"| {scenario.name:<25} | "
            f"{b:>11.2f} | "
            f"{u:>15.2f} | "
            f"{m:>15.2f} | "
            f"{u / m:>15.2f}x | "
            f"{b / m:>16.2f}x |"
        )
    print("+---------------------------+---------------+-----------------+-----------------+-----------------+------------------+")
    print()
    print("+---------------------------+---------------+-----------------+-----------------+")
    print("| Scenario                  | Builtin Mean  | glooext UV Mean | Myelon Mean     |")
    print("+---------------------------+---------------+-----------------+-----------------+")
    print(
        f"| {'init_process_group':<25} | "
        f"{builtin['init']['mean_us']:>11.2f} | "
        f"{ext_uv['init']['mean_us']:>15.2f} | "
        f"{ext_myelon['init']['mean_us']:>15.2f} |"
    )
    for scenario in SCENARIOS:
        print(
            f"| {scenario.name:<25} | "
            f"{builtin['scenarios'][scenario.name]['mean_us']:>11.2f} | "
            f"{ext_uv['scenarios'][scenario.name]['mean_us']:>15.2f} | "
            f"{ext_myelon['scenarios'][scenario.name]['mean_us']:>15.2f} |"
        )
    print("+---------------------------+---------------+-----------------+-----------------+")


def import_backend(extension_dir: Path) -> None:
    sys.path.insert(0, str(extension_dir))
    __import__("myelon_c10d_backend")


def verify_broadcast_int64(tensor: Any) -> bool:
    return int(tensor.item()) == 17


def verify_allreduce_min_tensor(tensor: Any, world_size: int) -> bool:
    expected = [10_000, -(10_000 + world_size - 1)]
    actual = [int(x) for x in tensor.tolist()]
    return actual == expected


def worker_main(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    extension_dir = args.extension_dir.resolve()
    if args.backend_name == "glooext":
        import_backend(extension_dir)

    init_start = time.perf_counter_ns()
    dist.init_process_group(
        backend=args.backend_name,
        init_method="env://",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=20),
    )
    dist.barrier()
    init_end = time.perf_counter_ns()

    results: Dict[str, Any] = {"init_ns": init_end - init_start, "scenarios": {}}

    def run_barrier() -> bool:
        dist.barrier()
        return True

    def run_broadcast_int64() -> bool:
        tensor = torch.tensor(17 if args.rank == 0 else -1, dtype=torch.int64)
        dist.broadcast(tensor, src=0)
        return verify_broadcast_int64(tensor)

    uid_payload = bytes([rank % 251 for rank in range(128)])

    def run_broadcast_object() -> bool:
        objects = [uid_payload] if args.rank == 0 else [None]
        dist.broadcast_object_list(objects, src=0)
        return isinstance(objects[0], (bytes, bytearray)) and len(objects[0]) == len(uid_payload)

    def run_allreduce_min() -> bool:
        free_memory = 10_000 + args.rank
        tensor = torch.tensor([free_memory, -free_memory], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return verify_allreduce_min_tensor(tensor, args.world_size)

    runners = {
        "barrier": run_barrier,
        "broadcast_int64": run_broadcast_int64,
        "broadcast_object_128b": run_broadcast_object,
        "allreduce_min_2xint64": run_allreduce_min,
    }

    for scenario in SCENARIOS:
        runner = runners[scenario.kind]
        dist.barrier()
        for _ in range(args.warmup):
            ok = runner()
            if not ok:
                raise RuntimeError(f"warmup verification failed for {scenario.name}")
        dist.barrier()
        samples_ns: List[int] = []
        verified = True
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            ok = runner()
            end = time.perf_counter_ns()
            samples_ns.append(end - start)
            verified = verified and ok
        dist.barrier()
        results["scenarios"][scenario.name] = {
            "samples_ns": samples_ns,
            "verified": verified,
        }

    output_path = args.results_dir / f"{args.backend_label}_rank{args.rank}.json"
    output_path.write_text(json.dumps(results))
    dist.destroy_process_group()
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(args)

    extension_dir = args.extension_dir.resolve()
    print(f"extension_dir={extension_dir}")
    build_extension(extension_dir)

    builtin = run_backend(args, extension_dir, backend_name="gloo", backend_label="builtin_gloo")
    ext_uv = run_backend(
        args,
        extension_dir,
        backend_name="glooext",
        backend_label="glooext_uv",
        extra_env={"MINISGL_C10D_GLOOEXT_TRANSPORT": "uv"},
    )
    ext_myelon = run_backend(
        args,
        extension_dir,
        backend_name="glooext",
        backend_label="glooext_myelon",
        extra_env={"MINISGL_C10D_GLOOEXT_TRANSPORT": "myelon"},
    )
    print_table(args, builtin, ext_uv, ext_myelon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
