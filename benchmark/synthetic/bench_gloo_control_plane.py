from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class BenchSpec:
    name: str
    description: str
    binary: str
    ranks: int
    sizes: List[str]
    warmup: int
    iterations: int


@dataclass
class BenchRow:
    size_label: str
    p50_us: float
    mean_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run mini-sglang-style control-plane proxy benchmarks against Gloo uv and "
            "the Myelon transport."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("../gloo/build-mpi-uv/gloo/mpi/benchmark"),
        help="Path containing mpi_bench_pingpong/broadcast/allreduce binaries.",
    )
    parser.add_argument("--mpirun", default="mpirun")
    return parser.parse_args()


def specs() -> List[BenchSpec]:
    return [
        BenchSpec(
            name="scheduler_broadcast",
            description=(
                "Proxy for scheduler rank0 broadcasting pending message counts and small "
                "control scalars across TP ranks."
            ),
            binary="mpi_bench_broadcast",
            ranks=4,
            sizes=["4", "16"],
            warmup=20,
            iterations=100,
        ),
        BenchSpec(
            name="nccl_uid_broadcast",
            description=(
                "Proxy for init_pynccl broadcast_object_list traffic, modeled as small CPU "
                "metadata fanout."
            ),
            binary="mpi_bench_broadcast",
            ranks=4,
            sizes=["256", "1KiB"],
            warmup=20,
            iterations=100,
        ),
        BenchSpec(
            name="free_mem_min_allreduce",
            description=(
                "Proxy for engine._sync_get_memory, which reduces two int64 CPU values "
                "across TP ranks."
            ),
            binary="mpi_bench_allreduce",
            ranks=4,
            sizes=["16"],
            warmup=20,
            iterations=100,
        ),
        BenchSpec(
            name="small_metadata_allreduce",
            description=(
                "Proxy for small CPU-side aggregate state sync beyond the exact free-memory "
                "shape."
            ),
            binary="mpi_bench_allreduce",
            ranks=4,
            sizes=["1KiB"],
            warmup=20,
            iterations=100,
        ),
    ]


def run_benchmark(
    mpirun: str,
    benchmark_dir: Path,
    spec: BenchSpec,
    transport: str,
) -> str:
    binary = benchmark_dir / spec.binary
    if not binary.exists():
        raise FileNotFoundError(f"missing benchmark binary: {binary}")

    cmd = [
        mpirun,
        "-n",
        str(spec.ranks),
        str(binary),
        f"--transport={transport}",
        f"--sizes={','.join(spec.sizes)}",
        f"--warmup={spec.warmup}",
        f"--iterations={spec.iterations}",
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout


def parse_rows(output: str) -> Dict[str, BenchRow]:
    rows: Dict[str, BenchRow] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.split("|")[1:-1]]
        if len(parts) < 8 or parts[0] == "Bytes" or parts[0].startswith("-"):
            continue
        try:
            rows[parts[0]] = BenchRow(
                size_label=parts[0],
                p50_us=float(parts[3]),
                mean_us=float(parts[7]),
            )
        except ValueError:
            continue
    if not rows:
        raise RuntimeError("failed to parse benchmark rows")
    return rows


def print_table(spec: BenchSpec, uv_rows: Dict[str, BenchRow], myelon_rows: Dict[str, BenchRow]) -> None:
    print(f"\n{spec.name}")
    print(f"  {spec.description}")
    print("+-------------+-------------+----------------+---------------+---------------+")
    print("| Size        | UV p50 us   | Myelon p50 us  | P50 speedup   | Mean speedup  |")
    print("+-------------+-------------+----------------+---------------+---------------+")
    for size_label in uv_rows:
        uv_row = uv_rows[size_label]
        myelon_row = myelon_rows[size_label]
        p50_speedup = uv_row.p50_us / myelon_row.p50_us
        mean_speedup = uv_row.mean_us / myelon_row.mean_us
        print(
            f"| {size_label:>11} | {uv_row.p50_us:>11.2f} | {myelon_row.p50_us:>14.2f} | "
            f"{p50_speedup:>11.2f}x | {mean_speedup:>11.2f}x |"
        )
    print("+-------------+-------------+----------------+---------------+---------------+")


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    print(f"benchmark_dir={benchmark_dir}")
    for spec in specs():
        uv_output = run_benchmark(args.mpirun, benchmark_dir, spec, "auto")
        myelon_output = run_benchmark(args.mpirun, benchmark_dir, spec, "myelon")
        print_table(spec, parse_rows(uv_output), parse_rows(myelon_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
