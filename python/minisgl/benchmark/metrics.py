from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence


class HasTics(Protocol):
    tics: List[float]


@dataclass(frozen=True)
class MetricStats:
    avg: float
    p50: float
    p90: float
    p99: float
    max: float

    @classmethod
    def from_times(cls, times: List[float], scale: float = 1.0) -> MetricStats:
        if len(times) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0)
        sorted_times = sorted(times)

        def _idx(frac: float) -> int:
            return min(int(len(sorted_times) * frac), len(sorted_times) - 1)

        return cls(
            avg=scale * sum(sorted_times) / len(sorted_times),
            p50=scale * sorted_times[_idx(0.5)],
            p90=scale * sorted_times[_idx(0.9)],
            p99=scale * sorted_times[_idx(0.99)],
            max=scale * sorted_times[-1],
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "avg": self.avg,
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "max": self.max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> MetricStats:
        return cls(
            avg=float(data["avg"]),
            p50=float(data["p50"]),
            p90=float(data["p90"]),
            p99=float(data["p99"]),
            max=float(data["max"]),
        )


@dataclass(frozen=True)
class BenchmarkSummary:
    schema_version: int
    num_requests: int
    num_tokens: int
    duration_s: float
    throughput_token_per_s: float
    throughput_req_per_s: float
    ttft_ms: MetricStats
    tpot_ms: MetricStats
    e2e_s: MetricStats

    @classmethod
    def from_tics_batches(cls, batches: Sequence[Sequence[float]]) -> BenchmarkSummary:
        if len(batches) == 0:
            raise ValueError("Cannot summarize empty benchmark data")
        if any(len(tics) < 2 for tics in batches):
            raise ValueError("Each benchmark trace must contain at least 2 timestamps")

        first_times: List[float] = []
        accum_times: List[float] = []
        e2e_times: List[float] = []

        for tics in batches:
            deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
            first_times.append(deltas[0])
            accum_times.extend(deltas[1:])
            e2e_times.append(tics[-1] - tics[0])

        min_time = min(min(tics) for tics in batches)
        max_time = max(max(tics) for tics in batches)
        duration_s = max_time - min_time
        if duration_s <= 0:
            raise ValueError("Duration must be positive")

        num_tokens = sum(len(tics) for tics in batches)
        num_requests = len(batches)

        return cls(
            schema_version=1,
            num_requests=num_requests,
            num_tokens=num_tokens,
            duration_s=duration_s,
            throughput_token_per_s=num_tokens / duration_s,
            throughput_req_per_s=num_requests / duration_s,
            ttft_ms=MetricStats.from_times(first_times, scale=1000.0),
            tpot_ms=MetricStats.from_times(accum_times, scale=1000.0),
            e2e_s=MetricStats.from_times(e2e_times, scale=1.0),
        )

    @classmethod
    def from_raw_results(cls, raw_data: Sequence[HasTics]) -> BenchmarkSummary:
        return cls.from_tics_batches([r.tics for r in raw_data])

    @classmethod
    def from_throughput(
        cls,
        *,
        num_requests: int,
        num_tokens: int,
        duration_s: float,
    ) -> BenchmarkSummary:
        if duration_s <= 0:
            raise ValueError("Duration must be positive")
        return cls(
            schema_version=1,
            num_requests=num_requests,
            num_tokens=num_tokens,
            duration_s=duration_s,
            throughput_token_per_s=num_tokens / duration_s,
            throughput_req_per_s=num_requests / duration_s,
            ttft_ms=MetricStats.from_times([]),
            tpot_ms=MetricStats.from_times([]),
            e2e_s=MetricStats.from_times([]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_requests": self.num_requests,
            "num_tokens": self.num_tokens,
            "duration_s": self.duration_s,
            "throughput_token_per_s": self.throughput_token_per_s,
            "throughput_req_per_s": self.throughput_req_per_s,
            "ttft_ms": self.ttft_ms.to_dict(),
            "tpot_ms": self.tpot_ms.to_dict(),
            "e2e_s": self.e2e_s.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkSummary:
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            num_requests=int(data["num_requests"]),
            num_tokens=int(data["num_tokens"]),
            duration_s=float(data["duration_s"]),
            throughput_token_per_s=float(data["throughput_token_per_s"]),
            throughput_req_per_s=float(data["throughput_req_per_s"]),
            ttft_ms=MetricStats.from_dict(data["ttft_ms"]),
            tpot_ms=MetricStats.from_dict(data["tpot_ms"]),
            e2e_s=MetricStats.from_dict(data["e2e_s"]),
        )


@dataclass(frozen=True)
class BenchmarkGate:
    min_throughput_token_per_s: float | None = None
    min_throughput_req_per_s: float | None = None
    max_avg_ttft_ms: float | None = None
    max_avg_tpot_ms: float | None = None
    max_avg_e2e_s: float | None = None

    def to_dict(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if self.min_throughput_token_per_s is not None:
            result["min_throughput_token_per_s"] = self.min_throughput_token_per_s
        if self.min_throughput_req_per_s is not None:
            result["min_throughput_req_per_s"] = self.min_throughput_req_per_s
        if self.max_avg_ttft_ms is not None:
            result["max_avg_ttft_ms"] = self.max_avg_ttft_ms
        if self.max_avg_tpot_ms is not None:
            result["max_avg_tpot_ms"] = self.max_avg_tpot_ms
        if self.max_avg_e2e_s is not None:
            result["max_avg_e2e_s"] = self.max_avg_e2e_s
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkGate:
        kwargs: Dict[str, float] = {}
        for key in (
            "min_throughput_token_per_s",
            "min_throughput_req_per_s",
            "max_avg_ttft_ms",
            "max_avg_tpot_ms",
            "max_avg_e2e_s",
        ):
            if key in data:
                kwargs[key] = float(data[key])
        return cls(**kwargs)


def evaluate_gates(summary: BenchmarkSummary, gate: BenchmarkGate) -> List[str]:
    failures: List[str] = []

    if gate.min_throughput_token_per_s is not None and (
        summary.throughput_token_per_s < gate.min_throughput_token_per_s
    ):
        failures.append(
            "throughput_token_per_s "
            f"{summary.throughput_token_per_s:.4f} < min {gate.min_throughput_token_per_s:.4f}"
        )
    if gate.min_throughput_req_per_s is not None and (
        summary.throughput_req_per_s < gate.min_throughput_req_per_s
    ):
        failures.append(
            "throughput_req_per_s "
            f"{summary.throughput_req_per_s:.4f} < min {gate.min_throughput_req_per_s:.4f}"
        )
    if gate.max_avg_ttft_ms is not None and summary.ttft_ms.avg > gate.max_avg_ttft_ms:
        failures.append(
            "avg_ttft_ms " f"{summary.ttft_ms.avg:.4f} > max {gate.max_avg_ttft_ms:.4f}"
        )
    if gate.max_avg_tpot_ms is not None and summary.tpot_ms.avg > gate.max_avg_tpot_ms:
        failures.append(
            "avg_tpot_ms " f"{summary.tpot_ms.avg:.4f} > max {gate.max_avg_tpot_ms:.4f}"
        )
    if gate.max_avg_e2e_s is not None and summary.e2e_s.avg > gate.max_avg_e2e_s:
        failures.append("avg_e2e_s " f"{summary.e2e_s.avg:.4f} > max {gate.max_avg_e2e_s:.4f}")

    return failures


def assert_gates(summary: BenchmarkSummary, gate: BenchmarkGate) -> None:
    failures = evaluate_gates(summary, gate)
    if failures:
        lines = "\n".join(f"- {f}" for f in failures)
        raise AssertionError(f"Benchmark gate check failed:\n{lines}")


def summary_lines(summary: BenchmarkSummary) -> List[str]:
    def _fmt(x: float) -> str:
        if x >= 1000:
            return f"{int(x):>6}"
        if x >= 10:
            return f"{x:>6.2f}"
        return f"{x:>6.4f}"

    return [
        f"Num requests: #{summary.num_requests}, Num tokens: #{summary.num_tokens}",
        "TTFT: "
        f"{_fmt(summary.ttft_ms.avg)} ms (p50: {_fmt(summary.ttft_ms.p50)} ms, "
        f"p90: {_fmt(summary.ttft_ms.p90)} ms, p99: {_fmt(summary.ttft_ms.p99)} ms, "
        f"max: {_fmt(summary.ttft_ms.max)} ms)",
        "TPOT: "
        f"{_fmt(summary.tpot_ms.avg)} ms (p50: {_fmt(summary.tpot_ms.p50)} ms, "
        f"p90: {_fmt(summary.tpot_ms.p90)} ms, p99: {_fmt(summary.tpot_ms.p99)} ms, "
        f"max: {_fmt(summary.tpot_ms.max)} ms)",
        "E2E:  "
        f"{_fmt(summary.e2e_s.avg)}  s (p50: {_fmt(summary.e2e_s.p50)}  s, "
        f"p90: {_fmt(summary.e2e_s.p90)}  s, p99: {_fmt(summary.e2e_s.p99)}  s, "
        f"max: {_fmt(summary.e2e_s.max)}  s)",
        f"Duration: {_fmt(summary.duration_s)} s",
        "Throughput: "
        f"{_fmt(summary.throughput_token_per_s)} token/s, "
        f"{_fmt(summary.throughput_req_per_s)} req/s",
    ]
