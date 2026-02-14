from __future__ import annotations

import math

from minisgl.benchmark.client import RawResult
from minisgl.benchmark.metrics import BenchmarkGate, BenchmarkSummary, evaluate_gates


def _make_raw() -> list[RawResult]:
    return [
        RawResult(
            input_len=16,
            output_len=3,
            message="a",
            tics=[10.00, 10.10, 10.20, 10.40],
        ),
        RawResult(
            input_len=24,
            output_len=2,
            message="b",
            tics=[10.05, 10.25, 10.45],
        ),
    ]


def test_summary_from_raw_results() -> None:
    summary = BenchmarkSummary.from_raw_results(_make_raw())

    assert summary.num_requests == 2
    assert summary.num_tokens == 7
    assert math.isclose(summary.duration_s, 0.45, rel_tol=1e-6)
    assert math.isclose(summary.throughput_token_per_s, 7 / 0.45, rel_tol=1e-6)
    assert math.isclose(summary.throughput_req_per_s, 2 / 0.45, rel_tol=1e-6)

    assert math.isclose(summary.ttft_ms.avg, 150.0, rel_tol=1e-6)
    assert math.isclose(summary.ttft_ms.p50, 200.0, rel_tol=1e-6)
    assert math.isclose(summary.tpot_ms.avg, (100 + 200 + 200) / 3, rel_tol=1e-6)
    assert math.isclose(summary.e2e_s.avg, 0.40, rel_tol=1e-6)


def test_summary_roundtrip_dict() -> None:
    summary = BenchmarkSummary.from_raw_results(_make_raw())
    loaded = BenchmarkSummary.from_dict(summary.to_dict())
    assert loaded == summary


def test_evaluate_gates_reports_failures() -> None:
    summary = BenchmarkSummary.from_raw_results(_make_raw())
    gate = BenchmarkGate(
        min_throughput_token_per_s=20.0,
        max_avg_ttft_ms=100.0,
        max_avg_tpot_ms=150.0,
    )
    failures = evaluate_gates(summary, gate)

    assert len(failures) == 3
    assert any("throughput_token_per_s" in msg for msg in failures)
    assert any("avg_ttft_ms" in msg for msg in failures)
    assert any("avg_tpot_ms" in msg for msg in failures)


def test_evaluate_gates_passes_when_thresholds_met() -> None:
    summary = BenchmarkSummary.from_raw_results(_make_raw())
    gate = BenchmarkGate(
        min_throughput_token_per_s=10.0,
        max_avg_ttft_ms=200.0,
        max_avg_tpot_ms=200.0,
        max_avg_e2e_s=1.0,
    )
    assert evaluate_gates(summary, gate) == []


def test_summary_from_throughput() -> None:
    summary = BenchmarkSummary.from_throughput(
        num_requests=8,
        num_tokens=400,
        duration_s=2.0,
    )
    assert summary.num_requests == 8
    assert summary.num_tokens == 400
    assert math.isclose(summary.throughput_token_per_s, 200.0, rel_tol=1e-6)
    assert math.isclose(summary.throughput_req_per_s, 4.0, rel_tol=1e-6)
    assert summary.ttft_ms.avg == 0.0
