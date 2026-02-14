from __future__ import annotations

from minisgl.benchmark.metrics import BenchmarkGate, BenchmarkSummary, MetricStats
from minisgl.benchmark.release_gate import evaluate_release_gate


def _summary(throughput: float, ttft_avg: float) -> BenchmarkSummary:
    return BenchmarkSummary(
        schema_version=1,
        num_requests=8,
        num_tokens=520,
        duration_s=1.0,
        throughput_token_per_s=throughput,
        throughput_req_per_s=8.0,
        ttft_ms=MetricStats(avg=ttft_avg, p50=ttft_avg, p90=ttft_avg, p99=ttft_avg, max=ttft_avg),
        tpot_ms=MetricStats(avg=4.5, p50=4.5, p90=4.5, p99=4.5, max=4.5),
        e2e_s=MetricStats(avg=0.33, p50=0.33, p90=0.33, p99=0.33, max=0.33),
    )


def test_release_gate_fails_perf_threshold():
    passed, failures, _ = evaluate_release_gate(
        perf_summary=_summary(throughput=900.0, ttft_avg=40.0),
        perf_gate=BenchmarkGate(min_throughput_token_per_s=1000.0),
    )
    assert passed is False
    assert any("throughput_token_per_s" in failure for failure in failures)


def test_release_gate_fails_parity():
    passed, failures, _ = evaluate_release_gate(
        perf_summary=_summary(throughput=1200.0, ttft_avg=40.0),
        parity_payload={"parity_passed": False},
    )
    assert passed is False
    assert any("parity gate failed" in failure for failure in failures)


def test_release_gate_fails_shadow_divergence_limit():
    passed, failures, _ = evaluate_release_gate(
        perf_summary=_summary(throughput=1200.0, ttft_avg=40.0),
        shadow_divergence_entries=3,
        max_shadow_divergence_entries=0,
    )
    assert passed is False
    assert any("shadow gate failed" in failure for failure in failures)


def test_release_gate_fails_stability_cv():
    passed, failures, report = evaluate_release_gate(
        perf_summary=_summary(throughput=1200.0, ttft_avg=40.0),
        stability_summaries=[_summary(1200.0, 40.0), _summary(900.0, 40.0), _summary(1300.0, 40.0)],
        max_throughput_cv=0.05,
    )
    assert passed is False
    assert any("throughput_cv" in failure for failure in failures)
    assert report["stability"]["throughput_cv"] > 0.05


def test_release_gate_passes_when_all_conditions_hold():
    passed, failures, report = evaluate_release_gate(
        perf_summary=_summary(throughput=1200.0, ttft_avg=35.0),
        perf_gate=BenchmarkGate(min_throughput_token_per_s=1000.0, max_avg_ttft_ms=50.0),
        parity_payload={"parity_passed": True},
        shadow_divergence_entries=0,
        max_shadow_divergence_entries=0,
        stability_summaries=[_summary(1200.0, 35.0), _summary(1180.0, 35.5), _summary(1210.0, 34.8)],
        max_throughput_cv=0.02,
        max_ttft_cv=0.02,
    )
    assert passed is True
    assert failures == []
    assert report["gate_passed"] is True
