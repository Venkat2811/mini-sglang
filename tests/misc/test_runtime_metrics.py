from __future__ import annotations

from minisgl.env import ENV
from minisgl.utils import (
    record_backend_fallback,
    record_backend_selection,
    record_scheduler_step,
    record_tokenizer_latency,
    runtime_metrics_snapshot,
)


def test_runtime_metrics_snapshot_accumulates_scheduler_tokenizer_and_backend_counts():
    old = ENV.RUNTIME_METRICS.value
    ENV.RUNTIME_METRICS.value = True
    runtime_metrics_snapshot(reset=True)
    try:
        record_scheduler_step(duration_ns=2_000_000, queue_prefill=3, queue_decode=1, inflight_tokens=9)
        record_scheduler_step(duration_ns=4_000_000, queue_prefill=5, queue_decode=2, inflight_tokens=12)
        record_tokenizer_latency(duration_ns=1_500_000, tokenize_count=8, detokenize_count=0)
        record_tokenizer_latency(duration_ns=2_500_000, tokenize_count=0, detokenize_count=11)
        record_backend_selection(component="cpu", backend="rust_hotpath")
        record_backend_selection(component="tokenizer", backend="rust_inprocess")
        record_backend_fallback(
            component="tokenizer",
            requested="rust_inprocess",
            selected="python",
            reason="init_error",
        )

        stats = runtime_metrics_snapshot()
        assert stats["enabled"] is True
        assert int(stats["scheduler_step_count"]) == 2
        assert int(stats["scheduler_step_ns"]) == 6_000_000
        assert int(stats["scheduler_step_ns_max"]) == 4_000_000
        assert int(stats["queue_prefill_max"]) == 5
        assert int(stats["queue_decode_max"]) == 2
        assert int(stats["inflight_tokens_max"]) == 12
        assert float(stats["scheduler_avg_step_us"]) > 0.0
        assert int(stats["tokenizer_call_count"]) == 2
        assert int(stats["tokenize_items"]) == 8
        assert int(stats["detokenize_items"]) == 11
        assert float(stats["tokenizer_avg_us"]) > 0.0
        assert stats["backend_selection_counts"] == {
            "cpu:rust_hotpath": 1,
            "tokenizer:rust_inprocess": 1,
        }
        assert stats["backend_fallback_counts"] == {
            "tokenizer:rust_inprocess->python:init_error": 1
        }
    finally:
        ENV.RUNTIME_METRICS.value = old
        runtime_metrics_snapshot(reset=True)


def test_runtime_metrics_can_be_disabled():
    old = ENV.RUNTIME_METRICS.value
    ENV.RUNTIME_METRICS.value = False
    runtime_metrics_snapshot(reset=True)
    try:
        record_scheduler_step(duration_ns=100, queue_prefill=1, queue_decode=1, inflight_tokens=1)
        record_tokenizer_latency(duration_ns=100, tokenize_count=1, detokenize_count=1)
        record_backend_selection(component="cpu", backend="python")
        record_backend_fallback(
            component="cpu",
            requested="rust_hotpath",
            selected="python",
            reason="module_load_error",
        )
        stats = runtime_metrics_snapshot()
        assert stats["enabled"] is False
        assert int(stats["scheduler_step_count"]) == 0
        assert int(stats["tokenizer_call_count"]) == 0
        assert stats["backend_selection_counts"] == {}
        assert stats["backend_fallback_counts"] == {}
    finally:
        ENV.RUNTIME_METRICS.value = old
        runtime_metrics_snapshot(reset=True)
