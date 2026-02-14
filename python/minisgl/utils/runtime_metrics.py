from __future__ import annotations

from threading import Lock
from typing import Dict

from minisgl.env import ENV


class _RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def _reset_locked(self) -> None:
        self.scheduler_step_count = 0
        self.scheduler_step_ns = 0
        self.scheduler_step_ns_max = 0
        self.scheduler_last_step_ns = 0
        self.queue_prefill_last = 0
        self.queue_decode_last = 0
        self.inflight_tokens_last = 0
        self.queue_prefill_max = 0
        self.queue_decode_max = 0
        self.inflight_tokens_max = 0

        self.tokenizer_call_count = 0
        self.tokenizer_ns = 0
        self.tokenizer_ns_max = 0
        self.tokenize_items = 0
        self.detokenize_items = 0

        self.backend_selection_counts: Dict[str, int] = {}
        self.backend_fallback_counts: Dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def record_scheduler_step(
        self,
        *,
        duration_ns: int,
        queue_prefill: int,
        queue_decode: int,
        inflight_tokens: int,
    ) -> None:
        if not bool(ENV.RUNTIME_METRICS):
            return
        with self._lock:
            self.scheduler_step_count += 1
            self.scheduler_step_ns += duration_ns
            self.scheduler_last_step_ns = duration_ns
            self.scheduler_step_ns_max = max(self.scheduler_step_ns_max, duration_ns)
            self.queue_prefill_last = queue_prefill
            self.queue_decode_last = queue_decode
            self.inflight_tokens_last = inflight_tokens
            self.queue_prefill_max = max(self.queue_prefill_max, queue_prefill)
            self.queue_decode_max = max(self.queue_decode_max, queue_decode)
            self.inflight_tokens_max = max(self.inflight_tokens_max, inflight_tokens)

    def record_tokenizer_latency(
        self,
        *,
        duration_ns: int,
        tokenize_count: int,
        detokenize_count: int,
    ) -> None:
        if not bool(ENV.RUNTIME_METRICS):
            return
        with self._lock:
            self.tokenizer_call_count += 1
            self.tokenizer_ns += duration_ns
            self.tokenizer_ns_max = max(self.tokenizer_ns_max, duration_ns)
            self.tokenize_items += tokenize_count
            self.detokenize_items += detokenize_count

    def record_backend_selection(self, *, component: str, backend: str) -> None:
        if not bool(ENV.RUNTIME_METRICS):
            return
        key = f"{component}:{backend}"
        with self._lock:
            self.backend_selection_counts[key] = self.backend_selection_counts.get(key, 0) + 1

    def record_backend_fallback(
        self,
        *,
        component: str,
        requested: str,
        selected: str,
        reason: str,
    ) -> None:
        if not bool(ENV.RUNTIME_METRICS):
            return
        key = f"{component}:{requested}->{selected}:{reason}"
        with self._lock:
            self.backend_fallback_counts[key] = self.backend_fallback_counts.get(key, 0) + 1

    def snapshot(self, *, reset: bool = False) -> Dict[str, int | float | bool | Dict[str, int]]:
        with self._lock:
            step_count = self.scheduler_step_count
            tok_calls = self.tokenizer_call_count
            payload: Dict[str, int | float | bool | Dict[str, int]] = {
                "enabled": bool(ENV.RUNTIME_METRICS),
                "scheduler_step_count": step_count,
                "scheduler_step_ns": self.scheduler_step_ns,
                "scheduler_step_ns_max": self.scheduler_step_ns_max,
                "scheduler_last_step_ns": self.scheduler_last_step_ns,
                "scheduler_avg_step_us": (self.scheduler_step_ns / step_count / 1000.0)
                if step_count
                else 0.0,
                "queue_prefill_last": self.queue_prefill_last,
                "queue_decode_last": self.queue_decode_last,
                "inflight_tokens_last": self.inflight_tokens_last,
                "queue_prefill_max": self.queue_prefill_max,
                "queue_decode_max": self.queue_decode_max,
                "inflight_tokens_max": self.inflight_tokens_max,
                "tokenizer_call_count": tok_calls,
                "tokenizer_ns": self.tokenizer_ns,
                "tokenizer_ns_max": self.tokenizer_ns_max,
                "tokenizer_avg_us": (self.tokenizer_ns / tok_calls / 1000.0) if tok_calls else 0.0,
                "tokenize_items": self.tokenize_items,
                "detokenize_items": self.detokenize_items,
                "backend_selection_counts": dict(self.backend_selection_counts),
                "backend_fallback_counts": dict(self.backend_fallback_counts),
            }
            if reset:
                self._reset_locked()
            return payload


_RUNTIME_METRICS = _RuntimeMetrics()


def runtime_metrics_snapshot(*, reset: bool = False) -> Dict[str, int | float | bool | Dict[str, int]]:
    return _RUNTIME_METRICS.snapshot(reset=reset)


def record_scheduler_step(
    *,
    duration_ns: int,
    queue_prefill: int,
    queue_decode: int,
    inflight_tokens: int,
) -> None:
    _RUNTIME_METRICS.record_scheduler_step(
        duration_ns=duration_ns,
        queue_prefill=queue_prefill,
        queue_decode=queue_decode,
        inflight_tokens=inflight_tokens,
    )


def record_tokenizer_latency(
    *,
    duration_ns: int,
    tokenize_count: int,
    detokenize_count: int,
) -> None:
    _RUNTIME_METRICS.record_tokenizer_latency(
        duration_ns=duration_ns,
        tokenize_count=tokenize_count,
        detokenize_count=detokenize_count,
    )


def record_backend_selection(*, component: str, backend: str) -> None:
    _RUNTIME_METRICS.record_backend_selection(component=component, backend=backend)


def record_backend_fallback(*, component: str, requested: str, selected: str, reason: str) -> None:
    _RUNTIME_METRICS.record_backend_fallback(
        component=component,
        requested=requested,
        selected=selected,
        reason=reason,
    )
