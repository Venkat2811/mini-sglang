#!/usr/bin/env python3
"""Side-by-side comparison across closure runs.

Reads multiple `run_closure_benchmark.py` JSON artifacts (3 per backend
recommended) and emits:

- per-backend mean, stddev, and coefficient of variation (CV)
- python-vs-rust delta percentages on the medians
- a Markdown table suitable for inclusion in the closure retrospective

The CV is the stability gate used by `release_gate.py`. CV > 0.10 on
throughput is the existing soft warning threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


METRICS = [
    ("throughput_token_per_s", "tok/s", False),
    ("throughput_req_per_s", "req/s", False),
    ("ttft_ms.avg", "TTFT avg ms", True),
    ("ttft_ms.p50", "TTFT p50 ms", True),
    ("ttft_ms.p99", "TTFT p99 ms", True),
    ("e2e_s.avg", "E2E avg s", True),
    ("e2e_s.p50", "E2E p50 s", True),
    ("e2e_s.p99", "E2E p99 s", True),
]


def _get(payload: Dict[str, Any], dotted: str) -> float:
    cur: Any = payload["summary"]
    for part in dotted.split("."):
        cur = cur[part]
    return float(cur)


def _load_runs(paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            print(f"missing artifact: {p}", file=sys.stderr)
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _stats_for(runs: List[Dict[str, Any]], metric: str) -> Tuple[float, float, float]:
    values = [_get(r, metric) for r in runs]
    if not values:
        return (0.0, 0.0, 0.0)
    mean = stats.fmean(values)
    sd = stats.pstdev(values) if len(values) > 1 else 0.0
    cv = sd / mean if mean else 0.0
    return (mean, sd, cv)


def _median(runs: List[Dict[str, Any]], metric: str) -> float:
    values = [_get(r, metric) for r in runs]
    if not values:
        return 0.0
    return stats.median(values)


def _delta_pct(rust: float, python: float, lower_better: bool) -> Tuple[float, str]:
    if python == 0:
        return (0.0, "n/a")
    pct = 100.0 * (rust - python) / python
    if lower_better:
        verdict = "BETTER" if pct < 0 else ("WORSE" if pct > 0 else "EQUAL")
    else:
        verdict = "BETTER" if pct > 0 else ("WORSE" if pct < 0 else "EQUAL")
    return (pct, verdict)


def _render_md(
    python_runs: List[Dict[str, Any]],
    rust_runs: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Closure A/B Side-By-Side")
    lines.append("")
    if python_runs:
        first = python_runs[0]
        m = first.get("machine_snapshot", {})
        gpu = m.get("gpu", {})
        lines.append("## Run Context")
        lines.append("")
        lines.append(f"- model: `{first.get('model_path', 'unknown')}`")
        lines.append(f"- prompts: `{first.get('profile', {}).get('prompts_path', 'unknown')}` "
                     f"({first.get('profile', {}).get('prompts_count', 'unknown')} prompts)")
        lines.append(f"- concurrency: `{first.get('profile', {}).get('concurrency', 'unknown')}`")
        lines.append(f"- streaming: `{first.get('profile', {}).get('streaming', 'unknown')}`")
        lines.append(f"- typed transport: `MINISGL_TYPED_TRANSPORT={first.get('profile', {}).get('typed_transport', '1')}`")
        lines.append(f"- gpu: `{gpu.get('device_name_0', 'unknown')}`")
        lines.append(f"- driver: `{gpu.get('driver_version_via_nvml', 'unknown')}`")
        lines.append(f"- python: `{m.get('python_version', 'unknown')}`")
        lines.append("")

    lines.append("## Per-Backend Stability (3 runs each)")
    lines.append("")
    lines.append("| Metric | Python mean | Python CV | Rust mean | Rust CV |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, label, _lower_better in METRICS:
        py_mean, _py_sd, py_cv = _stats_for(python_runs, key)
        ru_mean, _ru_sd, ru_cv = _stats_for(rust_runs, key)
        lines.append(
            f"| {label} | {py_mean:.2f} | {py_cv * 100:.2f}% | {ru_mean:.2f} | {ru_cv * 100:.2f}% |"
        )

    lines.append("")
    lines.append("## Rust vs Python (medians)")
    lines.append("")
    lines.append("| Metric | Python median | Rust median | Delta % | Verdict |")
    lines.append("|---|---:|---:|---:|:---|")
    for key, label, lower_better in METRICS:
        py = _median(python_runs, key)
        ru = _median(rust_runs, key)
        pct, verdict = _delta_pct(ru, py, lower_better)
        lines.append(f"| {label} | {py:.2f} | {ru:.2f} | {pct:+.2f}% | {verdict} |")

    lines.append("")
    lines.append("## Stability Gate")
    lines.append("")
    lines.append("Per `release_gate.py` defaults, throughput CV above 10% indicates an unstable run series.")
    py_tput_cv = _stats_for(python_runs, "throughput_token_per_s")[2]
    ru_tput_cv = _stats_for(rust_runs, "throughput_token_per_s")[2]
    lines.append(f"- python throughput CV: `{py_tput_cv * 100:.2f}%` "
                 f"({'PASS' if py_tput_cv <= 0.10 else 'FAIL'})")
    lines.append(f"- rust throughput CV:   `{ru_tput_cv * 100:.2f}%` "
                 f"({'PASS' if ru_tput_cv <= 0.10 else 'FAIL'})")
    lines.append("")
    lines.append("## Closure Outcome Per RFC 0001")
    lines.append("")
    py_tput_med = _median(python_runs, "throughput_token_per_s")
    ru_tput_med = _median(rust_runs, "throughput_token_per_s")
    if py_tput_med > 0:
        delta = 100.0 * (ru_tput_med - py_tput_med) / py_tput_med
        if delta > 2.0:
            outcome = "Outcome A: Rust positive (>= +2% online tok/s)"
        elif delta < -2.0:
            outcome = "Outcome C: Rust regression (< -2% online tok/s)"
        else:
            outcome = "Outcome B: Rust neutral (within +/- 2% online tok/s)"
        lines.append(f"- online tok/s delta: `{delta:+.2f}%`")
        lines.append(f"- {outcome}")
    else:
        lines.append("- could not compute online tok/s delta (python median is zero)")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", nargs="+", type=Path, required=True, help="Python-backend run JSONs (3 expected)")
    parser.add_argument("--rust", nargs="+", type=Path, required=True, help="Rust-backend run JSONs (3 expected)")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown path")
    args = parser.parse_args()

    python_runs = _load_runs(args.python)
    rust_runs = _load_runs(args.rust)
    if not python_runs or not rust_runs:
        print("need at least one run per backend", file=sys.stderr)
        return 2

    md = _render_md(python_runs, rust_runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"wrote {args.out}")
    # Echo the outcome line to stdout for easy grep.
    for line in md.splitlines():
        if line.startswith("- Outcome") or line.startswith("- online tok/s"):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
