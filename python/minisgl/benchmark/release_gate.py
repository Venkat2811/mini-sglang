from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import yaml
from minisgl.benchmark.metrics import BenchmarkGate, BenchmarkSummary, evaluate_gates


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gate(path: Path) -> BenchmarkGate:
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = _load_json(path)
    return BenchmarkGate.from_dict(data)


def _load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("release gate config must be a mapping")
    return data


def _summary_from_payload(payload: Dict[str, Any]) -> BenchmarkSummary:
    return BenchmarkSummary.from_dict(payload["summary"])


def _count_jsonl_entries(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            count += 1
    return count


def _cv(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    if avg == 0:
        return 0.0
    return pstdev(values) / avg


def evaluate_release_gate(
    *,
    perf_summary: BenchmarkSummary,
    perf_gate: BenchmarkGate | None = None,
    parity_payload: Dict[str, Any] | None = None,
    shadow_divergence_entries: int | None = None,
    max_shadow_divergence_entries: int | None = None,
    stability_summaries: List[BenchmarkSummary] | None = None,
    max_throughput_cv: float | None = None,
    max_ttft_cv: float | None = None,
) -> tuple[bool, List[str], Dict[str, Any]]:
    failures: List[str] = []

    if perf_gate is not None:
        failures.extend(evaluate_gates(perf_summary, perf_gate))

    if parity_payload is not None and not bool(parity_payload.get("parity_passed", False)):
        failures.append("parity gate failed: parity_passed is false")

    if (
        shadow_divergence_entries is not None
        and max_shadow_divergence_entries is not None
        and shadow_divergence_entries > max_shadow_divergence_entries
    ):
        failures.append(
            "shadow gate failed: "
            f"divergence_entries={shadow_divergence_entries} > max={max_shadow_divergence_entries}"
        )

    stability: Dict[str, Any] = {}
    if stability_summaries:
        throughput_values = [s.throughput_token_per_s for s in stability_summaries]
        ttft_values = [s.ttft_ms.avg for s in stability_summaries]
        throughput_cv = _cv(throughput_values)
        ttft_cv = _cv(ttft_values)
        stability = {
            "num_samples": len(stability_summaries),
            "throughput_values": throughput_values,
            "ttft_values": ttft_values,
            "throughput_cv": throughput_cv,
            "ttft_cv": ttft_cv,
        }

        if max_throughput_cv is not None and throughput_cv > max_throughput_cv:
            failures.append(
                "stability gate failed: "
                f"throughput_cv={throughput_cv:.6f} > max={max_throughput_cv:.6f}"
            )
        if max_ttft_cv is not None and ttft_cv > max_ttft_cv:
            failures.append(
                "stability gate failed: " f"ttft_cv={ttft_cv:.6f} > max={max_ttft_cv:.6f}"
            )

    report = {
        "gate_passed": len(failures) == 0,
        "gate_failures": failures,
        "stability": stability,
    }
    return len(failures) == 0, failures, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-SGLang release gate evaluator")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON release gate config")
    parser.add_argument("--perf-summary", type=Path, default=None, help="Benchmark JSON payload path")
    parser.add_argument("--perf-gate", type=Path, default=None, help="Perf gate config (JSON/YAML)")
    parser.add_argument("--parity-json", type=Path, default=None, help="Token parity JSON payload path")
    parser.add_argument("--shadow-jsonl", type=Path, default=None, help="Shadow divergence report JSONL")
    parser.add_argument(
        "--max-shadow-divergences",
        type=int,
        default=None,
        help="Max allowed shadow divergence entries",
    )
    parser.add_argument(
        "--stability-summaries",
        nargs="*",
        type=Path,
        default=None,
        help="List of benchmark JSON payloads for CV stability checks",
    )
    parser.add_argument("--max-throughput-cv", type=float, default=None)
    parser.add_argument("--max-ttft-cv", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON report path")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_config(args.config)

    perf_summary_path = args.perf_summary or (
        Path(cfg["perf_summary"]) if cfg.get("perf_summary") else None
    )
    perf_gate_path = args.perf_gate or (Path(cfg["perf_gate"]) if cfg.get("perf_gate") else None)
    parity_json_path = args.parity_json or (
        Path(cfg["parity_json"]) if cfg.get("parity_json") else None
    )
    shadow_jsonl_path = args.shadow_jsonl or (
        Path(cfg["shadow_jsonl"]) if cfg.get("shadow_jsonl") else None
    )
    max_shadow_divergences = (
        args.max_shadow_divergences
        if args.max_shadow_divergences is not None
        else cfg.get("max_shadow_divergences")
    )
    stability_summary_paths = args.stability_summaries
    if stability_summary_paths is None and cfg.get("stability_summaries"):
        stability_summary_paths = [Path(p) for p in cfg["stability_summaries"]]
    max_throughput_cv = (
        args.max_throughput_cv if args.max_throughput_cv is not None else cfg.get("max_throughput_cv")
    )
    max_ttft_cv = args.max_ttft_cv if args.max_ttft_cv is not None else cfg.get("max_ttft_cv")
    out_path = args.out or (Path(cfg["out"]) if cfg.get("out") else None)

    if perf_summary_path is None:
        raise SystemExit("perf summary path must be provided via --perf-summary or config")

    perf_payload = _load_json(perf_summary_path)
    perf_summary = _summary_from_payload(perf_payload)
    perf_gate = _load_gate(perf_gate_path) if perf_gate_path is not None else None

    parity_payload = _load_json(parity_json_path) if parity_json_path is not None else None
    shadow_entries = _count_jsonl_entries(shadow_jsonl_path) if shadow_jsonl_path is not None else None

    stability_summaries: List[BenchmarkSummary] = []
    if stability_summary_paths:
        stability_summaries = [_summary_from_payload(_load_json(path)) for path in stability_summary_paths]

    passed, failures, report = evaluate_release_gate(
        perf_summary=perf_summary,
        perf_gate=perf_gate,
        parity_payload=parity_payload,
        shadow_divergence_entries=shadow_entries,
        max_shadow_divergence_entries=max_shadow_divergences,
        stability_summaries=stability_summaries,
        max_throughput_cv=max_throughput_cv,
        max_ttft_cv=max_ttft_cv,
    )
    report["inputs"] = {
        "config": str(args.config) if args.config else None,
        "perf_summary": str(perf_summary_path),
        "perf_gate": str(perf_gate_path) if perf_gate_path else None,
        "parity_json": str(parity_json_path) if parity_json_path else None,
        "shadow_jsonl": str(shadow_jsonl_path) if shadow_jsonl_path else None,
        "max_shadow_divergences": max_shadow_divergences,
        "stability_summaries": [str(p) for p in stability_summary_paths or []],
        "max_throughput_cv": max_throughput_cv,
        "max_ttft_cv": max_ttft_cv,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not passed:
        lines = "\n".join(f"- {f}" for f in failures)
        raise SystemExit(f"Release gate failed:\n{lines}")


if __name__ == "__main__":
    main()
