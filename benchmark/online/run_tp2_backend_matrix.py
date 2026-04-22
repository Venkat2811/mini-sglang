from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_MODELS = {
    "qwen3_06b": "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B",
    "qwen3_4b": "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B",
}


VARIANTS = [
    {"name": "gloo", "backend": "gloo", "transport": None},
    {"name": "glooext_uv", "backend": "glooext", "transport": "uv"},
    {"name": "glooext_myelon", "backend": "glooext", "transport": "myelon"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retained TP=2 Mini-SGLang backend matrix.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--model-key",
        action="append",
        choices=sorted(DEFAULT_MODELS.keys()),
        help="Model key to run. May be passed multiple times. Defaults to all retained models.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--cache-type", type=str, default="naive")
    parser.add_argument("--attention-backend", type=str, default="fi")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=Path("/root/Documents/myelon-launch/mini-sglang/benchmark/synthetic/c10d_glooext"),
    )
    return parser.parse_args()


def wait_for_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    model_url = f"{base_url}/v1/models"
    last_error = "not started"
    while time.time() < deadline:
        try:
            with urlopen(model_url, timeout=2.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last_error = f"http {resp.status}"
        except URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


def percent_delta(candidate: float, baseline: float) -> float:
    return ((candidate - baseline) / baseline) * 100.0


def resolve_model_snapshot(root: Path) -> Path:
    if (root / "config.json").exists():
        return root
    refs_main = root / "refs" / "main"
    snapshots = root / "snapshots"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        candidate = snapshots / revision
        if candidate.exists():
            return candidate
    if snapshots.exists():
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"no model snapshot found under {root}")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    raw_dir = artifact_dir / "raw"
    analysis_dir = artifact_dir / "analysis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "serving_matrix_summary.json"

    model_keys = args.model_key or list(DEFAULT_MODELS.keys())
    bench_script = Path(__file__).with_name("bench_tp2_serving.py")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {
            "host": args.host,
            "port": args.port,
            "tp_size": args.tp_size,
            "dtype": args.dtype,
            "cache_type": args.cache_type,
            "attention_backend": args.attention_backend,
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": args.seed,
            "models": {},
        }

    for model_key in model_keys:
        model_root = Path(DEFAULT_MODELS[model_key])
        if not model_root.exists():
            raise FileNotFoundError(f"model root missing for {model_key}: {model_root}")
        model_path = resolve_model_snapshot(model_root)

        model_results: dict[str, Any] = {
            "model_path": str(model_path),
            "variants": {},
        }

        for variant_index, variant in enumerate(VARIANTS):
            variant_name = variant["name"]
            prefix = f"{model_key}_{variant_name}"
            server_log = raw_dir / f"{prefix}_server.log"
            bench_json = raw_dir / f"{prefix}_bench.json"
            launch_json = raw_dir / f"{prefix}_launch.json"
            variant_port = args.port + variant_index * 10
            variant_base_url = f"http://{args.host}:{variant_port}"

            env = os.environ.copy()
            env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"
            if variant["backend"] == "glooext":
                env["MINISGL_C10D_GLOOEXT_EXTENSION_DIR"] = str(args.extension_dir.resolve())
                env["MINISGL_C10D_GLOOEXT_TRANSPORT"] = str(variant["transport"])

            cmd = [
                sys.executable,
                "-m",
                "minisgl",
                "--model",
                str(model_path),
                "--tp-size",
                str(args.tp_size),
                "--dtype",
                args.dtype,
                "--cache-type",
                args.cache_type,
                "--attention-backend",
                args.attention_backend,
                "--tp-cpu-backend",
                str(variant["backend"]),
                "--host",
                args.host,
                "--port",
                str(variant_port),
            ]
            if variant["transport"] is not None:
                cmd.extend(["--tp-cpu-transport", str(variant["transport"])])

            launch_json.write_text(
                json.dumps(
                    {
                        "model_key": model_key,
                        "variant": variant_name,
                        "base_url": variant_base_url,
                        "cmd": cmd,
                        "env_subset": {
                            key: env[key]
                            for key in sorted(env)
                            if key.startswith("MINISGL_")
                        },
                    },
                    indent=2,
                )
            )

            proc: subprocess.Popen[Any] | None = None
            started_at = time.time()
            failure: str | None = None
            try:
                with server_log.open("w") as log_fp:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        cwd=Path(__file__).resolve().parents[2],
                        env=env,
                        start_new_session=True,
                    )
                    wait_for_ready(variant_base_url, args.startup_timeout_s)
                    subprocess.run(
                        [
                            sys.executable,
                            str(bench_script),
                            "--base-url",
                            variant_base_url,
                            "--batch-size",
                            str(args.batch_size),
                            "--input-len",
                            str(args.input_len),
                            "--output-len",
                            str(args.output_len),
                            "--seed",
                            str(args.seed),
                            "--output-json",
                            str(bench_json),
                        ],
                        check=True,
                        cwd=Path(__file__).resolve().parents[2],
                        env=env,
                    )
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)
            finally:
                if proc is not None:
                    terminate_process(proc)

            elapsed = time.time() - started_at
            record: dict[str, Any] = {
                "server_log": str(server_log),
                "bench_json": str(bench_json),
                "launch_json": str(launch_json),
                "elapsed_s": elapsed,
                "failure": failure,
            }
            if bench_json.exists():
                record["bench"] = json.loads(bench_json.read_text())
            model_results["variants"][variant_name] = record

            if failure is not None:
                break

        baseline = model_results["variants"].get("gloo", {}).get("bench")
        if baseline is not None:
            baseline_rps = baseline["request_throughput_rps"]
            baseline_ttft = baseline["ttft_s"]["p50"]
            for variant_name, record in model_results["variants"].items():
                bench = record.get("bench")
                if bench is None:
                    continue
                record["delta_vs_gloo_pct"] = {
                    "request_throughput_rps": percent_delta(bench["request_throughput_rps"], baseline_rps),
                    "ttft_p50": percent_delta(baseline_ttft, bench["ttft_s"]["p50"]),
                }

        summary["models"][model_key] = model_results

    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
