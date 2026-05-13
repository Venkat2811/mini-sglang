#!/usr/bin/env python3
"""Single-backend closure run wrapper.

Manages the full lifecycle for one mini-sglang server run:

- start `python -m minisgl` with the requested backend env vars
- poll `/v1/models` until ready or timeout
- issue chat-completion requests for every prompt in the JSONL file
- collect TTFT/TPOT/throughput summary using the existing benchmark client
- write the retained artifact JSON
- stop the server cleanly

This script does not modify any code on the hot path. It only orchestrates
the existing harness and captures the additional metadata fields that the
closure RFC requires.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _free_port_starting_at(start: int) -> int:
    """Pick the first available port at-or-above start. Used only when MASTER_PORT is not set."""
    port = start
    for _ in range(64):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"no free port near {start}")


def _wait_for_ready(base_url: str, timeout_s: float) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    last_err: str = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError as exc:
            last_err = str(exc)
        time.sleep(1.0)
    print(f"server not ready in {timeout_s}s; last error: {last_err}", file=sys.stderr)
    return False


def _gpu_snapshot() -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "device_name_0": torch.cuda.get_device_name(0),
                "driver_version_via_nvml": _try_nvml_driver(),
            }
    except Exception:
        pass
    return {"cuda_available": False}


def _try_nvml_driver() -> str:
    try:
        # Avoid hard dep on pynvml; parse nvidia-smi as a fallback.
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "unknown"


def _machine_snapshot() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_snapshot(),
    }


def _load_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("prompt")
            if isinstance(p, str) and p:
                prompts.append(p)
    return prompts


async def _run_one_chat(client, prompt: str, model: str, max_tokens: int) -> Tuple[float, float, int]:
    """Issue one non-streaming chat completion.

    Returns (ttft_s, total_s, output_tokens). TTFT is approximated as
    total time when streaming is off; we capture it separately via the
    streaming variant below for better fidelity.
    """
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
    )
    t1 = time.perf_counter()
    out_tokens = 0
    if resp.usage and resp.usage.completion_tokens:
        out_tokens = int(resp.usage.completion_tokens)
    elif resp.choices:
        # crude fallback: split on whitespace
        out_tokens = max(1, len(resp.choices[0].message.content.split()))
    return (t1 - t0, t1 - t0, out_tokens)


async def _run_one_chat_streaming(client, prompt: str, model: str, max_tokens: int) -> Tuple[float, float, int]:
    """Streaming version with proper TTFT measurement."""
    t0 = time.perf_counter()
    out_tokens = 0
    ttft = 0.0
    first_chunk_seen = False
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    async for chunk in stream:
        if not first_chunk_seen and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - t0
            first_chunk_seen = True
        if chunk.choices and chunk.choices[0].delta.content:
            out_tokens += 1
    t1 = time.perf_counter()
    return (ttft if ttft > 0 else (t1 - t0), t1 - t0, out_tokens)


async def _run_batch(
    base_url: str, model: str, prompts: List[str], max_tokens: int, streaming: bool, concurrency: int
) -> Dict[str, Any]:
    from openai import AsyncOpenAI

    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[float, float, int]] = []

    async with AsyncOpenAI(base_url=f"{base_url}/v1", api_key="") as client:

        async def _bounded(p: str):
            async with sem:
                fn = _run_one_chat_streaming if streaming else _run_one_chat
                return await fn(client, p, model, max_tokens)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(_bounded(p)) for p in prompts]
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
        t_total = time.perf_counter() - t0

    ttfts = sorted(r[0] for r in results)
    e2es = sorted(r[1] for r in results)
    tokens = [r[2] for r in results]
    total_out = sum(tokens)

    def _q(xs: List[float], frac: float) -> float:
        if not xs:
            return 0.0
        idx = min(int(len(xs) * frac), len(xs) - 1)
        return xs[idx]

    return {
        "num_requests": len(prompts),
        "num_tokens": total_out,
        "duration_s": t_total,
        "throughput_token_per_s": (total_out / t_total) if t_total > 0 else 0.0,
        "throughput_req_per_s": (len(prompts) / t_total) if t_total > 0 else 0.0,
        "ttft_ms": {
            "avg": 1000.0 * sum(ttfts) / max(1, len(ttfts)),
            "p50": 1000.0 * _q(ttfts, 0.5),
            "p90": 1000.0 * _q(ttfts, 0.9),
            "p99": 1000.0 * _q(ttfts, 0.99),
            "max": 1000.0 * (ttfts[-1] if ttfts else 0.0),
        },
        "e2e_s": {
            "avg": sum(e2es) / max(1, len(e2es)),
            "p50": _q(e2es, 0.5),
            "p90": _q(e2es, 0.9),
            "p99": _q(e2es, 0.99),
            "max": e2es[-1] if e2es else 0.0,
        },
        "streaming": streaming,
        "concurrency": concurrency,
    }


def _start_server(
    model_path: str,
    host: str,
    port: int,
    backend: str,
    log_path: Path,
    extra_env: Dict[str, str],
    cuda_graph_max_bs: int,
    memory_ratio: float,
    max_running_requests: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["MINISGL_CPU_BACKEND"] = backend
    env.update(extra_env)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", buffering=1)

    cmd = [
        sys.executable,
        "-m",
        "minisgl",
        "--model-path",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--cuda-graph-max-bs",
        str(cuda_graph_max_bs),
        "--memory-ratio",
        str(memory_ratio),
        "--max-running-requests",
        str(max_running_requests),
    ]
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so we can kill the whole group
    )


def _stop_server(proc: subprocess.Popen, timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10.0)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["python", "rust_hotpath", "rust_inprocess_ffi", "python_cpu"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True, type=Path, help="JSONL produced by sharegpt_prep.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-prompts", type=int, default=None, help="Cap; if omitted, use all prompts in the file")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64)
    parser.add_argument("--memory-ratio", type=float, default=0.8)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-log", type=Path, default=None)
    args = parser.parse_args()

    if not args.prompts.exists():
        print(f"prompts file does not exist: {args.prompts}", file=sys.stderr)
        return 2

    prompts = _load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        print("no prompts loaded", file=sys.stderr)
        return 2

    base_url = f"http://{args.host}:{args.port}"
    server_log = args.server_log or args.out.with_suffix(".server.log")

    extra_env: Dict[str, str] = {}
    master_port = os.environ.get("MASTER_PORT")
    if master_port:
        extra_env["MASTER_PORT"] = master_port

    print(f"[closure] starting server: backend={args.backend} port={args.port} master_port={master_port}")
    proc = _start_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        backend=args.backend,
        log_path=server_log,
        extra_env=extra_env,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        memory_ratio=args.memory_ratio,
        max_running_requests=args.max_running_requests,
    )

    try:
        if not _wait_for_ready(base_url, args.ready_timeout_s):
            print(f"[closure] server never became ready; see {server_log}", file=sys.stderr)
            _stop_server(proc)
            return 3

        print(f"[closure] running {len(prompts)} prompts at concurrency={args.concurrency}, streaming={args.streaming}")
        summary = asyncio.run(
            _run_batch(
                base_url=base_url,
                model=args.model_path,
                prompts=prompts,
                max_tokens=args.max_tokens,
                streaming=args.streaming,
                concurrency=args.concurrency,
            )
        )
    finally:
        print("[closure] stopping server")
        _stop_server(proc)

    payload = {
        "schema_version": 1,
        "kind": "closure_online_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_path": args.model_path,
        "run_index": args.run_index,
        "master_port": master_port,
        "base_url": base_url,
        "machine_snapshot": _machine_snapshot(),
        "profile": {
            "prompts_path": str(args.prompts),
            "prompts_count": len(prompts),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "streaming": args.streaming,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "memory_ratio": args.memory_ratio,
            "max_running_requests": args.max_running_requests,
            "typed_transport": os.environ.get("MINISGL_TYPED_TRANSPORT", "1"),
            "runtime_metrics": os.environ.get("MINISGL_RUNTIME_METRICS", "1"),
            "transport_latency_stats": os.environ.get("MINISGL_TRANSPORT_LATENCY_STATS", "0"),
            "shadow_enabled": os.environ.get("MINISGL_CPU_BACKEND_SHADOW", "0"),
            "shadow_report_path": os.environ.get("MINISGL_CPU_BACKEND_SHADOW_REPORT", ""),
        },
        "summary": summary,
        "server_log": str(server_log),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[closure] wrote artifact {args.out}")
    print(
        f"[closure] summary: tok/s={summary['throughput_token_per_s']:.2f} "
        f"req/s={summary['throughput_req_per_s']:.2f} "
        f"ttft_avg_ms={summary['ttft_ms']['avg']:.2f} "
        f"e2e_avg_s={summary['e2e_s']['avg']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
