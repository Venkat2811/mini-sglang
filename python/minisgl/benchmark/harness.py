from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from minisgl.benchmark.client import benchmark_one, benchmark_one_batch, generate_prompt, get_model_name
from minisgl.benchmark.metrics import BenchmarkGate, BenchmarkSummary, evaluate_gates, summary_lines
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)


def _default_output(mode: str) -> Path:
    return Path("0_venkat-worklog/baselines") / f"latest-{mode}.json"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_gate(path: Path | None) -> BenchmarkGate | None:
    if path is None:
        return None
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkGate.from_dict(data)


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@dataclass(frozen=True)
class OfflineDetails:
    init_time_s: float
    warmup_time_s: float
    warmup_tokens: int
    single_req_time_s: float
    single_out_tokens: int
    batch_time_s: float
    batch_out_tokens: int


def run_offline(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    dtype = _dtype_from_name(args.dtype)

    t0 = time.perf_counter()
    llm = LLM(
        args.model_path,
        dtype=dtype,
        max_seq_len_override=args.max_seq_len_override,
        max_extend_tokens=args.max_extend_tokens,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        memory_ratio=args.memory_ratio,
        num_page_override=args.num_page_override,
    )
    init_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    warm = llm.generate(
        [args.warmup_prompt],
        SamplingParams(temperature=0.0, top_k=1, max_tokens=args.warmup_tokens),
    )[0]
    warmup_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    single = llm.generate(
        [args.single_prompt],
        SamplingParams(temperature=0.0, top_k=1, max_tokens=args.single_output_tokens),
    )[0]
    single_req_time_s = time.perf_counter() - t0

    prompts: List[List[int]] = []
    params: List[SamplingParams] = []
    for _ in range(args.batch_size):
        in_len = random.randint(args.min_input_len, args.max_input_len)
        prompts.append([random.randint(100, 20000) for _ in range(in_len)])
        params.append(
            SamplingParams(
                temperature=0.0,
                top_k=1,
                ignore_eos=True,
                max_tokens=args.batch_output_tokens,
            )
        )

    t0 = time.perf_counter()
    batch_results = llm.generate(prompts, params)
    batch_time_s = time.perf_counter() - t0
    batch_out_tokens = sum(len(r["token_ids"]) for r in batch_results)

    summary = BenchmarkSummary.from_throughput(
        num_requests=args.batch_size,
        num_tokens=batch_out_tokens,
        duration_s=batch_time_s,
    )

    details = OfflineDetails(
        init_time_s=init_time_s,
        warmup_time_s=warmup_time_s,
        warmup_tokens=len(warm["token_ids"]),
        single_req_time_s=single_req_time_s,
        single_out_tokens=len(single["token_ids"]),
        batch_time_s=batch_time_s,
        batch_out_tokens=batch_out_tokens,
    )

    for line in summary_lines(summary):
        logger.info("[offline] %s", line)
    logger.info("[offline] init_time_s=%.4f warmup_time_s=%.4f", init_time_s, warmup_time_s)

    return {
        "mode": "offline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "profile": {
            "dtype": args.dtype,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "min_input_len": args.min_input_len,
            "max_input_len": args.max_input_len,
            "batch_output_tokens": args.batch_output_tokens,
            "max_seq_len_override": args.max_seq_len_override,
            "max_extend_tokens": args.max_extend_tokens,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "memory_ratio": args.memory_ratio,
            "num_page_override": args.num_page_override,
        },
        "summary": summary.to_dict(),
        "details": asdict(details),
    }


async def _run_online_async(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    async with OpenAI(base_url=f"{args.base_url}/v1", api_key="") as client:
        model = args.model_path or await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)

        warmup_prompt = generate_prompt(tokenizer, args.warmup_input_tokens)
        t0 = time.perf_counter()
        warm = await benchmark_one(
            client,
            warmup_prompt,
            args.warmup_output_tokens,
            model,
            pbar=False,
        )
        warmup_elapsed_s = time.perf_counter() - t0

        prompts = [
            generate_prompt(tokenizer, random.randint(args.min_input_len, args.max_input_len))
            for _ in range(args.batch_size)
        ]
        output_lengths = [args.batch_output_tokens] * args.batch_size

        t0 = time.perf_counter()
        raw = await benchmark_one_batch(
            client=client,
            prompts=prompts,
            output_lengths=output_lengths,
            model=model,
            pbar=False,
        )
        bench_wall_time_s = time.perf_counter() - t0

        summary = BenchmarkSummary.from_raw_results(raw)
        for line in summary_lines(summary):
            logger.info("[online] %s", line)
        logger.info(
            "[online] warmup_elapsed_s=%.4f warmup_tokens=%d",
            warmup_elapsed_s,
            len(warm.tics) - 1,
        )

        return {
            "mode": "online",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model_path": model,
            "profile": {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "min_input_len": args.min_input_len,
                "max_input_len": args.max_input_len,
                "batch_output_tokens": args.batch_output_tokens,
                "warmup_input_tokens": args.warmup_input_tokens,
                "warmup_output_tokens": args.warmup_output_tokens,
            },
            "summary": summary.to_dict(),
            "details": {
                "warmup_elapsed_s": warmup_elapsed_s,
                "warmup_tokens": len(warm.tics) - 1,
                "bench_wall_time_s": bench_wall_time_s,
            },
        }


def run_online(args: argparse.Namespace) -> Dict[str, Any]:
    return asyncio.run(_run_online_async(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini-SGLang baseline harness")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=None, help="Output JSON path")
    common.add_argument(
        "--gate",
        type=Path,
        default=None,
        help="Optional gate config (JSON/YAML) with BenchmarkGate fields",
    )

    p_offline = sub.add_parser("offline", parents=[common], help="Run offline baseline")
    p_offline.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p_offline.add_argument("--dtype", type=str, default="float16")
    p_offline.add_argument("--seed", type=int, default=42)
    p_offline.add_argument("--max-seq-len-override", type=int, default=2048)
    p_offline.add_argument("--max-extend-tokens", type=int, default=4096)
    p_offline.add_argument("--cuda-graph-max-bs", type=int, default=64)
    p_offline.add_argument("--memory-ratio", type=float, default=0.9)
    p_offline.add_argument("--num-page-override", type=int, default=None)
    p_offline.add_argument("--warmup-prompt", type=str, default="Hello")
    p_offline.add_argument("--warmup-tokens", type=int, default=16)
    p_offline.add_argument(
        "--single-prompt",
        type=str,
        default="Write one short sentence explaining why batching helps GPU utilization.",
    )
    p_offline.add_argument("--single-output-tokens", type=int, default=64)
    p_offline.add_argument("--batch-size", type=int, default=16)
    p_offline.add_argument("--min-input-len", type=int, default=64)
    p_offline.add_argument("--max-input-len", type=int, default=256)
    p_offline.add_argument("--batch-output-tokens", type=int, default=64)

    p_online = sub.add_parser("online", parents=[common], help="Run online baseline")
    p_online.add_argument("--base-url", type=str, default="http://127.0.0.1:1919")
    p_online.add_argument("--model-path", type=str, default=None)
    p_online.add_argument("--seed", type=int, default=123)
    p_online.add_argument("--batch-size", type=int, default=8)
    p_online.add_argument("--min-input-len", type=int, default=64)
    p_online.add_argument("--max-input-len", type=int, default=256)
    p_online.add_argument("--batch-output-tokens", type=int, default=64)
    p_online.add_argument("--warmup-input-tokens", type=int, default=32)
    p_online.add_argument("--warmup-output-tokens", type=int, default=16)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    out = args.out or _default_output(args.mode)

    payload = run_offline(args) if args.mode == "offline" else run_online(args)
    summary = BenchmarkSummary.from_dict(payload["summary"])

    gate = _load_gate(args.gate)
    if gate is not None:
        payload["gate"] = gate.to_dict()
        failures = evaluate_gates(summary, gate)
        payload["gate_failures"] = failures
        payload["gate_passed"] = len(failures) == 0
    else:
        payload["gate"] = None
        payload["gate_failures"] = []
        payload["gate_passed"] = None

    _write_json(out, payload)
    logger.info("Wrote benchmark summary to %s", out)
    if payload["gate_passed"] is False:
        lines = "\n".join(f"- {msg}" for msg in payload["gate_failures"])
        raise SystemExit(f"Benchmark gate check failed:\n{lines}")


if __name__ == "__main__":
    main()
