from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from statistics import mean

from minisgl.benchmark.client import benchmark_one_batch, generate_prompt, get_model_name
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retained TP2 serving benchmark for Mini-SGLang.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:1919")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    async with OpenAI(base_url=f"{args.base_url}/v1", api_key="") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        prompts = [generate_prompt(tokenizer, args.input_len) for _ in range(args.batch_size)]
        output_lens = [args.output_len for _ in range(args.batch_size)]
        results = await benchmark_one_batch(client, prompts, output_lens, model, pbar=False)

    per_request = []
    total_output_tokens = 0
    total_e2e = 0.0
    ttfts = []
    tpots = []
    e2es = []
    decode_rates = []
    for item in results:
        ttft = item.tics[1] - item.tics[0]
        e2e = item.tics[-1] - item.tics[0]
        decode_tokens = max(item.output_len - 1, 1)
        tpot = (item.tics[-1] - item.tics[1]) / decode_tokens if item.output_len > 1 else 0.0
        decode_rate = item.output_len / e2e if e2e > 0 else 0.0
        total_output_tokens += item.output_len
        total_e2e += e2e
        ttfts.append(ttft)
        tpots.append(tpot)
        e2es.append(e2e)
        decode_rates.append(decode_rate)
        per_request.append(
            {
                "input_len": item.input_len,
                "output_len": item.output_len,
                "ttft_s": ttft,
                "e2e_s": e2e,
                "tpot_s": tpot,
                "decode_tok_per_s": decode_rate,
            }
        )

    summary = {
        "model": model,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "request_throughput_rps": len(results) / max(e2es),
        "output_token_throughput_tok_s": total_output_tokens / max(e2es),
        "ttft_s": {
            "mean": mean(ttfts),
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "e2e_s": {
            "mean": mean(e2es),
            "p50": percentile(e2es, 0.50),
            "p95": percentile(e2es, 0.95),
            "p99": percentile(e2es, 0.99),
        },
        "tpot_s": {
            "mean": mean(tpots),
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
            "p99": percentile(tpots, 0.99),
        },
        "decode_tok_per_s": {
            "mean": mean(decode_rates),
            "p50": percentile(decode_rates, 0.50),
            "p95": percentile(decode_rates, 0.95),
            "p99": percentile(decode_rates, 0.99),
        },
        "per_request": per_request,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
