from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM


@dataclass(frozen=True)
class SetResult:
    name: str
    backend_a: str
    backend_b: str
    match: bool
    mismatch_count: int
    signature_a: str
    signature_b: str
    duration_a_s: float
    duration_b_s: float
    first_mismatch: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "match": self.match,
            "mismatch_count": self.mismatch_count,
            "signature_a": self.signature_a,
            "signature_b": self.signature_b,
            "duration_a_s": self.duration_a_s,
            "duration_b_s": self.duration_b_s,
            "first_mismatch": self.first_mismatch,
        }


def _default_output() -> Path:
    return Path("0_venkat-worklog/baselines/latest-token-parity.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _signature(token_lists: list[list[int]]) -> str:
    h = hashlib.sha256()
    for ids in token_lists:
        h.update(len(ids).to_bytes(4, byteorder="little", signed=False))
        for token in ids:
            h.update(int(token).to_bytes(4, byteorder="little", signed=True))
    return h.hexdigest()


def _run_worker(args: argparse.Namespace) -> None:
    if args.worker_input is None or args.worker_output is None:
        raise SystemExit("Both --worker-input and --worker-output are required in worker mode")
    payload = json.loads(args.worker_input.read_text(encoding="utf-8"))
    backend = payload["backend"]
    text_prompts = payload["text_prompts"]
    token_prompts = payload["token_prompts"]
    max_tokens = int(payload["max_tokens"])
    master_port = int(payload["master_port"])

    ENV.CPU_BACKEND.value = backend
    os.environ["MASTER_PORT"] = str(master_port)
    llm = LLM(
        payload["model_path"],
        dtype=torch.float16,
        max_seq_len_override=payload["max_seq_len_override"],
        max_extend_tokens=payload["max_extend_tokens"],
        cuda_graph_max_bs=payload["cuda_graph_max_bs"],
        memory_ratio=payload["memory_ratio"],
        num_page_override=payload["num_page_override"],
    )
    try:
        text_params = [SamplingParams(temperature=0.0, top_k=1, top_p=1.0, max_tokens=max_tokens) for _ in text_prompts]
        t0 = time.perf_counter()
        text_outputs = llm.generate(text_prompts, text_params)
        text_elapsed = time.perf_counter() - t0

        token_params = [
            SamplingParams(temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True, max_tokens=max_tokens)
            for _ in token_prompts
        ]
        t0 = time.perf_counter()
        token_outputs = llm.generate(token_prompts, token_params)
        token_elapsed = time.perf_counter() - t0
    finally:
        llm.shutdown()

    _write_json(
        args.worker_output,
        {
            "backend": backend,
            "text_duration_s": text_elapsed,
            "token_duration_s": token_elapsed,
            "text_token_ids": [list(out["token_ids"]) for out in text_outputs],
            "token_token_ids": [list(out["token_ids"]) for out in token_outputs],
        },
    )


def _run_backend(
    backend: str,
    text_prompts: list[str],
    token_prompts: list[list[int]],
    args: argparse.Namespace,
    master_port: int,
) -> tuple[list[list[int]], float, list[list[int]], float]:
    with tempfile.TemporaryDirectory(prefix="minisgl-token-parity-") as td:
        tmp_dir = Path(td)
        worker_input = tmp_dir / "worker-input.json"
        worker_output = tmp_dir / "worker-output.json"
        _write_json(
            worker_input,
            {
                "backend": backend,
                "master_port": master_port,
                "model_path": args.model_path,
                "max_tokens": args.max_tokens,
                "max_seq_len_override": args.max_seq_len_override,
                "max_extend_tokens": args.max_extend_tokens,
                "cuda_graph_max_bs": args.cuda_graph_max_bs,
                "memory_ratio": args.memory_ratio,
                "num_page_override": args.num_page_override,
                "text_prompts": text_prompts,
                "token_prompts": token_prompts,
            },
        )
        cmd = [
            sys.executable,
            "-m",
            "minisgl.benchmark.token_parity",
            "--worker-input",
            str(worker_input),
            "--worker-output",
            str(worker_output),
        ]
        subprocess.run(cmd, check=True)
        worker_result = json.loads(worker_output.read_text(encoding="utf-8"))
    return (
        worker_result["text_token_ids"],
        float(worker_result["text_duration_s"]),
        worker_result["token_token_ids"],
        float(worker_result["token_duration_s"]),
    )


def _compare_sets(
    name: str,
    backend_a: str,
    backend_b: str,
    out_a: list[list[int]],
    out_b: list[list[int]],
    dur_a: float,
    dur_b: float,
) -> SetResult:
    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    for idx, (a, b) in enumerate(zip(out_a, out_b)):
        if a != b:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "index": idx,
                    "len_a": len(a),
                    "len_b": len(b),
                    "preview_a": a[:16],
                    "preview_b": b[:16],
                }
    return SetResult(
        name=name,
        backend_a=backend_a,
        backend_b=backend_b,
        match=mismatch_count == 0,
        mismatch_count=mismatch_count,
        signature_a=_signature(out_a),
        signature_b=_signature(out_b),
        duration_a_s=dur_a,
        duration_b_s=dur_b,
        first_mismatch=first_mismatch,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    text_prompts = [
        "Summarize why batching improves GPU utilization in one sentence.",
        "List three reasons deterministic decoding helps parity checks.",
        "Explain CPU scheduler overhead in one concise paragraph.",
        "Write one short line about KV cache locality.",
    ]

    token_prompts: list[list[int]] = []
    for _ in range(args.token_prompt_count):
        in_len = rng.randint(args.min_input_len, args.max_input_len)
        token_prompts.append([rng.randint(100, 20000) for _ in range(in_len)])
    text_a, text_a_dur, token_a, token_a_dur = _run_backend(
        args.backend_a, text_prompts, token_prompts, args, master_port=args.master_port
    )
    text_b, text_b_dur, token_b, token_b_dur = _run_backend(
        args.backend_b, text_prompts, token_prompts, args, master_port=args.master_port + 1
    )

    sets = [
        _compare_sets(
            "text_prompts",
            args.backend_a,
            args.backend_b,
            text_a,
            text_b,
            text_a_dur,
            text_b_dur,
        ),
        _compare_sets(
            "token_prompts",
            args.backend_a,
            args.backend_b,
            token_a,
            token_b,
            token_a_dur,
            token_b_dur,
        ),
    ]

    parity_passed = all(item.match for item in sets)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "seed": args.seed,
        "backend_a": args.backend_a,
        "backend_b": args.backend_b,
        "profile": {
            "max_tokens": args.max_tokens,
            "token_prompt_count": args.token_prompt_count,
            "min_input_len": args.min_input_len,
            "max_input_len": args.max_input_len,
            "max_seq_len_override": args.max_seq_len_override,
            "max_extend_tokens": args.max_extend_tokens,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "memory_ratio": args.memory_ratio,
            "num_page_override": args.num_page_override,
        },
        "sets": [item.to_dict() for item in sets],
        "parity_passed": parity_passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic token parity check across CPU backends")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--backend-a", type=str, default="python")
    parser.add_argument("--backend-b", type=str, default="rust_hotpath")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--token-prompt-count", type=int, default=8)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-seq-len-override", type=int, default=2048)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--num-page-override", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=2360)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_input is not None or args.worker_output is not None:
        _run_worker(args)
        return
    out = args.out or _default_output()
    payload = run(args)
    _write_json(out, payload)
    print(f"wrote={out}")
    print(f"parity_passed={payload['parity_passed']}")
    for item in payload["sets"]:
        print(
            f"set={item['name']} match={item['match']} mismatches={item['mismatch_count']} "
            f"sig_a={item['signature_a'][:12]} sig_b={item['signature_b'][:12]}"
        )
    if not payload["parity_passed"]:
        raise SystemExit("token parity check failed")


if __name__ == "__main__":
    main()
