#!/usr/bin/env python3
"""Bounded ShareGPT subset preparation for the mini-sglang closure run.

Produces a JSONL file with one record per line:

    {"id": 0, "prompt": "<first user turn>"}

The corpus source is configurable. By default the script will try the
HuggingFace dataset `philschmid/sharegpt-raw` (used by similar Myelon
benchmark slices) and fall back to a local JSON file passed via `--input`.

This script intentionally does no chat-template rendering. The
mini-sglang server applies the model's chat template at request time
via `/v1/chat/completions`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional


def _load_local_json(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON shape in {path}: expected list")


def _load_hf_dataset(name: str) -> List[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            f"datasets package not installed; pip install datasets ({exc!r})"
        )
    ds = load_dataset(name, split="train")
    return [dict(item) for item in ds]


def _extract_first_user_turn(record: dict) -> Optional[str]:
    """Return the first user turn from a ShareGPT-style record.

    Handles two common shapes:
    - `{"conversations": [{"from": "human", "value": "..."}, ...]}`
    - `{"messages": [{"role": "user", "content": "..."}, ...]}`
    """
    convs = record.get("conversations")
    if isinstance(convs, list):
        for turn in convs:
            who = (turn.get("from") or turn.get("role") or "").lower()
            text = turn.get("value") or turn.get("content") or ""
            if who in {"human", "user"} and isinstance(text, str) and text.strip():
                return text.strip()
        return None
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            who = (msg.get("role") or "").lower()
            text = msg.get("content") or ""
            if who == "user" and isinstance(text, str) and text.strip():
                return text.strip()
        return None
    return None


def _filter_by_length(prompts: Iterable[str], min_chars: int, max_chars: int) -> List[str]:
    out: List[str] = []
    for p in prompts:
        n = len(p)
        if min_chars <= n <= max_chars:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=200, help="Number of prompts to retain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=200, help="Minimum prompt length in characters")
    parser.add_argument("--max-chars", type=int, default=4000, help="Maximum prompt length in characters")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local JSON or JSONL with ShareGPT-style records. If omitted, the script attempts to load --hf-dataset.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="philschmid/sharegpt-raw",
        help="HuggingFace dataset id to use when --input is not provided.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.input is not None:
        if not args.input.exists():
            raise SystemExit(f"input file does not exist: {args.input}")
        records = _load_local_json(args.input)
    else:
        records = _load_hf_dataset(args.hf_dataset)

    first_turns: List[str] = []
    for rec in records:
        p = _extract_first_user_turn(rec)
        if p:
            first_turns.append(p)

    filtered = _filter_by_length(first_turns, args.min_chars, args.max_chars)
    if len(filtered) < args.count:
        raise SystemExit(
            f"only {len(filtered)} prompts pass length filter [{args.min_chars}, {args.max_chars}]; "
            f"requested {args.count}. Try widening the filter."
        )

    rng.shuffle(filtered)
    sample = filtered[: args.count]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(sample):
            f.write(json.dumps({"id": idx, "prompt": prompt}, ensure_ascii=False) + "\n")

    print(f"wrote {args.count} prompts to {args.out}")
    print(f"source: {args.input if args.input else args.hf_dataset}")
    print(f"seed: {args.seed}; length filter: [{args.min_chars}, {args.max_chars}]")


if __name__ == "__main__":
    sys.exit(main())
