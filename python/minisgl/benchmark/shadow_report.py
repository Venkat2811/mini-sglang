from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CPU backend shadow divergence logs")
    parser.add_argument("--input", type=Path, required=True, help="Path to shadow report JSONL file")
    parser.add_argument("--top", type=int, default=10, help="Number of top kinds/reasons to print")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Treat missing input file as zero divergence instead of an error",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path: Path = args.input
    if not path.exists():
        if args.allow_missing:
            print("divergence_entries=0")
            print("top_kinds:")
            print("top_reasons:")
            return
        raise SystemExit(f"input file does not exist: {path}")

    total = 0
    kind_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()

    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        entry = json.loads(raw)
        total += 1
        kind_counter[str(entry.get("kind", "unknown"))] += 1
        reason_counter[str(entry.get("reason", "unknown"))] += 1

    print(f"divergence_entries={total}")
    print("top_kinds:")
    for kind, count in kind_counter.most_common(args.top):
        print(f"  {kind}: {count}")
    print("top_reasons:")
    for reason, count in reason_counter.most_common(args.top):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
