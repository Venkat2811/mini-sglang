from __future__ import annotations

import argparse
import json
from pathlib import Path

from minisgl.benchmark import token_parity


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        backend_a="python",
        backend_b="rust_hotpath",
        seed=123,
        max_tokens=8,
        token_prompt_count=2,
        min_input_len=4,
        max_input_len=4,
        shared_prefix_len=0,
        max_seq_len_override=512,
        max_extend_tokens=1024,
        cuda_graph_max_bs=1,
        memory_ratio=0.8,
        num_page_override=None,
        master_port=2450,
    )


def test_run_backend_uses_worker_subprocess(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        assert "--worker-input" in cmd
        assert "--worker-output" in cmd
        worker_input = Path(cmd[cmd.index("--worker-input") + 1])
        worker_output = Path(cmd[cmd.index("--worker-output") + 1])
        payload = json.loads(worker_input.read_text(encoding="utf-8"))
        captured["payload"] = payload
        worker_output.write_text(
            json.dumps(
                {
                    "backend": payload["backend"],
                    "text_duration_s": 0.1,
                    "token_duration_s": 0.2,
                    "text_token_ids": [[11, 12]],
                    "token_token_ids": [[21, 22]],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(token_parity.subprocess, "run", _fake_run)

    args = _args()
    text_ids, text_dur, token_ids, token_dur = token_parity._run_backend(
        backend="python",
        text_prompts=["hello"],
        token_prompts=[[100, 101]],
        args=args,
        master_port=2500,
    )

    assert text_ids == [[11, 12]]
    assert token_ids == [[21, 22]]
    assert text_dur == 0.1
    assert token_dur == 0.2
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["backend"] == "python"
    assert payload["master_port"] == 2500
    assert payload["text_prompts"] == ["hello"]
    assert payload["token_prompts"] == [[100, 101]]


def test_run_reports_parity_when_backends_match(monkeypatch):
    def _fake_run_backend(
        backend: str,
        text_prompts: list[str],
        token_prompts: list[list[int]],
        args: argparse.Namespace,
        master_port: int,
    ):
        assert len(text_prompts) == 4
        assert len(token_prompts) == args.token_prompt_count
        assert master_port in {args.master_port, args.master_port + 1}
        _ = backend
        return [[1, 2], [3]], 0.05, [[4], [5]], 0.06

    monkeypatch.setattr(token_parity, "_run_backend", _fake_run_backend)

    payload = token_parity.run(_args())
    assert payload["parity_passed"] is True
    assert [s["mismatch_count"] for s in payload["sets"]] == [0, 0]


def test_run_reports_mismatch_when_backends_diverge(monkeypatch):
    def _fake_run_backend(
        backend: str,
        text_prompts: list[str],
        token_prompts: list[list[int]],
        args: argparse.Namespace,
        master_port: int,
    ):
        assert len(text_prompts) == 4
        assert len(token_prompts) == args.token_prompt_count
        assert master_port in {args.master_port, args.master_port + 1}
        if backend == "python":
            return [[1, 2], [3]], 0.05, [[4], [5]], 0.06
        return [[1, 2], [3]], 0.05, [[4], [6]], 0.06

    monkeypatch.setattr(token_parity, "_run_backend", _fake_run_backend)

    payload = token_parity.run(_args())
    assert payload["parity_passed"] is False
    token_set = next(item for item in payload["sets"] if item["name"] == "token_prompts")
    assert token_set["match"] is False
    assert token_set["mismatch_count"] == 1
    assert token_set["first_mismatch"]["index"] == 1


def test_run_builds_shared_prefix_prompts(monkeypatch):
    seen_prompts: list[list[list[int]]] = []

    def _fake_run_backend(
        backend: str,
        text_prompts: list[str],
        token_prompts: list[list[int]],
        args: argparse.Namespace,
        master_port: int,
    ):
        _ = backend, text_prompts, args, master_port
        seen_prompts.append(token_prompts)
        return [[1], [2]], 0.01, [[3], [4]], 0.01

    args = _args()
    args.token_prompt_count = 3
    args.min_input_len = 8
    args.max_input_len = 8
    args.shared_prefix_len = 4
    monkeypatch.setattr(token_parity, "_run_backend", _fake_run_backend)

    payload = token_parity.run(args)
    assert payload["parity_passed"] is True
    assert len(seen_prompts) == 2
    token_prompts = seen_prompts[0]
    assert len(token_prompts) == 3
    first_prefix = token_prompts[0][:4]
    assert token_prompts[1][:4] == first_prefix
    assert token_prompts[2][:4] == first_prefix
