from __future__ import annotations

import sys
import types

import torch

from minisgl.core import SamplingParams
from minisgl.message import DetokenizeMsg, TokenizeMsg
from minisgl.tokenizer.rust_backend import RustTokenizerManagers


class _FakeWorker:
    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer_path = tokenizer_path
        self.last_prompts = None
        self.last_detokenize = None

    def tokenize(self, prompts):
        self.last_prompts = prompts
        t0 = torch.tensor([11, 12, 13], dtype=torch.int32)
        t1 = torch.tensor([21, 22], dtype=torch.int32)
        return [bytearray(t0.numpy().tobytes()), bytearray(t1.numpy().tobytes())]

    def detokenize(self, uids, next_tokens, finished):
        self.last_detokenize = (uids, next_tokens, finished)
        return [f"{uid}:{tok}:{int(done)}" for uid, tok, done in zip(uids, next_tokens, finished, strict=True)]


def test_rust_tokenizer_manager_tokenize_converts_i32_buffers(monkeypatch):
    holder = {}

    def _ctor(tokenizer_path: str):
        worker = _FakeWorker(tokenizer_path)
        holder["worker"] = worker
        return worker

    monkeypatch.setitem(sys.modules, "minisgl_cpu", types.SimpleNamespace(TokenizerWorker=_ctor))
    manager = RustTokenizerManagers("dummy-tokenizer-path")
    msgs = [
        TokenizeMsg(uid=1, text="hello", sampling_params=SamplingParams()),
        TokenizeMsg(
            uid=2,
            text=[{"role": "user", "content": "hi"}],
            sampling_params=SamplingParams(),
        ),
    ]

    out = manager.tokenize(msgs)
    assert [t.tolist() for t in out] == [[11, 12, 13], [21, 22]]
    assert holder["worker"].tokenizer_path == "dummy-tokenizer-path"
    assert holder["worker"].last_prompts == ["hello", [{"role": "user", "content": "hi"}]]


def test_rust_tokenizer_manager_detokenize_forwards_vectors(monkeypatch):
    holder = {}

    def _ctor(tokenizer_path: str):
        worker = _FakeWorker(tokenizer_path)
        holder["worker"] = worker
        return worker

    monkeypatch.setitem(sys.modules, "minisgl_cpu", types.SimpleNamespace(TokenizerWorker=_ctor))
    manager = RustTokenizerManagers("dummy")
    msgs = [
        DetokenizeMsg(uid=9, next_token=101, finished=False),
        DetokenizeMsg(uid=9, next_token=0, finished=True),
    ]

    out = manager.detokenize(msgs)
    assert out == ["9:101:0", "9:0:1"]
    assert holder["worker"].last_detokenize == ([9, 9], [101, 0], [False, True])
