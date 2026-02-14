from __future__ import annotations

from typing import List

import torch
from minisgl.message import DetokenizeMsg, TokenizeMsg


class RustTokenizerManagers:
    def __init__(self, tokenizer_path: str) -> None:
        import minisgl_cpu

        self._worker = minisgl_cpu.TokenizerWorker(tokenizer_path)

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        prompts = [msg.text for msg in msgs]
        buffers = self._worker.tokenize(prompts)
        out: List[torch.Tensor] = []
        for buffer in buffers:
            # Clone so tensor lifetime is decoupled from the Python buffer object.
            out.append(torch.frombuffer(memoryview(buffer), dtype=torch.int32).clone())
        return out

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        uids = [int(msg.uid) for msg in msgs]
        next_tokens = [int(msg.next_token) for msg in msgs]
        finished = [bool(msg.finished) for msg in msgs]
        return self._worker.detokenize(uids, next_tokens, finished)
