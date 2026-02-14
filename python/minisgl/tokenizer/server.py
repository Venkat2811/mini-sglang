from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import List

import torch
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from transformers import AutoTokenizer, LlamaTokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def _resolve_rust_tokenizer_path(model_or_path: str, logger) -> str:
    path = Path(model_or_path)
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidate = path / "tokenizer.json"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"tokenizer.json not found under directory: {model_or_path}")

    from huggingface_hub import hf_hub_download

    tokenizer_json = hf_hub_download(repo_id=model_or_path, filename="tokenizer.json")
    try:
        hf_hub_download(repo_id=model_or_path, filename="tokenizer_config.json")
    except Exception as exc:
        logger.warning("Could not prefetch tokenizer_config.json for '%s': %s", model_or_path, exc)
    return tokenizer_json


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")
    backend_mode = str(ENV.TOKENIZER_BACKEND)

    if backend_mode in {"rust_inprocess", "rust_tokenize_only"}:
        try:
            from .rust_backend import RustTokenizerManagers

            rust_tokenizer_path = _resolve_rust_tokenizer_path(tokenizer_path, logger)
            shared_manager = RustTokenizerManagers(rust_tokenizer_path)
            tokenize_manager = shared_manager
            if backend_mode == "rust_inprocess":
                detokenize_manager = shared_manager
            else:
                from .detokenize import DetokenizeManager

                tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
                detokenize_manager = DetokenizeManager(tokenizer)
            logger.info(
                "Tokenizer backend selected: %s (tokenizer_path=%s)",
                backend_mode,
                rust_tokenizer_path,
            )
        except Exception as exc:
            logger.warning(
                "Rust tokenizer backend requested but unavailable; fallback to python: %s",
                exc,
            )
            backend_mode = "python"

    if backend_mode not in {"python", "rust_inprocess", "rust_tokenize_only"}:
        logger.warning("Unknown tokenizer backend '%s'; fallback to python", backend_mode)
        backend_mode = "python"

    if backend_mode == "python":
        from .detokenize import DetokenizeManager
        from .tokenize import TokenizeManager

        tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        tokenize_manager = TokenizeManager(tokenizer)
        detokenize_manager = DetokenizeManager(tokenizer)
        logger.info("Tokenizer backend selected: python")

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
