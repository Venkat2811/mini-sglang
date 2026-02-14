from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)


@contextmanager
def typed_transport(enabled: bool):
    old = ENV.TYPED_TRANSPORT.value
    ENV.TYPED_TRANSPORT.value = enabled
    try:
        yield
    finally:
        ENV.TYPED_TRANSPORT.value = old


def test_backend_typed_roundtrip_and_legacy_compat():
    msg = BatchBackendMsg(
        [
            UserMsg(
                uid=7,
                input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
                sampling_params=SamplingParams(
                    temperature=0.6,
                    top_k=8,
                    top_p=0.95,
                    ignore_eos=True,
                    max_tokens=128,
                ),
            ),
            ExitMsg(),
        ]
    )

    with typed_transport(False):
        legacy = msg.encoder()

    with typed_transport(True):
        typed = msg.encoder()
        assert typed["__schema__"] == 1
        typed_out = BaseBackendMsg.decoder(typed)
        legacy_out = BaseBackendMsg.decoder(legacy)

    assert isinstance(typed_out, BatchBackendMsg)
    assert isinstance(legacy_out, BatchBackendMsg)
    assert isinstance(typed_out.data[0], UserMsg)
    assert typed_out.data[0].uid == 7
    assert typed_out.data[0].input_ids.tolist() == [1, 2, 3]
    assert typed_out.data[0].sampling_params.max_tokens == 128
    assert isinstance(typed_out.data[1], ExitMsg)
    assert legacy_out.data[0].input_ids.tolist() == [1, 2, 3]


def test_tokenizer_typed_roundtrip_and_legacy_compat():
    msg = BatchTokenizerMsg(
        [
            TokenizeMsg(
                uid=1,
                text=[{"role": "user", "content": "hi"}],
                sampling_params=SamplingParams(top_k=1, max_tokens=16),
            ),
            DetokenizeMsg(uid=1, next_token=123, finished=False),
            AbortMsg(uid=99),
        ]
    )

    with typed_transport(False):
        legacy = BaseTokenizerMsg.encoder(msg)

    with typed_transport(True):
        typed = BaseTokenizerMsg.encoder(msg)
        assert typed["__schema__"] == 1
        typed_out = BaseTokenizerMsg.decoder(typed)
        legacy_out = BaseTokenizerMsg.decoder(legacy)

    assert isinstance(typed_out, BatchTokenizerMsg)
    assert isinstance(typed_out.data[0], TokenizeMsg)
    assert typed_out.data[0].text == [{"role": "user", "content": "hi"}]
    assert isinstance(typed_out.data[1], DetokenizeMsg)
    assert typed_out.data[1].next_token == 123
    assert isinstance(typed_out.data[2], AbortMsg)
    assert isinstance(legacy_out.data[0], TokenizeMsg)


def test_frontend_typed_roundtrip_and_legacy_compat():
    msg = BatchFrontendMsg([UserReply(uid=5, incremental_output="ok", finished=True)])

    with typed_transport(False):
        legacy = BaseFrontendMsg.encoder(msg)

    with typed_transport(True):
        typed = BaseFrontendMsg.encoder(msg)
        assert typed["__schema__"] == 1
        typed_out = BaseFrontendMsg.decoder(typed)
        legacy_out = BaseFrontendMsg.decoder(legacy)

    assert isinstance(typed_out, BatchFrontendMsg)
    assert typed_out.data[0].incremental_output == "ok"
    assert typed_out.data[0].finished is True
    assert isinstance(legacy_out, BatchFrontendMsg)


def test_typed_decoder_rejects_unknown_schema():
    bad_payload = {"__schema__": 99, "f": "backend", "k": "exit"}
    with pytest.raises(ValueError, match="schema version"):
        BaseBackendMsg.decoder(bad_payload)
