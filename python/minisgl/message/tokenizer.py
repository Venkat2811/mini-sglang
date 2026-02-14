from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from minisgl.core import SamplingParams
from minisgl.env import ENV

from .utils import deserialize_type, serialize_type

_SCHEMA_VERSION = 1
_FAMILY = "tokenizer"


def _encode_sampling_params(params: SamplingParams) -> list[Any]:
    return [
        float(params.temperature),
        int(params.top_k),
        float(params.top_p),
        bool(params.ignore_eos),
        int(params.max_tokens),
    ]


def _decode_sampling_params(values: list[Any]) -> SamplingParams:
    if len(values) != 5:
        raise ValueError("invalid sampling params payload")
    return SamplingParams(
        temperature=float(values[0]),
        top_k=int(values[1]),
        top_p=float(values[2]),
        ignore_eos=bool(values[3]),
        max_tokens=int(values[4]),
    )


def _encode_typed(msg: BaseTokenizerMsg) -> Dict[str, Any]:
    if isinstance(msg, BatchTokenizerMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "batch",
            "items": [_encode_typed(item) for item in msg.data],
        }
    if isinstance(msg, DetokenizeMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "detok",
            "u": int(msg.uid),
            "n": int(msg.next_token),
            "fin": bool(msg.finished),
        }
    if isinstance(msg, TokenizeMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "tok",
            "u": int(msg.uid),
            "t": msg.text,
            "sp": _encode_sampling_params(msg.sampling_params),
        }
    if isinstance(msg, AbortMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "abort",
            "u": int(msg.uid),
        }
    raise ValueError(f"unsupported tokenizer msg type for typed transport: {type(msg)!r}")


def _decode_typed(payload: Dict[str, Any]) -> BaseTokenizerMsg:
    if int(payload.get("__schema__", -1)) != _SCHEMA_VERSION:
        raise ValueError("unsupported typed tokenizer schema version")
    if payload.get("f") != _FAMILY:
        raise ValueError("typed payload family mismatch for tokenizer")

    kind = payload.get("k")
    if kind == "batch":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("typed tokenizer batch payload missing items")
        return BatchTokenizerMsg(data=[_decode_typed(item) for item in items])
    if kind == "detok":
        return DetokenizeMsg(
            uid=int(payload["u"]),
            next_token=int(payload["n"]),
            finished=bool(payload["fin"]),
        )
    if kind == "tok":
        text = payload["t"]
        if not isinstance(text, (str, list)):
            raise ValueError("typed tokenizer tok payload has invalid text type")
        return TokenizeMsg(
            uid=int(payload["u"]),
            text=text,
            sampling_params=_decode_sampling_params(payload["sp"]),
        )
    if kind == "abort":
        return AbortMsg(uid=int(payload["u"]))
    raise ValueError(f"unsupported typed tokenizer kind: {kind!r}")


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        if bool(ENV.TYPED_TRANSPORT):
            return _encode_typed(msg)
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        if isinstance(json, dict) and "__schema__" in json:
            return _decode_typed(json)
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
