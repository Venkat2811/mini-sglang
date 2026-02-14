from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from minisgl.env import ENV

from .utils import deserialize_type, serialize_type

_SCHEMA_VERSION = 1
_FAMILY = "frontend"


def _encode_typed(msg: BaseFrontendMsg) -> Dict[str, Any]:
    if isinstance(msg, BatchFrontendMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "batch",
            "items": [_encode_typed(item) for item in msg.data],
        }
    if isinstance(msg, UserReply):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "reply",
            "u": int(msg.uid),
            "o": msg.incremental_output,
            "fin": bool(msg.finished),
        }
    raise ValueError(f"unsupported frontend msg type for typed transport: {type(msg)!r}")


def _decode_typed(payload: Dict[str, Any]) -> BaseFrontendMsg:
    if int(payload.get("__schema__", -1)) != _SCHEMA_VERSION:
        raise ValueError("unsupported typed frontend schema version")
    if payload.get("f") != _FAMILY:
        raise ValueError("typed payload family mismatch for frontend")

    kind = payload.get("k")
    if kind == "batch":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("typed frontend batch payload missing items")
        return BatchFrontendMsg(data=[_decode_typed(item) for item in items])
    if kind == "reply":
        return UserReply(
            uid=int(payload["u"]),
            incremental_output=str(payload["o"]),
            finished=bool(payload["fin"]),
        )
    raise ValueError(f"unsupported typed frontend kind: {kind!r}")


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        if bool(ENV.TYPED_TRANSPORT):
            return _encode_typed(msg)
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        if isinstance(json, dict) and "__schema__" in json:
            return _decode_typed(json)
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
