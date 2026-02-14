from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV

from .utils import deserialize_type, serialize_type

_SCHEMA_VERSION = 1
_FAMILY = "backend"


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


def _encode_typed_tensor_i32(tensor: torch.Tensor) -> bytes:
    if tensor.dim() != 1:
        raise ValueError("typed backend transport only supports 1D tensors")
    if not tensor.is_cpu:
        tensor = tensor.cpu()
    if tensor.dtype != torch.int32:
        tensor = tensor.to(dtype=torch.int32)
    return tensor.numpy().tobytes()


def _decode_typed_tensor_i32(buffer: bytes) -> torch.Tensor:
    np_tensor = np.frombuffer(buffer, dtype=np.int32)
    return torch.from_numpy(np_tensor.copy())


def _encode_typed(msg: BaseBackendMsg) -> Dict[str, Any]:
    if isinstance(msg, BatchBackendMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "batch",
            "items": [_encode_typed(item) for item in msg.data],
        }
    if isinstance(msg, ExitMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "exit",
        }
    if isinstance(msg, UserMsg):
        return {
            "__schema__": _SCHEMA_VERSION,
            "f": _FAMILY,
            "k": "user",
            "u": int(msg.uid),
            "ib": _encode_typed_tensor_i32(msg.input_ids),
            "sp": _encode_sampling_params(msg.sampling_params),
        }
    raise ValueError(f"unsupported backend msg type for typed transport: {type(msg)!r}")


def _decode_typed(payload: Dict[str, Any]) -> BaseBackendMsg:
    if int(payload.get("__schema__", -1)) != _SCHEMA_VERSION:
        raise ValueError("unsupported typed backend schema version")
    if payload.get("f") != _FAMILY:
        raise ValueError("typed payload family mismatch for backend")

    kind = payload.get("k")
    if kind == "batch":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError("typed backend batch payload missing items")
        return BatchBackendMsg(data=[_decode_typed(item) for item in items])
    if kind == "exit":
        return ExitMsg()
    if kind == "user":
        return UserMsg(
            uid=int(payload["u"]),
            input_ids=_decode_typed_tensor_i32(payload["ib"]),
            sampling_params=_decode_sampling_params(payload["sp"]),
        )
    raise ValueError(f"unsupported typed backend kind: {kind!r}")


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        if bool(ENV.TYPED_TRANSPORT):
            return _encode_typed(self)
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        if isinstance(json, dict) and "__schema__" in json:
            return _decode_typed(json)
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
