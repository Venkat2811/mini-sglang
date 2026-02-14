from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import torch

from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.core import Batch


logger = init_logger(__name__)


def _make_positions_python(batch: Batch, device: torch.device) -> torch.Tensor:
    indices_host = torch.empty(len(batch.out_loc), dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple_python(batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mapping_host = torch.empty(len(batch.out_loc), dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions


def _make_write_tuple_python(batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int32, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int32, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)


@dataclass
class CpuBackendStats:
    selected_backend: str
    python_calls: int = 0
    rust_calls: int = 0
    rust_fallbacks: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "selected_backend": self.selected_backend,
            "python_calls": self.python_calls,
            "rust_calls": self.rust_calls,
            "rust_fallbacks": self.rust_fallbacks,
        }


def _load_rust_module() -> Any:
    import minisgl_cpu

    return minisgl_cpu


class PythonCpuBackend:
    name = "python"

    def __init__(self):
        self.stats = CpuBackendStats(selected_backend=self.name)

    def make_positions(self, batch: Batch, device: torch.device) -> torch.Tensor:
        self.stats.python_calls += 1
        return _make_positions_python(batch, device)

    def make_input_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stats.python_calls += 1
        return _make_input_tuple_python(batch, device)

    def make_write_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stats.python_calls += 1
        return _make_write_tuple_python(batch, device)

    def snapshot(self) -> dict[str, int | str]:
        return self.stats.as_dict()


class RustHotpathCpuBackend:
    name = "rust_hotpath"

    def __init__(self):
        self.stats = CpuBackendStats(selected_backend=self.name)
        self.rust_mod = None
        try:
            self.rust_mod = _load_rust_module()
            logger.info("Rust CPU backend module loaded")
        except Exception as exc:
            logger.warning("Rust CPU backend requested but not available, falling back to python: %s", exc)

    @staticmethod
    def _req_shapes(batch: Batch, padded: bool) -> tuple[list[int], list[int], list[int], list[bool]]:
        reqs = batch.padded_reqs if padded else batch.reqs
        table_idxs = [req.table_idx for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        device_lens = [req.device_len for req in reqs]
        can_decode = [req.can_decode for req in reqs]
        return table_idxs, cached_lens, device_lens, can_decode

    def _fallback_positions(self, batch: Batch, device: torch.device) -> torch.Tensor:
        self.stats.rust_fallbacks += 1
        self.stats.python_calls += 1
        return _make_positions_python(batch, device)

    def _fallback_input(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stats.rust_fallbacks += 1
        self.stats.python_calls += 1
        return _make_input_tuple_python(batch, device)

    def _fallback_write(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stats.rust_fallbacks += 1
        self.stats.python_calls += 1
        return _make_write_tuple_python(batch, device)

    def make_positions(self, batch: Batch, device: torch.device) -> torch.Tensor:
        if self.rust_mod is None:
            return self._fallback_positions(batch, device)
        try:
            _, cached_lens, device_lens, _ = self._req_shapes(batch, padded=True)
            positions = self.rust_mod.make_positions(cached_lens, device_lens)
            self.stats.rust_calls += 1
            host = torch.tensor(positions, dtype=torch.int32, pin_memory=True)
            return host.to(device, non_blocking=True)
        except Exception as exc:
            logger.warning("Rust make_positions failed; fallback to python: %s", exc)
            return self._fallback_positions(batch, device)

    def make_input_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rust_mod is None:
            return self._fallback_input(batch, device)
        try:
            table_idxs, cached_lens, device_lens, _ = self._req_shapes(batch, padded=True)
            mapping, positions = self.rust_mod.make_input_mapping(
                table_idxs, cached_lens, device_lens, batch.positions.tolist()
            )
            self.stats.rust_calls += 1
            map_host = torch.tensor(mapping, dtype=torch.int32, pin_memory=True)
            pos_host = torch.tensor(positions, dtype=torch.int32, pin_memory=True)
            return map_host.to(device, non_blocking=True), pos_host.to(device, non_blocking=True)
        except Exception as exc:
            logger.warning("Rust make_input_mapping failed; fallback to python: %s", exc)
            return self._fallback_input(batch, device)

    def make_write_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rust_mod is None:
            return self._fallback_write(batch, device)
        try:
            table_idxs, _, device_lens, can_decode = self._req_shapes(batch, padded=False)
            req_mapping, write_mapping = self.rust_mod.make_write_mapping(
                table_idxs, device_lens, can_decode
            )
            self.stats.rust_calls += 1
            req_host = torch.tensor(req_mapping, dtype=torch.int32, pin_memory=True)
            write_host = torch.tensor(write_mapping, dtype=torch.int32, pin_memory=True)
            return req_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
        except Exception as exc:
            logger.warning("Rust make_write_mapping failed; fallback to python: %s", exc)
            return self._fallback_write(batch, device)

    def snapshot(self) -> dict[str, int | str]:
        return self.stats.as_dict()


def create_cpu_backend(mode: str):
    if mode == "python":
        return PythonCpuBackend()
    if mode == "rust_hotpath":
        return RustHotpathCpuBackend()
    logger.warning("Unknown CPU backend mode '%s'; fallback to python", mode)
    return PythonCpuBackend()
