from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import torch

from minisgl.env import ENV
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
        self._cached_batch_id: int | None = None
        self._cached_positions: torch.Tensor | None = None
        self._cached_input_mapping: torch.Tensor | None = None
        self._cached_write_req_mapping: torch.Tensor | None = None
        self._cached_write_pos: torch.Tensor | None = None
        self._cached_uses_remaining = 0
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
        self._invalidate_batch_cache()
        return _make_positions_python(batch, device)

    def _fallback_input(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stats.rust_fallbacks += 1
        self.stats.python_calls += 1
        self._invalidate_batch_cache()
        return _make_input_tuple_python(batch, device)

    def _fallback_write(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stats.rust_fallbacks += 1
        self.stats.python_calls += 1
        self._invalidate_batch_cache()
        return _make_write_tuple_python(batch, device)

    def _invalidate_batch_cache(self) -> None:
        self._cached_batch_id = None
        self._cached_positions = None
        self._cached_input_mapping = None
        self._cached_write_req_mapping = None
        self._cached_write_pos = None
        self._cached_uses_remaining = 0

    @staticmethod
    def _from_i32_buffer_to_device(buffer_obj: Any, device: torch.device) -> torch.Tensor:
        src = torch.frombuffer(buffer_obj, dtype=torch.int32)
        if device.type == "cpu":
            return src.clone()
        return src.to(device, non_blocking=True)

    def _ensure_cached_metadata(self, batch: Batch, device: torch.device) -> None:
        if self.rust_mod is None:
            raise RuntimeError("rust backend module not loaded")

        batch_id = id(batch)
        if self._cached_batch_id == batch_id and self._cached_uses_remaining > 0:
            return

        if not hasattr(self.rust_mod, "make_metadata_buffers"):
            self._invalidate_batch_cache()
            return

        table_idxs_padded, cached_lens, device_lens_padded, _ = self._req_shapes(batch, padded=True)
        table_idxs, _, device_lens, can_decode = self._req_shapes(batch, padded=False)
        (
            positions_buf,
            input_mapping_buf,
            write_req_mapping_buf,
            write_pos_buf,
        ) = self.rust_mod.make_metadata_buffers(
            table_idxs_padded,
            cached_lens,
            device_lens_padded,
            table_idxs,
            device_lens,
            can_decode,
        )
        self.stats.rust_calls += 1
        self._cached_positions = self._from_i32_buffer_to_device(positions_buf, device)
        self._cached_input_mapping = self._from_i32_buffer_to_device(input_mapping_buf, device)
        self._cached_write_req_mapping = self._from_i32_buffer_to_device(write_req_mapping_buf, device)
        self._cached_write_pos = self._from_i32_buffer_to_device(write_pos_buf, device)
        self._cached_batch_id = batch_id
        self._cached_uses_remaining = 3

    def _consume_cache_use(self) -> None:
        if self._cached_uses_remaining <= 0:
            return
        self._cached_uses_remaining -= 1
        if self._cached_uses_remaining == 0:
            self._invalidate_batch_cache()

    def make_positions(self, batch: Batch, device: torch.device) -> torch.Tensor:
        if self.rust_mod is None:
            return self._fallback_positions(batch, device)
        try:
            self._ensure_cached_metadata(batch, device)
            if self._cached_positions is not None:
                positions = self._cached_positions
                self._consume_cache_use()
                return positions
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
            self._ensure_cached_metadata(batch, device)
            if self._cached_input_mapping is not None and self._cached_positions is not None:
                input_mapping = self._cached_input_mapping
                positions = self._cached_positions
                self._consume_cache_use()
                return input_mapping, positions
            table_idxs, cached_lens, device_lens, _ = self._req_shapes(batch, padded=True)
            mapping = self.rust_mod.make_input_mapping(table_idxs, cached_lens, device_lens)
            self.stats.rust_calls += 1
            map_host = torch.tensor(mapping, dtype=torch.int32, pin_memory=True)
            return map_host.to(device, non_blocking=True), batch.positions
        except Exception as exc:
            logger.warning("Rust make_input_mapping failed; fallback to python: %s", exc)
            return self._fallback_input(batch, device)

    def make_write_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rust_mod is None:
            return self._fallback_write(batch, device)
        try:
            self._ensure_cached_metadata(batch, device)
            if self._cached_write_req_mapping is not None and self._cached_write_pos is not None:
                req_mapping = self._cached_write_req_mapping
                write_pos = self._cached_write_pos
                self._consume_cache_use()
                return req_mapping, write_pos
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


class ShadowCpuBackend:
    name = "shadow"

    def __init__(
        self,
        primary: PythonCpuBackend | RustHotpathCpuBackend,
        shadow: PythonCpuBackend | RustHotpathCpuBackend,
        report_path: str = "",
        max_diffs: int = 128,
    ):
        self.primary = primary
        self.shadow = shadow
        self.report_path = Path(report_path) if report_path else None
        self.max_diffs = max_diffs
        self.shadow_compares = 0
        self.shadow_divergences = 0
        self.shadow_logged = 0

    @staticmethod
    def _req_ids(batch: Batch) -> list[int]:
        return [int(req.uid) for req in batch.reqs]

    @staticmethod
    def _diff_reason(primary: torch.Tensor, shadow: torch.Tensor) -> str | None:
        if primary.shape != shadow.shape:
            return f"shape mismatch: {tuple(primary.shape)} != {tuple(shadow.shape)}"
        if primary.dtype != shadow.dtype:
            return f"dtype mismatch: {primary.dtype} != {shadow.dtype}"
        p = primary.detach().cpu()
        s = shadow.detach().cpu()
        if torch.equal(p, s):
            return None
        diff_idx = int(torch.nonzero(p != s, as_tuple=False).flatten()[0].item())
        p_flat, s_flat = p.flatten(), s.flatten()
        return f"value mismatch at idx={diff_idx}: {int(p_flat[diff_idx])} != {int(s_flat[diff_idx])}"

    def _append_report(self, payload: dict[str, Any]) -> None:
        if self.report_path is None or self.shadow_logged >= self.max_diffs:
            return
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        self.shadow_logged += 1

    def _record_divergence(self, kind: str, batch: Batch, reason: str) -> None:
        self.shadow_divergences += 1
        req_ids = self._req_ids(batch)
        logger.warning(
            "CPU backend shadow divergence kind=%s req_uids=%s reason=%s",
            kind,
            req_ids,
            reason,
        )
        self._append_report(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "req_uids": req_ids,
                "reason": reason,
                "primary_backend": self.primary.name,
                "shadow_backend": self.shadow.name,
            }
        )

    def _compare(self, kind: str, batch: Batch, primary: torch.Tensor, shadow: torch.Tensor) -> None:
        self.shadow_compares += 1
        reason = self._diff_reason(primary, shadow)
        if reason is not None:
            self._record_divergence(kind, batch, reason)

    def make_positions(self, batch: Batch, device: torch.device) -> torch.Tensor:
        primary = self.primary.make_positions(batch, device)
        try:
            shadow = self.shadow.make_positions(batch, device)
            self._compare("positions", batch, primary, shadow)
        except Exception as exc:
            self._record_divergence("positions_exception", batch, str(exc))
        return primary

    def make_input_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        primary_mapping, primary_positions = self.primary.make_input_tuple(batch, device)
        try:
            shadow_mapping, shadow_positions = self.shadow.make_input_tuple(batch, device)
            self._compare("input_mapping", batch, primary_mapping, shadow_mapping)
            self._compare("input_positions", batch, primary_positions, shadow_positions)
        except Exception as exc:
            self._record_divergence("input_exception", batch, str(exc))
        return primary_mapping, primary_positions

    def make_write_tuple(self, batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        primary_req, primary_write = self.primary.make_write_tuple(batch, device)
        try:
            shadow_req, shadow_write = self.shadow.make_write_tuple(batch, device)
            self._compare("write_req_mapping", batch, primary_req, shadow_req)
            self._compare("write_pos", batch, primary_write, shadow_write)
        except Exception as exc:
            self._record_divergence("write_exception", batch, str(exc))
        return primary_req, primary_write

    @staticmethod
    def _prefixed_stats(prefix: str, payload: dict[str, int | str]) -> dict[str, int | str]:
        return {f"{prefix}_{k}": v for k, v in payload.items()}

    def snapshot(self) -> dict[str, int | str]:
        stats: dict[str, int | str] = {
            "selected_backend": self.primary.name,
            "shadow_backend": self.shadow.name,
            "shadow_enabled": 1,
            "shadow_compares": self.shadow_compares,
            "shadow_divergences": self.shadow_divergences,
            "shadow_logged": self.shadow_logged,
        }
        stats.update(self._prefixed_stats("primary", self.primary.snapshot()))
        stats.update(self._prefixed_stats("shadow", self.shadow.snapshot()))
        return stats


def _create_backend_unwrapped(mode: str):
    if mode == "python":
        return PythonCpuBackend()
    if mode == "rust_hotpath":
        return RustHotpathCpuBackend()
    logger.warning("Unknown CPU backend mode '%s'; fallback to python", mode)
    return PythonCpuBackend()


def create_cpu_backend(mode: str):
    primary = _create_backend_unwrapped(mode)
    if not ENV.CPU_BACKEND_SHADOW:
        return primary
    shadow_mode = "rust_hotpath" if primary.name == "python" else "python"
    shadow = _create_backend_unwrapped(shadow_mode)
    report_path = str(ENV.CPU_BACKEND_SHADOW_REPORT).strip()
    max_diffs = ENV.CPU_BACKEND_SHADOW_MAX_DIFFS.value
    logger.info(
        "CPU backend shadow mode enabled: primary=%s shadow=%s report_path=%s max_diffs=%d",
        primary.name,
        shadow.name,
        report_path or "<disabled>",
        max_diffs,
    )
    return ShadowCpuBackend(
        primary=primary,
        shadow=shadow,
        report_path=report_path,
        max_diffs=max_diffs,
    )
