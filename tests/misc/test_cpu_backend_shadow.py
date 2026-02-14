from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from minisgl.core import Batch, Req, SamplingParams
from minisgl.scheduler import cpu_backend


@dataclass
class DummyCacheHandle:
    cached_len: int = 0


def _make_batch() -> Batch:
    req_a = Req(
        input_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        table_idx=3,
        cached_len=1,
        output_len=2,
        uid=100,
        sampling_params=SamplingParams(max_tokens=2),
        cache_handle=DummyCacheHandle(1),
    )
    req_b = Req(
        input_ids=torch.tensor([21, 22, 23], dtype=torch.int32),
        table_idx=5,
        cached_len=1,
        output_len=3,
        uid=101,
        sampling_params=SamplingParams(max_tokens=3),
        cache_handle=DummyCacheHandle(1),
    )
    batch = Batch(reqs=[req_a, req_b], phase="prefill")
    batch.padded_reqs = [req_a, req_b]
    total_extend = sum(req.extend_len for req in batch.padded_reqs)
    batch.out_loc = torch.arange(total_extend, dtype=torch.int32)
    batch.positions = torch.empty(total_extend, dtype=torch.int32)
    return batch


class FakeBackend:
    def __init__(
        self,
        name: str,
        *,
        positions: list[int],
        input_mapping: list[int],
        input_positions: list[int],
        write_req_mapping: list[int],
        write_pos: list[int],
        raise_in: str | None = None,
    ):
        self.name = name
        self.positions = positions
        self.input_mapping = input_mapping
        self.input_positions = input_positions
        self.write_req_mapping = write_req_mapping
        self.write_pos = write_pos
        self.raise_in = raise_in
        self.calls = 0

    def make_positions(self, _batch: Batch, device: torch.device) -> torch.Tensor:
        if self.raise_in == "positions":
            raise RuntimeError("forced positions failure")
        self.calls += 1
        return torch.tensor(self.positions, dtype=torch.int32).to(device)

    def make_input_tuple(self, _batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.raise_in == "input":
            raise RuntimeError("forced input failure")
        self.calls += 1
        return (
            torch.tensor(self.input_mapping, dtype=torch.int32).to(device),
            torch.tensor(self.input_positions, dtype=torch.int32).to(device),
        )

    def make_write_tuple(self, _batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.raise_in == "write":
            raise RuntimeError("forced write failure")
        self.calls += 1
        return (
            torch.tensor(self.write_req_mapping, dtype=torch.int32).to(device),
            torch.tensor(self.write_pos, dtype=torch.int32).to(device),
        )

    def snapshot(self) -> dict[str, int | str]:
        return {"selected_backend": self.name, "calls": self.calls}


def _baseline_backend(name: str, *, raise_in: str | None = None) -> FakeBackend:
    return FakeBackend(
        name,
        positions=[1, 2, 3, 1, 2],
        input_mapping=[3, 3, 3, 5, 5],
        input_positions=[1, 2, 3, 1, 2],
        write_req_mapping=[3, 5],
        write_pos=[4, 3],
        raise_in=raise_in,
    )


def test_shadow_backend_logs_mapping_divergence(tmp_path):
    batch = _make_batch()
    primary = _baseline_backend("python")
    shadow = _baseline_backend("rust_hotpath")
    shadow.input_mapping = [3, 9, 3, 5, 5]

    report_path = tmp_path / "shadow.jsonl"
    backend = cpu_backend.ShadowCpuBackend(
        primary=primary,
        shadow=shadow,
        report_path=str(report_path),
        max_diffs=16,
    )

    device = torch.device("cpu")
    batch.positions = backend.make_positions(batch, device)
    input_mapping, positions = backend.make_input_tuple(batch, device)
    write_mapping, write_pos = backend.make_write_tuple(batch, device)

    assert input_mapping.tolist() == [3, 3, 3, 5, 5]
    assert positions.tolist() == [1, 2, 3, 1, 2]
    assert write_mapping.tolist() == [3, 5]
    assert write_pos.tolist() == [4, 3]

    stats = backend.snapshot()
    assert stats["shadow_divergences"] >= 1
    assert stats["shadow_compares"] >= 4
    entries = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
    assert any(entry["kind"] == "input_mapping" for entry in entries)
    assert any(entry["req_uids"] == [100, 101] for entry in entries)


def test_shadow_backend_logs_shadow_exception(tmp_path):
    batch = _make_batch()
    primary = _baseline_backend("python")
    shadow = _baseline_backend("rust_hotpath", raise_in="write")

    report_path = tmp_path / "shadow.jsonl"
    backend = cpu_backend.ShadowCpuBackend(
        primary=primary,
        shadow=shadow,
        report_path=str(report_path),
        max_diffs=16,
    )

    device = torch.device("cpu")
    _ = backend.make_positions(batch, device)
    _ = backend.make_input_tuple(batch, device)
    write_mapping, write_pos = backend.make_write_tuple(batch, device)

    assert write_mapping.tolist() == [3, 5]
    assert write_pos.tolist() == [4, 3]
    stats = backend.snapshot()
    assert stats["shadow_divergences"] >= 1
    entries = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
    assert any(entry["kind"] == "write_exception" for entry in entries)


def test_create_cpu_backend_wraps_shadow_when_enabled(monkeypatch):
    monkeypatch.setattr(cpu_backend.ENV.CPU_BACKEND_SHADOW, "value", True)
    monkeypatch.setattr(cpu_backend.ENV.CPU_BACKEND_SHADOW_REPORT, "value", "")
    monkeypatch.setattr(cpu_backend.ENV.CPU_BACKEND_SHADOW_MAX_DIFFS, "value", 4)
    monkeypatch.setattr(cpu_backend.ENV.CPU_BACKEND_SHADOW_EVERY_N, "value", 1)

    created: list[str] = []

    def _fake_factory(mode: str):
        created.append(mode)
        return _baseline_backend(mode)

    monkeypatch.setattr(cpu_backend, "_create_backend_unwrapped", _fake_factory)
    backend = cpu_backend.create_cpu_backend("python")
    assert isinstance(backend, cpu_backend.ShadowCpuBackend)
    assert created == ["python", "rust_hotpath"]


def test_shadow_backend_compare_every_n_skips_shadow_calls(tmp_path):
    batch = _make_batch()
    primary = _baseline_backend("python")
    shadow = _baseline_backend("rust_hotpath")

    report_path = tmp_path / "shadow.jsonl"
    backend = cpu_backend.ShadowCpuBackend(
        primary=primary,
        shadow=shadow,
        report_path=str(report_path),
        max_diffs=16,
        compare_every_n=3,
    )

    device = torch.device("cpu")
    _ = backend.make_positions(batch, device)
    _ = backend.make_input_tuple(batch, device)
    _ = backend.make_write_tuple(batch, device)

    stats = backend.snapshot()
    assert stats["shadow_compares"] == 2
    assert stats["shadow_samples_skipped"] == 2
