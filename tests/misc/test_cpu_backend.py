from __future__ import annotations

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


def test_python_backend_selected_and_operational():
    backend = cpu_backend.create_cpu_backend("python")
    batch = _make_batch()
    device = torch.device("cpu")

    batch.positions = backend.make_positions(batch, device)
    input_mapping, positions = backend.make_input_tuple(batch, device)
    write_mapping, write_pos = backend.make_write_tuple(batch, device)

    assert backend.name == "python"
    assert positions.tolist() == [1, 2, 3, 1, 2]
    assert input_mapping.tolist() == [3, 3, 3, 5, 5]
    assert write_mapping.tolist() == [3, 5]
    assert write_pos.tolist() == [4, 3]
    assert backend.snapshot()["python_calls"] == 3


def test_rust_backend_import_failure_falls_back_to_python(monkeypatch):
    def _raise_import_error():
        raise ModuleNotFoundError("minisgl_cpu not available")

    monkeypatch.setattr(cpu_backend, "_load_rust_module", _raise_import_error)
    backend = cpu_backend.create_cpu_backend("rust_hotpath")
    batch = _make_batch()
    device = torch.device("cpu")

    batch.positions = backend.make_positions(batch, device)
    input_mapping, _ = backend.make_input_tuple(batch, device)
    write_mapping, _ = backend.make_write_tuple(batch, device)

    assert backend.name == "rust_hotpath"
    stats = backend.snapshot()
    assert stats["rust_calls"] == 0
    assert stats["rust_fallbacks"] == 3
    assert stats["python_calls"] == 3
    assert input_mapping.tolist() == [3, 3, 3, 5, 5]
    assert write_mapping.tolist() == [3, 5]


def test_rust_backend_runtime_error_fallback(monkeypatch):
    class FakeRustMod:
        @staticmethod
        def make_positions(_cached_lens, _device_lens):
            raise RuntimeError("forced failure")

        @staticmethod
        def make_input_mapping(table_idxs, cached_lens, device_lens, positions):
            _ = cached_lens, device_lens
            return (table_idxs, positions)

        @staticmethod
        def make_write_mapping(table_idxs, device_lens, can_decode):
            _ = device_lens
            write = [1 if flag else -1 for flag in can_decode]
            return (table_idxs, write)

    monkeypatch.setattr(cpu_backend, "_load_rust_module", lambda: FakeRustMod())
    backend = cpu_backend.create_cpu_backend("rust_hotpath")
    batch = _make_batch()
    device = torch.device("cpu")

    batch.positions = backend.make_positions(batch, device)
    input_mapping, _ = backend.make_input_tuple(batch, device)
    write_mapping, _ = backend.make_write_tuple(batch, device)

    stats = backend.snapshot()
    assert stats["rust_calls"] == 2  # input/write succeeded via fake rust module
    assert stats["rust_fallbacks"] == 1  # positions fell back
    assert stats["python_calls"] == 1
    assert batch.positions.tolist() == [1, 2, 3, 1, 2]
    assert input_mapping.tolist() == [3, 5]
    assert write_mapping.tolist() == [3, 5]
