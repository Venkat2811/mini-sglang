# Synthetic Gloo Control-Plane Probes

These scripts let this repo answer two questions without requiring a runnable Linux + CUDA
Mini-SGLang environment on the local host:

- does a `uv`-managed PyTorch wheel expose CPU `torch.distributed` Gloo here?
- how much would Mini-SGLang-style small CPU coordination traffic benefit from the custom
  Myelon-enabled Gloo build?

## 1. Probe the wheel-backed torch Gloo path

```bash
uv venv .venv-gloo --python 3.12
source .venv-gloo/bin/activate
uv pip install 'torch<2.10.0'
python benchmark/synthetic/torch_gloo_smoke.py
```

This script:

- verifies that `torch.distributed` and the built-in `gloo` backend work on CPU
- prints the torch library directory
- shows whether the wheel ships a standalone `libgloo`

If the wheel has no standalone `libgloo`, swapping in a custom Gloo build at runtime is not a
realistic path. At that point the practical options are:

- rebuild PyTorch against the custom Gloo
- add a custom c10d process-group backend plugin

## 2. Compare uv versus Myelon for Mini-SGLang-like control traffic

This script delegates to the external Gloo MPI benchmarks that already support
`--transport=auto` and `--transport=myelon`.

```bash
python benchmark/synthetic/bench_gloo_control_plane.py \
  --benchmark-dir ../gloo/build-mpi-uv/gloo/mpi/benchmark
```

The retained proxy shapes are:

- small rank-wide broadcast for scheduler counts
- small metadata broadcast for NCCL UID fanout
- 16-byte allreduce for `_sync_get_memory`
- 1 KiB allreduce for small CPU-side state sync

## 3. Compare real torch.distributed collectives through a custom c10d backend

This path uses an out-of-tree CPU-only c10d backend that links against the external
`libgloo.a` build and selects either `uv` or `myelon` transport underneath via
`MINISGL_C10D_GLOOEXT_TRANSPORT`.

Install the missing runtime/build helpers in the same `uv` venv first:

```bash
uv pip install --python .venv-gloo/bin/python numpy ninja
```

Then run:

```bash
python benchmark/synthetic/bench_c10d_control_plane.py \
  --world-size=4 \
  --warmup=20 \
  --iterations=100
```

This benchmark exercises the real PyTorch API layer for the exact Mini-SGLang-relevant CPU
operations:

- `dist.barrier()` for `SchedulerIOMixin.sync_all_ranks()`
- `dist.broadcast()` on a scalar `int64` tensor for scheduler count fanout
- `dist.broadcast_object_list()` for the PyNCCL unique-ID fanout shape
- `dist.all_reduce(..., ReduceOp.MIN)` on `[free, -free]` for `_sync_get_memory()`

The retained results from this path are in
`benchmark/synthetic/2026-04-21_c10d_results.md`.
