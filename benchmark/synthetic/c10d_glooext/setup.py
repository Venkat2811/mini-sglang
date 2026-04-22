from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def libuv_prefix() -> Path:
    candidates = []
    try:
        output = subprocess.check_output(["brew", "--prefix", "libuv"], text=True).strip()
        if output:
            candidates.append(Path(output))
    except Exception:
        pass
    candidates.extend(
        [
            Path("/opt/homebrew/opt/libuv"),
            Path("/usr/local/opt/libuv"),
            Path("/opt/homebrew"),
            Path("/usr/local"),
        ]
    )
    for candidate in candidates:
        if (candidate / "lib" / "libuv.dylib").exists():
            return candidate
    raise RuntimeError("failed to locate libuv; install it with Homebrew first")


WORKSPACE_ROOT = workspace_root()
GLOO_ROOT = WORKSPACE_ROOT / "gloo"
GLOO_BUILD_ROOT = GLOO_ROOT / "build-mpi-uv"
MYELON_ROOT = WORKSPACE_ROOT / "myelon-playground"
MYELON_RELEASE = MYELON_ROOT / "target" / "release"
LIBUV_PREFIX = libuv_prefix()

extra_objects = [
    str(GLOO_BUILD_ROOT / "gloo" / "libgloo.a"),
    str(MYELON_RELEASE / "libmyelon_gloo_ffi.a"),
]
for artifact in extra_objects:
    if not Path(artifact).exists():
        raise RuntimeError(f"missing required artifact: {artifact}")


extension = CppExtension(
    name="myelon_c10d_backend",
    sources=["src/backend.cpp"],
    include_dirs=[
        str(GLOO_ROOT),
        str(GLOO_BUILD_ROOT),
        str(MYELON_ROOT / "crates" / "myelon-gloo-ffi" / "include"),
    ],
    library_dirs=[str(LIBUV_PREFIX / "lib")],
    libraries=["uv"],
    extra_objects=extra_objects,
    extra_compile_args=["-std=c++17", "-O3"],
    extra_link_args=[f"-Wl,-rpath,{LIBUV_PREFIX / 'lib'}"],
)


setup(
    name="myelon-c10d-backend",
    version="0.0.1",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
