from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def env_or(default: Path, name: str) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


def libuv_link_settings() -> tuple[list[str], list[str]]:
    system = platform.system()
    if system == "Linux":
        # The Ubuntu images we use ship libuv via the standard linker path.
        return ([], [])

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
            return ([str(candidate / "lib")], [f"-Wl,-rpath,{candidate / 'lib'}"])
    raise RuntimeError("failed to locate libuv; install it first")


WORKSPACE_ROOT = workspace_root()
GLOO_ROOT = env_or(WORKSPACE_ROOT / "gloo", "MINISGL_C10D_GLOOEXT_GLOO_ROOT")
GLOO_BUILD_ROOT = env_or(GLOO_ROOT / "build-mpi-uv", "MINISGL_C10D_GLOOEXT_GLOO_BUILD_ROOT")
MYELON_ROOT = env_or(WORKSPACE_ROOT / "myelon-playground", "MINISGL_C10D_GLOOEXT_MYELON_ROOT")
MYELON_RELEASE = MYELON_ROOT / "target" / "release"
LIBRARY_DIRS, EXTRA_LINK_ARGS = libuv_link_settings()

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
    library_dirs=LIBRARY_DIRS,
    libraries=["uv"],
    extra_objects=extra_objects,
    extra_compile_args=["-std=c++17", "-O3"],
    extra_link_args=EXTRA_LINK_ARGS,
)


setup(
    name="myelon-c10d-backend",
    version="0.0.1",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
