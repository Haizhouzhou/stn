"""GPU environment helpers for Brian2CUDA Phase 4 runs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def infer_cuda_home() -> Path | None:
    """Return the CUDA installation root when it can be inferred."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        path = Path(cuda_home)
        if path.exists():
            return path

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    return Path(nvcc).resolve().parent.parent


def cuda_runtime_library_dir(cuda_home: str | Path | None = None) -> Path | None:
    """Return the CUDA runtime library directory used by Brian2CUDA binaries."""
    if cuda_home is None:
        home = infer_cuda_home()
    else:
        home = Path(cuda_home)
    if home is None:
        return None

    candidate = home / "targets" / "x86_64-linux" / "lib"
    if candidate.exists():
        return candidate
    return None


def ensure_cuda_runtime_libraries() -> str | None:
    """Prepend the CUDA runtime library directory to ``LD_LIBRARY_PATH`` when needed."""
    library_dir = cuda_runtime_library_dir()
    if library_dir is None:
        return None

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in existing.split(":") if entry]
    library_dir_str = str(library_dir)
    if library_dir_str not in entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join([library_dir_str, *entries])
    return library_dir_str


def visible_gpu_ids() -> list[int]:
    """Return integer GPU ids visible to the current environment."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids: list[int] = []
        for item in visible.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ids.append(int(item))
            except ValueError:
                continue
        if ids:
            return ids

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if completed.returncode != 0:
        return []

    ids = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            continue
    return ids
