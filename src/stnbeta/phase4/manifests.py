"""Manifest helpers for reproducible Phase 4 outputs."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"available": False, "returncode": None, "stdout": "", "stderr": ""}

    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_commit(cwd: str | Path | None = None) -> str | None:
    result = _run_command(["git", "rev-parse", "HEAD"])
    if result["available"] and result["returncode"] == 0:
        return result["stdout"] or None
    return None


def collect_runtime_manifest(
    *,
    backend: str,
    config_hash_value: str,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect a minimal runtime manifest for Phase 4 outputs."""
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config_hash": config_hash_value,
        "seed": seed,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "git_commit": _git_commit(),
        "nvidia_smi": _run_command(["nvidia-smi", "-L"]),
        "nvcc": _run_command(["nvcc", "--version"]),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a JSON manifest with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
