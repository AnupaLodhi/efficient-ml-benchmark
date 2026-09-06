"""Capture environment/hardware metadata so every experiment can be traced
back to the exact software and hardware conditions it ran under.

This is written to results/raw/environment.json by
experiments/check_environment.py, and every subsequent experiment script
should re-record a lightweight version of this alongside its own results
(hardware can differ between machines/runs).
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy
import pandas
import sklearn
import torch


def _safe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "logical_cores": None,
        "physical_cores": None,
        "torch_num_threads": torch.get_num_threads(),
    }
    try:
        import os

        info["logical_cores"] = os.cpu_count()
    except Exception:
        pass

    lscpu = _safe_run(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                info["model_name"] = line.split(":", 1)[1].strip()
            if line.startswith("Core(s) per socket:"):
                info["cores_per_socket"] = line.split(":", 1)[1].strip()
            if line.startswith("Socket(s):"):
                info["sockets"] = line.split(":", 1)[1].strip()
    return info


def get_memory_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["mem_total_kb"] = int(line.split()[1])
                if line.startswith("MemAvailable:"):
                    info["mem_available_kb"] = int(line.split()[1])
    except FileNotFoundError:
        pass
    return info


def get_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["devices"].append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                }
            )
    return info


def collect_environment_info() -> dict[str, Any]:
    """Collect a full reproducibility snapshot as a JSON-serializable dict."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "sklearn_version": sklearn.__version__,
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
    }
