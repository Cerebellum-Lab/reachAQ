from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GpuRuntimeStatus:
    is_available: bool
    backend: str = ""
    devices: Tuple[str, ...] = tuple()
    error: str = ""


def detect_gpu_runtime() -> GpuRuntimeStatus:
    """Detect GPU availability for live inference without requiring one backend."""

    statuses = (_detect_tensorflow_gpu(), _detect_torch_cuda())
    for status in statuses:
        if status.is_available:
            return status

    errors = "; ".join(
        f"{status.backend}: {status.error or 'no GPU devices found'}"
        for status in statuses
    )
    return GpuRuntimeStatus(False, backend="tensorflow/torch", error=errors)


def _detect_tensorflow_gpu() -> GpuRuntimeStatus:
    try:
        import tensorflow as tf
    except Exception as exc:
        return GpuRuntimeStatus(False, backend="tensorflow", error=str(exc))
    try:
        devices = tuple(device.name for device in tf.config.list_physical_devices("GPU"))
    except Exception as exc:
        return GpuRuntimeStatus(False, backend="tensorflow", error=str(exc))
    error = "" if devices else "no GPU devices found"
    return GpuRuntimeStatus(bool(devices), backend="tensorflow", devices=devices, error=error)


def _detect_torch_cuda() -> GpuRuntimeStatus:
    try:
        import torch
    except Exception as exc:
        return GpuRuntimeStatus(False, backend="torch", error=str(exc))
    try:
        if not torch.cuda.is_available():
            return GpuRuntimeStatus(False, backend="torch", error="CUDA is not available")
        devices = tuple(torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count()))
    except Exception as exc:
        return GpuRuntimeStatus(False, backend="torch", error=str(exc))
    return GpuRuntimeStatus(True, backend="torch", devices=devices)
