from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional, Tuple

from autotrainer.core.logging import get_verbose_logger, log_hardware_initialization


logger = get_verbose_logger(__name__)


@dataclass(frozen=True)
class GpuRuntimeStatus:
    is_available: bool
    backend: str = ""
    devices: Tuple[str, ...] = tuple()
    error: str = ""


def detect_gpu_runtime(*, required_backend: Optional[str] = None) -> GpuRuntimeStatus:
    """Detect a usable NVIDIA driver and GPU inference backend.

    The driver probe intentionally runs before importing TensorFlow or PyTorch.
    A machine using nouveau, or without a working ``nvidia-smi``, therefore
    fails in a few seconds instead of paying the cost of both framework imports.
    """

    detection_started = time.perf_counter()
    log_hardware_initialization(logger, "START | NVIDIA driver preflight")
    driver_status = _detect_nvidia_driver()
    if not driver_status.is_available:
        log_hardware_initialization(
            logger,
            "FAILED | NVIDIA driver preflight | elapsed=%.3fs error=%s",
            time.perf_counter() - detection_started,
            driver_status.error,
        )
        return driver_status
    log_hardware_initialization(
        logger,
        "READY | NVIDIA driver preflight | devices=%s elapsed=%.3fs",
        driver_status.devices,
        time.perf_counter() - detection_started,
    )

    detectors = (
        ("tensorflow", _detect_tensorflow_gpu),
        ("torch", _detect_torch_cuda),
    )
    if required_backend is not None:
        detectors = tuple(item for item in detectors if item[0] == required_backend)
        if not detectors:
            raise ValueError(f"Unsupported GPU backend: {required_backend!r}")

    statuses = []
    for backend, detector in detectors:
        backend_started = time.perf_counter()
        log_hardware_initialization(
            logger,
            "START | GPU runtime probe | backend=%s",
            backend,
        )
        status = detector()
        statuses.append(status)
        log_hardware_initialization(
            logger,
            "%s | GPU runtime probe | backend=%s devices=%s elapsed=%.3fs%s",
            "READY" if status.is_available else "UNAVAILABLE",
            backend,
            status.devices,
            time.perf_counter() - backend_started,
            "" if not status.error else f" error={status.error}",
        )
        if status.is_available:
            log_hardware_initialization(
                logger,
                "READY | live inference GPU | backend=%s devices=%s elapsed=%.3fs",
                status.backend,
                status.devices,
                time.perf_counter() - detection_started,
            )
            return status

    errors = "; ".join(
        f"{status.backend}: {status.error or 'no GPU devices found'}"
        for status in statuses
    )
    log_hardware_initialization(
        logger,
        "FAILED | live inference GPU | elapsed=%.3fs error=%s",
        time.perf_counter() - detection_started,
        errors,
    )
    backend = required_backend or "/".join(status.backend for status in statuses)
    return GpuRuntimeStatus(False, backend=backend, error=errors)


def _detect_nvidia_driver() -> GpuRuntimeStatus:
    if sys.platform.startswith("linux"):
        active_drivers = _linux_nvidia_display_drivers()
        if active_drivers and not any(driver.startswith("nvidia") for driver in active_drivers):
            return GpuRuntimeStatus(
                False,
                backend="nvidia-driver",
                error=(
                    "NVIDIA GPU found, but its active kernel driver is "
                    f"{', '.join(active_drivers)}; CUDA requires the proprietary NVIDIA driver"
                ),
            )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return GpuRuntimeStatus(
            False,
            backend="nvidia-driver",
            error="nvidia-smi is unavailable; the proprietary NVIDIA driver is not installed or active",
        )

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GpuRuntimeStatus(
            False,
            backend="nvidia-driver",
            error="nvidia-smi timed out after 3 seconds",
        )
    except OSError as exc:
        return GpuRuntimeStatus(False, backend="nvidia-driver", error=str(exc))

    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return GpuRuntimeStatus(
            False,
            backend="nvidia-driver",
            error=f"nvidia-smi could not communicate with the NVIDIA driver: {error}",
        )

    devices = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if not devices:
        return GpuRuntimeStatus(
            False,
            backend="nvidia-driver",
            error="nvidia-smi reported no NVIDIA GPU devices",
        )
    return GpuRuntimeStatus(True, backend="nvidia-driver", devices=devices)


def _linux_nvidia_display_drivers() -> Tuple[str, ...]:
    drivers = []
    for device_path in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if device_path.joinpath("vendor").read_text().strip().lower() != "0x10de":
                continue
            if not device_path.joinpath("class").read_text().strip().lower().startswith("0x03"):
                continue
            driver_path = device_path.joinpath("driver")
            driver = driver_path.resolve().name if driver_path.exists() else "unbound"
            if driver not in drivers:
                drivers.append(driver)
        except OSError:
            continue
    return tuple(drivers)


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


_CUDNN_PROBE_SOURCE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "with torch.no_grad():\n"
    "    probe = torch.randn(1, 8, 16, 16, device='cuda')\n"
    "    nn.Conv2d(8, 8, 3, padding=1).to('cuda')(probe)\n"
    "torch.cuda.synchronize()\n"
)

_CUDNN_PROBE_TIMEOUT_SECONDS = 120


def _probe_torch_cudnn() -> Optional[str]:
    """Run one convolution in a subprocess. Returns an error string, or None.

    This cannot be done in-process. A cuDNN that fails to load does not raise:
    it prints to stderr and aborts the interpreter, so a try/except around the
    convolution would take the caller down with it. Running it in a subprocess
    turns that abort into a non-zero return code we can report.

    The cost is one interpreter start plus a torch import, which is acceptable
    for a preflight that already pays for a framework import.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CUDNN_PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=_CUDNN_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"cuDNN probe timed out after {_CUDNN_PROBE_TIMEOUT_SECONDS} seconds"
    except OSError as exc:
        return f"cuDNN probe could not start: {exc}"
    if completed.returncode == 0:
        return None
    detail = (completed.stderr or completed.stdout or "").strip()
    # A signal shows as a negative return code; a core dump from cuDNN's loader
    # lands here rather than as an exception.
    first_line = detail.splitlines()[0] if detail else f"exit code {completed.returncode}"
    return first_line[:400]


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

    # A visible device is not a working one. torch.cuda.is_available() checks the
    # driver and CUDA runtime only; on this fleet cuDNN can be unloadable while
    # it still returns True, because nvidia-cudnn-cu11 and nvidia-cudnn-cu12
    # both install libcudnn.so.8 and whichever lands last wins. Inference needs
    # convolutions, so verify one actually runs.
    cudnn_error = _probe_torch_cudnn()
    if cudnn_error is not None:
        return GpuRuntimeStatus(
            False,
            backend="torch",
            devices=devices,
            error=f"CUDA is present but cuDNN cannot run a convolution: {cudnn_error}",
        )
    return GpuRuntimeStatus(True, backend="torch", devices=devices)
