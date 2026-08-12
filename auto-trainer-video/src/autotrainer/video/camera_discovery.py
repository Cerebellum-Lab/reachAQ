from __future__ import annotations

import multiprocessing
import time
import traceback
from dataclasses import dataclass
from queue import Empty
from typing import Callable, Optional, Tuple


DEFAULT_CAMERA_DISCOVERY_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True)
class CameraDiscoveryFailure:
    stage: str
    exception_type: str
    message: str
    traceback_text: str


class CameraDiscoveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        elapsed_seconds: float,
        exception_type: str = "",
        traceback_text: str = "",
    ):
        super().__init__(message)
        self.stage = stage
        self.elapsed_seconds = float(elapsed_seconds)
        self.exception_type = exception_type
        self.traceback_text = traceback_text


class CameraDiscoveryTimeout(CameraDiscoveryError):
    pass


def _spin_discovery_worker(result_queue) -> None:
    stage = "load_backend"
    try:
        from .video_manager import _get_spincam_cls

        spin_camera = _get_spincam_cls()
        stage = "enumerate"
        serials = tuple(str(serial) for serial in spin_camera.list())
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                CameraDiscoveryFailure(
                    stage=stage,
                    exception_type=exc.__class__.__name__,
                    message=str(exc) or exc.__class__.__name__,
                    traceback_text=traceback.format_exc(),
                ),
            )
        )
    else:
        result_queue.put(("ok", serials))


def discover_spin_cameras(
    *,
    timeout_seconds: float = DEFAULT_CAMERA_DISCOVERY_TIMEOUT_SECONDS,
    _worker_target: Optional[Callable] = None,
) -> Tuple[str, ...]:
    """Enumerate Spinnaker cameras without allowing a vendor call to hang."""

    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("camera discovery timeout must be greater than zero")
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker_target or _spin_discovery_worker,
        args=(result_queue,),
        name="SpinnakerDiscovery",
        daemon=True,
    )
    try:
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(2.0)
            raise CameraDiscoveryTimeout(
                f"Spinnaker camera discovery exceeded {timeout_seconds:g} seconds",
                stage="timeout",
                elapsed_seconds=time.perf_counter() - started,
                exception_type="TimeoutError",
            )
        try:
            state, payload = result_queue.get(timeout=0.25)
        except Empty as exc:
            raise CameraDiscoveryError(
                "Spinnaker discovery process exited without a result "
                f"(exit code {process.exitcode})",
                stage="worker_exit",
                elapsed_seconds=time.perf_counter() - started,
                exception_type="ChildProcessError",
            ) from exc
        if state == "ok":
            return tuple(payload)
        failure = payload
        raise CameraDiscoveryError(
            f"Spinnaker {failure.stage} failed: {failure.message}",
            stage=failure.stage,
            elapsed_seconds=time.perf_counter() - started,
            exception_type=failure.exception_type,
            traceback_text=failure.traceback_text,
        )
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        result_queue.close()
        result_queue.join_thread()
