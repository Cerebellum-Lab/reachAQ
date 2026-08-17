"""Isolated exact NI task verification used before the streaming worker."""

from __future__ import annotations

import dataclasses
import queue
from typing import Optional, Tuple

from autotrainer.core import NidaqSignalStreamConfiguration, NidaqTimingPlan
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.device import NidaqSignalStreamController


@dataclasses.dataclass(frozen=True)
class NidaqPreflightResult:
    status: str
    verified_tasks: Tuple[str, ...] = ()
    failing_task: str = ""
    stage: str = ""
    error_type: str = ""
    error: str = ""
    daqmx_code: Optional[int] = None
    corrective_action: str = ""
    multidevice_probe_status: str = "not_requested"
    multidevice_probe_error: str = ""

    @property
    def is_valid(self):
        return self.status == "verified"


def _preflight_worker(configuration, timing_plan, result_queue):
    controller = None
    verified = ()
    try:
        strategy = getattr(
            getattr(timing_plan, "task_graph", None),
            "strategy",
            "per_device",
        )
        requested = strategy in {"auto_multidevice", "forced_multidevice"}
        input_devices = {
            str(channel.physical_channel).strip("/").split("/", 1)[0]
            for channel in getattr(configuration, "channels", ())
        }
        multidevice_status = "not_requested"
        multidevice_error = ""
        if requested and len(input_devices) > 1:
            from tools.acquisition.model.nidaq_timing import (
                resolve_nidaq_multidevice_probe,
            )
            combined_plan = (
                resolve_nidaq_multidevice_probe(timing_plan, "verified")
                if isinstance(timing_plan, NidaqTimingPlan)
                else timing_plan
            )
            try:
                controller = NidaqSignalStreamController(
                    configuration,
                    timing_plan=combined_plan,
                )
                verified = controller.verify_tasks(commit=True)
                multidevice_status = "verified"
            except Exception as error:
                multidevice_error = str(error) or type(error).__name__
                if controller is not None:
                    try:
                        controller.close()
                    except Exception:
                        # Preserve the Verify/Commit error which selected the
                        # fallback. Cleanup is retried by the outer finalizer.
                        pass
                    controller = None
                if strategy == "forced_multidevice":
                    raise RuntimeError(
                        "Forced NI multidevice task verification failed: "
                        + multidevice_error
                    ) from error
                multidevice_status = "fallback_per_device"
        elif requested:
            multidevice_status = "not_applicable_single_device"

        if controller is None:
            from tools.acquisition.model.nidaq_timing import (
                resolve_nidaq_multidevice_probe,
            )
            selected_plan = (
                resolve_nidaq_multidevice_probe(timing_plan, multidevice_status)
                if isinstance(timing_plan, NidaqTimingPlan)
                else timing_plan
            )
            controller = NidaqSignalStreamController(
                configuration,
                timing_plan=selected_plan,
            )
            verified = controller.verify_tasks(commit=True)
        result = NidaqPreflightResult(
            "verified",
            verified_tasks=verified,
            multidevice_probe_status=multidevice_status,
            multidevice_probe_error=multidevice_error,
        )
    except Exception as error:
        result = NidaqPreflightResult(
            "failed",
            verified_tasks=verified,
            stage="create_verify_commit",
            error_type=type(error).__name__,
            error=str(error) or type(error).__name__,
            daqmx_code=getattr(error, "error_code", None),
            corrective_action=(
                "Check the named physical channels, timing routes, counter/port "
                "availability, and whether another process owns the resource."
            ),
        )
    finally:
        if controller is not None:
            try:
                controller.close()
            except Exception as cleanup_error:
                if result.is_valid:
                    result = NidaqPreflightResult(
                        "failed",
                        verified_tasks=verified,
                        stage="cleanup",
                        error_type=type(cleanup_error).__name__,
                        error=str(cleanup_error),
                        corrective_action="Restart NI acquisition after releasing the task resources.",
                    )
        result_queue.put(result)


def run_isolated_nidaq_preflight(
    configuration: NidaqSignalStreamConfiguration,
    timing_plan: NidaqTimingPlan,
    *,
    timeout_seconds: float = 15.0,
    mp_ctx=None,
) -> NidaqPreflightResult:
    context = get_mp_ctx() if mp_ctx is None else mp_ctx
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_preflight_worker,
        args=(configuration, timing_plan, result_queue),
        name="NidaqExactPreflight",
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(2)
        return NidaqPreflightResult(
            "failed",
            stage="timeout",
            error_type="TimeoutError",
            error=f"NI exact task preflight exceeded {timeout_seconds:g} seconds",
            corrective_action="Check the NI driver and release occupied DAQmx resources.",
        )
    try:
        return result_queue.get(timeout=1)
    except queue.Empty:
        return NidaqPreflightResult(
            "failed",
            stage="worker_exit",
            error_type="RuntimeError",
            error=f"NI preflight exited with code {process.exitcode} without a result",
            corrective_action="Inspect the NI-DAQmx runtime and application logs.",
        )
