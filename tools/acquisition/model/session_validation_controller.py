"""Nonblocking process owner for automatic and operator session validation."""

from __future__ import annotations

import dataclasses
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from autotrainer.core.multiproc import get_mp_ctx
from tools.acquisition.model.atomic_session_io import atomic_write_json
from tools.session_validation import ValidationProfile, validate_session


@dataclasses.dataclass(frozen=True)
class SessionValidationState:
    status: str = "idle"
    profile: str = ""
    session_path: str = ""
    report_path: str = ""
    started_perf: float = 0.0
    elapsed_seconds: float = 0.0
    counts: dict = dataclasses.field(default_factory=dict)
    error: str = ""

    @property
    def active(self):
        return self.status in {"pending", "running"}


def _validation_worker(session_path, profile, output_path, result_queue, reduced_priority):
    try:
        if reduced_priority and hasattr(os, "nice"):
            os.nice(10)
        report = validate_session(session_path, profile=profile)
        atomic_write_json(Path(output_path), report.to_record())
        result_queue.put(("finished", report.counts, report.exit_code, ""))
    except Exception as error:
        result_queue.put(("error", {}, 2, f"{type(error).__name__}: {error}"))


class SessionValidationController:
    def __init__(self, callback=None, *, mp_ctx=None):
        self._callback = callback
        self._mp_ctx = get_mp_ctx() if mp_ctx is None else mp_ctx
        self._lock = threading.RLock()
        self._state = SessionValidationState()
        self._process = None
        self._monitor = None

    @property
    def state(self):
        with self._lock:
            state = self._state
            if state.active:
                state = dataclasses.replace(
                    state,
                    elapsed_seconds=max(0.0, time.perf_counter() - state.started_perf),
                )
            return state

    def start(self, session_path, profile, *, automatic=False, reduced_priority=True):
        session_path = Path(session_path).resolve()
        profile = ValidationProfile(profile)
        with self._lock:
            if self._state.active:
                raise RuntimeError("Another session validation is already running")
            validation_dir = session_path / "validation"
            if automatic:
                if profile is not ValidationProfile.QUICK:
                    raise ValueError("Automatic validation must use the quick profile")
                output_path = validation_dir / "quick.json"
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                output_path = validation_dir / (
                    f"{profile.value}-{stamp}-{uuid.uuid4().hex[:8]}.json"
                )
            result_queue = self._mp_ctx.Queue(maxsize=1)
            process = self._mp_ctx.Process(
                target=_validation_worker,
                args=(
                    session_path.as_posix(), profile.value,
                    output_path.as_posix(), result_queue, reduced_priority,
                ),
                name=f"SessionValidation-{profile.value}",
                daemon=True,
            )
            self._process = process
            self._state = SessionValidationState(
                status="pending",
                profile=profile.value,
                session_path=session_path.as_posix(),
                report_path=output_path.as_posix(),
                started_perf=time.perf_counter(),
            )
            self._notify()
            process.start()
            self._state = dataclasses.replace(self._state, status="running")
            self._notify()
            monitor = threading.Thread(
                target=self._monitor_process,
                args=(process, result_queue),
                name=f"MonitorSessionValidation-{profile.value}",
                daemon=True,
            )
            self._monitor = monitor
            monitor.start()
            return output_path

    def cancel(self):
        with self._lock:
            process = self._process
            if process is None or not self._state.active:
                return False
            process.terminate()
            process.join(2)
            self._state = dataclasses.replace(
                self._state,
                status="cancelled",
                elapsed_seconds=time.perf_counter() - self._state.started_perf,
            )
            self._process = None
            self._notify()
            return True

    def wait(self, timeout=None):
        monitor = self._monitor
        if monitor is not None:
            monitor.join(timeout)
        return self.state

    def _monitor_process(self, process, result_queue):
        process.join()
        try:
            status, counts, exit_code, error = result_queue.get(timeout=1)
        except queue.Empty:
            status, counts, exit_code, error = (
                "error", {}, 2,
                f"validator exited with code {process.exitcode} without a result",
            )
        with self._lock:
            if process is not self._process:
                return
            final_status = (
                "error" if status == "error"
                else "fail" if exit_code == 1
                else "tool_error" if exit_code == 2
                else "warning" if counts.get("warning", 0)
                else "pass"
            )
            self._state = dataclasses.replace(
                self._state,
                status=final_status,
                elapsed_seconds=time.perf_counter() - self._state.started_perf,
                counts=dict(counts),
                error=error,
            )
            self._process = None
            self._notify()

    def _notify(self):
        if self._callback is not None:
            self._callback(self._state)
