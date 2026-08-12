from __future__ import annotations

import dataclasses
import enum
import time
from typing import Dict, Iterable, Mapping, Optional


def _subsystem_key(subsystem_id) -> str:
    return subsystem_id.value if isinstance(subsystem_id, enum.Enum) else str(subsystem_id)


class SubsystemId(str, enum.Enum):
    REACH_SYNCHRONIZATION = "reach_synchronization"
    TOP_CAPTURE = "top_capture"
    LIVE_INFERENCE = "live_inference"
    CAN_PELLET = "can_pellet"
    NIDAQ_STREAM = "nidaq_stream"
    LASER = "laser"
    SESSION_LOGS = "session_logs"
    OFFLINE_ANALYSIS = "offline_analysis"
    ANIMAL_REGISTRY = "animal_registry"
    RFID_READER = "rfid_reader"
    RUNTIME_DIAGNOSTICS = "runtime_diagnostics"

    @staticmethod
    def camera(camera_name: str) -> str:
        return f"camera.{camera_name}"


class SubsystemState(str, enum.Enum):
    DISABLED = "disabled"
    STARTING = "starting"
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclasses.dataclass(frozen=True)
class SubsystemStatus:
    subsystem_id: str
    state: SubsystemState
    reason: str = ""
    error: str = ""
    required_for_recording: bool = False
    generation: int = 0
    started_perf_time: Optional[float] = None
    finished_perf_time: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self.state is SubsystemState.READY

    @property
    def recording_blocker(self) -> Optional[str]:
        if not self.required_for_recording or self.is_ready:
            return None
        detail = self.error or self.reason or self.state.value
        return f"{self.subsystem_id}: {detail}"

    def transition(
        self,
        state: SubsystemState,
        *,
        reason: str = "",
        error: str = "",
        required_for_recording: Optional[bool] = None,
        generation: Optional[int] = None,
        now: Optional[float] = None,
    ) -> "SubsystemStatus":
        now = time.perf_counter() if now is None else float(now)
        started = self.started_perf_time
        finished = self.finished_perf_time
        if state is SubsystemState.STARTING:
            started = now
            finished = None
        elif state in {
            SubsystemState.READY,
            SubsystemState.BLOCKED,
            SubsystemState.FAILED,
            SubsystemState.STOPPED,
            SubsystemState.DISABLED,
        }:
            finished = now
        return dataclasses.replace(
            self,
            state=state,
            reason=reason,
            error=error,
            required_for_recording=(
                self.required_for_recording
                if required_for_recording is None
                else bool(required_for_recording)
            ),
            generation=self.generation if generation is None else int(generation),
            started_perf_time=started,
            finished_perf_time=finished,
        )


class SubsystemStatusRegistry:
    """Thread-safe ownership is provided by the AppModel lock.

    This class deliberately contains no callbacks. AppModel publishes one
    property-change event after replacing an immutable status value.
    """

    def __init__(self):
        self._statuses: Dict[str, SubsystemStatus] = {}

    @property
    def statuses(self) -> Mapping[str, SubsystemStatus]:
        return dict(self._statuses)

    def get(self, subsystem_id: str) -> Optional[SubsystemStatus]:
        return self._statuses.get(_subsystem_key(subsystem_id))

    def ensure(
        self,
        subsystem_id: str,
        *,
        state: SubsystemState = SubsystemState.DISABLED,
        required_for_recording: bool = False,
        reason: str = "",
    ) -> SubsystemStatus:
        subsystem_id = _subsystem_key(subsystem_id)
        status = self._statuses.get(subsystem_id)
        if status is None:
            status = SubsystemStatus(
                subsystem_id=subsystem_id,
                state=state,
                reason=reason,
                required_for_recording=required_for_recording,
            )
            self._statuses[subsystem_id] = status
        return status

    def transition(
        self,
        subsystem_id: str,
        state: SubsystemState,
        **kwargs,
    ) -> tuple[SubsystemStatus, Optional[SubsystemStatus]]:
        subsystem_id = _subsystem_key(subsystem_id)
        previous = self._statuses.get(subsystem_id)
        if previous is None:
            previous = SubsystemStatus(
                subsystem_id=subsystem_id,
                state=SubsystemState.STOPPED,
            )
        generation = kwargs.get("generation")
        if (
            generation is not None
            and int(generation) < previous.generation
        ):
            return previous, previous
        status = previous.transition(state, **kwargs)
        self._statuses[subsystem_id] = status
        return status, previous

    def begin_retry(
        self,
        subsystem_id: str,
        *,
        required_for_recording: Optional[bool] = None,
        reason: str = "",
    ) -> tuple[SubsystemStatus, Optional[SubsystemStatus]]:
        current = self.get(subsystem_id)
        generation = 1 if current is None else current.generation + 1
        return self.transition(
            subsystem_id,
            SubsystemState.STARTING,
            required_for_recording=required_for_recording,
            generation=generation,
            reason=reason,
        )

    def recording_blockers(self) -> tuple[str, ...]:
        return tuple(
            blocker
            for status in self._statuses.values()
            if (blocker := status.recording_blocker) is not None
        )

    def snapshot(self, subsystem_ids: Optional[Iterable[str]] = None) -> dict:
        if subsystem_ids is None:
            statuses = self._statuses.values()
        else:
            statuses = (
                self._statuses[subsystem_id]
                for subsystem_id in subsystem_ids
                if subsystem_id in self._statuses
            )
        return {
            status.subsystem_id: {
                **dataclasses.asdict(status),
                "state": status.state.value,
            }
            for status in statuses
        }
