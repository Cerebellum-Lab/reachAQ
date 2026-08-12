"""State owner for one manually controlled recording session."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.session_boundary import SessionBoundary


@dataclass(frozen=True)
class SessionGeneration:
    generation: int
    session_id: str


@dataclass
class RecordingSessionController:
    """Own session lifecycle, canonical boundary, and completeness state.

    Device orchestration remains in AppModel; the mutable state describing one
    recording no longer does. This keeps session state independent from camera,
    NI-DAQ, CAN, laser, and analysis implementations.
    """

    status: SessionRecordingStatus = SessionRecordingStatus.READY
    analysis_finished: bool = True
    analysis_started_perf: Optional[float] = None
    analysis_duration_seconds: Optional[float] = None
    pending_end_perf: Optional[float] = None
    boundary: Optional[SessionBoundary] = None
    hardware_status_at_record: Optional[dict] = None
    animal_snapshot: Optional[dict] = None
    data_complete: bool = True
    data_errors: Tuple[str, ...] = ()
    enabled_sources: Tuple[dict, ...] = ()
    end_actions: Tuple[dict, ...] = ()
    storage_telemetry: dict = field(default_factory=dict)
    _generation: int = field(default=0, init=False, repr=False)
    _session_id: Optional[str] = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def session_id(self) -> Optional[str]:
        with self._lock:
            return self._session_id

    @property
    def metadata_generation_id(self) -> Optional[str]:
        with self._lock:
            if self._session_id is None:
                return None
            return f"{self._session_id}-g{self._generation}"

    def token(self) -> Optional[SessionGeneration]:
        with self._lock:
            if self._session_id is None:
                return None
            return SessionGeneration(self._generation, self._session_id)

    def is_current(
        self,
        token: Optional[SessionGeneration],
        *,
        statuses: Optional[Tuple[SessionRecordingStatus, ...]] = None,
    ) -> bool:
        if token is None:
            return False
        with self._lock:
            current = (
                token.generation == self._generation
                and token.session_id == self._session_id
            )
            return current and (statuses is None or self.status in statuses)

    def begin_record(
        self,
        session_id: str,
        hardware_status: dict,
        *,
        animal_snapshot: Optional[dict] = None,
    ) -> Optional[Tuple[SessionRecordingStatus, SessionGeneration]]:
        with self._lock:
            if self.status is not SessionRecordingStatus.READY:
                return None
            self._generation += 1
            self._session_id = str(session_id)
            self._prepare_record_unlocked(
                hardware_status,
                animal_snapshot=animal_snapshot,
            )
            previous = self.status
            self.status = SessionRecordingStatus.ARMING
            return previous, SessionGeneration(self._generation, self._session_id)

    def transition(
        self,
        status: SessionRecordingStatus,
        *,
        expected: Optional[Tuple[SessionRecordingStatus, ...]] = None,
        token: Optional[SessionGeneration] = None,
    ) -> Optional[SessionRecordingStatus]:
        with self._lock:
            if token is not None and not self.is_current(token):
                return None
            if expected is not None and self.status not in expected:
                return None
            previous = self.status
            self.status = SessionRecordingStatus(status)
            return previous

    def prepare_record(
        self,
        hardware_status: dict,
        *,
        animal_snapshot: Optional[dict] = None,
    ) -> None:
        with self._lock:
            self._prepare_record_unlocked(
                hardware_status,
                animal_snapshot=animal_snapshot,
            )

    def _prepare_record_unlocked(
        self,
        hardware_status: dict,
        *,
        animal_snapshot: Optional[dict] = None,
    ) -> None:
        self.analysis_finished = False
        self.analysis_started_perf = None
        self.analysis_duration_seconds = None
        self.pending_end_perf = None
        self.boundary = None
        self.hardware_status_at_record = hardware_status
        self.animal_snapshot = animal_snapshot
        self.data_complete = True
        self.data_errors = ()
        self.enabled_sources = ()
        self.end_actions = ()
        self.storage_telemetry = {}

    def reserve_end_action(self, name: str, details: dict) -> bool:
        with self._lock:
            if any(action.get("name") == name for action in self.end_actions):
                return False
            self.end_actions = (*self.end_actions, {
                "name": str(name),
                "status": "pending",
                **dict(details),
            })
            return True

    def finish_end_action(self, name: str, **result) -> None:
        with self._lock:
            self.end_actions = tuple(
                ({**action, **result} if action.get("name") == name else action)
                for action in self.end_actions
            )

    def set_stream_result(self, result: dict) -> None:
        with self._lock:
            self.data_complete = bool(result.get("sessionComplete", True))
            self.data_errors = tuple(result.get("incompleteReasons", ()))
            self.enabled_sources = tuple(result.get("enabledSources", ()))

    def add_data_error(self, error: str) -> None:
        with self._lock:
            self.data_complete = False
            self.data_errors = (*self.data_errors, str(error))

    def set_storage_telemetry(self, telemetry: dict) -> None:
        with self._lock:
            self.storage_telemetry = dict(telemetry)

    def set_boundary(
        self,
        boundary: SessionBoundary,
        token: SessionGeneration,
    ) -> bool:
        with self._lock:
            if not self.is_current(token) or boundary.session_id != token.session_id:
                return False
            self.boundary = boundary
            return True

    def set_pending_end(
        self,
        end_perf: float,
        token: SessionGeneration,
    ) -> bool:
        with self._lock:
            if not self.is_current(token):
                return False
            self.pending_end_perf = float(end_perf)
            return True

    def take_pending_end(self, token: SessionGeneration) -> Optional[float]:
        with self._lock:
            if not self.is_current(token):
                return None
            value = self.pending_end_perf
            self.pending_end_perf = None
            return value

    def reset_after_abort(self) -> None:
        with self._lock:
            self.pending_end_perf = None
            self.boundary = None
            self.data_complete = True
            self.data_errors = ()
            self.enabled_sources = ()
            self.end_actions = ()
            self.storage_telemetry = {}
            self.analysis_finished = True
            self.analysis_started_perf = None
            self.analysis_duration_seconds = None
            self.animal_snapshot = None
