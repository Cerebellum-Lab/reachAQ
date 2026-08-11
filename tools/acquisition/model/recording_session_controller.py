"""State owner for one manually controlled recording session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.session_boundary import SessionBoundary


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
    data_complete: bool = True
    data_errors: Tuple[str, ...] = ()
    enabled_sources: Tuple[dict, ...] = ()

    def transition(self, status: SessionRecordingStatus) -> SessionRecordingStatus:
        previous = self.status
        self.status = SessionRecordingStatus(status)
        return previous

    def prepare_record(self, hardware_status: dict) -> None:
        self.analysis_finished = False
        self.analysis_started_perf = None
        self.analysis_duration_seconds = None
        self.pending_end_perf = None
        self.boundary = None
        self.hardware_status_at_record = hardware_status
        self.data_complete = True
        self.data_errors = ()
        self.enabled_sources = ()

    def set_stream_result(self, result: dict) -> None:
        self.data_complete = bool(result.get("sessionComplete", True))
        self.data_errors = tuple(result.get("incompleteReasons", ()))
        self.enabled_sources = tuple(result.get("enabledSources", ()))

    def add_data_error(self, error: str) -> None:
        self.data_complete = False
        self.data_errors = (*self.data_errors, str(error))

    def reset_after_abort(self) -> None:
        self.pending_end_perf = None
        self.boundary = None
        self.data_complete = True
        self.data_errors = ()
        self.enabled_sources = ()
        self.analysis_finished = True
        self.analysis_started_perf = None
        self.analysis_duration_seconds = None
