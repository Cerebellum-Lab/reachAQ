"""Exactly-once public session and pellet-trial lifecycle events."""

from __future__ import annotations

from typing import Optional, Set

from autotrainer.api import ApiEventKind


class SessionApiPublisher:
    def __init__(self, event_manager):
        self._event_manager = event_manager
        self._session_id: Optional[str] = None
        self._started_attempts: Set[str] = set()
        self._capture_ended_attempts: Set[str] = set()
        self._ended_attempts: Set[str] = set()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def session_started(self, session_id: str) -> bool:
        if self._session_id == str(session_id):
            return False
        if self._session_id is not None:
            raise RuntimeError("Cannot open a second API session before ending the first")
        self._session_id = str(session_id)
        self._started_attempts.clear()
        self._capture_ended_attempts.clear()
        self._ended_attempts.clear()
        self._event_manager.post_event_content(
            ApiEventKind.sessionStarted,
            data={
                "session_id": self._session_id,
                "is_analysis_deferred": True,
                "recording_scope": "continuous_session",
            },
        )
        return True

    def trial_started(self, attempt) -> bool:
        key = attempt.operation_id
        if key in self._started_attempts:
            return False
        self._require_session(attempt.session_id)
        self._started_attempts.add(key)
        self._event_manager.post_event_content(
            ApiEventKind.trialStarted,
            data=self._attempt_context(attempt, reason="pellet_send_dispatch"),
        )
        return True

    def trial_capture_ended(self, attempt) -> bool:
        key = attempt.operation_id
        if key in self._capture_ended_attempts:
            return False
        self._require_session(attempt.session_id)
        if key not in self._started_attempts:
            self.trial_started(attempt)
        self._capture_ended_attempts.add(key)
        self._event_manager.post_event_content(
            ApiEventKind.trialCaptureEnded,
            data=self._attempt_context(attempt),
        )
        return True

    def trial_ended(self, attempt) -> bool:
        key = attempt.operation_id
        if key in self._ended_attempts:
            return False
        self._require_session(attempt.session_id)
        if key not in self._capture_ended_attempts:
            self.trial_capture_ended(attempt)
        self._ended_attempts.add(key)
        result = (
            "analysis_succeeded"
            if attempt.outcome is not None and attempt.outcome.is_scored
            else "analysis_failed"
        )
        self._event_manager.post_event_content(
            ApiEventKind.trialEnded,
            data={
                **self._attempt_context(attempt),
                "result": result,
                "outcome": None if attempt.outcome is None else attempt.outcome.value,
                "error": attempt.error,
            },
        )
        return True

    def session_ended(self, summary: dict, *, aborted: bool = False) -> bool:
        if self._session_id is None:
            return False
        session_id = self._session_id
        self._event_manager.post_event_content(
            ApiEventKind.sessionEnded,
            data={
                "session_id": session_id,
                "capture_trial_count": int(summary.get("physical_attempts", 0)),
                "analysis_trial_count": int(summary.get("scored_trials", 0)),
                "failed_trial_count": int(summary.get("hardware_errors", 0))
                + int(summary.get("incomplete_attempts", 0)),
                "aborted": bool(aborted),
            },
        )
        self._session_id = None
        return True

    def _require_session(self, session_id: str) -> None:
        if self._session_id is None:
            self.session_started(session_id)
        if self._session_id != str(session_id):
            raise RuntimeError(
                f"Pellet trial belongs to {session_id}, active API session is "
                f"{self._session_id}"
            )

    @staticmethod
    def _attempt_context(attempt, **extra):
        return {
            "session_id": attempt.session_id,
            "trial_id": attempt.trial_id,
            "attempt_id": attempt.attempt_id,
            "attempt_label": attempt.attempt_label,
            "operation_id": attempt.operation_id,
            **extra,
        }
