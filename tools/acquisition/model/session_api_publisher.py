"""Exactly-once public session and pellet-trial lifecycle events."""

from __future__ import annotations

import collections
import functools
import threading
from typing import Optional, Set

from autotrainer.api import ApiEventKind
from autotrainer.core.event.event_manager import EventQueueFullError


def _publisher_locked(method):
    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class SessionApiPublisher:
    def __init__(self, event_manager):
        self._event_manager = event_manager
        self._lock = threading.RLock()
        self._pending_events = collections.deque(maxlen=1024)
        self._session_id: Optional[str] = None
        self._started_attempts: Set[str] = set()
        self._capture_ended_attempts: Set[str] = set()
        self._ended_attempts: Set[str] = set()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @_publisher_locked
    def session_started(self, session_id: str) -> bool:
        if self._session_id == str(session_id):
            return False
        if self._session_id is not None:
            raise RuntimeError("Cannot open a second API session before ending the first")
        session_id = str(session_id)
        self._post_event(
            ApiEventKind.sessionStarted,
            data={
                "session_id": session_id,
                "is_analysis_deferred": True,
                "recording_scope": "continuous_session",
            },
        )
        self._session_id = session_id
        self._started_attempts.clear()
        self._capture_ended_attempts.clear()
        self._ended_attempts.clear()
        return True

    @_publisher_locked
    def trial_started(self, attempt) -> bool:
        key = attempt.operation_id
        if key in self._started_attempts:
            return False
        self._require_session(attempt.session_id)
        self._post_event(
            ApiEventKind.trialStarted,
            data=self._attempt_context(attempt, reason="pellet_send_dispatch"),
        )
        self._started_attempts.add(key)
        return True

    @_publisher_locked
    def trial_capture_ended(self, attempt) -> bool:
        key = attempt.operation_id
        if key in self._capture_ended_attempts:
            return False
        self._require_session(attempt.session_id)
        if key not in self._started_attempts:
            self.trial_started(attempt)
        self._post_event(
            ApiEventKind.trialCaptureEnded,
            data=self._attempt_context(attempt),
        )
        self._capture_ended_attempts.add(key)
        return True

    @_publisher_locked
    def trial_ended(self, attempt) -> bool:
        key = attempt.operation_id
        if key in self._ended_attempts:
            return False
        self._require_session(attempt.session_id)
        if key not in self._capture_ended_attempts:
            self.trial_capture_ended(attempt)
        result = (
            "analysis_succeeded"
            if attempt.outcome is not None and attempt.outcome.is_scored
            else "analysis_failed"
        )
        self._post_event(
            ApiEventKind.trialEnded,
            data={
                **self._attempt_context(attempt),
                "result": result,
                "outcome": None if attempt.outcome is None else attempt.outcome.value,
                "error": attempt.error,
            },
        )
        self._ended_attempts.add(key)
        return True

    @_publisher_locked
    def session_ended(self, summary: dict, *, aborted: bool = False) -> bool:
        if self._session_id is None:
            return False
        session_id = self._session_id
        self._post_event(
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

    @property
    def pending_event_count(self) -> int:
        with self._lock:
            return len(self._pending_events)

    def flush_pending(self) -> int:
        """Retry the bounded lifecycle outbox without changing the public API."""
        with self._lock:
            while self._pending_events:
                kind, data = self._pending_events[0]
                try:
                    self._event_manager.post_event_content(kind, data=data)
                except EventQueueFullError:
                    break
                self._pending_events.popleft()
            return len(self._pending_events)

    def _post_event(self, kind, *, data) -> None:
        with self._lock:
            self.flush_pending()
            if self._pending_events:
                self._append_pending(kind, data)
                return
            try:
                self._event_manager.post_event_content(kind, data=data)
            except EventQueueFullError:
                self._append_pending(kind, data)

    def _append_pending(self, kind, data) -> None:
        if len(self._pending_events) >= self._pending_events.maxlen:
            raise RuntimeError(
                "Public session lifecycle outbox is full; restart acquisition "
                "before continuing pellet trials"
            )
        self._pending_events.append((kind, dict(data)))

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
