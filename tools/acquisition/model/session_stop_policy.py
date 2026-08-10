"""Automatic recording-session stop policy evaluation."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple


class SessionStopReason(str, enum.Enum):
    DURATION_LIMIT = "duration_limit"
    TRIAL_LIMIT = "trial_limit"
    PROTOCOL_COMPLETE = "protocol_complete"

    @property
    def display_name(self) -> str:
        return {
            self.DURATION_LIMIT: "Recording time reached",
            self.TRIAL_LIMIT: "Trial target reached",
            self.PROTOCOL_COMPLETE: "Protocol completed",
        }[self]


class SessionStopDecision(str, enum.Enum):
    NONE = "none"
    FINISH_ACTIVE_TRIAL = "finish_active_trial"
    STOP = "stop"
    TIMEOUT_ERROR = "timeout_error"


@dataclass(frozen=True)
class SessionStopConfiguration:
    duration_seconds: Optional[float] = None
    trial_limit: Optional[int] = None
    stop_on_protocol_complete: bool = False
    drain_timeout_seconds: float = 15.0

    def __post_init__(self):
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("Recording duration must be positive")
        if self.trial_limit is not None and self.trial_limit <= 0:
            raise ValueError("Trial target must be positive")
        if self.drain_timeout_seconds <= 0:
            raise ValueError("Stop drain timeout must be positive")


@dataclass(frozen=True)
class SessionStopEvaluation:
    decision: SessionStopDecision
    reason: Optional[SessionStopReason] = None
    triggered_reasons: Tuple[SessionStopReason, ...] = tuple()
    requested_perf_time: Optional[float] = None
    timeout_seconds: Optional[float] = None

    @property
    def is_error(self) -> bool:
        return self.decision is SessionStopDecision.TIMEOUT_ERROR


class SessionStopPolicy:
    """Request Stop at a threshold, then let the active trial finish normally."""

    def __init__(self, configuration: Optional[SessionStopConfiguration] = None):
        self.configuration = configuration or SessionStopConfiguration()
        self._started_perf_time: Optional[float] = None
        self._requested_perf_time: Optional[float] = None
        self._reason: Optional[SessionStopReason] = None
        self._triggered_reasons: Tuple[SessionStopReason, ...] = tuple()

    @property
    def is_started(self) -> bool:
        return self._started_perf_time is not None

    @property
    def is_stop_requested(self) -> bool:
        return self._requested_perf_time is not None

    def start(self, perf_time: float) -> None:
        self._started_perf_time = float(perf_time)
        self._requested_perf_time = None
        self._reason = None
        self._triggered_reasons = tuple()

    def reset(self) -> None:
        self._started_perf_time = None
        self._requested_perf_time = None
        self._reason = None
        self._triggered_reasons = tuple()

    def evaluate(
        self,
        perf_time: float,
        *,
        trial_count: int,
        protocol_complete: bool,
        trial_active: bool,
    ) -> SessionStopEvaluation:
        if self._started_perf_time is None:
            raise RuntimeError("Session stop policy has not been started")
        perf_time = float(perf_time)

        if self._requested_perf_time is None:
            triggered = self._get_triggered_reasons(
                perf_time,
                int(trial_count),
                bool(protocol_complete),
            )
            if not triggered:
                return SessionStopEvaluation(SessionStopDecision.NONE)
            self._requested_perf_time = perf_time
            self._reason = triggered[0]
            self._triggered_reasons = triggered

        if not trial_active:
            return self._evaluation(SessionStopDecision.STOP)

        drain_elapsed = perf_time - self._requested_perf_time
        if drain_elapsed >= self.configuration.drain_timeout_seconds:
            return self._evaluation(
                SessionStopDecision.TIMEOUT_ERROR,
                timeout_seconds=drain_elapsed,
            )
        return self._evaluation(SessionStopDecision.FINISH_ACTIVE_TRIAL)

    def _get_triggered_reasons(
        self,
        perf_time: float,
        trial_count: int,
        protocol_complete: bool,
    ) -> Tuple[SessionStopReason, ...]:
        config = self.configuration
        reasons = []
        if (
            config.duration_seconds is not None
            and perf_time - self._started_perf_time >= config.duration_seconds
        ):
            reasons.append(SessionStopReason.DURATION_LIMIT)
        if config.trial_limit is not None and trial_count >= config.trial_limit:
            reasons.append(SessionStopReason.TRIAL_LIMIT)
        if config.stop_on_protocol_complete and protocol_complete:
            reasons.append(SessionStopReason.PROTOCOL_COMPLETE)
        return tuple(reasons)

    def _evaluation(
        self,
        decision: SessionStopDecision,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> SessionStopEvaluation:
        return SessionStopEvaluation(
            decision=decision,
            reason=self._reason,
            triggered_reasons=self._triggered_reasons,
            requested_perf_time=self._requested_perf_time,
            timeout_seconds=timeout_seconds,
        )
