"""Optional inter-trial timing between pellet retrieval and the next send.

The policy measures from the moment a retrieval is dispatched to the moment the
next automatic send would go out.  When a trial becomes ready early the caller
holds it until the target elapses; when it becomes ready late the caller sends
immediately and records how late it was, because delaying further would only
compound the drift.

The policy owns no clock.  Callers pass monotonic ``perf_counter`` values, the
same convention as :mod:`tools.acquisition.model.session_stop_policy`, so it
stays testable and free of hardware and Qt.

Timing defaults off.  An armed-but-disabled policy always reports send-now, so
callers can wire it unconditionally.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, Optional


# Operators enter the target in seconds at millisecond resolution.
TIMING_RESOLUTION_SECONDS = 0.001
TIMING_DECIMALS = 3
MAX_TARGET_SECONDS = 3600.0


class InterTrialDecision(str, enum.Enum):
    SEND_NOW = "send_now"
    WAIT = "wait"

    @property
    def display_name(self) -> str:
        return {
            InterTrialDecision.SEND_NOW: "Send now",
            InterTrialDecision.WAIT: "Waiting for inter-trial target",
        }[self]


class InterTrialReason(str, enum.Enum):
    DISABLED = "disabled"
    NOT_ARMED = "not_armed"
    RETRY_BYPASS = "retry_bypass"
    TARGET_NOT_REACHED = "target_not_reached"
    TARGET_REACHED = "target_reached"
    READY_LATE = "ready_late"


@dataclasses.dataclass(frozen=True)
class InterTrialTimingConfiguration:
    """Operator configuration for the retrieval-to-next-send target."""

    enabled: bool = False
    target_seconds: Optional[float] = None
    apply_to_retries: bool = False

    def __post_init__(self):
        if not self.enabled:
            return
        if self.target_seconds is None:
            raise ValueError("Inter-trial timing requires a target when enabled")
        target = float(self.target_seconds)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("Inter-trial target must be positive")
        if target > MAX_TARGET_SECONDS:
            raise ValueError(
                f"Inter-trial target must be at most {MAX_TARGET_SECONDS:g} seconds"
            )
        rounded = round(target, TIMING_DECIMALS)
        object.__setattr__(self, "target_seconds", rounded)

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class InterTrialTimingEvaluation:
    """One decision about whether the next automatic send may go out."""

    decision: InterTrialDecision
    reason: InterTrialReason
    remaining_seconds: float = 0.0
    elapsed_seconds: Optional[float] = None
    lateness_seconds: Optional[float] = None
    target_seconds: Optional[float] = None

    @property
    def should_wait(self) -> bool:
        return self.decision is InterTrialDecision.WAIT

    @property
    def was_late(self) -> bool:
        return self.reason is InterTrialReason.READY_LATE

    def to_record(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["decision"] = self.decision.value
        record["reason"] = self.reason.value
        return record


class InterTrialTimingPolicy:
    """Hold the next automatic send until the configured target elapses."""

    def __init__(
        self,
        configuration: Optional[InterTrialTimingConfiguration] = None,
    ):
        self._configuration = configuration or InterTrialTimingConfiguration()
        self._armed_perf_time: Optional[float] = None

    @property
    def configuration(self) -> InterTrialTimingConfiguration:
        return self._configuration

    @property
    def is_enabled(self) -> bool:
        return self._configuration.enabled

    @property
    def is_armed(self) -> bool:
        return self._armed_perf_time is not None

    def arm(self, dispatch_perf_time: float) -> None:
        """Start measuring from a dispatched pellet retrieval."""

        dispatch_perf_time = float(dispatch_perf_time)
        if not math.isfinite(dispatch_perf_time):
            raise ValueError("Retrieval dispatch time must be finite")
        self._armed_perf_time = dispatch_perf_time

    def disarm(self) -> None:
        self._armed_perf_time = None

    def evaluate(
        self,
        ready_perf_time: float,
        *,
        is_retry: bool = False,
    ) -> InterTrialTimingEvaluation:
        """Decide whether a send that is ready now may go out."""

        target = self._configuration.target_seconds
        if not self._configuration.enabled:
            return InterTrialTimingEvaluation(
                decision=InterTrialDecision.SEND_NOW,
                reason=InterTrialReason.DISABLED,
            )
        if is_retry and not self._configuration.apply_to_retries:
            return InterTrialTimingEvaluation(
                decision=InterTrialDecision.SEND_NOW,
                reason=InterTrialReason.RETRY_BYPASS,
                target_seconds=target,
            )
        if self._armed_perf_time is None:
            # Nothing to measure from; never invent a wait.
            return InterTrialTimingEvaluation(
                decision=InterTrialDecision.SEND_NOW,
                reason=InterTrialReason.NOT_ARMED,
                target_seconds=target,
            )

        ready_perf_time = float(ready_perf_time)
        if not math.isfinite(ready_perf_time):
            raise ValueError("Ready time must be finite")
        elapsed = round(ready_perf_time - self._armed_perf_time, TIMING_DECIMALS)
        remaining = round(target - elapsed, TIMING_DECIMALS)

        if remaining > 0.0:
            return InterTrialTimingEvaluation(
                decision=InterTrialDecision.WAIT,
                reason=InterTrialReason.TARGET_NOT_REACHED,
                remaining_seconds=remaining,
                elapsed_seconds=elapsed,
                target_seconds=target,
            )
        if remaining == 0.0:
            return InterTrialTimingEvaluation(
                decision=InterTrialDecision.SEND_NOW,
                reason=InterTrialReason.TARGET_REACHED,
                elapsed_seconds=elapsed,
                lateness_seconds=0.0,
                target_seconds=target,
            )
        # Ready late: send immediately and report the measured lateness rather
        # than compounding the drift by waiting further.
        return InterTrialTimingEvaluation(
            decision=InterTrialDecision.SEND_NOW,
            reason=InterTrialReason.READY_LATE,
            elapsed_seconds=elapsed,
            lateness_seconds=round(-remaining, TIMING_DECIMALS),
            target_seconds=target,
        )
