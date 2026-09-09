"""Tone 1 to Tone 2 deadline enforcement.

With Lock Timing on, the interval between Tone 1 and Tone 2 is fixed.  If the
gate is blocked or its evidence is stale when the deadline arrives, Tone 2 is
skipped and the trial resets rather than firing late, because a late go cue
silently corrupts every latency measured from it.

With Lock Timing off, a genuinely blocked gate may extend the trial: Tone 2
fires once the gate has been clear for the configured post-clear delay, and a
re-block restarts that delay.  A trial that was never blocked receives no
extension.

The gate in reachAQ is pose-derived reach state plus pellet presence evidence
from the board, not a camera ROI.  Evidence older than the configured staleness
bound is treated as unknown rather than as clear, so a stalled evidence source
cannot silently authorize a cue.

The policy owns no clock and touches no hardware.  Callers pass monotonic
``perf_counter`` values, matching
:mod:`tools.acquisition.model.session_stop_policy`.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, Optional


DEFAULT_GATE_STALENESS_MS = 500
MAX_GATE_STALENESS_MS = 10_000
MAX_POST_CLEAR_DELAY_MS = 60_000


class CueDecision(str, enum.Enum):
    WAIT = "wait"
    FIRE = "fire"
    SKIP_AND_RESET = "skip_and_reset"

    @property
    def display_name(self) -> str:
        return {
            CueDecision.WAIT: "Waiting for the cue deadline",
            CueDecision.FIRE: "Fire Tone 2",
            CueDecision.SKIP_AND_RESET: "Skip Tone 2 and reset",
        }[self]


class CueCancelReason(str, enum.Enum):
    """Why a cue was not delivered.

    The ``LOCK_DEADLINE_`` reasons are distinct from their unlocked
    counterparts so evidence shows whether a fixed deadline caused the reset.
    """

    NONE = "none"
    HOST_REQUEST = "host_request"
    PELLET_MISSING = "pellet_missing"
    EARLY_REACH = "early_reach"
    GATE_STATE_STALE = "gate_state_stale"
    LOCK_DEADLINE_PELLET_MISSING = "lock_deadline_pellet_missing"
    LOCK_DEADLINE_EARLY_REACH = "lock_deadline_early_reach"
    LOCK_DEADLINE_GATE_STATE_STALE = "lock_deadline_gate_state_stale"


@dataclasses.dataclass(frozen=True)
class CueGateState:
    """One observation of whether the cue may be delivered.

    ``pellet_present`` is ``None`` when presence is unknown.  ``observed_at``
    is the monotonic time the evidence was captured, not the time it was read.
    """

    observed_at: float
    pellet_present: Optional[bool] = None
    reach_active: bool = False

    def __post_init__(self):
        if not math.isfinite(float(self.observed_at)):
            raise ValueError("Gate observation time must be finite")


@dataclasses.dataclass(frozen=True)
class CueTimingConfiguration:
    lock_timing: bool = True
    post_clear_delay_ms: int = 0
    gate_staleness_ms: int = DEFAULT_GATE_STALENESS_MS

    def __post_init__(self):
        delay = int(self.post_clear_delay_ms)
        if not 0 <= delay <= MAX_POST_CLEAR_DELAY_MS:
            raise ValueError(
                f"Post-clear delay must be between 0 and {MAX_POST_CLEAR_DELAY_MS} ms"
            )
        if self.lock_timing and delay:
            raise ValueError(
                "Post-clear delay applies only when Lock Timing is off"
            )
        staleness = int(self.gate_staleness_ms)
        if not 1 <= staleness <= MAX_GATE_STALENESS_MS:
            raise ValueError(
                f"Gate staleness must be between 1 and {MAX_GATE_STALENESS_MS} ms"
            )
        object.__setattr__(self, "post_clear_delay_ms", delay)
        object.__setattr__(self, "gate_staleness_ms", staleness)

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CueTimingEvaluation:
    decision: CueDecision
    reason: CueCancelReason = CueCancelReason.NONE
    remaining_seconds: float = 0.0
    deadline_perf_time: Optional[float] = None
    lock_timing: bool = True
    was_blocked: bool = False
    extension_seconds: float = 0.0

    @property
    def should_fire(self) -> bool:
        return self.decision is CueDecision.FIRE

    @property
    def should_reset(self) -> bool:
        return self.decision is CueDecision.SKIP_AND_RESET

    def to_record(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["decision"] = self.decision.value
        record["reason"] = self.reason.value
        return record


class CueTimingPolicy:
    """Decide when Tone 2 may fire for one trial."""

    def __init__(self, configuration: Optional[CueTimingConfiguration] = None):
        self._configuration = configuration or CueTimingConfiguration()
        self._deadline: Optional[float] = None
        self._clear_since: Optional[float] = None
        self._was_blocked = False

    @property
    def configuration(self) -> CueTimingConfiguration:
        return self._configuration

    @property
    def deadline_perf_time(self) -> Optional[float]:
        return self._deadline

    @property
    def is_started(self) -> bool:
        return self._deadline is not None

    def start(self, tone_1_perf_time: float, cue_interval_ms: int) -> float:
        """Arm the trial at Tone 1 and return the Tone 2 deadline."""

        tone_1_perf_time = float(tone_1_perf_time)
        if not math.isfinite(tone_1_perf_time):
            raise ValueError("Tone 1 time must be finite")
        cue_interval_ms = int(cue_interval_ms)
        if cue_interval_ms <= 0:
            raise ValueError("Cue interval must be positive")
        self._deadline = tone_1_perf_time + cue_interval_ms / 1000.0
        self._clear_since = None
        self._was_blocked = False
        return self._deadline

    def reset(self) -> None:
        self._deadline = None
        self._clear_since = None
        self._was_blocked = False

    def cancel(self, reason: CueCancelReason) -> CueTimingEvaluation:
        """Cancel the cue for an external reason, such as a host stop."""

        evaluation = CueTimingEvaluation(
            decision=CueDecision.SKIP_AND_RESET,
            reason=CueCancelReason(reason),
            deadline_perf_time=self._deadline,
            lock_timing=self._configuration.lock_timing,
            was_blocked=self._was_blocked,
        )
        self.reset()
        return evaluation

    def evaluate(
        self,
        now_perf_time: float,
        gate: CueGateState,
    ) -> CueTimingEvaluation:
        """Decide what to do with Tone 2 at ``now_perf_time``."""

        if self._deadline is None:
            raise RuntimeError("Cue timing was not started for this trial")
        now_perf_time = float(now_perf_time)
        if not math.isfinite(now_perf_time):
            raise ValueError("Evaluation time must be finite")

        blocked_reason = self._blocked_reason(now_perf_time, gate)
        if blocked_reason is None:
            if self._clear_since is None:
                # The clear run starts when the gate was observed clear, not
                # when the observation was read, so evidence age does not
                # silently extend the post-clear delay.
                self._clear_since = float(gate.observed_at)
        else:
            # A re-block restarts the unlocked post-clear delay.
            self._clear_since = None
            self._was_blocked = True

        remaining = self._deadline - now_perf_time
        if remaining > 0.0:
            return self._evaluation(
                CueDecision.WAIT,
                CueCancelReason.NONE,
                remaining_seconds=remaining,
            )

        if self._configuration.lock_timing:
            if blocked_reason is not None:
                return self._deadline_reset(blocked_reason)
            return self._evaluation(CueDecision.FIRE, CueCancelReason.NONE)

        if blocked_reason is not None:
            # Unlocked trials may wait out a genuine block indefinitely; the
            # caller decides when to give up.
            return self._evaluation(
                CueDecision.WAIT,
                CueCancelReason.NONE,
                remaining_seconds=0.0,
            )

        extension = self._post_clear_remaining(now_perf_time)
        if extension > 0.0:
            return self._evaluation(
                CueDecision.WAIT,
                CueCancelReason.NONE,
                remaining_seconds=extension,
            )
        return self._evaluation(
            CueDecision.FIRE,
            CueCancelReason.NONE,
            extension_seconds=max(0.0, now_perf_time - self._deadline),
        )

    def _post_clear_remaining(self, now_perf_time: float) -> float:
        """Return how much of the post-clear delay is still outstanding."""

        if not self._was_blocked or not self._configuration.post_clear_delay_ms:
            return 0.0
        if self._clear_since is None:
            return self._configuration.post_clear_delay_ms / 1000.0
        elapsed = now_perf_time - self._clear_since
        return max(0.0, self._configuration.post_clear_delay_ms / 1000.0 - elapsed)

    def _blocked_reason(
        self,
        now_perf_time: float,
        gate: CueGateState,
    ) -> Optional[CueCancelReason]:
        age_ms = (now_perf_time - float(gate.observed_at)) * 1000.0
        if age_ms > self._configuration.gate_staleness_ms:
            return CueCancelReason.GATE_STATE_STALE
        if gate.pellet_present is None:
            return CueCancelReason.GATE_STATE_STALE
        if not gate.pellet_present:
            return CueCancelReason.PELLET_MISSING
        if gate.reach_active:
            return CueCancelReason.EARLY_REACH
        return None

    def _deadline_reset(self, reason: CueCancelReason) -> CueTimingEvaluation:
        locked = {
            CueCancelReason.PELLET_MISSING: (
                CueCancelReason.LOCK_DEADLINE_PELLET_MISSING
            ),
            CueCancelReason.EARLY_REACH: (
                CueCancelReason.LOCK_DEADLINE_EARLY_REACH
            ),
            CueCancelReason.GATE_STATE_STALE: (
                CueCancelReason.LOCK_DEADLINE_GATE_STATE_STALE
            ),
        }[reason]
        return self._evaluation(CueDecision.SKIP_AND_RESET, locked)

    def _evaluation(
        self,
        decision: CueDecision,
        reason: CueCancelReason,
        *,
        remaining_seconds: float = 0.0,
        extension_seconds: float = 0.0,
    ) -> CueTimingEvaluation:
        return CueTimingEvaluation(
            decision=decision,
            reason=reason,
            remaining_seconds=max(0.0, round(remaining_seconds, 6)),
            deadline_perf_time=self._deadline,
            lock_timing=self._configuration.lock_timing,
            was_blocked=self._was_blocked,
            extension_seconds=max(0.0, round(extension_seconds, 6)),
        )
