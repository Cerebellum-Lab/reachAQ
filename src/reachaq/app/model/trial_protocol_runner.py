"""Run a training plan on pellet-delivery trial boundaries."""

from __future__ import annotations

from typing import Callable, Optional

from reachaq.behavior.training import (
    SessionResult,
    TrainingProgressState,
    TrialDirective,
    TrialKind,
    TrialResult,
)

from reachaq.behavior.pellet_trial import TrialOutcome


# Outcomes that never count as a plan trial: the subject had no fair trial.
_NOT_COUNTED = {
    TrialOutcome.HARDWARE_ERROR,
    TrialOutcome.ABORTED,
    TrialOutcome.INCOMPLETE,
}

_KIND = {
    TrialOutcome.SUCCESS: TrialKind.SUCCESS,
    TrialOutcome.FAILURE: TrialKind.FAILURE,
    TrialOutcome.NO_REACH: TrialKind.NO_REACH,
    TrialOutcome.PELLET_MISSING: TrialKind.PELLET_MISSING,
    TrialOutcome.PELLET_PRECHECK_FAILED: TrialKind.PELLET_MISSING,
    TrialOutcome.PELLET_PRECHECK_AUTHORIZATION_TIMEOUT: TrialKind.PELLET_MISSING,
}


class TrialProtocolRunner:
    """Evaluate the selected training plan exactly once per completed pellet trial.

    Recordings are not plan trials. The pellet-cycle controller reports each
    presented pellet (``begin_trial``) and each finalized logical trial
    (``record_trial_outcome``); the plan's predicates are evaluated then, and
    with automatic advance the phase change is applied here.
    """

    def __init__(
        self,
        *,
        automatic_advance: bool = False,
        on_protocol_complete: Optional[Callable[[], None]] = None,
        on_trial_recorded: Optional[Callable[[str], None]] = None,
    ):
        self.automatic_advance = bool(automatic_advance)
        self._on_protocol_complete = on_protocol_complete
        # Called with the attempt label after each counted trial, once any
        # phase change is applied, so the next trial's directive is current.
        self._on_trial_recorded = on_trial_recorded
        self._plan = None
        self._active_attempt_label: Optional[str] = None
        self._protocol_complete = False

    @property
    def plan(self):
        return self._plan

    @property
    def active_attempt_label(self) -> Optional[str]:
        return self._active_attempt_label

    @property
    def protocol_complete(self) -> bool:
        return self._protocol_complete

    def attach(self, plan, algorithm, pellet_device) -> None:
        self.detach()
        plan.attach(algorithm, pellet_device, None)
        self._plan = plan

    def detach(self) -> None:
        plan = self._plan
        if plan is None:
            return
        self.cancel_active_trial()
        plan.detach()
        self._plan = None
        self._protocol_complete = False

    def next_trial_directive(self, draw_key: str) -> Optional[TrialDirective]:
        """What the current phase's trial actions ask of the next SEND."""
        plan = self._plan
        return None if plan is None else plan.next_trial_directive(draw_key)

    def begin_trial(self, attempt_label: str) -> bool:
        plan = self._plan
        if plan is None or plan.current_phase is None:
            return False
        if self._active_attempt_label is not None:
            if self._active_attempt_label == str(attempt_label):
                return False
            raise RuntimeError(
                "Cannot begin protocol trial while another attempt is active"
            )
        self._active_attempt_label = str(attempt_label)
        plan.current_phase.progress.progress_resumed()
        return True

    def cancel_active_trial(self) -> bool:
        if self._active_attempt_label is None:
            return False
        plan = self._plan
        if plan is not None and plan.current_phase is not None:
            plan.current_phase.progress.progress_paused()
        self._active_attempt_label = None
        return True

    def finish_trial(
        self,
        attempt_label: str,
        outcome: TrialOutcome,
        **counts,
    ) -> bool:
        """Count and evaluate one non-hardware pellet trial exactly once."""
        if self._active_attempt_label != str(attempt_label):
            return False
        self._active_attempt_label = None
        return self.record_trial_outcome(attempt_label, outcome, **counts)

    def record_trial_outcome(
        self,
        attempt_label: str,
        outcome: TrialOutcome,
        *,
        reach_count: int = 0,
        success_count: int = 0,
        consumption_count: int = 0,
    ) -> bool:
        """Apply one finalized pellet-trial result after session analysis."""
        plan = self._plan
        phase = None if plan is None else plan.current_phase
        if phase is None:
            return False
        if self._active_attempt_label == str(attempt_label):
            self._active_attempt_label = None

        outcome = TrialOutcome(outcome)
        if outcome in _NOT_COUNTED:
            phase.progress.progress_paused()
            plan._on_progress_updated()
            return False

        kind = _KIND.get(outcome, TrialKind.OTHER)
        result = plan.record_trial(TrialResult(
            kind=kind,
            reaches=int(reach_count),
            successful_reaches=int(success_count),
            pellets_consumed=int(consumption_count),
            pellet_presented=kind is not TrialKind.PELLET_MISSING,
        ))

        if self.automatic_advance:
            if result is SessionResult.FALLBACK and plan.can_fallback:
                plan.fallback()
            elif result is SessionResult.ADVANCE:
                if plan.can_advance:
                    plan.advance()
                else:
                    self._mark_protocol_complete()
        plan._on_progress_updated()
        if self._on_trial_recorded is not None:
            self._on_trial_recorded(str(attempt_label))
        return True

    def _mark_protocol_complete(self) -> None:
        if self._protocol_complete:
            return
        self._protocol_complete = True
        self._plan.progress_state = TrainingProgressState.Complete
        if self._on_protocol_complete is not None:
            self._on_protocol_complete()
