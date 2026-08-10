"""Adapt retained training protocols to pellet-delivery trial boundaries."""

from __future__ import annotations

from typing import Callable, Optional

from autotrainer.core.interfaces import CaptureAnalysisResult
from autotrainer.training import TrainingProgressState
from autotrainer.training.training_phase import SessionResult

from autotrainer.behavior.pellet_trial import TrialOutcome


class TrialProtocolRunner:
    """Run a selected protocol without treating recordings as protocol trials.

    The installed training package still provides protocol parsing, actions,
    predicates, progress serialization, and manual phase navigation. This
    adapter deliberately disconnects its recording lifecycle callbacks and
    evaluates those same phase rules exactly once per completed pellet trial.
    """

    _RECORDING_CALLBACKS = (
        ("session_starting", "_on_session_starting"),
        ("session_capture_ending", "_on_session_capture_ending"),
        ("session_ending", "_on_session_ending"),
    )

    def __init__(
        self,
        *,
        automatic_advance: bool = False,
        on_protocol_complete: Optional[Callable[[], None]] = None,
    ):
        self.automatic_advance = bool(automatic_advance)
        self._on_protocol_complete = on_protocol_complete
        self._plan = None
        self._algorithm = None
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
        self._algorithm = algorithm
        self._set_recording_callbacks_enabled(False)

    def detach(self) -> None:
        plan = self._plan
        if plan is None:
            return
        self.cancel_active_trial()
        # TrainingPlan.detach expects to remove its original callbacks.
        self._set_recording_callbacks_enabled(True)
        plan.detach()
        self._plan = None
        self._algorithm = None
        self._protocol_complete = False

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
    ) -> bool:
        """Count and evaluate one non-hardware pellet trial exactly once."""
        if self._active_attempt_label != str(attempt_label):
            return False
        plan = self._plan
        phase = None if plan is None else plan.current_phase
        self._active_attempt_label = None
        if phase is None:
            return False

        outcome = TrialOutcome(outcome)
        if outcome in {
            TrialOutcome.HARDWARE_ERROR,
            TrialOutcome.ABORTED,
            TrialOutcome.INCOMPLETE,
        }:
            phase.progress.progress_paused()
            plan._on_progress_updated()
            return False

        phase.capture_ended()
        phase.progress.session_count += 1
        context = plan._system_context
        context.capture_analysis_result = CaptureAnalysisResult.CAPTURE_ONLY
        try:
            result = phase.session_ended(context)
        finally:
            context.capture_analysis_result = None

        if self.automatic_advance:
            if result is SessionResult.FALLBACK and plan.can_fallback:
                plan.fallback()
            elif result is SessionResult.ADVANCE:
                if plan.can_advance:
                    plan.advance()
                else:
                    self._mark_protocol_complete()
        plan._on_progress_updated()
        return True

    def _mark_protocol_complete(self) -> None:
        if self._protocol_complete:
            return
        self._protocol_complete = True
        self._plan.progress_state = TrainingProgressState.Complete
        if self._on_protocol_complete is not None:
            self._on_protocol_complete()

    def _set_recording_callbacks_enabled(self, enabled: bool) -> None:
        if self._plan is None or self._algorithm is None:
            return
        for event_name, callback_name in self._RECORDING_CALLBACKS:
            event = getattr(self._algorithm, event_name)
            callback = getattr(self._plan, callback_name)
            if enabled:
                event += callback
            else:
                event -= callback
