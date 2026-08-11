"""Pellet-attempt lifecycle ownership for a continuous recording session."""

from __future__ import annotations

import math
from typing import Optional

from autotrainer.behavior import (
    HardwareErrorKind,
    PelletTrialLedger,
    TrialAccountingConfiguration,
    TrialCountBasis,
    TrialOutcome,
)
from autotrainer.core import SystemCommandKind
from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import CanFailure, CanFailureKind


logger = get_verbose_logger(__name__)


class PelletCycleController:
    """Coordinate the ledger, public lifecycle, protocol, and persistence.

    AppModel still receives hardware and analysis events, but this controller is
    the single owner of how those events mutate pellet-attempt state.
    """

    def __init__(self, session_api, protocol_runner, session_data_recorder):
        self._session_api = session_api
        self._protocol_runner = protocol_runner
        self._session_data_recorder = session_data_recorder
        self._ledger: Optional[PelletTrialLedger] = None

    @property
    def ledger(self) -> Optional[PelletTrialLedger]:
        return self._ledger

    @ledger.setter
    def ledger(self, value: Optional[PelletTrialLedger]) -> None:
        """Compatibility seam for tests and controlled session restoration."""
        self._ledger = value

    @property
    def active_attempt(self):
        return None if self._ledger is None else self._ledger.active_attempt

    def start_session(
        self,
        session_id: str,
        configuration: TrialAccountingConfiguration,
    ) -> PelletTrialLedger:
        self._ledger = PelletTrialLedger(session_id, configuration)
        return self._ledger

    def reset(self) -> None:
        self._ledger = None

    def count(self, basis: Optional[TrialCountBasis] = None) -> int:
        return 0 if self._ledger is None else self._ledger.count(basis)

    def summary(self) -> dict:
        return {} if self._ledger is None else self._ledger.summary()

    def begin_send(
        self,
        perf_time: float,
        wall_time: float,
        *,
        operation_id: str,
        pellet_position=None,
        planned_shift=None,
        applied_shift=None,
        protocol_context=None,
    ):
        ledger = self._require_ledger()
        if ledger.active_attempt is not None:
            previous = ledger.close_active_for_analysis(perf_time, wall_time)
            self._session_api.trial_capture_ended(previous)
            self._protocol_runner.cancel_active_trial()
        attempt = ledger.begin_send(
            perf_time,
            wall_time,
            operation_id=operation_id,
            pellet_position=pellet_position,
            planned_shift=planned_shift,
            applied_shift=applied_shift,
            protocol_context=protocol_context,
        )
        self._session_api.trial_started(attempt)
        return attempt

    def acknowledge_presentation(
        self,
        perf_time: float,
        wall_time: float,
        *,
        operation_id: Optional[str],
    ):
        ledger = self._require_ledger()
        attempt = ledger.active_attempt
        if attempt is None:
            return None
        if operation_id is not None and attempt.operation_id != operation_id:
            logger.error(
                "Ignoring pellet presentation acknowledgement with mismatched "
                "context: active=%s received=%s",
                attempt.operation_id,
                operation_id,
            )
            return None
        presented = ledger.acknowledge_presentation(perf_time, wall_time)
        self._protocol_runner.begin_trial(presented.attempt_label)
        return presented

    def finish_active(
        self,
        perf_time: float,
        wall_time: float,
        *,
        retry: bool = False,
    ):
        ledger = self._ledger
        if ledger is None or ledger.active_attempt is None:
            return None
        attempt = ledger.close_active_for_analysis(
            perf_time,
            wall_time,
            retry=retry,
        )
        self._session_api.trial_capture_ended(attempt)
        self._protocol_runner.cancel_active_trial()
        return attempt

    def finalize_active_incomplete(
        self,
        perf_time: float,
        wall_time: float,
        *,
        error: str,
    ):
        ledger = self._ledger
        if ledger is None or ledger.active_attempt is None:
            return None
        finalized = ledger.finalize(
            TrialOutcome.INCOMPLETE,
            perf_time,
            wall_time,
            error=error,
        )
        self._session_api.trial_ended(finalized)
        self._protocol_runner.cancel_active_trial()
        return finalized

    def finalize_hardware_failure(
        self,
        failure: CanFailure,
        *,
        in_session: bool,
    ):
        ledger = self._ledger
        attempt = None if ledger is None else ledger.active_attempt
        if (
            attempt is None
            and ledger is not None
            and in_session
            and failure.command is SystemCommandKind.SEND_PELLET
        ):
            attempt = ledger.begin_send(
                failure.perf_time,
                failure.wall_time,
                operation_id=failure.context,
            )
            self._session_api.trial_started(attempt)
        if attempt is None:
            return None
        if failure.context is not None and failure.context != attempt.operation_id:
            logger.info(
                "CAN failure does not match active pellet attempt: "
                "active=%s failure_context=%s",
                attempt.operation_id,
                failure.context,
            )
            return None

        if failure.kind is CanFailureKind.ACKNOWLEDGEMENT_TIMEOUT:
            kind = HardwareErrorKind.ACKNOWLEDGEMENT_TIMEOUT
        elif failure.kind is CanFailureKind.TRANSPORT:
            kind = HardwareErrorKind.TRANSPORT_FAILURE
        else:
            command = failure.command
            is_motor_command = (
                isinstance(command, SystemCommandKind)
                and 200 <= int(command) < 300
            )
            kind = (
                HardwareErrorKind.MOTOR_FAILURE
                if is_motor_command
                else HardwareErrorKind.COMMAND_FAILURE
            )

        finalized = ledger.finalize_hardware_error(
            kind,
            failure.perf_time,
            failure.wall_time,
            error=failure.error,
        )
        self._session_api.trial_ended(finalized)
        self._protocol_runner.cancel_active_trial()
        logger.error(
            "pellet attempt %s finalized as %s: %s",
            finalized.attempt_label,
            kind.value,
            failure.error,
        )
        return finalized

    def reconcile_analysis(
        self,
        project,
        result,
        *,
        recording_start_perf_time: float,
        frame_rate: float,
        finalized_perf_time: float,
        finalized_wall_time: float,
        tone_references=(),
        laser_references=(),
    ):
        ledger = self._require_ledger()
        finalized = ledger.reconcile_analysis(
            result,
            recording_start_perf_time=recording_start_perf_time,
            frame_rate=frame_rate,
            finalized_perf_time=finalized_perf_time,
            finalized_wall_time=finalized_wall_time,
            tone_references=tone_references,
            laser_references=laser_references,
        )
        for attempt in finalized:
            self._session_api.trial_ended(attempt)
            if attempt.logical_trial_complete:
                self._protocol_runner.record_trial_outcome(
                    attempt.attempt_label,
                    attempt.outcome,
                )
        self._session_data_recorder.update_persisted_trial_ledger(
            project,
            ledger.to_records(),
            ledger.summary(),
        )
        return finalized

    def finalize_pending_without_analysis(
        self,
        project,
        *,
        perf_time: float,
        wall_time: float,
        reason: str,
    ):
        ledger = self._ledger
        if ledger is None:
            return ()
        finalized = ledger.finalize_pending_without_analysis(
            perf_time,
            wall_time,
            reason=reason,
        )
        for attempt in finalized:
            self._session_api.trial_ended(attempt)
        if finalized:
            self._session_data_recorder.update_persisted_trial_ledger(
                project,
                ledger.to_records(),
                ledger.summary(),
            )
        return finalized

    def snapshot_for_stop(self, perf_time: float, wall_time: float) -> None:
        ledger = self._ledger
        if ledger is None:
            return
        if ledger.active_attempt is not None:
            self.finish_active(perf_time, wall_time)
        self._session_data_recorder.set_trial_ledger(
            ledger.to_records(),
            ledger.summary(),
        )

    def end_session(self, *, aborted: bool = False) -> None:
        if self._ledger is not None:
            self._session_api.session_ended(
                self._ledger.summary(),
                aborted=aborted,
            )

    def abort(self, *, perf_time: float, wall_time: float) -> None:
        ledger = self._ledger
        if ledger is not None and ledger.active_attempt is not None:
            finalized = ledger.finalize(
                TrialOutcome.ABORTED,
                perf_time,
                wall_time,
                error="recording aborted by operator",
            )
            self._session_api.trial_ended(finalized)
        self.end_session(aborted=True)
        self.reset()

    def sync_behavior_counts(self, algorithm) -> None:
        """Project all four UI counts from the one authoritative summary."""
        summary = self.summary()
        algorithm.pellet_reaches = int(summary.get("reaches", 0))
        algorithm.pellets_presented = int(summary.get("pellets_presented", 0))
        algorithm.successful_reaches = int(summary.get("successful_reaches", 0))
        algorithm.pellets_consumed = int(summary.get("pellets_consumed", 0))

    def _require_ledger(self) -> PelletTrialLedger:
        if self._ledger is None:
            raise RuntimeError("No active pellet-trial ledger")
        return self._ledger


def offset_record(value):
    if value is None:
        return None
    result = {}
    for name in ("x", "y", "z"):
        item = float(getattr(value, name))
        result[name] = item if math.isfinite(item) else None
    return result
