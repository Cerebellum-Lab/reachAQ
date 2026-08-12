"""Authoritative pellet-delivery trial and physical-attempt accounting."""

from __future__ import annotations

import dataclasses
import enum
import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from autotrainer.core.reach_event import ReachEventMethod, ReachEventOutcome


class AttemptAssignmentPolicy(str, enum.Enum):
    """How behavioral retries receive public trial numbers."""

    RETRY_WITHIN_TRIAL = "retry_within_trial"
    EVERY_ATTEMPT_IS_TRIAL = "every_attempt_is_trial"
    SUCCESSFUL_PRESENTATIONS_ONLY = "successful_presentations_only"

    @property
    def display_name(self) -> str:
        return {
            self.RETRY_WITHIN_TRIAL: "Retry within the same trial",
            self.EVERY_ATTEMPT_IS_TRIAL: "Count every attempt as a new trial",
            self.SUCCESSFUL_PRESENTATIONS_ONLY: (
                "Count only successful pellet presentations"
            ),
        }[self]


class RetrySettingsPolicy(str, enum.Enum):
    REUSE = "reuse"
    RESAMPLE = "resample"

    @property
    def display_name(self) -> str:
        return {
            self.REUSE: "Reuse the original trial settings",
            self.RESAMPLE: "Choose new settings for the retry",
        }[self]


class TrialCountBasis(str, enum.Enum):
    STARTED = "started"
    PRESENTED = "presented"
    COMPLETED = "completed"
    SCORED = "scored"

    @property
    def display_name(self) -> str:
        return {
            self.STARTED: "Trials started",
            self.PRESENTED: "Pellets presented",
            self.COMPLETED: "Trials completed",
            self.SCORED: "Scored trials",
        }[self]


class TrialOutcome(str, enum.Enum):
    PENDING_ANALYSIS = "pending_analysis"
    SUCCESS = "success"
    FAILURE = "failure"
    PELLET_MISSING = "pellet_missing"
    NO_REACH = "no_reach"
    INCOMPLETE = "incomplete"
    ABORTED = "aborted"
    HARDWARE_ERROR = "hardware_error"

    @property
    def is_scored(self) -> bool:
        return self in {
            TrialOutcome.SUCCESS,
            TrialOutcome.FAILURE,
            TrialOutcome.PELLET_MISSING,
            TrialOutcome.NO_REACH,
        }


class HardwareErrorKind(str, enum.Enum):
    MOTOR_FAILURE = "motor_failure"
    COMMAND_FAILURE = "command_failure"
    TRANSPORT_FAILURE = "transport_failure"
    ACKNOWLEDGEMENT_TIMEOUT = "acknowledgement_timeout"
    OPERATION_UNKNOWN = "operation_unknown"


@dataclass(frozen=True)
class TrialAccountingConfiguration:
    assignment_policy: AttemptAssignmentPolicy = (
        AttemptAssignmentPolicy.RETRY_WITHIN_TRIAL
    )
    retry_settings_policy: RetrySettingsPolicy = RetrySettingsPolicy.REUSE
    count_basis: TrialCountBasis = TrialCountBasis.COMPLETED
    counted_outcomes: FrozenSet[TrialOutcome] = frozenset({
        TrialOutcome.SUCCESS,
        TrialOutcome.FAILURE,
        TrialOutcome.PELLET_MISSING,
        TrialOutcome.NO_REACH,
    })

    def __post_init__(self):
        if TrialOutcome.HARDWARE_ERROR in self.counted_outcomes:
            raise ValueError("Hardware errors can never be counted as trials")


@dataclass(frozen=True)
class PelletTrialAttempt:
    session_id: str
    operation_id: str
    trial_id: Optional[int]
    attempt_id: int
    send_perf_time: float
    send_wall_time: float
    send_ack_perf_time: Optional[float] = None
    send_ack_wall_time: Optional[float] = None
    finalized_perf_time: Optional[float] = None
    finalized_wall_time: Optional[float] = None
    capture_end_perf_time: Optional[float] = None
    capture_end_wall_time: Optional[float] = None
    outcome: Optional[TrialOutcome] = None
    hardware_error_kind: Optional[HardwareErrorKind] = None
    error: str = ""
    logical_trial_complete: bool = False
    retry_settings_policy: Optional[RetrySettingsPolicy] = None
    pellet_position: Optional[Dict[str, float]] = None
    planned_shift: Optional[Dict[str, float]] = None
    applied_shift: Optional[Dict[str, float]] = None
    protocol_context: Optional[Dict[str, Any]] = None
    reach_count: int = 0
    success_count: int = 0
    consumption_count: int = 0
    reach_event_indices: Tuple[int, ...] = ()
    tone_references: Tuple[Dict[str, Any], ...] = ()
    laser_references: Tuple[Dict[str, Any], ...] = ()
    pellet_presence: str = "unknown"
    pellet_misplacement: str = "unknown"
    analysis_window_start_perf: Optional[float] = None
    analysis_window_end_perf: Optional[float] = None
    tracking_coverage: Optional[float] = None
    tracking_missing_frame_ids: Tuple[int, ...] = ()
    tracking_interpolated_points: int = 0
    tracking_long_gap_count: int = 0
    analysis_duration_seconds: Optional[float] = None
    recommended_shift: Optional[Dict[str, float]] = None

    @property
    def is_presented(self) -> bool:
        return self.send_ack_perf_time is not None

    @property
    def is_finalized(self) -> bool:
        return self.outcome not in {None, TrialOutcome.PENDING_ANALYSIS}

    @property
    def attempt_label(self) -> str:
        if self.trial_id is None:
            return f"unindexed.{self.attempt_id}"
        return f"{self.trial_id}.{self.attempt_id}"

    def to_dict(self) -> dict:
        result = dataclasses.asdict(self)
        for key in (
            "outcome",
            "hardware_error_kind",
            "retry_settings_policy",
        ):
            value = result[key]
            if isinstance(value, enum.Enum):
                result[key] = value.value
        result["attempt_label"] = self.attempt_label
        return result


class PelletTrialLedger:
    """Track logical trials separately from physical pellet-send attempts.

    A send dispatch begins an attempt. A successful device acknowledgement marks
    presentation. Hardware failures are retained as attempt errors but are
    excluded from every trial-count basis by construction.
    """

    def __init__(
        self,
        session_id: str,
        configuration: Optional[TrialAccountingConfiguration] = None,
    ):
        self.session_id = str(session_id)
        self.configuration = configuration or TrialAccountingConfiguration()
        self._attempts = []
        self._active_index: Optional[int] = None
        self._next_trial_id = 1
        self._retry_trial_id: Optional[int] = None
        self._retry_attempt_id = 0
        self._unindexed_attempt_id = 0
        self._analysis_counts = {
            "reaches": 0,
            "successful_reaches": 0,
            "pellets_consumed": 0,
            "unassigned_reaches": 0,
            "unassigned_successful_reaches": 0,
            "unassigned_pellets_consumed": 0,
        }

    @property
    def active_attempt(self) -> Optional[PelletTrialAttempt]:
        if self._active_index is None:
            return None
        return self._attempts[self._active_index]

    @property
    def attempts(self) -> Tuple[PelletTrialAttempt, ...]:
        return tuple(self._attempts)

    @property
    def planned_trial_id(self) -> int:
        """Logical row that the next send attempt will use after retries."""
        return int(
            self._retry_trial_id
            if self._retry_trial_id is not None
            else self._next_trial_id
        )

    def begin_send(
        self,
        perf_time: float,
        wall_time: float,
        *,
        operation_id: Optional[str] = None,
        pellet_position: Optional[Dict[str, float]] = None,
        planned_shift: Optional[Dict[str, float]] = None,
        applied_shift: Optional[Dict[str, float]] = None,
        protocol_context: Optional[Dict[str, Any]] = None,
    ) -> PelletTrialAttempt:
        if self.active_attempt is not None:
            raise RuntimeError("Cannot begin a pellet send while an attempt is active")

        policy = self.configuration.assignment_policy
        retry_trial_id = self._retry_trial_id
        if retry_trial_id is not None:
            trial_id = retry_trial_id
            attempt_id = self._retry_attempt_id + 1
        elif policy is AttemptAssignmentPolicy.SUCCESSFUL_PRESENTATIONS_ONLY:
            trial_id = None
            self._unindexed_attempt_id += 1
            attempt_id = self._unindexed_attempt_id
        else:
            trial_id = self._next_trial_id
            self._next_trial_id += 1
            attempt_id = 1

        attempt = PelletTrialAttempt(
            session_id=self.session_id,
            operation_id=str(operation_id or uuid.uuid4()),
            trial_id=trial_id,
            attempt_id=attempt_id,
            send_perf_time=float(perf_time),
            send_wall_time=float(wall_time),
            retry_settings_policy=(
                self.configuration.retry_settings_policy
                if retry_trial_id is not None
                else None
            ),
            pellet_position=pellet_position,
            planned_shift=planned_shift,
            applied_shift=applied_shift,
            protocol_context=protocol_context,
        )
        self._attempts.append(attempt)
        self._active_index = len(self._attempts) - 1
        self._retry_trial_id = None
        self._retry_attempt_id = 0
        return attempt

    def acknowledge_presentation(
        self,
        perf_time: float,
        wall_time: float,
    ) -> PelletTrialAttempt:
        attempt = self._require_active()
        if attempt.is_presented:
            raise RuntimeError(f"Attempt {attempt.attempt_label} is already presented")
        trial_id = attempt.trial_id
        attempt_id = attempt.attempt_id
        if trial_id is None:
            trial_id = self._next_trial_id
            self._next_trial_id += 1
            attempt_id = 1
        return self._replace_active(
            trial_id=trial_id,
            attempt_id=attempt_id,
            send_ack_perf_time=float(perf_time),
            send_ack_wall_time=float(wall_time),
        )

    def finalize(
        self,
        outcome: TrialOutcome,
        perf_time: float,
        wall_time: float,
        *,
        retry: bool = False,
        error: str = "",
    ) -> PelletTrialAttempt:
        outcome = TrialOutcome(outcome)
        if outcome is TrialOutcome.HARDWARE_ERROR:
            raise ValueError("Use finalize_hardware_error for hardware failures")
        attempt = self._require_active()
        if attempt.is_finalized:
            raise RuntimeError(f"Attempt {attempt.attempt_label} is already finalized")

        policy = self.configuration.assignment_policy
        logical_complete = not retry
        should_retry_same_trial = (
            retry
            and policy is AttemptAssignmentPolicy.RETRY_WITHIN_TRIAL
            and attempt.trial_id is not None
        )
        if policy is AttemptAssignmentPolicy.EVERY_ATTEMPT_IS_TRIAL:
            logical_complete = True
        elif policy is AttemptAssignmentPolicy.SUCCESSFUL_PRESENTATIONS_ONLY:
            logical_complete = attempt.is_presented and not retry

        finalized = self._replace_active(
            finalized_perf_time=float(perf_time),
            finalized_wall_time=float(wall_time),
            outcome=outcome,
            error=str(error or ""),
            logical_trial_complete=logical_complete,
        )
        self._active_index = None
        if should_retry_same_trial:
            self._retry_trial_id = finalized.trial_id
            self._retry_attempt_id = finalized.attempt_id
        return finalized

    def close_active_for_analysis(
        self,
        perf_time: float,
        wall_time: float,
        *,
        retry: bool = False,
    ) -> PelletTrialAttempt:
        """Close the capture window without inventing a behavioral outcome."""
        attempt = self._require_active()
        if attempt.outcome is not None:
            raise RuntimeError(f"Attempt {attempt.attempt_label} is already closed")
        policy = self.configuration.assignment_policy
        should_retry_same_trial = (
            retry
            and policy is AttemptAssignmentPolicy.RETRY_WITHIN_TRIAL
            and attempt.trial_id is not None
        )
        logical_complete = not retry
        if policy is AttemptAssignmentPolicy.EVERY_ATTEMPT_IS_TRIAL:
            logical_complete = True
        elif policy is AttemptAssignmentPolicy.SUCCESSFUL_PRESENTATIONS_ONLY:
            logical_complete = attempt.is_presented and not retry
        closed = self._replace_active(
            capture_end_perf_time=float(perf_time),
            capture_end_wall_time=float(wall_time),
            outcome=TrialOutcome.PENDING_ANALYSIS,
            logical_trial_complete=logical_complete,
        )
        self._active_index = None
        if should_retry_same_trial:
            self._retry_trial_id = closed.trial_id
            self._retry_attempt_id = closed.attempt_id
        return closed

    def finalize_pending(
        self,
        trial_id: Optional[int],
        attempt_id: int,
        outcome: TrialOutcome,
        perf_time: float,
        wall_time: float,
        *,
        error: str = "",
        reach_count: int = 0,
        success_count: int = 0,
        consumption_count: int = 0,
        reach_event_indices: Tuple[int, ...] = (),
        tone_references: Tuple[Dict[str, Any], ...] = (),
        laser_references: Tuple[Dict[str, Any], ...] = (),
        pellet_presence: str = "unknown",
        pellet_misplacement: str = "unknown",
        analysis_window_start_perf: Optional[float] = None,
        analysis_window_end_perf: Optional[float] = None,
        tracking_coverage: Optional[float] = None,
        tracking_missing_frame_ids: Tuple[int, ...] = (),
        tracking_interpolated_points: int = 0,
        tracking_long_gap_count: int = 0,
        analysis_duration_seconds: Optional[float] = None,
        recommended_shift: Optional[Dict[str, float]] = None,
    ) -> PelletTrialAttempt:
        """Apply one offline-analysis result to a provisionally closed attempt."""
        outcome = TrialOutcome(outcome)
        if outcome in {TrialOutcome.PENDING_ANALYSIS, TrialOutcome.HARDWARE_ERROR}:
            raise ValueError("Pending attempts require a behavioral terminal outcome")
        for index, attempt in enumerate(self._attempts):
            if (
                attempt.trial_id != (
                    None if trial_id is None else int(trial_id)
                )
                or attempt.attempt_id != int(attempt_id)
            ):
                continue
            if attempt.outcome is not TrialOutcome.PENDING_ANALYSIS:
                raise RuntimeError(
                    f"Attempt {attempt.attempt_label} is not pending analysis"
                )
            finalized = dataclasses.replace(
                attempt,
                finalized_perf_time=float(perf_time),
                finalized_wall_time=float(wall_time),
                outcome=outcome,
                error=str(error or ""),
                reach_count=int(reach_count),
                success_count=int(success_count),
                consumption_count=int(consumption_count),
                reach_event_indices=tuple(int(value) for value in reach_event_indices),
                tone_references=tuple(tone_references),
                laser_references=tuple(laser_references),
                pellet_presence=str(pellet_presence),
                pellet_misplacement=str(pellet_misplacement),
                analysis_window_start_perf=(
                    None
                    if analysis_window_start_perf is None
                    else float(analysis_window_start_perf)
                ),
                analysis_window_end_perf=(
                    None
                    if analysis_window_end_perf is None
                    else float(analysis_window_end_perf)
                ),
                tracking_coverage=(
                    None if tracking_coverage is None else float(tracking_coverage)
                ),
                tracking_missing_frame_ids=tuple(
                    int(value) for value in tracking_missing_frame_ids
                ),
                tracking_interpolated_points=int(tracking_interpolated_points),
                tracking_long_gap_count=int(tracking_long_gap_count),
                analysis_duration_seconds=(
                    None
                    if analysis_duration_seconds is None
                    else float(analysis_duration_seconds)
                ),
                recommended_shift=(
                    None if recommended_shift is None else dict(recommended_shift)
                ),
            )
            self._attempts[index] = finalized
            return finalized
        raise KeyError(f"Unknown trial attempt {trial_id}.{attempt_id}")

    def finalize_hardware_error(
        self,
        kind: HardwareErrorKind,
        perf_time: float,
        wall_time: float,
        *,
        error: str,
    ) -> PelletTrialAttempt:
        attempt = self._require_active()
        kind = HardwareErrorKind(kind)
        finalized = self._replace_active(
            finalized_perf_time=float(perf_time),
            finalized_wall_time=float(wall_time),
            outcome=TrialOutcome.HARDWARE_ERROR,
            hardware_error_kind=kind,
            error=str(error),
            logical_trial_complete=False,
        )
        self._active_index = None
        # Hardware failures never consume a logical trial. Retry the reserved
        # number even when behavioral attempts are otherwise separate trials.
        if finalized.trial_id is not None:
            self._retry_trial_id = finalized.trial_id
            self._retry_attempt_id = finalized.attempt_id
        return finalized

    def reconcile_analysis(
        self,
        result,
        *,
        recording_start_perf_time: float,
        frame_rate: float,
        finalized_perf_time: float,
        finalized_wall_time: float,
        tone_references: Iterable[Dict[str, Any]] = (),
        laser_references: Iterable[Dict[str, Any]] = (),
    ) -> Tuple[PelletTrialAttempt, ...]:
        """Map session-frame analysis events into physical attempt windows."""
        if frame_rate <= 0:
            raise ValueError("Analysis reconciliation requires a positive frame rate")
        reach_events = tuple(result.reach_events)
        other_events = tuple(result.other_events)
        all_events = (*reach_events, *other_events)
        tones = tuple(tone_references)
        lasers = tuple(laser_references)
        pending = tuple(
            attempt
            for attempt in self._attempts
            if attempt.outcome is TrialOutcome.PENDING_ANALYSIS
        )
        finalized = []
        for pending_index, attempt in enumerate(pending):
            window_start = attempt.send_perf_time
            next_start = (
                pending[pending_index + 1].send_perf_time
                if pending_index + 1 < len(pending)
                else math.inf
            )
            window_end = min(
                next_start,
                attempt.capture_end_perf_time
                if attempt.capture_end_perf_time is not None
                else math.inf,
            )

            def event_perf(event) -> float:
                return recording_start_perf_time + float(event.init) / frame_rate

            indexed_reaches = tuple(
                (index, event)
                for index, event in enumerate(reach_events)
                if window_start <= event_perf(event) < window_end
            )
            assigned_events = tuple(
                event
                for event in all_events
                if window_start <= event_perf(event) < window_end
            )
            consumed = sum(
                event.outcome == ReachEventOutcome.EATEN
                for event in assigned_events
            )
            successful = sum(
                event.outcome == ReachEventOutcome.EATEN
                and event.method == ReachEventMethod.RIGHT_HAND
                for event in assigned_events
            )
            reach_count = len(indexed_reaches)
            if not attempt.is_presented:
                outcome = TrialOutcome.INCOMPLETE
                error = "pellet send was not acknowledged as presented"
            elif consumed:
                outcome = TrialOutcome.SUCCESS
                error = ""
            elif reach_count:
                outcome = TrialOutcome.FAILURE
                error = ""
            else:
                # Absence of a reach is not evidence that the pellet itself was
                # absent. PELLET_MISSING is reserved for direct live tracking
                # evidence supplied by the pellet-state tracker.
                outcome = TrialOutcome.NO_REACH
                error = ""

            finalized.append(self.finalize_pending(
                attempt.trial_id,
                attempt.attempt_id,
                outcome,
                finalized_perf_time,
                finalized_wall_time,
                error=error,
                reach_count=reach_count,
                success_count=successful,
                consumption_count=consumed,
                reach_event_indices=tuple(index for index, _ in indexed_reaches),
                tone_references=self._references_in_window(
                    tones, window_start, window_end
                ),
                laser_references=self._references_in_window(
                    lasers, window_start, window_end
                ),
            ))
        finalized_operation_ids = {
            attempt.operation_id for attempt in finalized
        }
        self._reindex_analyzed_attempts()
        assigned_reaches = sum(attempt.reach_count for attempt in self._attempts)
        assigned_successes = sum(
            attempt.success_count for attempt in self._attempts
        )
        assigned_consumed = sum(
            attempt.consumption_count for attempt in self._attempts
        )
        self._analysis_counts = {
            "reaches": int(result.total_reaches),
            "successful_reaches": int(result.successful_reaches),
            "pellets_consumed": int(result.food_consumed),
            "unassigned_reaches": max(
                0, int(result.total_reaches) - assigned_reaches
            ),
            "unassigned_successful_reaches": max(
                0, int(result.successful_reaches) - assigned_successes
            ),
            "unassigned_pellets_consumed": max(
                0, int(result.food_consumed) - assigned_consumed
            ),
        }
        return tuple(
            attempt
            for attempt in self._attempts
            if attempt.operation_id in finalized_operation_ids
        )

    def finalize_pending_without_analysis(
        self,
        perf_time: float,
        wall_time: float,
        *,
        reason: str = "post-session analysis was not performed",
    ) -> Tuple[PelletTrialAttempt, ...]:
        finalized = []
        for attempt in tuple(self._attempts):
            if attempt.outcome is TrialOutcome.PENDING_ANALYSIS:
                finalized.append(self.finalize_pending(
                    attempt.trial_id,
                    attempt.attempt_id,
                    TrialOutcome.INCOMPLETE,
                    perf_time,
                    wall_time,
                    error=reason,
                ))
        self._reindex_analyzed_attempts()
        return tuple(
            current
            for current in self._attempts
            if any(
                current.operation_id == previous.operation_id
                for previous in finalized
            )
        )

    def count(self, basis: Optional[TrialCountBasis] = None) -> int:
        basis = TrialCountBasis(basis or self.configuration.count_basis)
        grouped = self._grouped_attempts()
        total = 0
        for attempts in grouped.values():
            non_hardware = tuple(
                attempt
                for attempt in attempts
                if attempt.outcome is not TrialOutcome.HARDWARE_ERROR
            )
            if not non_hardware:
                continue
            if basis is TrialCountBasis.STARTED:
                qualifies = any(
                    attempt.is_presented or attempt.is_finalized
                    for attempt in non_hardware
                )
            elif basis is TrialCountBasis.PRESENTED:
                qualifies = any(attempt.is_presented for attempt in non_hardware)
            elif basis is TrialCountBasis.COMPLETED:
                qualifies = any(
                    attempt.logical_trial_complete
                    and (
                        attempt.outcome is TrialOutcome.PENDING_ANALYSIS
                        or attempt.outcome in self.configuration.counted_outcomes
                    )
                    for attempt in non_hardware
                )
            else:
                qualifies = any(
                    attempt.logical_trial_complete
                    and attempt.outcome is not None
                    and attempt.outcome.is_scored
                    and attempt.outcome in self.configuration.counted_outcomes
                    for attempt in non_hardware
                )
            total += int(qualifies)
        return total

    def summary(self) -> dict:
        return {
            "physical_attempts": len(self._attempts),
            "hardware_errors": sum(
                attempt.outcome is TrialOutcome.HARDWARE_ERROR
                for attempt in self._attempts
            ),
            "incomplete_attempts": sum(
                attempt.outcome in {TrialOutcome.INCOMPLETE, TrialOutcome.ABORTED}
                for attempt in self._attempts
            ),
            "pending_analysis_attempts": sum(
                attempt.outcome is TrialOutcome.PENDING_ANALYSIS
                for attempt in self._attempts
            ),
            "trials_started": self.count(TrialCountBasis.STARTED),
            "pellets_presented": self.count(TrialCountBasis.PRESENTED),
            "trials_completed": self.count(TrialCountBasis.COMPLETED),
            "scored_trials": self.count(TrialCountBasis.SCORED),
            "configured_trial_count": self.count(),
            **self._analysis_counts,
        }

    def to_records(self) -> Tuple[dict, ...]:
        return tuple(attempt.to_dict() for attempt in self._attempts)

    def _grouped_attempts(self) -> Dict[int, Tuple[PelletTrialAttempt, ...]]:
        grouped = {}
        for attempt in self._attempts:
            if attempt.trial_id is None:
                continue
            grouped.setdefault(attempt.trial_id, []).append(attempt)
        return {
            trial_id: tuple(attempts)
            for trial_id, attempts in grouped.items()
        }

    def _require_active(self) -> PelletTrialAttempt:
        attempt = self.active_attempt
        if attempt is None:
            raise RuntimeError("No pellet-send attempt is active")
        return attempt

    def _replace_active(self, **changes) -> PelletTrialAttempt:
        attempt = self._require_active()
        replacement = dataclasses.replace(attempt, **changes)
        self._attempts[self._active_index] = replacement
        return replacement

    @staticmethod
    def _references_in_window(references, start, end):
        selected = []
        for reference in references:
            perf_time = next((
                reference.get(key)
                for key in (
                    "perf_time",
                    "perfTime",
                    "eventPerfTime",
                    "edgePerfTime",
                )
                if reference.get(key) is not None
            ), None)
            if perf_time is not None and start <= float(perf_time) < end:
                selected.append(dict(reference))
        return tuple(selected)

    def _reindex_analyzed_attempts(self) -> None:
        """Apply behavioral retry numbering once outcomes are known."""
        policy = self.configuration.assignment_policy
        if policy is AttemptAssignmentPolicy.SUCCESSFUL_PRESENTATIONS_ONLY:
            next_trial = 1
            unindexed = 0
            for index, attempt in enumerate(self._attempts):
                if attempt.is_presented:
                    replacement = dataclasses.replace(
                        attempt,
                        trial_id=next_trial,
                        attempt_id=1,
                        logical_trial_complete=attempt.outcome is not TrialOutcome.HARDWARE_ERROR,
                    )
                    next_trial += 1
                else:
                    unindexed += 1
                    replacement = dataclasses.replace(
                        attempt,
                        trial_id=None,
                        attempt_id=unindexed,
                        logical_trial_complete=False,
                    )
                self._attempts[index] = replacement
            self._next_trial_id = next_trial
            self._unindexed_attempt_id = unindexed
            return

        trial_id = 1
        attempt_id = 1
        non_hardware_indices = [
            index
            for index, attempt in enumerate(self._attempts)
            if attempt.outcome is not TrialOutcome.HARDWARE_ERROR
        ]
        last_non_hardware = (
            non_hardware_indices[-1] if non_hardware_indices else None
        )
        for index, attempt in enumerate(self._attempts):
            is_hardware = attempt.outcome is TrialOutcome.HARDWARE_ERROR
            if policy is AttemptAssignmentPolicy.EVERY_ATTEMPT_IS_TRIAL:
                retry = is_hardware
            else:
                retry = is_hardware or (
                    index != last_non_hardware
                    and attempt.outcome in {
                        TrialOutcome.FAILURE,
                        TrialOutcome.PELLET_MISSING,
                    }
                )
            self._attempts[index] = dataclasses.replace(
                attempt,
                trial_id=trial_id,
                attempt_id=attempt_id,
                logical_trial_complete=not retry,
                retry_settings_policy=(
                    self.configuration.retry_settings_policy
                    if attempt_id > 1
                    else None
                ),
            )
            if retry:
                attempt_id += 1
            else:
                trial_id += 1
                attempt_id = 1
        self._next_trial_id = trial_id
