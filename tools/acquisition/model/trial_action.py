"""Pure compilation and generation-owned preparation for pellet-trial actions."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import threading
import time
import uuid
from typing import Callable, Mapping, Optional, Tuple

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.stimulus_trigger_profile import (
    StimulusTriggerCategory,
    profile_record as stimulus_trigger_profile_record,
    select_trigger,
    validate_profile as validate_stimulus_trigger_profile,
)

from autotrainer.core.delay_distribution import (
    EXPONENTIAL_CDF_TAU_MS,
    DelayDistributionProfile,
    DelayPreset,
    build_delay_distribution_profile,
    normalize_delay_preset,
    preset_values,
    select_delay,
)

from tools.acquisition.model.cue_timer import CueTimer
from tools.acquisition.model.cue_timing import (
    CueCancelReason,
    CueDecision,
    CueGateState,
    CueTimingConfiguration,
    CueTimingPolicy,
)
from tools.acquisition.model.reach_state_source import ReachStateResolver

from tools.acquisition.model.trial_protocol_schedule import (
    OFFSET_STIMULUS_TRIGGERS,
    LaserTriggerRoute,
    PelletLane,
    PelletPositionMode,
    RetryAssignment,
    StimulusAssignment,
    StimulusTrigger,
    TrialProtocolRow,
)


logger = get_verbose_logger(__name__)


# How long an unlocked trial keeps waiting for a blocked gate before giving up.
# CueTimingPolicy deliberately waits indefinitely and leaves that decision to
# its caller, so the bound lives here.
DEFAULT_CUE_MAX_WAIT_SECONDS = 5.0

# How often the gate is re-read while an unlocked trial waits out a block.
DEFAULT_CUE_POLL_SECONDS = 0.010


@dataclasses.dataclass(frozen=True)
class ToneProfile:
    profile_id: str
    revision: int
    frequency_hz: int
    duration_ms: int

    def __post_init__(self):
        if not self.profile_id:
            raise ValueError("Tone profile ID cannot be empty")
        if self.revision < 1 or self.frequency_hz <= 0 or self.duration_ms <= 0:
            raise ValueError("Tone revision, frequency, and duration must be positive")

    def to_record(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CueIntervalProfile:
    """Reusable Tone 1 to Tone 2 interval distribution.

    The profile owns the configuration; the resolved probability weights are
    derived by :mod:`autotrainer.core.delay_distribution` so the published
    supports stay in one place.
    """

    profile_id: str
    revision: int
    preset: str = DelayPreset.PUBLISHED_4S.value
    values: Tuple[int, ...] = ()
    tau_ms: int = EXPONENTIAL_CDF_TAU_MS
    manual_probabilities: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        if not self.profile_id:
            raise ValueError("Cue interval profile ID cannot be empty")
        if self.revision < 1:
            raise ValueError("Cue interval profile revision must be positive")
        # Build once so a malformed profile fails at construction rather than
        # part-way through compiling a trial.
        self.distribution()

    def distribution(self) -> DelayDistributionProfile:
        """Return the resolved distribution for this profile."""

        preset = normalize_delay_preset(self.preset)
        values = self.values or preset_values(preset)
        if not values:
            raise ValueError(
                f"Cue interval profile {self.profile_id!r} requires cue intervals"
            )
        return build_delay_distribution_profile(
            values,
            preset,
            tau_ms=self.tau_ms,
            manual_probabilities=self.manual_probabilities,
        )

    def to_record(self):
        record = dataclasses.asdict(self)
        record["values"] = list(self.values)
        if self.manual_probabilities is not None:
            record["manual_probabilities"] = list(self.manual_probabilities)
        return record


@dataclasses.dataclass(frozen=True)
class StimulusTriggerProfile:
    """Reusable weighted set of stimulus trigger categories.

    Weights are conditional on stimulation already having been selected for
    the trial, so they total 100% among enabled categories.
    """

    profile_id: str
    revision: int
    categories: Tuple[StimulusTriggerCategory, ...] = ()

    def __post_init__(self):
        if not self.profile_id:
            raise ValueError("Stimulus trigger profile ID cannot be empty")
        if self.revision < 1:
            raise ValueError("Stimulus trigger profile revision must be positive")
        object.__setattr__(
            self,
            "categories",
            tuple(
                item
                if isinstance(item, StimulusTriggerCategory)
                else StimulusTriggerCategory(**dict(item))
                for item in self.categories
            ),
        )
        for category in self.categories:
            trigger = StimulusTrigger(category.trigger)
            if category.is_offset_trigger and trigger not in OFFSET_STIMULUS_TRIGGERS:
                raise ValueError(
                    f"{trigger.value} stimulus triggers do not take a lead time"
                )
        # Reject an unusable profile at construction rather than mid-trial.
        validate_stimulus_trigger_profile(self.categories)

    def validated(self, *, offset_upper_bound_ms: Optional[int] = None):
        return validate_stimulus_trigger_profile(
            self.categories, offset_upper_bound_ms=offset_upper_bound_ms
        )

    def to_record(self):
        return {
            "profile_id": self.profile_id,
            "revision": self.revision,
            **stimulus_trigger_profile_record(self.categories),
        }


@dataclasses.dataclass(frozen=True)
class LaserPulseProfile:
    profile_id: str
    revision: int
    channel_id: int
    amplitude_volts: float
    pulse_duration_ms: float
    pulse_count: int = 1
    frequency_hz: Optional[float] = None
    baseline_ms: float = 0.0
    post_stim_ms: float = 0.0
    pmt_open_lead_ms: float = 0.0
    pmt_close_lag_ms: float = 0.0
    trigger_route: LaserTriggerRoute = LaserTriggerRoute.HARDWARE_STIM3
    trigger_terminal: str = ""
    trigger_pulse_us: int = 1000

    def __post_init__(self):
        object.__setattr__(self, "trigger_route", LaserTriggerRoute(self.trigger_route))
        numeric = (
            self.amplitude_volts,
            self.pulse_duration_ms,
            self.baseline_ms,
            self.post_stim_ms,
            self.pmt_open_lead_ms,
            self.pmt_close_lag_ms,
        )
        if not self.profile_id or self.revision < 1 or self.channel_id < 1:
            raise ValueError("Laser profile identity, revision, and channel are required")
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Laser profile values must be finite")
        if self.pulse_duration_ms <= 0 or self.pulse_count < 1:
            raise ValueError("Laser pulse duration/count must be positive")
        if not 100 <= int(self.trigger_pulse_us) <= 5_000_000:
            raise ValueError("STIM3 trigger pulse must be within 100 us..5 s")
        if any(value < 0 for value in numeric[2:]):
            raise ValueError("Laser timing margins cannot be negative")
        if self.pulse_count > 1 and (self.frequency_hz is None or self.frequency_hz <= 0):
            raise ValueError("Multi-pulse profiles require a positive frequency")
        if self.trigger_route is LaserTriggerRoute.NONE:
            raise ValueError("Laser profiles require an explicit trigger route")
        if (
            self.trigger_route is LaserTriggerRoute.HARDWARE_STIM3
            and not self.trigger_terminal
        ):
            raise ValueError("Hardware STIM3 laser profiles require an NI trigger terminal")

    def to_record(self):
        result = dataclasses.asdict(self)
        result["trigger_route"] = self.trigger_route.value
        return result


@dataclasses.dataclass(frozen=True)
class TrialCompileContext:
    session_id: str
    session_generation: int
    protocol_id: str
    protocol_revision: int
    logical_trial_id: int
    attempt_id: int
    session_seed: int
    animal_base_dcs: Tuple[float, float, float]
    lane_offsets_dcs: Mapping[PelletLane, Tuple[float, float, float]]
    automatic_target_dcs: Optional[Tuple[float, float, float]] = None
    automatic_generation: Optional[int] = None
    automatic_reach_ids: Tuple[str, ...] = ()
    automatic_policy: Optional[Mapping[str, object]] = None
    automatic_recommendation: Optional[Mapping[str, object]] = None


@dataclasses.dataclass(frozen=True)
class CompiledTrialRecipe:
    operation_id: str
    session_id: str
    session_generation: int
    protocol_id: str
    protocol_revision: int
    logical_trial_id: int
    attempt_id: int
    requested_row: Mapping[str, object]
    resolved_dcs_target: Tuple[float, float, float]
    resolved_motor_target: Tuple[float, float, float]
    position_evidence: Mapping[str, object]
    stimulus_selected: bool
    stimulus_seed: int
    stimulus_draw: float
    tone_profile: Optional[ToneProfile]
    laser_profile: Optional[LaserPulseProfile]
    cue_tone_profile: Optional[ToneProfile] = None
    cue_interval_ms: Optional[int] = None
    cue_interval_seed: Optional[int] = None
    cue_interval_draw: Optional[float] = None
    cue_interval_selection: Optional[Mapping[str, object]] = None
    cue_interval_distribution: Optional[Mapping[str, object]] = None
    resolved_stimulus_trigger: str = StimulusTrigger.NONE.value
    resolved_trigger_offset_ms: int = 0
    stimulus_trigger_seed: Optional[int] = None
    stimulus_trigger_draw: Optional[float] = None
    stimulus_trigger_selection: Optional[Mapping[str, object]] = None

    def to_record(self):
        return {
            "operation_id": self.operation_id,
            "session_id": self.session_id,
            "session_generation": self.session_generation,
            "protocol_id": self.protocol_id,
            "protocol_revision": self.protocol_revision,
            "logical_trial_id": self.logical_trial_id,
            "attempt_id": self.attempt_id,
            "requested_row": dict(self.requested_row),
            "resolved_dcs_target": list(self.resolved_dcs_target),
            "resolved_motor_target": list(self.resolved_motor_target),
            "position_evidence": dict(self.position_evidence),
            "stimulus_selected": self.stimulus_selected,
            "stimulus_seed": self.stimulus_seed,
            "stimulus_draw": self.stimulus_draw,
            "tone_profile": None if self.tone_profile is None else self.tone_profile.to_record(),
            "laser_profile": None if self.laser_profile is None else self.laser_profile.to_record(),
            "cue_tone_profile": (
                None
                if self.cue_tone_profile is None
                else self.cue_tone_profile.to_record()
            ),
            "cue_interval_ms": self.cue_interval_ms,
            "cue_interval_seed": self.cue_interval_seed,
            "cue_interval_draw": self.cue_interval_draw,
            "cue_interval_selection": (
                None
                if self.cue_interval_selection is None
                else dict(self.cue_interval_selection)
            ),
            "cue_interval_distribution": (
                None
                if self.cue_interval_distribution is None
                else dict(self.cue_interval_distribution)
            ),
            "resolved_stimulus_trigger": self.resolved_stimulus_trigger,
            "resolved_trigger_offset_ms": self.resolved_trigger_offset_ms,
            "stimulus_trigger_seed": self.stimulus_trigger_seed,
            "stimulus_trigger_draw": self.stimulus_trigger_draw,
            "stimulus_trigger_selection": (
                None
                if self.stimulus_trigger_selection is None
                else dict(self.stimulus_trigger_selection)
            ),
        }


class TrialActionCompiler:
    """Compile a resolved row without touching hardware or application state."""

    def __init__(
        self,
        *,
        tone_profiles: Mapping[str, ToneProfile] = None,
        laser_profiles: Mapping[str, LaserPulseProfile] = None,
        cue_interval_profiles: Mapping[str, CueIntervalProfile] = None,
        stimulus_trigger_profiles: Mapping[str, StimulusTriggerProfile] = None,
        dcs_to_motor: Callable[[Tuple[float, float, float]], Tuple[float, float, float]],
    ):
        self._tone_profiles = dict(tone_profiles or {})
        self._laser_profiles = dict(laser_profiles or {})
        self._cue_interval_profiles = dict(cue_interval_profiles or {})
        self._stimulus_trigger_profiles = dict(stimulus_trigger_profiles or {})
        self._dcs_to_motor = dcs_to_motor

    def compile(self, row: TrialProtocolRow, context: TrialCompileContext):
        row.validate(runnable=True)
        if row.trial_id != context.logical_trial_id:
            raise ValueError("Resolved row does not match the planned logical trial")
        base = _finite_vector(context.animal_base_dcs, "animal base DCS")
        lane = _finite_vector(
            context.lane_offsets_dcs.get(row.position_lane, (0.0, 0.0, 0.0)),
            f"{row.position_lane.value} lane offset",
        )
        baseline = _add(base, lane)
        position_evidence = {
            "mode": row.position_mode.value,
            "base_dcs": list(base),
            "lane": row.position_lane.value,
            "lane_offset_dcs": list(lane),
            "automatic_generation": context.automatic_generation,
            "automatic_reach_ids": list(context.automatic_reach_ids),
            "automatic_policy": (
                None
                if context.automatic_policy is None
                else dict(context.automatic_policy)
            ),
            "automatic_recommendation": (
                None
                if context.automatic_recommendation is None
                else dict(context.automatic_recommendation)
            ),
        }
        if row.position_mode is PelletPositionMode.FIXED_MANUAL:
            target = _add(baseline, (
                row.shift_x_mm,
                row.shift_y_mm,
                row.shift_z_mm,
            ))
            position_evidence["manual_offset_dcs"] = [
                row.shift_x_mm,
                row.shift_y_mm,
                row.shift_z_mm,
            ]
        elif row.position_mode is PelletPositionMode.REACH_DERIVED_AUTOMATIC:
            if context.automatic_target_dcs is None:
                target = baseline
                recommendation = context.automatic_recommendation or {}
                position_evidence.update(
                    automatic_status=(
                        "recommendation_only"
                        if recommendation
                        and not recommendation.get("apply_automatically", True)
                        else "insufficient_history"
                    ),
                    fallback="categorical_lane_baseline",
                )
            else:
                target = _finite_vector(
                    context.automatic_target_dcs,
                    "automatic target DCS",
                )
                position_evidence["automatic_status"] = "applied"
        else:
            target = baseline
        motor = _finite_vector(self._dcs_to_motor(target), "motor target")

        stimulus_seed = _stimulus_seed(row, context)
        stimulus_draw = _stable_draw(stimulus_seed)
        if row.stimulus_assignment is StimulusAssignment.DISABLED:
            selected = False
        elif row.stimulus_assignment is StimulusAssignment.ALWAYS:
            selected = True
        else:
            selected = stimulus_draw * 100.0 < row.stimulus_probability_percent

        tone = None
        if row.tone_profile_id:
            tone = self._tone_profiles.get(row.tone_profile_id)
            if tone is None:
                raise ValueError(f"Unknown tone profile {row.tone_profile_id!r}")
        cue_tone = None
        cue_interval_ms = None
        cue_interval_seed = None
        cue_interval_draw = None
        cue_interval_selection = None
        cue_interval_distribution = None
        if row.cue_tone_profile_id:
            cue_tone = self._tone_profiles.get(row.cue_tone_profile_id)
            if cue_tone is None:
                raise ValueError(
                    f"Unknown cue tone profile {row.cue_tone_profile_id!r}"
                )
            if row.cue_interval_profile_id:
                interval_profile = self._cue_interval_profiles.get(
                    row.cue_interval_profile_id
                )
                if interval_profile is None:
                    raise ValueError(
                        "Unknown cue interval profile "
                        f"{row.cue_interval_profile_id!r}"
                    )
                distribution = interval_profile.distribution()
                # A separate seed domain keeps the cue interval independent of
                # the stimulus draw for the same trial.
                cue_interval_seed = _domain_seed(row, context, "cue_interval")
                cue_interval_draw = _stable_draw(cue_interval_seed)
                selection = select_delay(distribution, cue_interval_draw)
                cue_interval_ms = selection.delay_ms
                cue_interval_selection = selection.to_record()
                cue_interval_distribution = distribution.to_record()
            else:
                cue_interval_ms = int(row.cue_interval_fixed_ms)

        resolved_trigger = row.stimulus_trigger
        resolved_offset_ms = int(row.pre_reveal_ms)
        stimulus_trigger_seed = None
        stimulus_trigger_draw = None
        stimulus_trigger_selection = None
        if row.stimulus_assignment is StimulusAssignment.RANDOMIZED:
            trigger_profile = self._stimulus_trigger_profiles.get(
                row.stimulus_trigger_profile_id
            )
            if trigger_profile is None:
                raise ValueError(
                    "Unknown stimulus trigger profile "
                    f"{row.stimulus_trigger_profile_id!r}"
                )
            # A Tone 2 lead time has to fit inside the shortest cue interval
            # this row can draw, otherwise it could never be delivered.
            categories = trigger_profile.validated(
                offset_upper_bound_ms=_shortest_cue_interval(
                    row, cue_interval_ms, cue_interval_distribution
                )
            )
            for category in categories:
                if (
                    category.enabled
                    and StimulusTrigger(category.trigger) is StimulusTrigger.TONE_2
                    and not row.cue_tone_profile_id
                ):
                    raise ValueError(
                        "Tone 2 stimulus triggers require a configured cue tone"
                    )
            if selected:
                stimulus_trigger_seed = _domain_seed(
                    row, context, "stimulus_trigger"
                )
                stimulus_trigger_draw = _stable_draw(stimulus_trigger_seed)
                selection = select_trigger(categories, stimulus_trigger_draw)
                resolved_trigger = StimulusTrigger(selection.category.trigger)
                resolved_offset_ms = int(selection.category.offset_ms or 0)
                stimulus_trigger_selection = selection.to_record()
            else:
                resolved_trigger = StimulusTrigger.NONE
                resolved_offset_ms = 0

        laser = None
        if row.laser_profile_id:
            laser = self._laser_profiles.get(row.laser_profile_id)
            if laser is None:
                raise ValueError(f"Unknown laser profile {row.laser_profile_id!r}")
            if laser.trigger_route is not row.laser_trigger_route:
                raise ValueError("Protocol row and laser profile trigger routes differ")
            if (
                resolved_trigger is StimulusTrigger.PRE_REVEAL
                and int(laser.trigger_pulse_us) >= resolved_offset_ms * 1000
            ):
                raise ValueError(
                    "Pre-reveal interval must be longer than the STIM3 trigger pulse"
                )
        if not selected:
            laser = None

        return CompiledTrialRecipe(
            operation_id=str(uuid.uuid4()),
            session_id=context.session_id,
            session_generation=context.session_generation,
            protocol_id=context.protocol_id,
            protocol_revision=context.protocol_revision,
            logical_trial_id=context.logical_trial_id,
            attempt_id=context.attempt_id,
            requested_row=row.to_record(),
            resolved_dcs_target=target,
            resolved_motor_target=motor,
            position_evidence=position_evidence,
            stimulus_selected=selected,
            stimulus_seed=stimulus_seed,
            stimulus_draw=stimulus_draw,
            tone_profile=tone,
            laser_profile=laser,
            cue_tone_profile=cue_tone,
            cue_interval_ms=cue_interval_ms,
            cue_interval_seed=cue_interval_seed,
            cue_interval_draw=cue_interval_draw,
            cue_interval_selection=cue_interval_selection,
            cue_interval_distribution=cue_interval_distribution,
            resolved_stimulus_trigger=resolved_trigger.value,
            resolved_trigger_offset_ms=resolved_offset_ms,
            stimulus_trigger_seed=stimulus_trigger_seed,
            stimulus_trigger_draw=stimulus_trigger_draw,
            stimulus_trigger_selection=stimulus_trigger_selection,
        )


class PreparedState(str, enum.Enum):
    PREPARING = "preparing"
    PREPARED = "prepared"
    SEND_ACCEPTED = "send_accepted"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclasses.dataclass(frozen=True)
class TrialActionObservation:
    state: PreparedState
    perf_time: float
    detail: str = ""


class PreparedTrialOperation:
    """Thread-safe state/evidence owner for one frozen physical attempt."""

    _TRANSITIONS = {
        PreparedState.PREPARING: {PreparedState.PREPARED, PreparedState.FAILED, PreparedState.CANCELLED},
        PreparedState.PREPARED: {PreparedState.SEND_ACCEPTED, PreparedState.FAILED, PreparedState.CANCELLED},
        PreparedState.SEND_ACCEPTED: {PreparedState.ACTIVE, PreparedState.FAILED, PreparedState.CANCELLED},
        PreparedState.ACTIVE: {PreparedState.COMPLETED, PreparedState.FAILED, PreparedState.CANCELLED},
    }
    TERMINAL = {PreparedState.COMPLETED, PreparedState.FAILED, PreparedState.CANCELLED}

    def __init__(self, recipe: CompiledTrialRecipe):
        self.recipe = recipe
        self._lock = threading.RLock()
        self._state = PreparedState.PREPARING
        self._observations = [TrialActionObservation(self._state, time.perf_counter())]
        self._action_records = {}

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def observations(self):
        with self._lock:
            return tuple(self._observations)

    def transition(self, state: PreparedState, detail=""):
        state = PreparedState(state)
        with self._lock:
            if state == self._state:
                return False
            if self._state in self.TERMINAL or state not in self._TRANSITIONS[self._state]:
                raise RuntimeError(f"Invalid prepared operation transition {self._state.value} -> {state.value}")
            self._state = state
            self._observations.append(
                TrialActionObservation(state, time.perf_counter(), str(detail or ""))
            )
            return True

    def require_generation(self, generation: int):
        if int(generation) != self.recipe.session_generation:
            raise RuntimeError("Prepared operation belongs to a stale session generation")

    def set_action_record(self, name: str, record) -> None:
        with self._lock:
            self._action_records[str(name)] = record

    def to_record(self):
        with self._lock:
            return {
                "recipe": self.recipe.to_record(),
                "state": self._state.value,
                "observations": [
                    {
                        "state": item.state.value,
                        "perf_time": item.perf_time,
                        "detail": item.detail,
                    }
                    for item in self._observations
                ],
                "actions": dict(self._action_records),
            }


class TrialActionExecutor:
    """Own preparation, SEND permission, and cleanup for frozen recipes.

    Hardware behavior is injected through narrow callbacks so this owner never
    reaches into Qt, CAN transports, or DAQmx directly. The application invokes
    ``prepare`` on its command worker, not the UI thread.
    """

    def __init__(
        self,
        *,
        move_absolute: Callable[[Tuple[float, float, float]], object],
        configure_cover: Callable[[str, CompiledTrialRecipe], object],
        play_tone: Callable[[ToneProfile, str], object],
        prepare_laser: Callable[[LaserPulseProfile, CompiledTrialRecipe], object],
        cancel_laser: Callable[[object], None],
        release_laser: Callable[[object], None] = lambda _handle: None,
        prepare_detector: Callable[[CompiledTrialRecipe], object] = lambda _recipe: None,
        activate_detector: Callable[[object], None] = lambda _handle: None,
        cancel_detector: Callable[[object], None] = lambda _handle: None,
        trigger_hardware_stimulus: Callable[
            [LaserPulseProfile, CompiledTrialRecipe, str], object
        ] = lambda _profile, _recipe, _detail: None,
        cue_timing_configuration: Optional[CueTimingConfiguration] = None,
        reach_state_resolver: Optional[ReachStateResolver] = None,
        pellet_presence_provider: Optional[Callable[[], object]] = None,
        cue_timer_factory: Optional[Callable[[], CueTimer]] = None,
        cue_max_wait_seconds: float = DEFAULT_CUE_MAX_WAIT_SECONDS,
        cue_poll_seconds: float = DEFAULT_CUE_POLL_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._move_absolute = move_absolute
        self._configure_cover = configure_cover
        self._play_tone = play_tone
        self._prepare_laser = prepare_laser
        self._cancel_laser = cancel_laser
        self._release_laser = release_laser
        self._prepare_detector = prepare_detector
        self._activate_detector = activate_detector
        self._cancel_detector = cancel_detector
        self._trigger_hardware_stimulus = trigger_hardware_stimulus
        self._lock = threading.RLock()
        self._operation: Optional[PreparedTrialOperation] = None
        self._laser_handle = None
        self._detector_handle = None
        self._send_context: Optional[str] = None

        # Cue pair. The policy owns the decision, the timer owns the deadline,
        # and this owner only routes between them. All of it is inert unless a
        # recipe carries both a cue tone and a drawn cue interval.
        self._cue_policy = CueTimingPolicy(cue_timing_configuration)
        self._reach_state = reach_state_resolver
        self._pellet_presence_provider = pellet_presence_provider
        self._cue_timer_factory = cue_timer_factory or CueTimer
        self._cue_max_wait_seconds = max(0.0, float(cue_max_wait_seconds))
        self._cue_poll_seconds = max(0.001, float(cue_poll_seconds))
        self._clock = clock
        self._cue_timer: Optional[CueTimer] = None
        self._cue_recipe: Optional[CompiledTrialRecipe] = None
        self._cue_expiry: Optional[float] = None

    @property
    def operation(self):
        with self._lock:
            return self._operation

    def matches_send_context(self, context) -> bool:
        with self._lock:
            return self._send_context is not None and self._send_context == str(context)

    def prepare(self, recipe: CompiledTrialRecipe) -> PreparedTrialOperation:
        with self._lock:
            if (
                self._operation is not None
                and self._operation.state not in PreparedTrialOperation.TERMINAL
            ):
                raise RuntimeError("Another pellet-trial operation is still active")
            operation = self._operation = PreparedTrialOperation(recipe)
            self._laser_handle = None
            self._detector_handle = None
            self._send_context = None
        try:
            self._move_absolute(recipe.resolved_motor_target)
            self._observe("motor target acknowledged")
            cover = recipe.requested_row["cover_policy"]
            self._configure_cover(str(cover), recipe)
            self._observe(f"cover policy prepared: {cover}")
            row = recipe.requested_row
            if (
                recipe.tone_profile is not None
                and row["tone_phase"] in {"before_send", "embedded_in_sequence"}
            ):
                phase = row["tone_phase"]
                self._play_tone(recipe.tone_profile, phase)
                self._observe(f"{phase} tone prepared/acknowledged")
            if recipe.laser_profile is not None:
                self._laser_handle = self._prepare_laser(recipe.laser_profile, recipe)
                self._snapshot_laser_action()
                self._observe(
                    "laser prepared: " + recipe.laser_profile.trigger_route.value
                )
                if row["laser_phase"] == "before_send":
                    self._trigger_laser_if_direct("before_send")
            if recipe.stimulus_selected and row["stimulus_trigger"] == "first_reach":
                self._detector_handle = self._prepare_detector(recipe)
                self._observe("stim-camera First Reach detector prepared")
            operation.transition(PreparedState.PREPARED)
            return operation
        except Exception as error:
            self._cancel_laser_safely()
            operation.transition(PreparedState.FAILED, f"{type(error).__name__}: {error}")
            raise

    def require_send_permission(self, operation_id: str, generation: int):
        with self._lock:
            operation = self._require_operation(operation_id)
            operation.require_generation(generation)
            if operation.state is not PreparedState.PREPARED:
                raise RuntimeError(
                    f"Pellet SEND requires Prepared state, found {operation.state.value}"
                )
            return operation

    def bind_send(self, operation_id: str, generation: int, send_context: str):
        with self._lock:
            operation = self.require_send_permission(operation_id, generation)
            self._send_context = str(send_context)
            operation.transition(
                PreparedState.SEND_ACCEPTED,
                f"pellet SEND queued: {self._send_context}",
            )
            return operation

    def acknowledge_presentation(self, send_context: str):
        with self._lock:
            operation = self._require_current()
            if self._send_context != str(send_context):
                raise RuntimeError("Pellet acknowledgement does not match prepared SEND")
            operation.transition(PreparedState.ACTIVE, "pellet presentation acknowledged")
            if self._detector_handle is not None:
                self._activate_detector(self._detector_handle)
                self._observe("stim-camera First Reach detector activated")
            self.execute_phase("pellet_presentation")
            return operation

    def execute_phase(self, phase: str):
        with self._lock:
            operation = self._require_current()
            recipe = operation.recipe
            row = recipe.requested_row
            if recipe.tone_profile is not None and row["tone_phase"] == phase:
                self._play_tone(recipe.tone_profile, phase)
                self._observe(f"{phase} tone acknowledged")
                # Tone 1 has just sounded, so the cue interval starts here.
                self._arm_cue_pair(recipe, phase)
            # A laser is already armed. Phase execution records the semantic
            # trigger point; the configured STIM3/NI route owns physical start.
            if recipe.laser_profile is not None and row["laser_phase"] == phase:
                self._trigger_laser_if_direct(phase)
                self._observe(f"{phase} laser trigger enabled")

    def trigger_stimulus(self, operation_id: str, generation: int, *, detail="stimulus"):
        """Accept one generation-tagged detector trigger for the prepared laser."""
        with self._lock:
            operation = self._require_operation(operation_id)
            operation.require_generation(generation)
            profile = operation.recipe.laser_profile
            if profile is None:
                raise RuntimeError("Stimulus trigger has no prepared laser profile")
            if profile.trigger_route is LaserTriggerRoute.DIRECT_NI_SOFTWARE:
                self._trigger_laser_if_direct(detail)
                self._observe(f"{detail} direct NI trigger accepted")
            elif profile.trigger_route is LaserTriggerRoute.HARDWARE_STIM3:
                self._trigger_hardware_stimulus(profile, operation.recipe, detail)
                self._observe(f"{detail} firmware STIM3 trigger acknowledged")
            else:
                raise RuntimeError(f"Unsupported stimulus route: {profile.trigger_route}")

    def _trigger_laser_if_direct(self, detail):
        operation = self._require_current()
        profile = operation.recipe.laser_profile
        if profile is None or profile.trigger_route is not LaserTriggerRoute.DIRECT_NI_SOFTWARE:
            return False
        trigger = getattr(self._laser_handle, "trigger", None)
        if trigger is None:
            raise RuntimeError("Prepared direct NI laser operation cannot be triggered")
        trigger()
        self._snapshot_laser_action()
        return True

    def complete(self, detail=""):
        with self._lock:
            operation = self._require_current()
            self._cancel_cue_pair()
            self._await_laser_terminal_for_cycle()
            if operation.state is PreparedState.SEND_ACCEPTED:
                operation.transition(PreparedState.ACTIVE, "cycle completion")
            operation.transition(PreparedState.COMPLETED, detail)
            if self._laser_handle is not None:
                self._snapshot_laser_action()
                self._release_laser(self._laser_handle)
            self._laser_handle = None
            self._cancel_detector_safely()
            return operation

    def fail(self, error):
        with self._lock:
            operation = self._require_current()
            self._cancel_cue_pair(CueCancelReason.HOST_REQUEST)
            self._cancel_laser_safely()
            self._cancel_detector_safely()
            if operation.state not in PreparedTrialOperation.TERMINAL:
                operation.transition(
                    PreparedState.FAILED,
                    f"{type(error).__name__}: {error}",
                )
            return operation

    def cancel(self, *, generation: Optional[int] = None, reason="cancelled"):
        with self._lock:
            operation = self._operation
            if operation is None:
                return None
            if generation is not None:
                operation.require_generation(generation)
            self._cancel_cue_pair(CueCancelReason.HOST_REQUEST)
            self._cancel_laser_safely()
            self._cancel_detector_safely()
            if operation.state not in PreparedTrialOperation.TERMINAL:
                operation.transition(PreparedState.CANCELLED, reason)
            return operation

    # --- cue pair -----------------------------------------------------------

    def _arm_cue_pair(self, recipe: CompiledTrialRecipe, phase: str) -> None:
        """
        Start the Tone 1 to Tone 2 interval, if this recipe has one.

        Only for a host-played tone. When the tone phase is
        ``embedded_in_sequence`` the pellet board owns the sequence timing, and
        a host timer would be describing a cue it does not deliver.
        """
        cue_tone = recipe.cue_tone_profile
        interval_ms = recipe.cue_interval_ms
        if cue_tone is None or not interval_ms:
            return
        if phase == "embedded_in_sequence":
            self._observe(
                "cue pair not host-timed: the board owns the embedded sequence"
            )
            return

        self._cancel_cue_pair()
        tone_1_perf_time = self._clock()
        deadline = self._cue_policy.start(tone_1_perf_time, interval_ms)
        self._cue_recipe = recipe
        self._cue_expiry = deadline + self._cue_max_wait_seconds
        self._observe(f"cue pair armed: Tone 2 due in {interval_ms} ms")
        self._schedule_cue(deadline)

    def _schedule_cue(self, deadline_perf_time: float) -> None:
        # A fresh timer per arm: the previous one's thread is the caller when
        # this runs from a WAIT re-arm, and a timer refuses to reschedule while
        # its own thread is still alive.
        timer = self._cue_timer = self._cue_timer_factory()
        timer.schedule(deadline_perf_time, self._on_cue_deadline)

    def _cancel_cue_pair(self, reason: Optional[CueCancelReason] = None) -> None:
        timer, self._cue_timer = self._cue_timer, None
        if timer is not None:
            timer.cancel()
        if reason is not None and self._cue_policy.is_started:
            evaluation = self._cue_policy.cancel(reason)
            self._observe_safely(f"Tone 2 cancelled: {evaluation.reason.value}")
        else:
            self._cue_policy.reset()
        self._cue_recipe = None
        self._cue_expiry = None

    def _cue_gate_state(self, now_perf_time: float) -> CueGateState:
        """
        Build one gate observation from whatever evidence is available.

        ``observed_at`` is the *oldest* contributing observation, so staleness
        is judged by the least fresh evidence rather than the freshest. With no
        provider at all the observation is "now with nothing known", which the
        policy treats as unknown presence rather than as a clear gate.
        """
        pellet_present = None
        reach_active = False
        observed_at = now_perf_time

        provider = self._pellet_presence_provider
        if provider is not None:
            try:
                observation = provider()
            except Exception as err:
                logger.warning("pellet presence provider failed: %s", err)
                observation = None
            if observation is not None:
                try:
                    pellet_present, presence_at = observation
                    observed_at = min(observed_at, float(presence_at))
                except (TypeError, ValueError):
                    logger.warning(
                        "pellet presence provider returned %r, expected (present, time)",
                        observation,
                    )
                    pellet_present = None

        if self._reach_state is not None:
            reach = self._reach_state.observe(now_perf_time)
            if reach is not None:
                reach_active, reach_at = reach
                observed_at = min(observed_at, reach_at)

        return CueGateState(
            observed_at=observed_at,
            pellet_present=pellet_present,
            reach_active=reach_active,
        )

    def _on_cue_deadline(self, fired_at: float, lateness_seconds: float) -> None:
        with self._lock:
            recipe = self._cue_recipe
            if recipe is None or not self._cue_policy.is_started:
                return  # completed, failed or cancelled while the timer waited

            gate = self._cue_gate_state(fired_at)
            try:
                evaluation = self._cue_policy.evaluate(fired_at, gate)
            except Exception as err:
                logger.exception("cue evaluation failed: %s", err)
                self._observe_safely(f"Tone 2 abandoned: {type(err).__name__}")
                self._cancel_cue_pair()
                return

            if evaluation.decision is CueDecision.FIRE:
                self._fire_cue_tone(recipe, lateness_seconds, evaluation)
                self._cancel_cue_pair()
                return

            if evaluation.decision is CueDecision.SKIP_AND_RESET:
                self._observe_safely(
                    f"Tone 2 skipped and reset: {evaluation.reason.value}"
                )
                self._cancel_cue_pair()
                return

            # WAIT. An unlocked trial may wait out a genuine block, but not
            # forever, so the caller-owned bound applies here.
            if self._cue_expiry is not None and fired_at >= self._cue_expiry:
                self._observe_safely(
                    "Tone 2 abandoned: the gate never cleared within "
                    f"{self._cue_max_wait_seconds:g}s"
                )
                self._cancel_cue_pair(CueCancelReason.HOST_REQUEST)
                return

            delay = evaluation.remaining_seconds or self._cue_poll_seconds
            self._schedule_cue(fired_at + max(delay, 0.0))

    def _fire_cue_tone(self, recipe, lateness_seconds, evaluation) -> None:
        try:
            self._play_tone(recipe.cue_tone_profile, "tone_2")
        except Exception as err:
            logger.exception("Tone 2 send failed: %s", err)
            self._observe_safely(f"Tone 2 send failed: {type(err).__name__}: {err}")
            return
        # The achieved lateness is recorded rather than assumed. It covers the
        # host timer only; the tone transport's own latency is not measured
        # here and has to be checked against recorded NI-DAQ edges.
        self._observe_safely(
            f"Tone 2 fired {lateness_seconds * 1000.0:.3f} ms after its deadline"
            + (f", extended {evaluation.extension_seconds * 1000.0:.3f} ms"
               if evaluation.extension_seconds else "")
        )

    def _observe_safely(self, detail):
        """Record an observation from the timer thread, tolerating a finished trial."""
        try:
            self._observe(detail)
        except Exception:
            logger.debug("cue observation dropped, no current operation: %s", detail)

    def _observe(self, detail):
        operation = self._require_current()
        with operation._lock:
            operation._observations.append(
                TrialActionObservation(operation.state, time.perf_counter(), detail)
            )

    def _cancel_laser_safely(self):
        handle, self._laser_handle = self._laser_handle, None
        if handle is not None:
            self._cancel_laser(handle)
            self._snapshot_laser_action(handle)

    def operation_record(self):
        with self._lock:
            self._snapshot_laser_action()
            operation = self._require_current()
            if self._detector_handle is not None:
                operation.set_action_record(
                    "stim_detector", dict(self._detector_handle),
                )
            return operation.to_record()

    def _snapshot_laser_action(self, handle=None):
        operation = self._operation
        handle = self._laser_handle if handle is None else handle
        if operation is None or handle is None:
            return
        to_record = getattr(handle, "to_record", None)
        if to_record is not None:
            operation.set_action_record("laser", to_record())

    def _await_laser_terminal_for_cycle(self):
        handle = self._laser_handle
        if handle is None:
            return
        state = getattr(getattr(handle, "state", None), "value", None)
        if state in {"armed", "triggered"}:
            profile = self._require_current().recipe.laser_profile
            duration = 0.0
            if profile is not None:
                duration = profile.pulse_duration_ms / 1000.0
                if profile.pulse_count > 1 and profile.frequency_hz:
                    duration += (profile.pulse_count - 1) / profile.frequency_hz
                duration += (
                    profile.baseline_ms
                    + profile.post_stim_ms
                    + profile.pmt_open_lead_ms
                    + profile.pmt_close_lag_ms
                ) / 1000.0
            wait = getattr(handle, "wait", None)
            if wait is not None:
                try:
                    wait(timeout=max(1.0, duration + 1.0))
                except Exception:
                    self._snapshot_laser_action()
        self._snapshot_laser_action()
        state = getattr(getattr(handle, "state", None), "value", None)
        if state not in {None, "completed"}:
            error = getattr(handle, "error", None)
            self._cancel_laser_safely()
            operation = self._require_current()
            if operation.state not in operation.TERMINAL:
                operation.transition(
                    PreparedState.FAILED,
                    "laser operation did not complete: "
                    + (str(error) if error is not None else str(state)),
                )
            raise RuntimeError(
                "Prepared laser operation did not complete successfully: "
                + (str(error) if error is not None else str(state))
            )

    def _cancel_detector_safely(self):
        handle, self._detector_handle = self._detector_handle, None
        if handle is not None:
            self._cancel_detector(handle)

    def _require_operation(self, operation_id):
        operation = self._require_current()
        if operation.recipe.operation_id != str(operation_id):
            raise RuntimeError("Prepared operation ID does not match")
        return operation

    def _require_current(self):
        if self._operation is None:
            raise RuntimeError("No prepared pellet-trial operation exists")
        return self._operation


def _finite_vector(values, name):
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _shortest_cue_interval(row, cue_interval_ms, distribution_record):
    """Return the shortest cue interval this row can produce, if it has one."""

    if not row.cue_tone_profile_id:
        return None
    if distribution_record:
        values = distribution_record.get("values") or ()
        return min(int(value) for value in values) if values else None
    return None if cue_interval_ms is None else int(cue_interval_ms)


def _attempt_component(row, context):
    return (
        context.attempt_id
        if row.retry_assignment is RetryAssignment.RESAMPLE
        else 0
    )


def _stimulus_seed(row, context):
    # The payload is deliberately untagged so previously recorded stimulus
    # seeds and draws remain reproducible.
    payload = json.dumps((
        int(context.session_seed),
        context.protocol_id,
        int(context.protocol_revision),
        int(context.logical_trial_id),
        int(_attempt_component(row, context)),
    ), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _domain_seed(row, context, domain):
    """Return an independent seed stream for one named draw domain."""

    payload = json.dumps((
        str(domain),
        int(context.session_seed),
        context.protocol_id,
        int(context.protocol_revision),
        int(context.logical_trial_id),
        int(_attempt_component(row, context)),
    ), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_draw(seed):
    # Divide the upper 53 bits by 2**53 to obtain an exact reproducible [0, 1).
    return (int(seed) >> 11) / float(1 << 53)
