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

from tools.acquisition.model.trial_protocol_schedule import (
    LaserTriggerRoute,
    PelletLane,
    PelletPositionMode,
    RetryAssignment,
    StimulusAssignment,
    TrialProtocolRow,
)


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
        }


class TrialActionCompiler:
    """Compile a resolved row without touching hardware or application state."""

    def __init__(
        self,
        *,
        tone_profiles: Mapping[str, ToneProfile] = None,
        laser_profiles: Mapping[str, LaserPulseProfile] = None,
        dcs_to_motor: Callable[[Tuple[float, float, float]], Tuple[float, float, float]],
    ):
        self._tone_profiles = dict(tone_profiles or {})
        self._laser_profiles = dict(laser_profiles or {})
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
                position_evidence.update(
                    automatic_status="insufficient_history",
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
        laser = None
        if row.laser_profile_id:
            laser = self._laser_profiles.get(row.laser_profile_id)
            if laser is None:
                raise ValueError(f"Unknown laser profile {row.laser_profile_id!r}")
            if laser.trigger_route is not row.laser_trigger_route:
                raise ValueError("Protocol row and laser profile trigger routes differ")
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
            }


def _finite_vector(values, name):
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _stimulus_seed(row, context):
    attempt_component = (
        context.attempt_id
        if row.retry_assignment is RetryAssignment.RESAMPLE
        else 0
    )
    payload = json.dumps((
        int(context.session_seed),
        context.protocol_id,
        int(context.protocol_revision),
        int(context.logical_trial_id),
        int(attempt_component),
    ), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_draw(seed):
    # Divide the upper 53 bits by 2**53 to obtain an exact reproducible [0, 1).
    return (int(seed) >> 11) / float(1 << 53)
