"""Training plans running on reachAQ pellet trials.

A plan is an ordered list of phases. One phase is current; each completed
pellet trial is recorded against it, then its advance and fallback predicates
are evaluated (advance first). Phase changes are applied by the caller, which
decides whether automatic advance is on. Before each SEND the current phase
gives a ``TrialDirective``: the pellet-target offset, tone and pause its trial
actions call for.
"""

import contextlib
import dataclasses
import enum
import hashlib
import random
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from reachaq.core import ObservableObject
from reachaq.core.logging import get_verbose_logger

from .schema import (
    AllOf,
    AnyOf,
    Condition,
    DeviceSettings,
    JitterPelletTarget,
    Metric,
    MovePelletTarget,
    MoveWhen,
    Pause,
    PhaseDocument,
    PlanDocument,
    Predicate,
    Tone,
    Vector,
    ZERO,
)

logger = get_verbose_logger(__name__)

PROGRESS_VERSION = 1
# Most recent outcomes kept per visit for windowed success rates.
OUTCOME_HISTORY = 500


class TrainingProgressState(str, enum.Enum):
    NotStarted = "not_started"
    InProgress = "in_progress"
    Complete = "complete"


class SessionResult(str, enum.Enum):
    """What a phase's predicates say after a trial."""
    CONTINUE = "continue"
    ADVANCE = "advance"
    FALLBACK = "fallback"


class LoadProgressResult(str, enum.Enum):
    OK = "ok"
    # A phase changed since the progress was saved; loading with
    # force_update resets those phases.
    OUT_OF_DATE = "out_of_date"
    # Unreadable progress; loading with force_update resets all of it.
    UNKNOWN_ERROR = "unknown_error"


class TrialKind(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    NO_REACH = "no_reach"
    PELLET_MISSING = "pellet_missing"
    OTHER = "other"


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """One completed, counted pellet trial."""
    kind: TrialKind
    reaches: int = 0
    successful_reaches: int = 0
    pellets_consumed: int = 0
    pellet_presented: bool = True


@dataclasses.dataclass(frozen=True)
class TrialDirective:
    plan_id: str
    phase_id: str
    target_offset_dcs: Vector = ZERO
    tone: Optional[Tone] = None
    pause_seconds: float = 0.0

    def to_record(self) -> Dict[str, Any]:
        return {
            "planId": self.plan_id,
            "phaseId": self.phase_id,
            "targetOffsetDcs": list(self.target_offset_dcs),
            "tone": None if self.tone is None else {
                "frequencyHz": self.tone.frequency_hz, "durationMs": self.tone.duration_ms},
            "pauseSeconds": self.pause_seconds,
        }


class PhaseProgress:
    """Counters for the current visit to a phase, plus lifetime totals."""

    def __init__(self):
        self.reset_visit()
        self.phase_attempts = 0
        self.lifetime_trials = 0
        self.lifetime_reaches = 0
        self.lifetime_successful_reaches = 0
        self.lifetime_pellets_presented = 0
        self.lifetime_pellets_consumed = 0
        self.lifetime_seconds = 0.0

    def reset_visit(self):
        self.timestamp: Optional[datetime] = None
        self.session_count = 0  # counted trials in this visit; the views' name
        self.successes = 0
        self.failures = 0
        self.no_reach = 0
        self.total_reaches = 0
        self.successful_reaches = 0
        self.pellets_presented = 0
        self.pellets_consumed = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.visit_seconds = 0.0
        self.target_offset: Vector = ZERO
        self.outcomes: Deque[bool] = deque(maxlen=OUTCOME_HISTORY)
        self._running_since: Optional[float] = None

    @property
    def time_in_training(self) -> float:
        """Seconds of trial time in this visit (shown in the progress panel)."""
        running = 0.0 if self._running_since is None else time.monotonic() - self._running_since
        return self.visit_seconds + running

    def progress_resumed(self):
        if self._running_since is None:
            self._running_since = time.monotonic()
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def progress_paused(self):
        if self._running_since is not None:
            elapsed = time.monotonic() - self._running_since
            self.visit_seconds += elapsed
            self.lifetime_seconds += elapsed
            self._running_since = None

    def record(self, trial: TrialResult):
        self.progress_paused()
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self.session_count += 1
        self.lifetime_trials += 1
        success = trial.kind is TrialKind.SUCCESS
        self.successes += success
        self.failures += trial.kind is TrialKind.FAILURE
        self.no_reach += trial.kind is TrialKind.NO_REACH
        if success:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        elif trial.kind in (TrialKind.FAILURE, TrialKind.NO_REACH):
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        if trial.kind is not TrialKind.PELLET_MISSING and trial.kind is not TrialKind.OTHER:
            self.outcomes.append(success)
        self.total_reaches += trial.reaches
        self.successful_reaches += trial.successful_reaches
        self.pellets_consumed += trial.pellets_consumed
        self.pellets_presented += bool(trial.pellet_presented)
        self.lifetime_reaches += trial.reaches
        self.lifetime_successful_reaches += trial.successful_reaches
        self.lifetime_pellets_consumed += trial.pellets_consumed
        self.lifetime_pellets_presented += bool(trial.pellet_presented)

    def metric(self, condition: Condition) -> float:
        m = condition.metric
        if m is Metric.SUCCESS_RATE:
            history = list(self.outcomes)
            if condition.window is not None:
                if len(history) < condition.window:
                    return float("nan")  # not enough trials yet: never true
                history = history[-condition.window:]
            return float("nan") if not history else sum(history) / len(history)
        return float({
            Metric.TRIALS: self.session_count,
            Metric.SUCCESSES: self.successes,
            Metric.FAILURES: self.failures,
            Metric.NO_REACH: self.no_reach,
            Metric.REACHES: self.total_reaches,
            Metric.SUCCESSFUL_REACHES: self.successful_reaches,
            Metric.PELLETS_PRESENTED: self.pellets_presented,
            Metric.PELLETS_CONSUMED: self.pellets_consumed,
            Metric.CONSECUTIVE_SUCCESSES: self.consecutive_successes,
            Metric.CONSECUTIVE_FAILURES: self.consecutive_failures,
            Metric.TIME_IN_PHASE: self.time_in_training,
        }[m])

    def to_record(self, fingerprint: str) -> Dict[str, Any]:
        return {
            "fingerprint": fingerprint,
            "visit": {
                "startedAt": None if self.timestamp is None else self.timestamp.isoformat(),
                "trials": self.session_count,
                "successes": self.successes,
                "failures": self.failures,
                "noReach": self.no_reach,
                "reaches": self.total_reaches,
                "successfulReaches": self.successful_reaches,
                "pelletsPresented": self.pellets_presented,
                "pelletsConsumed": self.pellets_consumed,
                "consecutiveSuccesses": self.consecutive_successes,
                "consecutiveFailures": self.consecutive_failures,
                "seconds": self.time_in_training,
                "targetOffsetDcs": list(self.target_offset),
                "outcomes": "".join("1" if o else "0" for o in self.outcomes),
            },
            "attempts": self.phase_attempts,
            "lifetime": {
                "trials": self.lifetime_trials,
                "reaches": self.lifetime_reaches,
                "successfulReaches": self.lifetime_successful_reaches,
                "pelletsPresented": self.lifetime_pellets_presented,
                "pelletsConsumed": self.lifetime_pellets_consumed,
                "seconds": self.lifetime_seconds + (
                    0.0 if self._running_since is None else time.monotonic() - self._running_since),
            },
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "PhaseProgress":
        p = cls()
        visit, lifetime = record["visit"], record["lifetime"]
        started = visit["startedAt"]
        p.timestamp = None if started is None else datetime.fromisoformat(started)
        p.session_count = int(visit["trials"])
        p.successes = int(visit["successes"])
        p.failures = int(visit["failures"])
        p.no_reach = int(visit["noReach"])
        p.total_reaches = int(visit["reaches"])
        p.successful_reaches = int(visit["successfulReaches"])
        p.pellets_presented = int(visit["pelletsPresented"])
        p.pellets_consumed = int(visit["pelletsConsumed"])
        p.consecutive_successes = int(visit["consecutiveSuccesses"])
        p.consecutive_failures = int(visit["consecutiveFailures"])
        p.visit_seconds = float(visit["seconds"])
        offset = visit["targetOffsetDcs"]
        if len(offset) != 3:
            raise ValueError("targetOffsetDcs must have three values")
        p.target_offset = tuple(float(v) for v in offset)  # type: ignore[assignment]
        outcomes = visit["outcomes"]
        if set(outcomes) - {"0", "1"}:
            raise ValueError("outcomes must be a 0/1 string")
        p.outcomes.extend(c == "1" for c in outcomes)
        p.phase_attempts = int(record["attempts"])
        p.lifetime_trials = int(lifetime["trials"])
        p.lifetime_reaches = int(lifetime["reaches"])
        p.lifetime_successful_reaches = int(lifetime["successfulReaches"])
        p.lifetime_pellets_presented = int(lifetime["pelletsPresented"])
        p.lifetime_pellets_consumed = int(lifetime["pelletsConsumed"])
        p.lifetime_seconds = float(lifetime["seconds"])
        return p


def evaluate(predicate: Optional[Predicate], progress: PhaseProgress) -> bool:
    if predicate is None:
        return False
    if isinstance(predicate, AllOf):
        return all(evaluate(t, progress) for t in predicate.terms)
    if isinstance(predicate, AnyOf):
        return any(evaluate(t, progress) for t in predicate.terms)
    value = progress.metric(predicate)
    return value == value and predicate.op.holds(value, predicate.value)  # NaN never holds


class TrainingPhase(ObservableObject):

    def __init__(self, document: PhaseDocument):
        super().__init__()
        self.document = document
        self.progress = PhaseProgress()

    def __repr__(self):
        return f"TrainingPhase({self.phase_id!r})"

    phase_id = property(lambda self: self.document.phase_id)
    name = property(lambda self: self.document.name)
    description = property(lambda self: self.document.description)
    device = property(lambda self: self.document.device)
    advance_predicate = property(lambda self: self.document.advance)
    fallback_predicate = property(lambda self: self.document.fallback)
    session_actions = property(lambda self: self.document.trial_actions)

    is_pellet_delivery_enabled = property(lambda self: self.document.device.pellet_delivery is not False)
    is_pellet_cover_enabled = property(lambda self: bool(self.document.device.pellet_cover))
    is_pellet_shift_enabled = property(lambda self: bool(self.document.device.pellet_shift))

    def _send(self, axis: int) -> Optional[float]:
        offset = self.document.device.send_offset
        return None if offset is None else offset[axis]

    send_x = property(lambda self: self._send(0))
    send_y = property(lambda self: self._send(1))
    send_z = property(lambda self: self._send(2))

    def record_trial(self, trial: TrialResult) -> SessionResult:
        progress = self.progress
        progress.record(trial)
        for action in self.document.trial_actions:
            if isinstance(action, MovePelletTarget) and _move_applies(action.when, trial.kind):
                progress.target_offset = _clamped_move(progress.target_offset, action)
        self.property_changed("progress", progress, None)
        if evaluate(self.document.advance, progress):
            return SessionResult.ADVANCE
        if evaluate(self.document.fallback, progress):
            return SessionResult.FALLBACK
        return SessionResult.CONTINUE

    def directive(self, plan_id: str, draw_key: str) -> TrialDirective:
        offset = self.document.device.send_offset or ZERO
        offset = _add(offset, self.progress.target_offset)
        tone, pause = None, 0.0
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"{plan_id}|{self.phase_id}|{draw_key}".encode("utf-8")).digest()[:8], "big"))
        for action in self.document.trial_actions:
            if isinstance(action, JitterPelletTarget):
                offset = _add(offset, tuple(rng.uniform(-h, h) if h else 0.0 for h in action.half_width))
            elif isinstance(action, Tone):
                tone = action
            elif isinstance(action, Pause):
                pause = action.min_seconds if action.max_seconds == action.min_seconds else rng.uniform(
                    action.min_seconds, action.max_seconds)
        return TrialDirective(plan_id, self.phase_id, tuple(round(v, 4) for v in offset), tone, round(pause, 3))


def _move_applies(when: MoveWhen, kind: TrialKind) -> bool:
    if when is MoveWhen.ALWAYS:
        return kind in (TrialKind.SUCCESS, TrialKind.FAILURE, TrialKind.NO_REACH)
    if when is MoveWhen.SUCCESS:
        return kind is TrialKind.SUCCESS
    return kind in (TrialKind.FAILURE, TrialKind.NO_REACH)


def _clamped_move(current: Vector, action: MovePelletTarget) -> Vector:
    moved = []
    for value, delta, limit in zip(current, action.delta, action.limits):
        value += delta
        if limit is not None:
            value = min(max(value, limit[0]), limit[1])
        moved.append(round(value, 6))
    return tuple(moved)  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


class TrainingPlan(ObservableObject):
    """A plan document with one animal's progress through it.

    Events: ``property_changed(name, value, old)`` (``current_phase``,
    ``progress_state``, ``is_automatic``) and ``progress_updated()``.
    """

    def __init__(self, document: PlanDocument):
        super().__init__(("progress_updated",))
        self.document = document
        self.phases: List[TrainingPhase] = [TrainingPhase(p) for p in document.phases]
        self._current_index = 0
        self._progress_state = TrainingProgressState.NotStarted
        self._is_automatic = False
        self.training_start_timestamp: Optional[datetime] = None
        self._algorithm = None
        self._pellet_device = None
        self._saved_device: Dict[str, bool] = {}
        self._applied_device: Dict[str, bool] = {}

    def __repr__(self):
        return f"TrainingPlan({self.plan_id!r})"

    plan_id = property(lambda self: self.document.plan_id)
    name = property(lambda self: self.document.name)
    description = property(lambda self: self.document.description)

    @property
    def current_phase(self) -> Optional[TrainingPhase]:
        return self.phases[self._current_index] if self.phases else None

    @property
    def can_advance(self) -> bool:
        return self._current_index < len(self.phases) - 1

    @property
    def can_fallback(self) -> bool:
        return self._current_index > 0

    @property
    def is_automatic(self) -> bool:
        return self._is_automatic

    @is_automatic.setter
    def is_automatic(self, value: bool):
        old, self._is_automatic = self._is_automatic, bool(value)
        self._on_property_changed("is_automatic", self._is_automatic, old)

    @property
    def progress_state(self) -> TrainingProgressState:
        return self._progress_state

    @progress_state.setter
    def progress_state(self, value: TrainingProgressState):
        old, self._progress_state = self._progress_state, TrainingProgressState(value)
        self._on_property_changed("progress_state", self._progress_state, old)

    # Totals across every phase (lifetime counters).
    total_session_count = property(lambda self: sum(p.progress.lifetime_trials for p in self.phases))
    total_reaches = property(lambda self: sum(p.progress.lifetime_reaches for p in self.phases))
    total_successful_reaches = property(
        lambda self: sum(p.progress.lifetime_successful_reaches for p in self.phases))
    total_pellets_presented = property(lambda self: sum(p.progress.lifetime_pellets_presented for p in self.phases))
    total_pellets_consumed = property(lambda self: sum(p.progress.lifetime_pellets_consumed for p in self.phases))

    @property
    def total_time(self) -> float:
        """Seconds of trial time across every phase."""
        return sum(p.progress.lifetime_seconds for p in self.phases) + sum(
            p.progress.time_in_training - p.progress.visit_seconds for p in self.phases)

    # Attachment to the running behaviour.

    def attach(self, algorithm, pellet_device, _unused=None) -> None:
        """Hold the algorithm and pellet device and apply the current phase's
        device settings; ``detach()`` restores what they were."""
        self._algorithm = algorithm
        self._pellet_device = pellet_device
        self._saved_device = {}
        self._applied_device = {}
        self._apply_device(self.current_phase)

    def detach(self) -> None:
        if self.current_phase is not None:
            self.current_phase.progress.progress_paused()
        self._restore_device(set(self._applied_device))
        self._algorithm = None
        self._pellet_device = None

    @contextlib.contextmanager
    def device_settings_restored(self):
        """The configuration's own device values for the duration, e.g. while
        the configuration is saved, then the phase's values again."""
        applied = dict(self._applied_device)
        algorithm = self._algorithm
        if algorithm is None or not applied:
            yield
            return
        saved = dict(self._saved_device)
        for attribute in applied:
            setattr(algorithm, attribute, saved[attribute])
        try:
            yield
        finally:
            for attribute, value in applied.items():
                setattr(algorithm, attribute, value)

    @property
    def pellet_device(self):
        return self._pellet_device

    def resume(self) -> None:
        if self._progress_state is TrainingProgressState.NotStarted:
            self.progress_state = TrainingProgressState.InProgress
        if self.training_start_timestamp is None:
            self.training_start_timestamp = datetime.now()
        phase = self.current_phase
        if phase is not None and phase.progress.phase_attempts == 0:
            phase.progress.phase_attempts = 1

    def _apply_device(self, phase: Optional[TrainingPhase]) -> None:
        algorithm = self._algorithm
        if algorithm is None or phase is None:
            return
        device: DeviceSettings = phase.document.device
        wanted = {
            attribute: value
            for attribute, value in (
                ("pellet_delivery_enabled", device.pellet_delivery),
                ("pellet_cover_enabled", device.pellet_cover),
                ("intersession_pellet_shift_enabled", device.pellet_shift),
            )
            if value is not None
        }
        # A setting the previous phase set and this one leaves out goes back
        # to the configuration's value.
        self._restore_device(set(self._applied_device) - set(wanted))
        for attribute, value in wanted.items():
            self._saved_device.setdefault(attribute, getattr(algorithm, attribute))
            setattr(algorithm, attribute, value)
            self._applied_device[attribute] = value

    def _restore_device(self, attributes) -> None:
        algorithm = self._algorithm
        for attribute in attributes:
            applied = self._applied_device.pop(attribute)
            saved = self._saved_device.pop(attribute)
            # An operator change made since the plan set it wins.
            if algorithm is not None and getattr(algorithm, attribute) == applied:
                setattr(algorithm, attribute, saved)

    # Trials.

    def record_trial(self, trial: TrialResult) -> SessionResult:
        phase = self.current_phase
        if phase is None:
            return SessionResult.CONTINUE
        self.resume()
        result = phase.record_trial(trial)
        logger.info("Training plan %s phase %s: %s trials, result %s",
                    self.plan_id, phase.phase_id, phase.progress.session_count, result.value)
        return result

    def next_trial_directive(self, draw_key: str) -> Optional[TrialDirective]:
        phase = self.current_phase
        return None if phase is None else phase.directive(self.plan_id, draw_key)

    def advance(self) -> None:
        if not self.can_advance:
            raise RuntimeError(f"{self.plan_id}: already at the last phase")
        self._move_to(self._current_index + 1)

    def fallback(self) -> None:
        if not self.can_fallback:
            raise RuntimeError(f"{self.plan_id}: already at the first phase")
        self._move_to(self._current_index - 1)

    def _move_to(self, index: int) -> None:
        old = self.current_phase
        if old is not None:
            old.progress.progress_paused()
        self._current_index = index
        phase = self.current_phase
        phase.progress.reset_visit()
        phase.progress.phase_attempts += 1
        if self._progress_state is TrainingProgressState.Complete:
            self.progress_state = TrainingProgressState.InProgress
        self._apply_device(phase)
        self.property_changed("current_phase", phase, old)
        self._on_progress_updated()

    def _on_progress_updated(self) -> None:
        self.progress_updated()

    # Persistence in the animal file's training.protocolProgress list.

    def serialize_progress(self) -> Dict[str, Any]:
        phase = self.current_phase
        return {
            "plan_id": self.plan_id,
            "progressVersion": PROGRESS_VERSION,
            "state": self._progress_state.value,
            "currentPhaseId": None if phase is None else phase.phase_id,
            "trainingStartedAt": (
                None if self.training_start_timestamp is None else self.training_start_timestamp.isoformat()),
            "phases": {p.phase_id: p.progress.to_record(p.document.fingerprint) for p in self.phases},
        }

    def deserialize_progress(self, record: Dict[str, Any], *, force_update: bool = False) -> LoadProgressResult:
        try:
            parsed = self._parse_progress(record)
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            logger.warning("%s: unreadable progress (%s)", self.plan_id, err)
            if not force_update:
                return LoadProgressResult.UNKNOWN_ERROR
            self._reset_progress()
            return LoadProgressResult.OK
        state, current_id, started, phases = parsed
        stale = [p.phase_id for p in self.phases
                 if p.phase_id in phases and phases[p.phase_id][0] != p.document.fingerprint]
        removed_current = current_id is not None and current_id not in {p.phase_id for p in self.phases}
        if (stale or removed_current) and not force_update:
            return LoadProgressResult.OUT_OF_DATE
        for p in self.phases:
            saved = phases.get(p.phase_id)
            p.progress = PhaseProgress() if saved is None or p.phase_id in stale else saved[1]
        ids = [p.phase_id for p in self.phases]
        self._current_index = 0 if removed_current or current_id is None else ids.index(current_id)
        if self.current_phase is not None and self.current_phase.phase_id in stale:
            state = TrainingProgressState.InProgress if state is TrainingProgressState.Complete else state
        self._progress_state = state
        self.training_start_timestamp = started
        return LoadProgressResult.OK

    def _parse_progress(self, record: Dict[str, Any]):
        if record.get("plan_id") != self.plan_id:
            raise ValueError(f"progress is for plan {record.get('plan_id')!r}")
        if record.get("progressVersion") != PROGRESS_VERSION:
            raise ValueError(f"unsupported progressVersion {record.get('progressVersion')!r}")
        state = TrainingProgressState(record["state"])
        current_id = record["currentPhaseId"]
        started = record["trainingStartedAt"]
        phases: Dict[str, Tuple[str, PhaseProgress]] = {
            phase_id: (str(value["fingerprint"]), PhaseProgress.from_record(value))
            for phase_id, value in record["phases"].items()
        }
        return state, current_id, None if started is None else datetime.fromisoformat(started), phases

    def _reset_progress(self) -> None:
        for p in self.phases:
            p.progress = PhaseProgress()
        self._current_index = 0
        self._progress_state = TrainingProgressState.NotStarted
        self.training_start_timestamp = None
