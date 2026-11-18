"""reachAQ training-plan documents (schema version 1).

A plan is one JSON file in the configuration's ``training_plans/`` directory::

    {
      "schemaVersion": 1,
      "planId": "reach-shaping",
      "name": "Reach shaping",
      "description": "...",
      "phases": [
        {
          "phaseId": "close-pellet",
          "name": "Close pellet",
          "description": "...",
          "device": {"pelletDelivery": true, "pelletCover": false, "pelletShift": false,
                     "sendOffset": {"x": 0.0, "y": -1.0, "z": 0.0}},
          "advance": {"all": [{"metric": "trials", "op": ">=", "value": 20},
                              {"metric": "successRate", "op": ">=", "value": 0.6, "window": 20}]},
          "fallback": {"metric": "consecutiveFailures", "op": ">=", "value": 10},
          "trialActions": [
            {"action": "movePelletTarget", "when": "success", "dy": 0.25,
             "limits": {"y": [-2.0, 2.0]}},
            {"action": "jitterPelletTarget", "x": 0.2, "y": 0.0, "z": 0.0},
            {"action": "tone", "frequencyHz": 4000, "durationMs": 200},
            {"action": "pause", "minSeconds": 3.0, "maxSeconds": 6.0}
          ]
        }
      ]
    }

Every ``device`` key is optional; an absent key leaves that setting as the
configuration has it. Offsets are DCS millimetres relative to the subject's
pellet target. Metrics are counted over the current visit to the phase.
"""

import dataclasses
import enum
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = 1

Vector = Tuple[float, float, float]
ZERO: Vector = (0.0, 0.0, 0.0)


class PlanFormatError(ValueError):
    """A plan document that does not follow the schema."""


class Metric(str, enum.Enum):
    TRIALS = "trials"
    SUCCESSES = "successes"
    FAILURES = "failures"
    NO_REACH = "noReach"
    REACHES = "reaches"
    SUCCESSFUL_REACHES = "successfulReaches"
    PELLETS_PRESENTED = "pelletsPresented"
    PELLETS_CONSUMED = "pelletsConsumed"
    SUCCESS_RATE = "successRate"
    CONSECUTIVE_SUCCESSES = "consecutiveSuccesses"
    CONSECUTIVE_FAILURES = "consecutiveFailures"
    TIME_IN_PHASE = "timeInPhase"


class Comparison(str, enum.Enum):
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"
    EQ = "=="

    def holds(self, left: float, right: float) -> bool:
        return {
            Comparison.GE: left >= right,
            Comparison.GT: left > right,
            Comparison.LE: left <= right,
            Comparison.LT: left < right,
            Comparison.EQ: left == right,
        }[self]


@dataclasses.dataclass(frozen=True)
class Condition:
    metric: Metric
    op: Comparison
    value: float
    # successRate only: the last N counted trials instead of the whole visit.
    window: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class AllOf:
    terms: Tuple["Predicate", ...]


@dataclasses.dataclass(frozen=True)
class AnyOf:
    terms: Tuple["Predicate", ...]


Predicate = Union[Condition, AllOf, AnyOf]


class MoveWhen(str, enum.Enum):
    ALWAYS = "always"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclasses.dataclass(frozen=True)
class MovePelletTarget:
    delta: Vector
    when: MoveWhen = MoveWhen.ALWAYS
    # Per axis (min, max) on the accumulated move offset; None is unbounded.
    limits: Tuple[Optional[Tuple[float, float]], ...] = (None, None, None)


@dataclasses.dataclass(frozen=True)
class JitterPelletTarget:
    half_width: Vector


@dataclasses.dataclass(frozen=True)
class Tone:
    frequency_hz: int
    duration_ms: int


@dataclasses.dataclass(frozen=True)
class Pause:
    min_seconds: float
    max_seconds: float


TrialAction = Union[MovePelletTarget, JitterPelletTarget, Tone, Pause]


@dataclasses.dataclass(frozen=True)
class DeviceSettings:
    pellet_delivery: Optional[bool] = None
    pellet_cover: Optional[bool] = None
    pellet_shift: Optional[bool] = None
    send_offset: Optional[Vector] = None


@dataclasses.dataclass(frozen=True)
class PhaseDocument:
    phase_id: str
    name: str
    description: str
    device: DeviceSettings
    advance: Optional[Predicate]
    fallback: Optional[Predicate]
    trial_actions: Tuple[TrialAction, ...]
    # Hash of the phase's rules and actions: saved progress for a phase whose
    # behaviour has changed since is out of date.
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class PlanDocument:
    plan_id: str
    name: str
    description: str
    phases: Tuple[PhaseDocument, ...]


def parse_plan(document: Mapping[str, Any], *, source: str = "plan") -> PlanDocument:
    where = source
    _require_mapping(document, where)
    _only_keys(document, {"schemaVersion", "planId", "name", "description", "phases"}, where)
    version = document.get("schemaVersion")
    if version != SCHEMA_VERSION:
        raise PlanFormatError(f"{where}: schemaVersion must be {SCHEMA_VERSION}, got {version!r}")
    plan_id = _identifier(document.get("planId"), f"{where}.planId")
    phases_raw = document.get("phases")
    if not isinstance(phases_raw, list) or not phases_raw:
        raise PlanFormatError(f"{where}.phases must be a non-empty list")
    phases = tuple(_parse_phase(p, f"{where}.phases[{i}]") for i, p in enumerate(phases_raw))
    ids = [p.phase_id for p in phases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PlanFormatError(f"{where}: duplicated phaseId {', '.join(duplicates)}")
    return PlanDocument(
        plan_id=plan_id,
        name=_text(document.get("name", plan_id), f"{where}.name"),
        description=_text(document.get("description", ""), f"{where}.description"),
        phases=phases,
    )


def _parse_phase(raw: Any, where: str) -> PhaseDocument:
    _require_mapping(raw, where)
    _only_keys(raw, {"phaseId", "name", "description", "device", "advance", "fallback", "trialActions"}, where)
    phase_id = _identifier(raw.get("phaseId"), f"{where}.phaseId")
    actions_raw = raw.get("trialActions", [])
    if not isinstance(actions_raw, list):
        raise PlanFormatError(f"{where}.trialActions must be a list")
    actions = tuple(_parse_action(a, f"{where}.trialActions[{i}]") for i, a in enumerate(actions_raw))
    for kind in (Tone, Pause, JitterPelletTarget):
        if sum(isinstance(a, kind) for a in actions) > 1:
            raise PlanFormatError(f"{where}.trialActions: at most one {_ACTION_NAMES[kind]}")
    return PhaseDocument(
        phase_id=phase_id,
        name=_text(raw.get("name", phase_id), f"{where}.name"),
        description=_text(raw.get("description", ""), f"{where}.description"),
        device=_parse_device(raw.get("device", {}), f"{where}.device"),
        advance=_parse_optional_predicate(raw.get("advance"), f"{where}.advance"),
        fallback=_parse_optional_predicate(raw.get("fallback"), f"{where}.fallback"),
        trial_actions=actions,
        # Renaming or rewording a phase does not make saved progress out of date.
        fingerprint=hashlib.sha256(json.dumps(
            {k: v for k, v in raw.items() if k not in ("name", "description")},
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16],
    )


def _parse_device(raw: Any, where: str) -> DeviceSettings:
    _require_mapping(raw, where)
    _only_keys(raw, {"pelletDelivery", "pelletCover", "pelletShift", "sendOffset"}, where)
    offset = raw.get("sendOffset")
    return DeviceSettings(
        pellet_delivery=_optional_bool(raw.get("pelletDelivery"), f"{where}.pelletDelivery"),
        pellet_cover=_optional_bool(raw.get("pelletCover"), f"{where}.pelletCover"),
        pellet_shift=_optional_bool(raw.get("pelletShift"), f"{where}.pelletShift"),
        send_offset=None if offset is None else _vector(offset, f"{where}.sendOffset"),
    )


def _parse_optional_predicate(raw: Any, where: str) -> Optional[Predicate]:
    return None if raw is None else _parse_predicate(raw, where)


def _parse_predicate(raw: Any, where: str) -> Predicate:
    _require_mapping(raw, where)
    for key, cls in (("all", AllOf), ("any", AnyOf)):
        if key in raw:
            _only_keys(raw, {key}, where)
            terms = raw[key]
            if not isinstance(terms, list) or not terms:
                raise PlanFormatError(f"{where}.{key} must be a non-empty list")
            return cls(tuple(_parse_predicate(t, f"{where}.{key}[{i}]") for i, t in enumerate(terms)))
    _only_keys(raw, {"metric", "op", "value", "window"}, where)
    metric = _enum(Metric, raw.get("metric"), f"{where}.metric")
    op = _enum(Comparison, raw.get("op"), f"{where}.op")
    value = _number(raw.get("value"), f"{where}.value")
    window = raw.get("window")
    if window is not None:
        if metric is not Metric.SUCCESS_RATE:
            raise PlanFormatError(f"{where}.window applies only to successRate")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise PlanFormatError(f"{where}.window must be a positive integer")
    return Condition(metric, op, value, window)


_ACTION_NAMES = {
    MovePelletTarget: "movePelletTarget",
    JitterPelletTarget: "jitterPelletTarget",
    Tone: "tone",
    Pause: "pause",
}


def _parse_action(raw: Any, where: str) -> TrialAction:
    _require_mapping(raw, where)
    name = raw.get("action")
    if name == "movePelletTarget":
        _only_keys(raw, {"action", "when", "dx", "dy", "dz", "limits"}, where)
        delta = tuple(_number(raw.get(k, 0.0), f"{where}.{k}") for k in ("dx", "dy", "dz"))
        limits_raw = raw.get("limits", {})
        _require_mapping(limits_raw, f"{where}.limits")
        _only_keys(limits_raw, {"x", "y", "z"}, f"{where}.limits")
        limits = []
        for axis in "xyz":
            pair = limits_raw.get(axis)
            if pair is None:
                limits.append(None)
                continue
            if not isinstance(pair, list) or len(pair) != 2:
                raise PlanFormatError(f"{where}.limits.{axis} must be [min, max]")
            low, high = (_number(v, f"{where}.limits.{axis}") for v in pair)
            if low > high:
                raise PlanFormatError(f"{where}.limits.{axis}: min is above max")
            limits.append((low, high))
        return MovePelletTarget(
            delta=delta,  # type: ignore[arg-type]
            when=_enum(MoveWhen, raw.get("when", "always"), f"{where}.when"),
            limits=tuple(limits),
        )
    if name == "jitterPelletTarget":
        _only_keys(raw, {"action", "x", "y", "z"}, where)
        half = tuple(_number(raw.get(k, 0.0), f"{where}.{k}") for k in "xyz")
        if any(v < 0 for v in half):
            raise PlanFormatError(f"{where}: jitter half-widths must not be negative")
        return JitterPelletTarget(half)  # type: ignore[arg-type]
    if name == "tone":
        _only_keys(raw, {"action", "frequencyHz", "durationMs"}, where)
        frequency = raw.get("frequencyHz")
        duration = raw.get("durationMs")
        for key, value, low, high in (("frequencyHz", frequency, 100, 20000), ("durationMs", duration, 1, 10000)):
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise PlanFormatError(f"{where}.{key} must be an integer in [{low}, {high}]")
        return Tone(frequency, duration)
    if name == "pause":
        _only_keys(raw, {"action", "seconds", "minSeconds", "maxSeconds"}, where)
        if "seconds" in raw:
            if "minSeconds" in raw or "maxSeconds" in raw:
                raise PlanFormatError(f"{where}: give seconds or minSeconds/maxSeconds, not both")
            low = high = _number(raw["seconds"], f"{where}.seconds")
        else:
            low = _number(raw.get("minSeconds"), f"{where}.minSeconds")
            high = _number(raw.get("maxSeconds"), f"{where}.maxSeconds")
        if not 0.0 <= low <= high <= 600.0:
            raise PlanFormatError(f"{where}: pause must satisfy 0 <= min <= max <= 600 seconds")
        return Pause(low, high)
    raise PlanFormatError(f"{where}.action must be one of {', '.join(_ACTION_NAMES.values())}, got {name!r}")


def _require_mapping(value: Any, where: str) -> None:
    if not isinstance(value, Mapping):
        raise PlanFormatError(f"{where} must be an object")


def _only_keys(value: Mapping[str, Any], allowed: set, where: str) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        raise PlanFormatError(f"{where}: unknown key(s) {', '.join(extra)}")


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PlanFormatError(f"{where} must be a non-empty string without surrounding spaces")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise PlanFormatError(f"{where} must be a string")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlanFormatError(f"{where} must be a finite number")
    return float(value)


def _optional_bool(value: Any, where: str) -> Optional[bool]:
    if value is not None and not isinstance(value, bool):
        raise PlanFormatError(f"{where} must be true or false")
    return value


def _vector(value: Any, where: str) -> Vector:
    _require_mapping(value, where)
    _only_keys(value, {"x", "y", "z"}, where)
    return tuple(_number(value.get(k, 0.0), f"{where}.{k}") for k in "xyz")  # type: ignore[return-value]


def _enum(cls, value: Any, where: str):
    try:
        return cls(value)
    except ValueError:
        raise PlanFormatError(
            f"{where} must be one of {', '.join(m.value for m in cls)}, got {value!r}") from None


def predicate_terms(predicate: Optional[Predicate]) -> Sequence[Condition]:
    """Every condition in a predicate tree, for display."""
    if predicate is None:
        return ()
    if isinstance(predicate, Condition):
        return (predicate,)
    return tuple(c for t in predicate.terms for c in predicate_terms(t))


def describe_predicate(predicate: Optional[Predicate]) -> str:
    if predicate is None:
        return "never"
    if isinstance(predicate, Condition):
        window = f" (last {predicate.window})" if predicate.window else ""
        return f"{predicate.metric.value}{window} {predicate.op.value} {predicate.value:g}"
    joiner = " and " if isinstance(predicate, AllOf) else " or "
    return "(" + joiner.join(describe_predicate(t) for t in predicate.terms) + ")"


def phase_record(phase: PhaseDocument) -> Dict[str, Any]:
    """Short JSON-safe summary of a phase, for logs and session evidence."""
    return {"phaseId": phase.phase_id, "fingerprint": phase.fingerprint}
