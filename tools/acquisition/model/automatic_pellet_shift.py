"""Generation-owned absolute pellet targets from eligible reach history."""

from __future__ import annotations

import dataclasses
import enum
import math
import statistics
import threading
from collections import deque
from typing import FrozenSet, Optional, Tuple


class ShiftWindowMethod(str, enum.Enum):
    LEGACY_BATCH = "legacy_batch"
    SLIDING_LAST_X = "sliding_last_x"


@dataclasses.dataclass(frozen=True)
class AutomaticShiftPolicy:
    policy_id: str = "default"
    window_method: ShiftWindowMethod = ShiftWindowMethod.LEGACY_BATCH
    window_size: int = 15
    eligible_outcomes: FrozenSet[str] = frozenset({"failure"})
    target_reach_offset_dcs: Tuple[float, float, float] = (1.5, -3.0, 1.0)
    deadbands_mm: Tuple[float, float, float] = (0.5, 1.0, 0.5)
    maximum_update_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    maximum_absolute_mm: Tuple[float, float, float] = (5.0, 5.0, 5.0)
    minimum_y_dcs: Optional[float] = None
    apply_automatically: bool = True

    def __post_init__(self):
        object.__setattr__(self, "window_method", ShiftWindowMethod(self.window_method))
        if not self.policy_id or not 1 <= int(self.window_size) <= 10_000:
            raise ValueError("Automatic shift policy requires an ID and window size 1..10000")
        for name in (
            "target_reach_offset_dcs",
            "deadbands_mm",
            "maximum_update_mm",
            "maximum_absolute_mm",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
            if name != "target_reach_offset_dcs" and any(value < 0 for value in values):
                raise ValueError(f"{name} cannot contain negative limits")
            object.__setattr__(self, name, values)
        if self.minimum_y_dcs is not None and not math.isfinite(self.minimum_y_dcs):
            raise ValueError("minimum_y_dcs must be finite")


@dataclasses.dataclass(frozen=True)
class ReachPositionObservation:
    reach_id: str
    outcome: str
    closest_offset_dcs: Tuple[float, float, float]

    def __post_init__(self):
        values = tuple(float(value) for value in self.closest_offset_dcs)
        if not self.reach_id or len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("Reach observation requires an ID and three finite DCS values")
        object.__setattr__(self, "closest_offset_dcs", values)


@dataclasses.dataclass(frozen=True)
class AutomaticShiftGeneration:
    generation: int
    policy_id: str
    window_method: ShiftWindowMethod
    reach_ids: Tuple[str, ...]
    reduced_reach_offset_dcs: Tuple[float, float, float]
    recommended_shift_dcs: Tuple[float, float, float]
    baseline_dcs: Tuple[float, float, float]
    prior_target_dcs: Tuple[float, float, float]
    resolved_target_dcs: Tuple[float, float, float]
    apply_automatically: bool

    def to_record(self):
        result = dataclasses.asdict(self)
        result["window_method"] = self.window_method.value
        return result


class AutomaticPelletShiftController:
    """Match legacy batches or calculate a true last-X sliding recommendation."""

    def __init__(self, policy: AutomaticShiftPolicy):
        self._lock = threading.RLock()
        self._policy = policy
        self._buffer = deque(maxlen=(
            None
            if policy.window_method is ShiftWindowMethod.LEGACY_BATCH
            else policy.window_size
        ))
        self._seen_reach_ids = set()
        self._generation = 0
        self._latest: Optional[AutomaticShiftGeneration] = None
        self._accepted_generation = 0
        self._accepted_target: Optional[Tuple[float, float, float]] = None

    @property
    def policy(self):
        return self._policy

    @property
    def latest(self):
        with self._lock:
            return self._latest

    @property
    def accepted_target(self):
        with self._lock:
            return self._accepted_target

    def reset(self):
        with self._lock:
            self._buffer.clear()
            self._seen_reach_ids.clear()
            self._generation = 0
            self._latest = None
            self._accepted_generation = 0
            self._accepted_target = None

    def add(
        self,
        observation: ReachPositionObservation,
        *,
        baseline_dcs: Tuple[float, float, float],
    ) -> Optional[AutomaticShiftGeneration]:
        with self._lock:
            if observation.reach_id in self._seen_reach_ids:
                return None
            self._seen_reach_ids.add(observation.reach_id)
            if observation.outcome not in self._policy.eligible_outcomes:
                return None
            self._buffer.append(observation)
            if len(self._buffer) < self._policy.window_size:
                return None
            window = tuple(self._buffer)
            if self._policy.window_method is ShiftWindowMethod.LEGACY_BATCH:
                # Preserve the retained autotrainer behavior exactly: when one
                # callback pushes the count over X, use the entire current
                # batch and then clear it.
                self._buffer.clear()
            baseline = _vector(baseline_dcs, "baseline_dcs")
            prior = self._accepted_target or baseline
            reduced = tuple(
                statistics.fmean(item.closest_offset_dcs[axis] for item in window)
                for axis in range(3)
            )
            raw_shift = tuple(
                reduced[axis] - self._policy.target_reach_offset_dcs[axis]
                for axis in range(3)
            )
            recommended = tuple(
                0.0 if abs(value) <= self._policy.deadbands_mm[axis]
                else _bounded(value, self._policy.maximum_update_mm[axis])
                for axis, value in enumerate(raw_shift)
            )
            candidate = tuple(prior[axis] + recommended[axis] for axis in range(3))
            target = tuple(
                baseline[axis] + _bounded(
                    candidate[axis] - baseline[axis],
                    self._policy.maximum_absolute_mm[axis],
                )
                for axis in range(3)
            )
            if self._policy.minimum_y_dcs is not None:
                target = (target[0], max(target[1], self._policy.minimum_y_dcs), target[2])
            self._generation += 1
            self._latest = AutomaticShiftGeneration(
                generation=self._generation,
                policy_id=self._policy.policy_id,
                window_method=self._policy.window_method,
                reach_ids=tuple(item.reach_id for item in window),
                reduced_reach_offset_dcs=reduced,
                recommended_shift_dcs=recommended,
                baseline_dcs=baseline,
                prior_target_dcs=prior,
                resolved_target_dcs=target,
                apply_automatically=self._policy.apply_automatically,
            )
            return self._latest

    def accept(self, generation: int) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            latest = self._latest
            if latest is None or latest.generation != int(generation):
                raise RuntimeError("Automatic shift generation is stale or unknown")
            if generation <= self._accepted_generation:
                return None
            self._accepted_generation = generation
            if latest.apply_automatically:
                self._accepted_target = latest.resolved_target_dcs
            return self._accepted_target


def _vector(values, name):
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _bounded(value, limit):
    return min(float(limit), max(-float(limit), float(value)))
