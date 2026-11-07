"""Weighted categorical stimulus trigger profiles.

A profile is an ordered set of stimulus trigger categories, each carrying a
conditional selection weight in percent.  Once a trial has been gated in by the
protocol's stimulus percentage, exactly one enabled category is drawn from the
profile, so the weights are conditional on stimulation occurring rather than
absolute per-trial probabilities.

This module is deliberately independent from Qt, hardware, and the protocol
trigger vocabulary.  A category identifies its trigger by an opaque string so
the acquisition layer can supply its own ``StimulusTrigger`` values without
inverting the package dependency.  Categories that fire at a fixed lead time
carry ``offset_ms``; the caller supplies the upper bound those offsets must
respect, because only the protocol layer knows the active cue interval.

Selection is driven by a caller-supplied uniform value in ``[0, 1)`` so the
acquisition layer keeps its seeded, reproducible per-trial draw, matching
:mod:`autotrainer.core.delay_distribution`.  This module never calls into
``random`` itself.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


STIMULUS_TRIGGER_PROFILE_SCHEMA_VERSION = 1
STIMULUS_TRIGGER_SAMPLING_ALGORITHM = "weighted_categorical_with_replacement_v1"

# Percentages are operator-entered to three decimals, so equality against 100
# is checked with half-a-thousandth of tolerance rather than exactly.
PERCENTAGE_DECIMALS = 3
PERCENTAGE_TOTAL_TOLERANCE = 0.0005


@dataclasses.dataclass(frozen=True)
class StimulusTriggerCategory:
    """One weighted stimulus trigger category."""

    category_id: str
    trigger: str
    label: str = ""
    enabled: bool = True
    percentage: float = 0.0
    offset_ms: Optional[int] = None

    def __post_init__(self):
        if not str(self.category_id).strip():
            raise ValueError("Stimulus trigger category ID cannot be empty")
        if not str(self.trigger).strip():
            raise ValueError(
                f"Stimulus trigger category {self.category_id!r} has no trigger"
            )

    @property
    def is_offset_trigger(self) -> bool:
        return self.offset_ms is not None

    @property
    def probability(self) -> float:
        """Conditional selection probability given that stimulation occurs."""

        return self.percentage / 100.0

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StimulusTriggerSelection:
    """One resolved category drawn from a validated profile."""

    category: StimulusTriggerCategory
    probability: float
    index: int
    sampling_algorithm: str = STIMULUS_TRIGGER_SAMPLING_ALGORITHM

    def to_record(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["category"] = self.category.to_record()
        return record


def _normalize_percentage(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Trigger percentages must be numeric")
    try:
        percentage = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Trigger percentages must be numeric") from exc
    if not math.isfinite(percentage):
        raise ValueError("Trigger percentages must be finite")
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("Trigger percentages must be between zero and 100")
    return round(percentage, PERCENTAGE_DECIMALS)


def _normalize_offset(value: Any, category_id: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"Trigger offset for {category_id!r} must be integer milliseconds"
        )
    try:
        offset = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Trigger offset for {category_id!r} must be integer milliseconds"
        ) from exc
    if offset != value:
        raise ValueError(
            f"Trigger offset for {category_id!r} must be integer milliseconds"
        )
    if offset < 0:
        raise ValueError(f"Trigger offset for {category_id!r} cannot be negative")
    return offset


def _as_category(entry: Any) -> StimulusTriggerCategory:
    if isinstance(entry, StimulusTriggerCategory):
        source: Dict[str, Any] = dataclasses.asdict(entry)
    else:
        try:
            source = dict(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Stimulus trigger categories must be mappings or categories"
            ) from exc
    category_id = str(source.get("category_id", "")).strip()
    trigger = str(source.get("trigger", "")).strip()
    label = str(source.get("label", "") or "").strip() or category_id
    return StimulusTriggerCategory(
        category_id=category_id,
        trigger=trigger,
        label=label,
        enabled=bool(source.get("enabled", True)),
        percentage=_normalize_percentage(source.get("percentage", 0.0)),
        offset_ms=_normalize_offset(source.get("offset_ms"), category_id),
    )


def normalize_profile(
    profile: Iterable[Any],
) -> Tuple[StimulusTriggerCategory, ...]:
    """Coerce saved entries into normalized categories, preserving order."""

    normalized = tuple(_as_category(entry) for entry in profile)
    if not normalized:
        raise ValueError("A stimulus trigger profile requires at least one category")
    seen = set()
    for category in normalized:
        if category.category_id in seen:
            raise ValueError(
                f"Duplicate stimulus trigger category ID {category.category_id!r}"
            )
        seen.add(category.category_id)
    return normalized


def enabled_categories(
    profile: Sequence[StimulusTriggerCategory],
) -> Tuple[StimulusTriggerCategory, ...]:
    return tuple(category for category in profile if category.enabled)


def has_offset_trigger(profile: Iterable[Any]) -> bool:
    """Report whether any enabled, weighted category fires at a lead time."""

    return any(
        category.is_offset_trigger
        and category.enabled
        and category.percentage > 0.0
        for category in normalize_profile(profile)
    )


def validate_profile(
    profile: Iterable[Any],
    *,
    offset_upper_bound_ms: Optional[int] = None,
) -> Tuple[StimulusTriggerCategory, ...]:
    """Validate selection weights and any lead-time constraint.

    ``offset_upper_bound_ms`` is the shortest cue interval the profile may run
    against.  An offset trigger must fire strictly before that interval elapses,
    otherwise it could never be delivered.
    """

    normalized = normalize_profile(profile)
    enabled = enabled_categories(normalized)
    if not enabled:
        raise ValueError("Enable at least one stimulus trigger category")

    zero_weighted = [
        category.label for category in enabled if category.percentage <= 0.0
    ]
    if zero_weighted:
        raise ValueError(
            "Enabled trigger percentages must be greater than zero: "
            + ", ".join(zero_weighted)
        )

    total = round(
        sum(category.percentage for category in enabled), PERCENTAGE_DECIMALS
    )
    if abs(total - 100.0) > PERCENTAGE_TOTAL_TOLERANCE:
        raise ValueError(
            f"Enabled trigger percentages total {total:.3f}%; "
            "renormalize or distribute evenly to reach 100%"
        )

    if offset_upper_bound_ms is not None:
        bound = int(offset_upper_bound_ms)
        invalid = [
            category
            for category in enabled
            if category.is_offset_trigger and category.offset_ms >= bound
        ]
        if invalid:
            described = ", ".join(
                f"{category.label} ({category.offset_ms} ms)"
                for category in invalid
            )
            raise ValueError(
                "Trigger offsets must be shorter than the minimum cue interval "
                f"({bound} ms): {described}"
            )
    return normalized


def _rounded_to_100(raw_values: Sequence[float]) -> Tuple[float, ...]:
    """Round to three decimals and put the residual on the largest share."""

    if not raw_values:
        raise ValueError("Enable at least one stimulus trigger category")
    rounded = [round(float(value), PERCENTAGE_DECIMALS) for value in raw_values]
    residual = round(100.0 - sum(rounded), PERCENTAGE_DECIMALS)
    largest = max(range(len(raw_values)), key=lambda index: raw_values[index])
    rounded[largest] = round(rounded[largest] + residual, PERCENTAGE_DECIMALS)
    return tuple(rounded)


def _with_percentages(
    profile: Sequence[StimulusTriggerCategory],
    percentages: Sequence[float],
) -> Tuple[StimulusTriggerCategory, ...]:
    enabled_indexes = [
        index for index, category in enumerate(profile) if category.enabled
    ]
    if len(percentages) != len(enabled_indexes):
        raise ValueError("Percentage count does not match the enabled category count")
    updates = dict(zip(enabled_indexes, percentages))
    return tuple(
        dataclasses.replace(category, percentage=updates[index])
        if index in updates
        else category
        for index, category in enumerate(profile)
    )


def distribute_evenly(
    profile: Iterable[Any],
) -> Tuple[StimulusTriggerCategory, ...]:
    """Give every enabled category an equal share of 100%."""

    normalized = normalize_profile(profile)
    count = len(enabled_categories(normalized))
    if count <= 0:
        raise ValueError("Enable at least one stimulus trigger category")
    return _with_percentages(normalized, _rounded_to_100([100.0 / count] * count))


def renormalize(
    profile: Iterable[Any],
) -> Tuple[StimulusTriggerCategory, ...]:
    """Scale existing enabled weights proportionally so they total 100%."""

    normalized = normalize_profile(profile)
    existing = [category.percentage for category in enabled_categories(normalized)]
    if not existing:
        raise ValueError("Enable at least one stimulus trigger category")
    total = sum(existing)
    if total <= 0.0:
        raise ValueError(
            "Enabled percentages total zero; enter a positive percentage or "
            "distribute evenly"
        )
    if any(value <= 0.0 for value in existing):
        raise ValueError(
            "Enabled percentages must be positive before proportional "
            "normalization; enter positive values or distribute evenly"
        )
    raw = [(value / total) * 100.0 for value in existing]
    return _with_percentages(normalized, _rounded_to_100(raw))


def select_trigger(
    profile: Iterable[Any],
    random_value: float,
    *,
    offset_upper_bound_ms: Optional[int] = None,
) -> StimulusTriggerSelection:
    """Draw one enabled category from a single uniform value in ``[0, 1)``.

    The caller owns the draw so acquisition can keep its seeded, reproducible
    per-trial value.
    """

    validated = validate_profile(
        profile, offset_upper_bound_ms=offset_upper_bound_ms
    )
    try:
        draw = float(random_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("random_value must be numeric") from exc
    if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise ValueError("random_value must be in the half-open interval [0, 1)")

    enabled = enabled_categories(validated)
    cumulative = 0.0
    for index, category in enumerate(enabled):
        cumulative += category.probability
        # The final category absorbs the rounding residual so a draw just under
        # one always resolves.
        if draw < cumulative or index == len(enabled) - 1:
            return StimulusTriggerSelection(
                category=category,
                probability=category.probability,
                index=index,
            )
    raise RuntimeError("Stimulus trigger selection failed")


def profile_record(
    profile: Sequence[StimulusTriggerCategory],
) -> Dict[str, Any]:
    """Return a plain mapping of the profile for session metadata."""

    return {
        "schema_version": STIMULUS_TRIGGER_PROFILE_SCHEMA_VERSION,
        "sampling_algorithm": STIMULUS_TRIGGER_SAMPLING_ALGORITHM,
        "categories": [category.to_record() for category in profile],
    }
