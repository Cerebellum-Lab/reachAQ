"""Cue-interval delay distributions for reach protocols.

This module is deliberately independent from Qt, hardware, and acquisition
state.  It turns a configured set of candidate cue intervals into a fully
resolved, immutable probability profile that the protocol layer can compile and
that session metadata can record verbatim.

Two families are supported:

* **Published presets** carry fixed support and fixed weights transcribed from
  the source publications, with the 305 ms safety adaptation already applied.
  They are read-only: editing their support is rejected rather than silently
  recalculated.
* **Custom distributions** quantize the continuous shifted-exponential CDF into
  nearest-value buckets, or accept an operator-entered manual probability
  vector.

Bucket boundaries for a calculated custom distribution are the adjacent-value
midpoints.  The first bucket starts at the distribution offset and the last
bucket extends to infinity, so the discrete weights always total one over the
full support rather than truncating the tail.

Selection is driven by a caller-supplied uniform value so the acquisition layer
can keep its seeded, reproducible per-trial draw.  This module never calls into
``random`` itself.
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


DELAY_DISTRIBUTION_SCHEMA_VERSION = 1

# The 305 ms floor is a rig safety adaptation applied to every published
# support; intervals shorter than this are not deliverable.
MINIMUM_CUE_INTERVAL_MS = 305

EXPONENTIAL_CDF_OFFSET_MS = 200
EXPONENTIAL_CDF_TAU_MS = 1400
MINIMUM_EXPONENTIAL_CDF_TAU_MS = 1
MAXIMUM_EXPONENTIAL_CDF_TAU_MS = 60000

DELAY_DISTRIBUTION_PLOT_CONTRACT = (
    "discrete_pmf_cdf_vs_continuous_shifted_exponential_v1"
)
HAZARD_REFERENCE = "continuous_shifted_exponential_constant_after_offset"
FINITE_SUPPORT_HAZARD = "quantized_not_exactly_constant"

PUBLISHED_4S_PROBABILITY_SOURCE = "paper_table_4s_weights_305_adaptation_v1"
PUBLISHED_5S_PROBABILITY_SOURCE = "paper_table_5s_weights_305_adaptation_v1"
PUBLISHED_SHORT_PROBABILITY_SOURCE = (
    "paper_table_short_weights_normalized_305_adaptation_v1"
)
CALCULATED_PROBABILITY_SOURCE = (
    "custom_shifted_exponential_cdf_midpoint_full_support_v2"
)
MANUAL_PROBABILITY_SOURCE = "custom_manual_probabilities_v1"


class DelayPreset(str, Enum):
    """Selector for a published support or an editable custom distribution."""

    PUBLISHED_4S = "published_4s"
    PUBLISHED_5S = "published_5s"
    PUBLISHED_SHORT = "published_short"
    CUSTOM = "custom"

    @property
    def label(self) -> str:
        return _PRESET_LABELS[self]

    @property
    def is_published(self) -> bool:
        return self is not DelayPreset.CUSTOM


class DelayProbabilityMode(str, Enum):
    """How the weights for a profile were produced."""

    PUBLISHED = "published"
    CALCULATED = "calculated"
    MANUAL = "manual"


DEFAULT_DELAY_PRESET = DelayPreset.PUBLISHED_4S
DEFAULT_PROBABILITY_MODE = DelayProbabilityMode.CALCULATED

_PRESET_LABELS = {
    DelayPreset.PUBLISHED_4S: "Published 4 s preset",
    DelayPreset.PUBLISHED_5S: "Published 5 s preset",
    DelayPreset.PUBLISHED_SHORT: "Published short-delay preset",
    DelayPreset.CUSTOM: "Calculated custom distribution",
}

PUBLISHED_4S_VALUES = (305, 500, 700, 900, 1200, 2000, 3200, 4000)
PUBLISHED_4S_PROBABILITIES = (0.125, 0.125, 0.10, 0.10, 0.10, 0.30, 0.10, 0.05)

PUBLISHED_5S_VALUES = (305, 500, 700, 900, 1200, 2000, 3200, 5000)
PUBLISHED_5S_PROBABILITIES = (0.125, 0.125, 0.10, 0.10, 0.10, 0.30, 0.07, 0.08)

PUBLISHED_SHORT_VALUES = (305, 600, 1200, 1800, 2400, 3600)
# The published short-delay weights are reported to two decimals and total
# 0.99; they are renormalized here rather than edited.
PUBLISHED_SHORT_REPORTED_PROBABILITIES = (0.22, 0.30, 0.19, 0.11, 0.09, 0.08)
PUBLISHED_SHORT_REPORTED_PROBABILITY_SUM = sum(
    PUBLISHED_SHORT_REPORTED_PROBABILITIES
)
PUBLISHED_SHORT_PROBABILITIES = tuple(
    probability / PUBLISHED_SHORT_REPORTED_PROBABILITY_SUM
    for probability in PUBLISHED_SHORT_REPORTED_PROBABILITIES
)


@dataclasses.dataclass(frozen=True)
class _PublishedPreset:
    preset: DelayPreset
    values: Tuple[int, ...]
    probabilities: Tuple[float, ...]
    reported_probabilities: Tuple[float, ...]
    source: str
    preset_name: str
    cdf_tau_ms: float
    cdf_offset_ms: float
    probability_normalization: str


_PUBLISHED_PRESETS = (
    _PublishedPreset(
        preset=DelayPreset.PUBLISHED_4S,
        values=PUBLISHED_4S_VALUES,
        probabilities=PUBLISHED_4S_PROBABILITIES,
        reported_probabilities=PUBLISHED_4S_PROBABILITIES,
        source=PUBLISHED_4S_PROBABILITY_SOURCE,
        preset_name="Published 4 s preset (305 ms safety adaptation)",
        cdf_tau_ms=EXPONENTIAL_CDF_TAU_MS,
        cdf_offset_ms=EXPONENTIAL_CDF_OFFSET_MS,
        probability_normalization="not_required",
    ),
    _PublishedPreset(
        preset=DelayPreset.PUBLISHED_5S,
        values=PUBLISHED_5S_VALUES,
        probabilities=PUBLISHED_5S_PROBABILITIES,
        reported_probabilities=PUBLISHED_5S_PROBABILITIES,
        source=PUBLISHED_5S_PROBABILITY_SOURCE,
        preset_name="Published 5 s preset (305 ms safety adaptation)",
        cdf_tau_ms=EXPONENTIAL_CDF_TAU_MS,
        cdf_offset_ms=EXPONENTIAL_CDF_OFFSET_MS,
        probability_normalization="not_required",
    ),
    _PublishedPreset(
        preset=DelayPreset.PUBLISHED_SHORT,
        values=PUBLISHED_SHORT_VALUES,
        probabilities=PUBLISHED_SHORT_PROBABILITIES,
        reported_probabilities=PUBLISHED_SHORT_REPORTED_PROBABILITIES,
        source=PUBLISHED_SHORT_PROBABILITY_SOURCE,
        preset_name=(
            "Published short-delay preset (305 ms safety adaptation; "
            "reported weights normalized from 0.99)"
        ),
        cdf_tau_ms=900.0,
        cdf_offset_ms=300.0,
        probability_normalization="proportional_from_reported_sum_0.99",
    ),
)

_PUBLISHED_BY_PRESET = {item.preset: item for item in _PUBLISHED_PRESETS}
_PUBLISHED_BY_VALUES = {item.values: item for item in _PUBLISHED_PRESETS}


@dataclasses.dataclass(frozen=True)
class DelaySelection:
    """One resolved cue interval drawn from a profile."""

    delay_ms: int
    probability: float
    index: int

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DelayDistributionProfile:
    """An immutable, fully resolved cue-interval distribution."""

    values: Tuple[int, ...]
    probabilities: Tuple[float, ...]
    preset: DelayPreset
    probability_mode: DelayProbabilityMode
    source: str
    probability_normalization: str
    cdf_tau_ms: float
    cdf_offset_ms: float
    categorical_cdf_at_values: Tuple[float, ...]
    continuous_reference_cdf_at_values: Tuple[float, ...]
    continuous_reference_pdf_per_second_at_values: Tuple[float, ...]
    cdf_max_abs_error: float
    continuous_mass_below_first_value: float
    continuous_mass_above_last_value: float
    preset_name: Optional[str] = None
    reported_probabilities: Optional[Tuple[float, ...]] = None
    reported_probability_sum: Optional[float] = None
    bucket_boundaries_ms: Optional[Tuple[Any, ...]] = None
    schema_version: int = DELAY_DISTRIBUTION_SCHEMA_VERSION
    plot_contract: str = DELAY_DISTRIBUTION_PLOT_CONTRACT
    hazard_reference: str = HAZARD_REFERENCE
    finite_support_hazard: str = FINITE_SUPPORT_HAZARD

    @property
    def continuous_hazard_rate_per_second(self) -> float:
        return 1000.0 / self.cdf_tau_ms

    def to_record(self) -> Dict[str, Any]:
        """Return a plain mapping suitable for session metadata."""

        record = dataclasses.asdict(self)
        record["preset"] = self.preset.value
        record["probability_mode"] = self.probability_mode.value
        record["continuous_hazard_rate_per_second"] = (
            self.continuous_hazard_rate_per_second
        )
        return record


def normalize_cue_intervals(values: Iterable[int]) -> Tuple[int, ...]:
    """Return sorted, unique, deliverable integer millisecond intervals."""

    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Cue intervals must be integer milliseconds")
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cue intervals must be integer milliseconds") from exc
        if numeric != value:
            raise ValueError("Cue intervals must be integer milliseconds")
        normalized.append(numeric)

    if not normalized:
        raise ValueError("At least one cue interval is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Cue intervals must be unique")
    normalized.sort()
    if normalized[0] < MINIMUM_CUE_INTERVAL_MS:
        raise ValueError(
            f"Cue intervals must be {MINIMUM_CUE_INTERVAL_MS} ms or longer"
        )
    return tuple(normalized)


def infer_delay_preset(values: Iterable[int]) -> DelayPreset:
    """Infer a published preset from a saved support, else custom."""

    published = _PUBLISHED_BY_VALUES.get(normalize_cue_intervals(values))
    return DelayPreset.CUSTOM if published is None else published.preset


def normalize_delay_preset(
    value: Any,
    legacy_values: Optional[Iterable[int]] = None,
) -> DelayPreset:
    """Normalize a saved selector, inferring from legacy support when asked."""

    if isinstance(value, DelayPreset):
        return value
    try:
        return DelayPreset(str(value or "").strip().lower())
    except ValueError:
        pass
    if legacy_values is not None:
        return infer_delay_preset(legacy_values)
    return DEFAULT_DELAY_PRESET


def normalize_probability_mode(value: Any) -> DelayProbabilityMode:
    if isinstance(value, DelayProbabilityMode):
        return value
    try:
        return DelayProbabilityMode(str(value or "").strip().lower())
    except ValueError:
        return DEFAULT_PROBABILITY_MODE


def normalize_tau_ms(value: Any) -> int:
    """Clamp a configured tau into the supported range."""

    if isinstance(value, bool):
        return EXPONENTIAL_CDF_TAU_MS
    try:
        tau_ms = int(value)
    except (TypeError, ValueError):
        return EXPONENTIAL_CDF_TAU_MS
    return max(
        MINIMUM_EXPONENTIAL_CDF_TAU_MS,
        min(MAXIMUM_EXPONENTIAL_CDF_TAU_MS, tau_ms),
    )


def preset_values(preset: Any) -> Optional[Tuple[int, ...]]:
    """Return the fixed support for a published preset, or None for custom."""

    published = _PUBLISHED_BY_PRESET.get(normalize_delay_preset(preset))
    return None if published is None else published.values


def shifted_exponential_cdf(
    delay_ms: float,
    tau_ms: float = EXPONENTIAL_CDF_TAU_MS,
    offset_ms: float = EXPONENTIAL_CDF_OFFSET_MS,
) -> float:
    """Return the continuous shifted-exponential CDF at a delay in ms."""

    delay_ms = float(delay_ms)
    tau_ms = float(tau_ms)
    offset_ms = float(offset_ms)
    if tau_ms <= 0.0:
        raise ValueError("tau_ms must be greater than zero")
    if delay_ms <= offset_ms:
        return 0.0
    if math.isinf(delay_ms):
        return 1.0
    return -math.expm1(-(delay_ms - offset_ms) / tau_ms)


def shifted_exponential_pdf_per_second(
    delay_ms: float,
    tau_ms: float = EXPONENTIAL_CDF_TAU_MS,
    offset_ms: float = EXPONENTIAL_CDF_OFFSET_MS,
) -> float:
    """Return the continuous shifted-exponential density in inverse seconds."""

    delay_ms = float(delay_ms)
    tau_ms = float(tau_ms)
    offset_ms = float(offset_ms)
    if tau_ms <= 0.0:
        raise ValueError("tau_ms must be greater than zero")
    if delay_ms < offset_ms or math.isinf(delay_ms):
        return 0.0
    return (1000.0 / tau_ms) * math.exp(-(delay_ms - offset_ms) / tau_ms)


def normalize_manual_probabilities(
    values: Iterable[int],
    probabilities: Iterable[float],
) -> Tuple[float, ...]:
    """Return manual weights aligned to the support, without totalling them."""

    ordered_values = normalize_cue_intervals(values)
    try:
        normalized = tuple(float(value) for value in probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError("Manual delay probabilities must be numeric") from exc
    if len(normalized) != len(ordered_values):
        raise ValueError("Manual probabilities must match the custom cue intervals")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in normalized
    ):
        raise ValueError("Manual delay probabilities must be between zero and one")
    return normalized


def validate_manual_probabilities(
    values: Iterable[int],
    probabilities: Iterable[float],
) -> Tuple[float, ...]:
    """Return manual weights, requiring them to total exactly 100.000%."""

    normalized = normalize_manual_probabilities(values, probabilities)
    total = sum(normalized)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Manual delay probabilities total {total:.6%}; "
            "they must total 100.000%"
        )
    return normalized


def _cdf_max_abs_error(
    values: Sequence[int],
    probabilities: Sequence[float],
    tau_ms: float,
    offset_ms: float,
) -> float:
    """Return the Kolmogorov distance from the continuous reference CDF."""

    cumulative = 0.0
    maximum_error = 0.0
    for value, probability in zip(values, probabilities):
        reference = shifted_exponential_cdf(float(value), tau_ms, offset_ms)
        maximum_error = max(maximum_error, abs(cumulative - reference))
        cumulative += probability
        maximum_error = max(maximum_error, abs(cumulative - reference))
    return maximum_error


def _calculated_probabilities(
    values: Sequence[int],
    tau_ms: float,
) -> Tuple[Tuple[float, ...], Tuple[Any, ...]]:
    """Quantize the shifted-exponential CDF into nearest-value buckets."""

    if len(values) == 1:
        return (1.0,), (float(EXPONENTIAL_CDF_OFFSET_MS), None)

    boundaries: list = [float(EXPONENTIAL_CDF_OFFSET_MS)]
    boundaries.extend(
        (left + right) / 2.0 for left, right in zip(values, values[1:])
    )
    boundaries.append(None)

    weights = [
        max(
            0.0,
            (
                1.0
                if upper is None
                else shifted_exponential_cdf(
                    upper, tau_ms, EXPONENTIAL_CDF_OFFSET_MS
                )
            )
            - shifted_exponential_cdf(lower, tau_ms, EXPONENTIAL_CDF_OFFSET_MS),
        )
        for lower, upper in zip(boundaries, boundaries[1:])
    ]
    # The final bucket carries the tail so the discrete weights total one.
    weights[-1] += 1.0 - sum(weights)
    return tuple(weights), tuple(boundaries)


def _reference_series(
    values: Sequence[int],
    probabilities: Sequence[float],
    tau_ms: float,
    offset_ms: float,
) -> Dict[str, Any]:
    categorical_cdf = []
    cumulative = 0.0
    for probability in probabilities:
        cumulative += float(probability)
        categorical_cdf.append(cumulative)
    categorical_cdf[-1] = 1.0
    first = float(values[0])
    last = float(values[-1])
    return {
        "categorical_cdf_at_values": tuple(categorical_cdf),
        "continuous_reference_cdf_at_values": tuple(
            shifted_exponential_cdf(value, tau_ms, offset_ms) for value in values
        ),
        "continuous_reference_pdf_per_second_at_values": tuple(
            shifted_exponential_pdf_per_second(value, tau_ms, offset_ms)
            for value in values
        ),
        "cdf_max_abs_error": _cdf_max_abs_error(
            values, probabilities, tau_ms, offset_ms
        ),
        "continuous_mass_below_first_value": shifted_exponential_cdf(
            first, tau_ms, offset_ms
        ),
        "continuous_mass_above_last_value": 1.0
        - shifted_exponential_cdf(last, tau_ms, offset_ms),
    }


def build_delay_distribution_profile(
    values: Iterable[int],
    preset: Any = None,
    *,
    tau_ms: Any = EXPONENTIAL_CDF_TAU_MS,
    manual_probabilities: Optional[Iterable[float]] = None,
) -> DelayDistributionProfile:
    """Return the resolved profile for the supplied cue intervals.

    With no explicit selector, a support that exactly matches a published preset
    keeps that preset's reported weights.  An explicit published selector
    requires its fixed support; an explicit custom selector always calculates or
    accepts manual weights.
    """

    ordered_values = normalize_cue_intervals(values)

    if preset is None:
        published = _PUBLISHED_BY_VALUES.get(ordered_values)
    else:
        # An explicit selector must be recognized; unlike a saved value it is
        # never silently defaulted.
        if isinstance(preset, DelayPreset):
            resolved_preset = preset
        else:
            try:
                resolved_preset = DelayPreset(str(preset).strip().lower())
            except ValueError as exc:  # noqa: TRY003 - explicit selector
                raise ValueError(
                    f"Unknown cue interval delay preset: {preset!r}"
                ) from exc
        published = _PUBLISHED_BY_PRESET.get(resolved_preset)
        if published is not None and ordered_values != published.values:
            raise ValueError(
                "Published delay preset values do not match its fixed support"
            )

    if published is not None:
        if manual_probabilities is not None:
            raise ValueError(
                "Published delay presets do not accept manual probabilities"
            )
        return DelayDistributionProfile(
            values=ordered_values,
            probabilities=published.probabilities,
            preset=published.preset,
            probability_mode=DelayProbabilityMode.PUBLISHED,
            source=published.source,
            preset_name=published.preset_name,
            reported_probabilities=published.reported_probabilities,
            reported_probability_sum=sum(published.reported_probabilities),
            probability_normalization=published.probability_normalization,
            cdf_tau_ms=published.cdf_tau_ms,
            cdf_offset_ms=published.cdf_offset_ms,
            bucket_boundaries_ms=None,
            **_reference_series(
                ordered_values,
                published.probabilities,
                published.cdf_tau_ms,
                published.cdf_offset_ms,
            ),
        )

    custom_tau_ms = float(normalize_tau_ms(tau_ms))
    if manual_probabilities is not None:
        probabilities = validate_manual_probabilities(
            ordered_values, manual_probabilities
        )
        boundaries: Optional[Tuple[Any, ...]] = None
        probability_mode = DelayProbabilityMode.MANUAL
        source = MANUAL_PROBABILITY_SOURCE
        normalization = "operator_entered_sum_1"
    else:
        probabilities, boundaries = _calculated_probabilities(
            ordered_values, custom_tau_ms
        )
        probability_mode = DelayProbabilityMode.CALCULATED
        source = CALCULATED_PROBABILITY_SOURCE
        normalization = "not_applicable"

    return DelayDistributionProfile(
        values=ordered_values,
        probabilities=probabilities,
        preset=DelayPreset.CUSTOM,
        probability_mode=probability_mode,
        source=source,
        preset_name=None,
        reported_probabilities=None,
        reported_probability_sum=None,
        probability_normalization=normalization,
        cdf_tau_ms=custom_tau_ms,
        cdf_offset_ms=float(EXPONENTIAL_CDF_OFFSET_MS),
        bucket_boundaries_ms=(
            None
            if boundaries is None
            else tuple(
                "+infinity" if boundary is None else boundary
                for boundary in boundaries
            )
        ),
        **_reference_series(
            ordered_values,
            probabilities,
            custom_tau_ms,
            float(EXPONENTIAL_CDF_OFFSET_MS),
        ),
    )


def select_delay(
    profile: DelayDistributionProfile,
    random_value: float,
) -> DelaySelection:
    """Select one cue interval from a single uniform value in ``[0, 1)``.

    The caller owns the draw so acquisition can keep its seeded, reproducible
    per-trial value.
    """

    values = profile.values
    probabilities = profile.probabilities
    if len(values) != len(probabilities) or not values:
        raise ValueError("Profile values and probabilities must be aligned")
    random_value = float(random_value)
    if not 0.0 <= random_value < 1.0:
        raise ValueError("random_value must be in the half-open interval [0, 1)")

    cumulative = 0.0
    last_positive_index = None
    for index, probability in enumerate(probabilities):
        if probability < 0.0:
            raise ValueError("Probabilities cannot be negative")
        if probability > 0.0:
            last_positive_index = index
        cumulative += probability
        if probability > 0.0 and random_value < cumulative:
            return DelaySelection(
                delay_ms=int(values[index]),
                probability=float(probability),
                index=index,
            )

    if last_positive_index is None or not math.isclose(
        cumulative, 1.0, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("Probabilities must sum to one")
    return DelaySelection(
        delay_ms=int(values[last_positive_index]),
        probability=float(probabilities[last_positive_index]),
        index=last_positive_index,
    )
