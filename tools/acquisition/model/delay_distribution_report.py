"""Realized cue-interval reporting for a recorded session.

A configured distribution says what should happen; this module records what
actually did.  At normal Record stop the session keeps the configuration, the
realized per-interval occurrence counts and percentages, and the deviation from
the configured weights, so a session can be audited without replaying every
trial recipe.

The report carries every series a renderer needs for the categorical CDF and
the PMF against the continuous reference PDF, so plotting is a presentation
step over this record rather than a second source of truth.

This module is pure: it aggregates already-drawn selections and never draws.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from autotrainer.core.delay_distribution import DelayDistributionProfile


DELAY_REPORT_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class RealizedDelay:
    """One interval that a trial actually drew."""

    delay_ms: int
    logical_trial_id: Optional[int] = None
    attempt_id: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DelayOccurrence:
    """How often one configured interval was drawn."""

    delay_ms: int
    count: int
    percentage: float
    configured_probability: float

    @property
    def realized_probability(self) -> float:
        return self.percentage / 100.0

    @property
    def deviation(self) -> float:
        """Realized minus configured probability."""

        return self.realized_probability - self.configured_probability

    def to_record(self) -> Dict[str, Any]:
        record = dataclasses.asdict(self)
        record["realized_probability"] = self.realized_probability
        record["deviation"] = self.deviation
        return record


@dataclasses.dataclass(frozen=True)
class DelayDistributionReport:
    """Configuration plus realized draws for one recorded session."""

    profile_id: str
    total_draws: int
    occurrences: Tuple[DelayOccurrence, ...]
    distribution: Mapping[str, Any]
    unexpected_delays: Tuple[int, ...] = ()
    schema_version: int = DELAY_REPORT_SCHEMA_VERSION

    @property
    def max_abs_deviation(self) -> float:
        if not self.occurrences:
            return 0.0
        return max(abs(item.deviation) for item in self.occurrences)

    @property
    def values(self) -> Tuple[int, ...]:
        return tuple(item.delay_ms for item in self.occurrences)

    @property
    def counts(self) -> Tuple[int, ...]:
        return tuple(item.count for item in self.occurrences)

    def plot_series(self) -> Dict[str, Any]:
        """Return everything a renderer needs, without rendering anything."""

        realized_cdf = []
        cumulative = 0.0
        for item in self.occurrences:
            cumulative += item.realized_probability
            realized_cdf.append(cumulative)
        return {
            "plot_contract": self.distribution.get("plot_contract"),
            "values_ms": list(self.values),
            "realized_counts": list(self.counts),
            "realized_pmf": [item.realized_probability for item in self.occurrences],
            "realized_cdf": realized_cdf,
            "configured_pmf": [
                item.configured_probability for item in self.occurrences
            ],
            "configured_cdf": list(
                self.distribution.get("categorical_cdf_at_values", ())
            ),
            "reference_cdf": list(
                self.distribution.get("continuous_reference_cdf_at_values", ())
            ),
            "reference_pdf_per_second": list(
                self.distribution.get(
                    "continuous_reference_pdf_per_second_at_values", ()
                )
            ),
        }

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "total_draws": self.total_draws,
            "max_abs_deviation": self.max_abs_deviation,
            "occurrences": [item.to_record() for item in self.occurrences],
            "unexpected_delays_ms": list(self.unexpected_delays),
            "distribution": dict(self.distribution),
            "plot_series": self.plot_series(),
        }


def build_delay_report(
    profile_id: str,
    distribution: DelayDistributionProfile,
    realized: Iterable[Any],
) -> DelayDistributionReport:
    """Summarize the intervals a session actually drew.

    ``realized`` accepts :class:`RealizedDelay` values, plain integers, or the
    selection records the compiler writes into each trial recipe.
    """

    counts: Dict[int, int] = {value: 0 for value in distribution.values}
    unexpected: Dict[int, int] = {}
    total = 0
    for item in realized:
        delay_ms = _delay_of(item)
        total += 1
        if delay_ms in counts:
            counts[delay_ms] += 1
        else:
            # An interval outside the configured support means the protocol
            # changed mid-session; surface it rather than folding it in.
            unexpected[delay_ms] = unexpected.get(delay_ms, 0) + 1

    occurrences = tuple(
        DelayOccurrence(
            delay_ms=value,
            count=counts[value],
            percentage=(counts[value] / total * 100.0) if total else 0.0,
            configured_probability=float(probability),
        )
        for value, probability in zip(
            distribution.values, distribution.probabilities
        )
    )
    return DelayDistributionReport(
        profile_id=str(profile_id),
        total_draws=total,
        occurrences=occurrences,
        distribution=distribution.to_record(),
        unexpected_delays=tuple(sorted(unexpected)),
    )


def _delay_of(item: Any) -> int:
    if isinstance(item, RealizedDelay):
        return int(item.delay_ms)
    if isinstance(item, Mapping):
        if "delay_ms" not in item:
            raise ValueError("Realized delay record is missing delay_ms")
        return int(item["delay_ms"])
    if isinstance(item, bool):
        raise ValueError("Realized delays must be integer milliseconds")
    try:
        delay_ms = int(item)
    except (TypeError, ValueError) as exc:
        raise ValueError("Realized delays must be integer milliseconds") from exc
    if delay_ms != item or not math.isfinite(float(delay_ms)):
        raise ValueError("Realized delays must be integer milliseconds")
    return delay_ms
