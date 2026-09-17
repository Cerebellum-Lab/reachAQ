"""Relative adjustments to the pellet load motion.

reachAQ already stores the pellet load sequence as an editable list of motor
steps (``load_pellet`` in the compound movement configuration), so unlike the
reach-training reference the motion is host-owned rather than firmware-owned
and needs no new command path.

What is missing is the ability to nudge a proven sequence.  Editing the step
list to move a scoop 1 mm deeper means retyping absolute targets and losing the
record of what the original motion was.  These offsets are relative: zero
preserves the existing motion exactly, negative moves toward home, positive
moves away from it, and the original step list stays intact underneath.

Offsets address a step by axis and by which occurrence of that axis it is, so
the separate approach, scoop, and retract legs of one axis can be adjusted
independently, matching the stage granularity of the reference.  Only absolute
positional steps are adjusted; relative moves and non-positional steps such as
arms, tones, and predefined actions are left untouched, because shifting a
relative move changes the travel rather than the endpoint.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Iterable, List, Mapping, Tuple


PELLET_LOAD_OFFSET_SCHEMA_VERSION = 1

# Absolute positional step types the offsets may adjust.
POSITIONAL_AXES = ("x", "y", "z")
MAX_ABS_OFFSET_MM = 25.0


@dataclasses.dataclass(frozen=True)
class PelletLoadOffset:
    """One relative adjustment to a positional step."""

    axis: str
    occurrence: int
    offset_mm: float

    def __post_init__(self):
        axis = str(self.axis).strip().lower()
        if axis not in POSITIONAL_AXES:
            raise ValueError(
                f"Pellet load offsets apply to {', '.join(POSITIONAL_AXES)}, "
                f"not {self.axis!r}"
            )
        occurrence = int(self.occurrence)
        if occurrence < 1:
            raise ValueError("Offset occurrence is 1-based and must be positive")
        offset = float(self.offset_mm)
        if not math.isfinite(offset) or abs(offset) > MAX_ABS_OFFSET_MM:
            raise ValueError(
                f"Pellet load offset must be finite and within "
                f"+/-{MAX_ABS_OFFSET_MM:g} mm"
            )
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "occurrence", occurrence)
        object.__setattr__(self, "offset_mm", offset)

    @property
    def key(self) -> Tuple[str, int]:
        return (self.axis, self.occurrence)

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PelletLoadOffsets:
    """A named set of relative load adjustments."""

    offsets: Tuple[PelletLoadOffset, ...] = ()
    schema_version: int = PELLET_LOAD_OFFSET_SCHEMA_VERSION

    def __post_init__(self):
        normalized = tuple(
            item
            if isinstance(item, PelletLoadOffset)
            else PelletLoadOffset(**dict(item))
            for item in self.offsets
        )
        seen = set()
        for offset in normalized:
            if offset.key in seen:
                raise ValueError(
                    f"Duplicate pellet load offset for {offset.axis} "
                    f"occurrence {offset.occurrence}"
                )
            seen.add(offset.key)
        object.__setattr__(self, "offsets", normalized)
        if self.schema_version != PELLET_LOAD_OFFSET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported pellet load offset schema {self.schema_version}"
            )

    @property
    def is_identity(self) -> bool:
        """True when the offsets leave the original motion unchanged."""

        return all(offset.offset_mm == 0.0 for offset in self.offsets)

    def by_key(self) -> Dict[Tuple[str, int], float]:
        return {offset.key: offset.offset_mm for offset in self.offsets}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PelletLoadOffsets":
        return cls(
            offsets=tuple(record.get("offsets", ())),
            schema_version=int(
                record.get("schema_version", PELLET_LOAD_OFFSET_SCHEMA_VERSION)
            ),
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "offsets": [offset.to_record() for offset in self.offsets],
        }


def apply_load_offsets(
    steps: Iterable[Mapping[str, Any]],
    offsets: PelletLoadOffsets,
) -> List[Dict[str, Any]]:
    """Return the step list with the offsets applied.

    Unknown axis occurrences are an error rather than a silent no-op: an offset
    that addresses a step which does not exist means the configuration and the
    sequence have drifted apart.
    """

    remaining = offsets.by_key()
    counts: Dict[str, int] = {}
    adjusted: List[Dict[str, Any]] = []

    for step in steps:
        step = dict(step)
        if len(step) != 1:
            raise ValueError(f"Motor step must hold exactly one action: {step!r}")
        (axis, value), = step.items()
        axis = str(axis)
        if axis not in POSITIONAL_AXES:
            adjusted.append(step)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            # A positional step that is not a plain number is left as authored.
            adjusted.append(step)
            continue
        counts[axis] = counts.get(axis, 0) + 1
        offset = remaining.pop((axis, counts[axis]), None)
        if offset:
            numeric += offset
            # Keep integral targets integral so the emitted steps still match
            # the authored style.
            value = int(numeric) if float(numeric).is_integer() else numeric
        adjusted.append({axis: value})

    if remaining:
        described = ", ".join(
            f"{axis} occurrence {occurrence}"
            for axis, occurrence in sorted(remaining)
        )
        raise ValueError(
            f"Pellet load offsets address steps that do not exist: {described}"
        )
    return adjusted
