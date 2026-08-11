"""Editable, ordered pellet-delivery settings for future logical trials."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Tuple


@dataclass(frozen=True)
class TrialProtocolRow:
    trial_id: int
    pellet_behavior: str = "Standard"
    shift_x_mm: float = 0.0
    shift_y_mm: float = 0.0
    shift_z_mm: float = 0.0
    cover: bool = True
    tone: str = "None"
    laser: str = "None"

    def to_record(self) -> dict:
        return asdict(self)


class TrialProtocolSchedule:
    """In-memory schedule; every used row is snapshotted into the trial ledger."""

    EDITABLE_FIELDS = frozenset({
        "pellet_behavior",
        "shift_x_mm",
        "shift_y_mm",
        "shift_z_mm",
        "cover",
        "tone",
        "laser",
    })

    def __init__(self, rows: Iterable[TrialProtocolRow] = ()):
        self._rows = {int(row.trial_id): row for row in rows}

    @classmethod
    def with_placeholder_rows(cls, count: int = 15) -> "TrialProtocolSchedule":
        rows = []
        for trial_id in range(1, int(count) + 1):
            rows.append(TrialProtocolRow(
                trial_id=trial_id,
                pellet_behavior="Standard" if trial_id % 5 else "Retract after cue",
                tone="Cue A" if trial_id % 3 == 0 else "None",
                laser="Pulse A" if trial_id % 5 == 0 else "None",
            ))
        return cls(rows)

    @property
    def rows(self) -> Tuple[TrialProtocolRow, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def row(self, trial_id: int) -> TrialProtocolRow:
        trial_id = int(trial_id)
        if trial_id not in self._rows:
            self._rows[trial_id] = TrialProtocolRow(trial_id=trial_id)
        return self._rows[trial_id]

    def update(self, trial_id: int, field: str, value) -> TrialProtocolRow:
        if field not in self.EDITABLE_FIELDS:
            raise ValueError(f"Unknown trial protocol field: {field}")
        row = self.row(trial_id)
        if field in {"shift_x_mm", "shift_y_mm", "shift_z_mm"}:
            value = float(value)
        elif field == "cover":
            value = bool(value)
        else:
            value = str(value).strip()
        row = replace(row, **{field: value})
        self._rows[row.trial_id] = row
        return row

    def to_records(self) -> Tuple[dict, ...]:
        return tuple(row.to_record() for row in self.rows)
