"""Typed, hierarchical pellet-trial protocols.

This module is deliberately independent from Qt and hardware controllers.  It
turns a reusable protocol document into immutable, fully resolved rows that can
later be compiled by the execution layer.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


PROTOCOL_SCHEMA_VERSION = 1
DEFAULT_TRIAL_COUNT = 15
MAX_ABS_SHIFT_MM = 50.0
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ProtocolValue(str, Enum):
    """String enum with stable serialized values and user-facing labels."""

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


class PelletBehavior(ProtocolValue):
    STANDARD_CYCLE = "standard_cycle"
    SEND_AND_HOLD = "send_and_hold"
    SEND_THEN_RETRACT = "send_then_retract"
    MANUAL_TRIAL = "manual_trial"


class PelletPositionMode(ProtocolValue):
    BASE = "base"
    FIXED_MANUAL = "fixed_manual"
    REACH_DERIVED_AUTOMATIC = "reach_derived_automatic"


class PelletLane(ProtocolValue):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"


class CoverPolicy(ProtocolValue):
    KEEP_CURRENT = "keep_current"
    COVER = "cover"
    REVEAL = "reveal"


class ActionPhase(ProtocolValue):
    NONE = "none"
    BEFORE_SEND = "before_send"
    EMBEDDED_IN_SEQUENCE = "embedded_in_sequence"
    PELLET_PRESENTATION = "pellet_presentation"
    RETRACT = "retract"


class StimulusTrigger(ProtocolValue):
    NONE = "none"
    TONE_1 = "tone_1"
    TONE_2 = "tone_2"
    FIRST_REACH = "first_reach"
    PRE_REVEAL = "pre_reveal"
    ROI_1 = "roi_1"
    ROI_2 = "roi_2"


class StimulusAssignment(ProtocolValue):
    DISABLED = "disabled"
    ALWAYS = "always"
    PERCENTAGE = "percentage"
    RANDOMIZED = "randomized"


class RetryAssignment(ProtocolValue):
    REPEAT = "repeat"
    RESAMPLE = "resample"


class LaserTriggerRoute(ProtocolValue):
    NONE = "none"
    HARDWARE_STIM3 = "hardware_stim3"
    DIRECT_NI_SOFTWARE = "direct_ni_software"


class AutomaticWindowMethod(ProtocolValue):
    LEGACY_BATCH = "legacy_batch"
    SLIDING_LAST_X = "sliding_last_x"


_ENUM_FIELDS = {
    "pellet_behavior": PelletBehavior,
    "position_mode": PelletPositionMode,
    "position_lane": PelletLane,
    "cover_policy": CoverPolicy,
    "tone_phase": ActionPhase,
    "laser_phase": ActionPhase,
    "laser_trigger_route": LaserTriggerRoute,
    "stimulus_trigger": StimulusTrigger,
    "stimulus_assignment": StimulusAssignment,
    "retry_assignment": RetryAssignment,
    "automatic_window_method": AutomaticWindowMethod,
}
_FLOAT_FIELDS = {
    "shift_x_mm",
    "shift_y_mm",
    "shift_z_mm",
    "stimulus_probability_percent",
}
_INT_FIELDS = {"pre_reveal_ms", "automatic_window_size"}
_STRING_FIELDS = {
    "tone_profile_id",
    "laser_profile_id",
    "automatic_shift_policy_id",
    "stimulus_category",
}


def _identifier(value: str, *, field: str, allow_empty: bool = False) -> str:
    value = str(value).strip().lower()
    if allow_empty and not value:
        return ""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{field} must start with a lowercase letter or number and contain "
            "only lowercase letters, numbers, '.', '_' or '-'"
        )
    return value


def _finite_shift(value, *, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or abs(value) > MAX_ABS_SHIFT_MM:
        raise ValueError(
            f"{field} must be finite and within +/-{MAX_ABS_SHIFT_MM:g} mm"
        )
    return value


@dataclass(frozen=True)
class TrialProtocolRow:
    """One fully resolved logical pellet-trial recipe."""

    trial_id: int
    enabled: bool = False
    pellet_behavior: PelletBehavior = PelletBehavior.STANDARD_CYCLE
    position_mode: PelletPositionMode = PelletPositionMode.BASE
    position_lane: PelletLane = PelletLane.CENTER
    shift_x_mm: float = 0.0
    shift_y_mm: float = 0.0
    shift_z_mm: float = 0.0
    automatic_shift_policy_id: str = "default"
    automatic_window_method: AutomaticWindowMethod = (
        AutomaticWindowMethod.LEGACY_BATCH
    )
    automatic_window_size: int = 15
    cover_policy: CoverPolicy = CoverPolicy.KEEP_CURRENT
    tone_profile_id: str = ""
    tone_phase: ActionPhase = ActionPhase.NONE
    laser_profile_id: str = ""
    laser_phase: ActionPhase = ActionPhase.NONE
    laser_trigger_route: LaserTriggerRoute = LaserTriggerRoute.NONE
    stimulus_category: str = "none"
    stimulus_assignment: StimulusAssignment = StimulusAssignment.DISABLED
    stimulus_probability_percent: float = 100.0
    stimulus_trigger: StimulusTrigger = StimulusTrigger.NONE
    pre_reveal_ms: int = 0
    retry_assignment: RetryAssignment = RetryAssignment.REPEAT

    # Compatibility for the old table while its UI is migrated in a later
    # bounded commit.
    @property
    def cover(self) -> bool:
        return self.cover_policy is not CoverPolicy.REVEAL

    @property
    def tone(self) -> str:
        return self.tone_profile_id or "None"

    @property
    def laser(self) -> str:
        return self.laser_profile_id or "None"

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "TrialProtocolRow":
        if "trial_id" not in record:
            raise ValueError("Protocol row is missing trial_id")
        row = cls(trial_id=int(record["trial_id"]))
        updates = {key: value for key, value in record.items() if key != "trial_id"}
        # Accept only the immediately preceding placeholder schema as a local
        # development migration. It was never a released session schema.
        if "cover" in updates and "cover_policy" not in updates:
            updates["cover_policy"] = (
                CoverPolicy.COVER if bool(updates.pop("cover"))
                else CoverPolicy.REVEAL
            )
        if "tone" in updates and "tone_profile_id" not in updates:
            tone = str(updates.pop("tone")).strip()
            updates["tone_profile_id"] = "" if tone.lower() == "none" else tone
        if "laser" in updates and "laser_profile_id" not in updates:
            laser = str(updates.pop("laser")).strip()
            updates["laser_profile_id"] = "" if laser.lower() == "none" else laser
        return row.with_updates(updates)

    def with_updates(
        self,
        values: Mapping[str, object],
        *,
        validate: bool = True,
    ) -> "TrialProtocolRow":
        known = {item.name for item in fields(self)} - {"trial_id"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                "Unknown trial protocol field(s): " + ", ".join(sorted(unknown))
            )
        normalized = {}
        for field, value in values.items():
            if field in _ENUM_FIELDS:
                normalized[field] = _ENUM_FIELDS[field](value)
            elif field in {"shift_x_mm", "shift_y_mm", "shift_z_mm"}:
                normalized[field] = _finite_shift(value, field=field)
            elif field == "stimulus_probability_percent":
                value = float(value)
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError(
                        "stimulus_probability_percent must be between 0 and 100"
                    )
                normalized[field] = value
            elif field == "pre_reveal_ms":
                value = int(value)
                if not 0 <= value <= 60_000:
                    raise ValueError("pre_reveal_ms must be between 0 and 60000")
                normalized[field] = value
            elif field == "automatic_window_size":
                value = int(value)
                if not 1 <= value <= 10_000:
                    raise ValueError(
                        "automatic_window_size must be between 1 and 10000"
                    )
                normalized[field] = value
            elif field == "enabled":
                normalized[field] = bool(value)
            elif field in _STRING_FIELDS:
                normalized[field] = _identifier(
                    value,
                    field=field,
                    allow_empty=field in {"tone_profile_id", "laser_profile_id"},
                )
            else:  # Defensive; all current fields are covered above.
                normalized[field] = value
        candidate = replace(self, **normalized)
        if validate:
            candidate.validate()
        return candidate

    def validate(self, *, runnable: bool = False) -> None:
        if int(self.trial_id) < 1:
            raise ValueError("trial_id must be positive")
        for field in ("shift_x_mm", "shift_y_mm", "shift_z_mm"):
            _finite_shift(getattr(self, field), field=field)
        if self.position_mode is not PelletPositionMode.FIXED_MANUAL and any(
            getattr(self, field) != 0.0
            for field in ("shift_x_mm", "shift_y_mm", "shift_z_mm")
        ):
            raise ValueError("Manual XYZ offsets require fixed_manual position mode")
        if self.tone_phase is ActionPhase.NONE and self.tone_profile_id:
            raise ValueError("A tone profile requires a non-none tone phase")
        if self.tone_phase is not ActionPhase.NONE and not self.tone_profile_id:
            raise ValueError("A tone phase requires a tone profile")
        if self.laser_trigger_route is LaserTriggerRoute.NONE:
            if self.laser_profile_id or self.laser_phase is not ActionPhase.NONE:
                raise ValueError("Laser actions require a trigger route")
        elif not self.laser_profile_id or self.laser_phase is ActionPhase.NONE:
            raise ValueError("Laser trigger route requires profile and phase")
        if self.stimulus_trigger is StimulusTrigger.PRE_REVEAL:
            if self.pre_reveal_ms <= 0:
                raise ValueError("Pre-reveal trigger requires a positive delay")
            if self.cover_policy is not CoverPolicy.REVEAL:
                raise ValueError("Pre-reveal trigger requires Reveal cover policy")
            if self.laser_trigger_route is not LaserTriggerRoute.HARDWARE_STIM3:
                raise ValueError(
                    "Pre-reveal trigger requires the Hardware STIM3 route"
                )
        elif self.pre_reveal_ms:
            raise ValueError("pre_reveal_ms applies only to pre_reveal trigger")
        if self.stimulus_assignment is StimulusAssignment.DISABLED:
            if self.stimulus_trigger is not StimulusTrigger.NONE:
                raise ValueError("Disabled stimulus assignment cannot have a trigger")
        elif self.stimulus_trigger is StimulusTrigger.NONE:
            raise ValueError("Enabled stimulus assignment requires a trigger")
        if (
            runnable
            and self.stimulus_trigger
            in {StimulusTrigger.ROI_1, StimulusTrigger.ROI_2}
        ):
            raise ValueError("ROI1/ROI2 triggers are reserved for a future release")
        if (
            runnable
            and self.laser_trigger_route is LaserTriggerRoute.DIRECT_NI_SOFTWARE
            and self.stimulus_trigger is not StimulusTrigger.FIRST_REACH
        ):
            raise ValueError(
                "Direct NI software start currently requires the First Reach trigger"
            )
        if (
            runnable
            and self.stimulus_trigger
            in {StimulusTrigger.TONE_1, StimulusTrigger.TONE_2}
            and self.laser_trigger_route is not LaserTriggerRoute.HARDWARE_STIM3
        ):
            raise ValueError(
                "Tone-triggered stimulation requires a verified hardware trigger input"
            )
        if (
            runnable
            and
            self.stimulus_assignment is not StimulusAssignment.DISABLED
            and not self.laser_profile_id
        ):
            raise ValueError("Enabled stimulation requires a laser profile")
        if runnable:
            if not self.enabled:
                raise ValueError("Protocol row is disabled")

    def to_record(self) -> dict:
        result = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = value.value if isinstance(value, Enum) else value
        return result


@dataclass(frozen=True)
class ProtocolPatch:
    """Validated partial row update used by protocol scopes."""

    values: Tuple[Tuple[str, object], ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ProtocolPatch":
        if "trial_id" in values:
            raise ValueError("A protocol patch cannot change trial_id")
        normalized = TrialProtocolRow(trial_id=1).with_updates(
            values,
            validate=False,
        ).to_record()
        return cls(tuple(
            (key, normalized[key]) for key in sorted(values)
        ))

    def to_mapping(self) -> dict:
        return dict(self.values)

    def apply(
        self,
        row: TrialProtocolRow,
        *,
        validate: bool = False,
    ) -> TrialProtocolRow:
        return row.with_updates(self.to_mapping(), validate=validate)


@dataclass(frozen=True)
class ProtocolScope:
    """Named non-row scope: epoch, block, or bulk-selected trials."""

    name: str
    trial_ids: Tuple[int, ...]
    patch: ProtocolPatch
    parent_epoch: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        trial_ids: Sequence[int],
        values: Mapping[str, object],
        *,
        parent_epoch: str = "",
    ) -> "ProtocolScope":
        name = _identifier(name, field="scope name")
        trial_ids = tuple(sorted({int(value) for value in trial_ids}))
        if not trial_ids or trial_ids[0] < 1:
            raise ValueError("Protocol scope requires positive trial IDs")
        return cls(
            name=name,
            trial_ids=trial_ids,
            patch=ProtocolPatch.from_mapping(values),
            parent_epoch=(
                _identifier(parent_epoch, field="parent epoch", allow_empty=True)
            ),
        )

    def to_record(self) -> dict:
        return {
            "name": self.name,
            "trial_ids": list(self.trial_ids),
            "values": self.patch.to_mapping(),
            "parent_epoch": self.parent_epoch,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ProtocolScope":
        return cls.create(
            record["name"],
            record["trial_ids"],
            record.get("values", {}),
            parent_epoch=record.get("parent_epoch", ""),
        )


@dataclass(frozen=True)
class TrialOverride:
    trial_id: int
    patch: ProtocolPatch

    @classmethod
    def create(cls, trial_id: int, values: Mapping[str, object]):
        trial_id = int(trial_id)
        if trial_id < 1:
            raise ValueError("Trial override ID must be positive")
        return cls(trial_id, ProtocolPatch.from_mapping(values))

    def to_record(self) -> dict:
        return {"trial_id": self.trial_id, "values": self.patch.to_mapping()}


@dataclass(frozen=True)
class ResolvedTrialProtocolRow:
    row: TrialProtocolRow
    sources: Tuple[Tuple[str, str], ...]

    def to_record(self) -> dict:
        result = self.row.to_record()
        result["value_sources"] = dict(self.sources)
        return result


@dataclass(frozen=True)
class TrialProtocolDocument:
    protocol_id: str
    name: str
    revision: int = 1
    trial_count: int = DEFAULT_TRIAL_COUNT
    defaults: ProtocolPatch = ProtocolPatch()
    epochs: Tuple[ProtocolScope, ...] = ()
    blocks: Tuple[ProtocolScope, ...] = ()
    bulk_overrides: Tuple[ProtocolScope, ...] = ()
    trial_overrides: Tuple[TrialOverride, ...] = ()
    description: str = ""
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self, "protocol_id", _identifier(self.protocol_id, field="protocol_id")
        )
        if not str(self.name).strip():
            raise ValueError("Protocol name cannot be empty")
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported protocol schema {self.schema_version}; expected "
                f"{PROTOCOL_SCHEMA_VERSION}"
            )
        if int(self.revision) < 1:
            raise ValueError("Protocol revision must be positive")
        if not 1 <= int(self.trial_count) <= 100_000:
            raise ValueError("trial_count must be between 1 and 100000")
        self._validate_scopes()

    @classmethod
    def disabled_default(
        cls,
        *,
        count: int = DEFAULT_TRIAL_COUNT,
    ) -> "TrialProtocolDocument":
        return cls(
            protocol_id="disabled-default",
            name="Disabled example",
            trial_count=int(count),
            description="Safe disabled rows; select or create a protocol to run.",
        )

    def _validate_scopes(self) -> None:
        trial_limit = set(range(1, int(self.trial_count) + 1))
        epoch_names = set()
        epoch_members = set()
        epoch_by_name = {}
        for epoch in self.epochs:
            if epoch.parent_epoch:
                raise ValueError("Epoch cannot have parent_epoch")
            if epoch.name in epoch_names:
                raise ValueError(f"Duplicate epoch name: {epoch.name}")
            members = set(epoch.trial_ids)
            if not members <= trial_limit:
                raise ValueError(f"Epoch {epoch.name} contains out-of-range trials")
            overlap = members & epoch_members
            if overlap:
                raise ValueError(
                    f"Epoch {epoch.name} overlaps trials {sorted(overlap)}"
                )
            epoch_names.add(epoch.name)
            epoch_members.update(members)
            epoch_by_name[epoch.name] = members

        block_members: Dict[str, set] = {}
        block_names = set()
        for block in self.blocks:
            if block.name in block_names:
                raise ValueError(f"Duplicate block name: {block.name}")
            if block.parent_epoch not in epoch_by_name:
                raise ValueError(
                    f"Block {block.name} has unknown parent epoch "
                    f"{block.parent_epoch!r}"
                )
            members = set(block.trial_ids)
            if not members <= epoch_by_name[block.parent_epoch]:
                raise ValueError(
                    f"Block {block.name} contains trials outside parent epoch"
                )
            overlap = members & block_members.setdefault(block.parent_epoch, set())
            if overlap:
                raise ValueError(
                    f"Block {block.name} overlaps trials {sorted(overlap)}"
                )
            block_members[block.parent_epoch].update(members)
            block_names.add(block.name)

        for scope in self.bulk_overrides:
            if not set(scope.trial_ids) <= trial_limit:
                raise ValueError(
                    f"Bulk scope {scope.name} contains out-of-range trials"
                )
        seen_trials = set()
        for override in self.trial_overrides:
            if override.trial_id not in trial_limit:
                raise ValueError("Trial override is outside protocol range")
            if override.trial_id in seen_trials:
                raise ValueError(
                    f"Duplicate individual override for trial {override.trial_id}"
                )
            seen_trials.add(override.trial_id)

    def resolve(self) -> Tuple[ResolvedTrialProtocolRow, ...]:
        rows = {}
        sources = {}
        for trial_id in range(1, int(self.trial_count) + 1):
            row = TrialProtocolRow(trial_id=trial_id)
            row, row_sources = self._apply_patch(
                row,
                {},
                self.defaults,
                "protocol",
            )
            rows[trial_id] = row
            sources[trial_id] = row_sources

        for prefix, scopes in (
            ("epoch", self.epochs),
            ("block", self.blocks),
            ("bulk", self.bulk_overrides),
        ):
            for scope in scopes:
                for trial_id in scope.trial_ids:
                    rows[trial_id], sources[trial_id] = self._apply_patch(
                        rows[trial_id],
                        sources[trial_id],
                        scope.patch,
                        f"{prefix}:{scope.name}",
                    )
        for override in self.trial_overrides:
            trial_id = override.trial_id
            rows[trial_id], sources[trial_id] = self._apply_patch(
                rows[trial_id],
                sources[trial_id],
                override.patch,
                f"trial:{trial_id}",
            )
        resolved = []
        for trial_id in sorted(rows):
            rows[trial_id].validate()
            resolved.append(ResolvedTrialProtocolRow(
                row=rows[trial_id],
                sources=tuple(sorted(sources[trial_id].items())),
            ))
        return tuple(resolved)

    @staticmethod
    def _apply_patch(row, sources, patch, label):
        values = patch.to_mapping()
        if not values:
            return row, dict(sources)
        row = patch.apply(row)
        sources = dict(sources)
        for field in values:
            sources[field] = label
        return row, sources

    def to_record(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "trial_count": self.trial_count,
            "defaults": self.defaults.to_mapping(),
            "epochs": [item.to_record() for item in self.epochs],
            "blocks": [item.to_record() for item in self.blocks],
            "bulk_overrides": [item.to_record() for item in self.bulk_overrides],
            "trial_overrides": [item.to_record() for item in self.trial_overrides],
            "resolved_rows": [item.to_record() for item in self.resolve()],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]):
        return cls(
            schema_version=int(record.get("schema_version", -1)),
            protocol_id=record["protocol_id"],
            name=record["name"],
            description=record.get("description", ""),
            revision=int(record.get("revision", 1)),
            trial_count=int(record.get("trial_count", DEFAULT_TRIAL_COUNT)),
            defaults=ProtocolPatch.from_mapping(record.get("defaults", {})),
            epochs=tuple(
                ProtocolScope.from_record(item) for item in record.get("epochs", ())
            ),
            blocks=tuple(
                ProtocolScope.from_record(item) for item in record.get("blocks", ())
            ),
            bulk_overrides=tuple(
                ProtocolScope.from_record(item)
                for item in record.get("bulk_overrides", ())
            ),
            trial_overrides=tuple(
                TrialOverride.create(item["trial_id"], item.get("values", {}))
                for item in record.get("trial_overrides", ())
            ),
        )


class TrialProtocolSchedule:
    """Thread-safe expanded schedule used by the application runtime."""

    EDITABLE_FIELDS = frozenset(
        item.name for item in fields(TrialProtocolRow) if item.name != "trial_id"
    )

    def __init__(
        self,
        rows: Iterable[TrialProtocolRow] = (),
        *,
        document: Optional[TrialProtocolDocument] = None,
        sources: Optional[Mapping[int, Mapping[str, str]]] = None,
    ):
        self._lock = threading.RLock()
        self._document = document
        self._rows = {int(row.trial_id): row for row in rows}
        self._sources = {
            int(key): dict(value) for key, value in (sources or {}).items()
        }

    @classmethod
    def from_document(cls, document: TrialProtocolDocument):
        resolved = document.resolve()
        return cls(
            (item.row for item in resolved),
            document=document,
            sources={item.row.trial_id: dict(item.sources) for item in resolved},
        )

    @classmethod
    def with_placeholder_rows(cls, count: int = DEFAULT_TRIAL_COUNT):
        return cls.from_document(
            TrialProtocolDocument.disabled_default(count=int(count))
        )

    @property
    def document(self) -> Optional[TrialProtocolDocument]:
        return self._document

    @property
    def rows(self) -> Tuple[TrialProtocolRow, ...]:
        with self._lock:
            return tuple(self._rows[key] for key in sorted(self._rows))

    def row(self, trial_id: int) -> TrialProtocolRow:
        with self._lock:
            trial_id = int(trial_id)
            if trial_id not in self._rows:
                self._rows[trial_id] = TrialProtocolRow(trial_id=trial_id)
            return self._rows[trial_id]

    def update(self, trial_id: int, field: str, value) -> TrialProtocolRow:
        with self._lock:
            if field not in self.EDITABLE_FIELDS:
                raise ValueError(f"Unknown trial protocol field: {field}")
            row = self.row(trial_id).with_updates({field: value})
            self._rows[row.trial_id] = row
            self._sources.setdefault(row.trial_id, {})[field] = (
                f"trial:{row.trial_id}"
            )
            return row

    def to_records(self) -> Tuple[dict, ...]:
        with self._lock:
            result = []
            for row in self.rows:
                record = row.to_record()
                record["value_sources"] = dict(self._sources.get(row.trial_id, {}))
                result.append(record)
            return tuple(result)
