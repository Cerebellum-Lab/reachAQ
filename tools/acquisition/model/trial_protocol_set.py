"""Reusable trial groups, and the experiments composed from them.

A set holds trial content only. It deliberately has no epochs and no blocks:
those slots belong to the compiled experiment, which uses one epoch per set
appearance to carry provenance through the existing resolution order.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from tools.acquisition.model.trial_protocol_schedule import (
    DEFAULT_TRIAL_COUNT,
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
    TrialProtocolDocument,
    normalize_identifier,
)


SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrialProtocolSet:
    """One reusable, independently saved group of trials."""

    set_id: str
    name: str
    revision: int = 1
    trial_count: int = DEFAULT_TRIAL_COUNT
    defaults: ProtocolPatch = ProtocolPatch()
    bulk_overrides: Tuple[ProtocolScope, ...] = ()
    trial_overrides: Tuple[TrialOverride, ...] = ()
    description: str = ""
    schema_version: int = SET_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self, "set_id", normalize_identifier(self.set_id, field="set_id")
        )
        if not str(self.name).strip():
            raise ValueError("Set name cannot be empty")
        if self.schema_version != SET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported set schema {}; expected {}".format(
                    self.schema_version, SET_SCHEMA_VERSION
                )
            )
        if int(self.revision) < 1:
            raise ValueError("Set revision must be positive")
        if not 1 <= int(self.trial_count) <= 100_000:
            raise ValueError("trial_count must be between 1 and 100000")
        object.__setattr__(self, "bulk_overrides", tuple(self.bulk_overrides))
        object.__setattr__(self, "trial_overrides", tuple(self.trial_overrides))
        self._validate()

    def _validate(self) -> None:
        limit = set(range(1, int(self.trial_count) + 1))
        for scope in self.bulk_overrides:
            if scope.parent_epoch:
                raise ValueError(
                    "Set scope {} cannot name a parent epoch; epochs belong to "
                    "the compiled experiment".format(scope.name)
                )
            if not set(scope.trial_ids) <= limit:
                raise ValueError(
                    "Set scope {} contains out-of-range trials".format(scope.name)
                )
        seen = set()
        for override in self.trial_overrides:
            if override.trial_id not in limit:
                raise ValueError("Set trial override is outside the set range")
            if override.trial_id in seen:
                raise ValueError(
                    "Duplicate trial override for trial {}".format(override.trial_id)
                )
            seen.add(override.trial_id)

    def to_record(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "set_id": self.set_id,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "trial_count": self.trial_count,
            "defaults": self.defaults.to_mapping(),
            "bulk_overrides": [item.to_record() for item in self.bulk_overrides],
            "trial_overrides": [item.to_record() for item in self.trial_overrides],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "TrialProtocolSet":
        stored = int(record.get("schema_version", -1))
        if stored != SET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported set schema {}; expected {}".format(
                    stored, SET_SCHEMA_VERSION
                )
            )
        return cls(
            schema_version=SET_SCHEMA_VERSION,
            set_id=record["set_id"],
            name=record["name"],
            description=record.get("description", ""),
            revision=int(record.get("revision", 1)),
            trial_count=int(record.get("trial_count", DEFAULT_TRIAL_COUNT)),
            defaults=ProtocolPatch.from_mapping(record.get("defaults", {})),
            bulk_overrides=tuple(
                ProtocolScope.from_record(item)
                for item in record.get("bulk_overrides", ())
            ),
            trial_overrides=tuple(
                TrialOverride.create(item["trial_id"], item.get("values", {}))
                for item in record.get("trial_overrides", ())
            ),
        )


EXPERIMENT_SCHEMA_VERSION = 1

#: Upper bound on how many times one set may appear through a single entry.
#: Large enough for any real session, small enough that a typo cannot compile a
#: million-trial document.
MAX_SET_REPEAT = 100


@dataclass(frozen=True)
class ExperimentSetEntry:
    """One appearance of a set inside an experiment, pinned to a revision."""

    set_id: str
    set_revision: int
    repeat: int = 1
    shuffle_trials: bool = False
    shuffle_seed: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(
            self, "set_id", normalize_identifier(self.set_id, field="set_id")
        )
        if int(self.set_revision) < 1:
            raise ValueError("Set revision must be positive")
        if not 1 <= int(self.repeat) <= MAX_SET_REPEAT:
            raise ValueError(
                "Set repeat must be between 1 and {}".format(MAX_SET_REPEAT)
            )
        if self.shuffle_seed is not None:
            object.__setattr__(self, "shuffle_seed", int(self.shuffle_seed))

    def to_record(self) -> dict:
        return {
            "set_id": self.set_id,
            "set_revision": int(self.set_revision),
            "repeat": int(self.repeat),
            "shuffle_trials": bool(self.shuffle_trials),
            "shuffle_seed": self.shuffle_seed,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ExperimentSetEntry":
        return cls(
            set_id=record["set_id"],
            set_revision=int(record["set_revision"]),
            repeat=int(record.get("repeat", 1)),
            shuffle_trials=bool(record.get("shuffle_trials", False)),
            shuffle_seed=record.get("shuffle_seed"),
        )


@dataclass(frozen=True)
class ExperimentComposition:
    """An ordered list of set appearances that compiles to one protocol."""

    experiment_id: str
    name: str
    revision: int = 1
    entries: Tuple[ExperimentSetEntry, ...] = ()
    description: str = ""
    schema_version: int = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self,
            "experiment_id",
            normalize_identifier(self.experiment_id, field="experiment_id"),
        )
        if not str(self.name).strip():
            raise ValueError("Experiment name cannot be empty")
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported experiment schema {}; expected {}".format(
                    self.schema_version, EXPERIMENT_SCHEMA_VERSION
                )
            )
        if int(self.revision) < 1:
            raise ValueError("Experiment revision must be positive")
        # An empty experiment is allowed so it can be built up in the editor.
        # compile_experiment refuses to compile one.
        object.__setattr__(self, "entries", tuple(self.entries))

    def with_entries(
        self, entries: Tuple["ExperimentSetEntry", ...]
    ) -> "ExperimentComposition":
        """Same experiment, different set list. Used by the builder."""
        return dataclasses.replace(self, entries=tuple(entries))

    def to_record(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "entries": [item.to_record() for item in self.entries],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "ExperimentComposition":
        stored = int(record.get("schema_version", -1))
        if stored != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported experiment schema {}; expected {}".format(
                    stored, EXPERIMENT_SCHEMA_VERSION
                )
            )
        return cls(
            schema_version=EXPERIMENT_SCHEMA_VERSION,
            experiment_id=record["experiment_id"],
            name=record["name"],
            description=record.get("description", ""),
            revision=int(record.get("revision", 1)),
            entries=tuple(
                ExperimentSetEntry.from_record(item)
                for item in record.get("entries", ())
            ),
        )


def set_from_document(
    document: TrialProtocolDocument,
    *,
    set_id: str,
    name: str,
) -> TrialProtocolSet:
    """Snapshot a protocol's trial content as a reusable set.

    Epochs and blocks are refused rather than flattened. Within a protocol they
    are the older, protocol-local grouping; a set is the cross-protocol one, and
    the compiler writes epochs itself. Flattening an epoch patch into the bulk
    layer would also change its precedence relative to blocks, so a silent
    conversion could alter what the trials actually do.
    """
    if document.epochs or document.blocks:
        raise ValueError(
            "Protocol {} uses epochs or blocks, which a set cannot hold. Express "
            "those groups as bulk overrides, or build the experiment from "
            "several sets instead.".format(document.protocol_id)
        )
    return TrialProtocolSet(
        set_id=set_id,
        name=name,
        trial_count=int(document.trial_count),
        defaults=document.defaults,
        bulk_overrides=tuple(document.bulk_overrides),
        trial_overrides=tuple(document.trial_overrides),
        description=document.description,
    )


def document_from_set(
    protocol_set: TrialProtocolSet,
    *,
    protocol_id: str,
    name: str,
) -> TrialProtocolDocument:
    """Materialise a set as an ordinary protocol so the editor can revise it."""
    return TrialProtocolDocument(
        protocol_id=protocol_id,
        name=name,
        trial_count=int(protocol_set.trial_count),
        defaults=protocol_set.defaults,
        bulk_overrides=tuple(protocol_set.bulk_overrides),
        trial_overrides=tuple(protocol_set.trial_overrides),
        description=protocol_set.description,
    )
