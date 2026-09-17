"""Reusable trial groups, and the experiments composed from them.

A set holds trial content only. It deliberately has no epochs and no blocks:
those slots belong to the compiled experiment, which uses one epoch per set
appearance to carry provenance through the existing resolution order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from tools.acquisition.model.trial_protocol_schedule import (
    DEFAULT_TRIAL_COUNT,
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
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
