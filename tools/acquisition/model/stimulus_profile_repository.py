"""Atomic persistence for reusable tone, cue interval and laser profiles."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

from tools.acquisition.model.atomic_session_io import atomic_write_json
from tools.acquisition.model.automatic_pellet_shift import AutomaticShiftPolicy
from tools.acquisition.model.trial_action import (
    CueIntervalProfile,
    LaserPulseProfile,
    ToneProfile,
)


PROFILE_SCHEMA_VERSION = 3


def _cue_interval_profile(record: Mapping[str, object]) -> CueIntervalProfile:
    """Rebuild a cue interval profile, restoring its tuple fields."""

    values = record.get("values") or ()
    manual = record.get("manual_probabilities")
    return CueIntervalProfile(
        profile_id=str(record["profile_id"]),
        revision=int(record["revision"]),
        preset=str(record.get("preset", "published_4s")),
        values=tuple(int(item) for item in values),
        tau_ms=int(record.get("tau_ms", 1400)),
        manual_probabilities=(
            None if manual is None else tuple(float(item) for item in manual)
        ),
    )


@dataclass(frozen=True)
class StimulusProfileLibrary:
    revision: int = 1
    tone_profiles: Tuple[ToneProfile, ...] = ()
    cue_interval_profiles: Tuple[CueIntervalProfile, ...] = ()
    laser_profiles: Tuple[LaserPulseProfile, ...] = ()
    automatic_shift_profiles: Tuple[AutomaticShiftPolicy, ...] = (
        AutomaticShiftPolicy(),
    )
    schema_version: int = PROFILE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported stimulus-profile schema {self.schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        if int(self.revision) < 1:
            raise ValueError("Stimulus-profile library revision must be positive")
        for kind, profiles in (
            ("tone", self.tone_profiles),
            ("cue interval", self.cue_interval_profiles),
            ("laser", self.laser_profiles),
            ("automatic shift", self.automatic_shift_profiles),
        ):
            identifiers = [
                getattr(profile, "profile_id", getattr(profile, "policy_id", ""))
                for profile in profiles
            ]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Duplicate {kind} profile ID")

    @classmethod
    def safe_default(cls):
        return cls(tone_profiles=(
            ToneProfile("tone-1", 1, 5_000, 100),
            ToneProfile("tone-2", 1, 6_000, 100),
        ))

    @classmethod
    def from_record(cls, record: Mapping[str, object]):
        schema_version = int(record.get("schema_version", 0))
        if schema_version not in {1, 2, PROFILE_SCHEMA_VERSION}:
            raise ValueError(
                f"Unsupported stimulus-profile schema {schema_version}"
            )
        return cls(
            schema_version=PROFILE_SCHEMA_VERSION,
            revision=int(record.get("revision", 0)),
            tone_profiles=tuple(
                ToneProfile(**item) for item in record.get("tone_profiles", ())
            ),
            cue_interval_profiles=tuple(
                _cue_interval_profile(item)
                for item in record.get("cue_interval_profiles", ())
            ),
            laser_profiles=tuple(
                LaserPulseProfile(**item)
                for item in record.get("laser_profiles", ())
            ),
            automatic_shift_profiles=tuple(
                AutomaticShiftPolicy.from_record(item)
                for item in record.get(
                    "automatic_shift_profiles",
                    (AutomaticShiftPolicy().to_record(),),
                )
            ),
        )

    def to_record(self):
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "tone_profiles": [profile.to_record() for profile in self.tone_profiles],
            "cue_interval_profiles": [
                profile.to_record() for profile in self.cue_interval_profiles
            ],
            "laser_profiles": [profile.to_record() for profile in self.laser_profiles],
            "automatic_shift_profiles": [
                profile.to_record() for profile in self.automatic_shift_profiles
            ],
        }


class StimulusProfileRepository:
    """Own one versioned profile library without hiding malformed input."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._library = StimulusProfileLibrary.safe_default()
        self._error = ""

    @property
    def library(self):
        with self._lock:
            return self._library

    @property
    def error(self):
        with self._lock:
            return self._error

    def load(self):
        with self._lock:
            if not self.path.exists():
                self._library = StimulusProfileLibrary.safe_default()
                self._error = ""
                return self._library
            try:
                with self.path.open("r", encoding="utf-8") as stream:
                    library = StimulusProfileLibrary.from_record(json.load(stream))
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                return self._library
            self._library = library
            self._error = ""
            return library

    def save(self, library: StimulusProfileLibrary, *, expected_revision: int):
        with self._lock:
            if self._library.revision != int(expected_revision):
                raise RuntimeError(
                    "Stimulus-profile library changed: expected revision "
                    f"{expected_revision}, found {self._library.revision}"
                )
            candidate = StimulusProfileLibrary(
                revision=self._library.revision + 1,
                tone_profiles=tuple(library.tone_profiles),
                laser_profiles=tuple(library.laser_profiles),
                automatic_shift_profiles=tuple(library.automatic_shift_profiles),
            )
            atomic_write_json(self.path, candidate.to_record())
            self._library = candidate
            self._error = ""
            return candidate
