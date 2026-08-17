import json

import pytest

from tools.acquisition.model.stimulus_profile_repository import (
    StimulusProfileLibrary,
    StimulusProfileRepository,
)
from tools.acquisition.model.trial_action import LaserPulseProfile, ToneProfile


def test_profile_library_round_trip_and_revision(tmp_path):
    repository = StimulusProfileRepository(tmp_path / "stimulus_profiles.json")
    original = repository.load()
    assert tuple(profile.profile_id for profile in original.tone_profiles) == (
        "tone-1",
        "tone-2",
    )

    saved = repository.save(
        StimulusProfileLibrary(
            revision=original.revision,
            tone_profiles=(*original.tone_profiles, ToneProfile("cue", 1, 4000, 50)),
            laser_profiles=(LaserPulseProfile(
                "pulse-a", 1, 1, 2.5, 5.0,
                trigger_terminal="/Dev1/PFI0",
            ),),
        ),
        expected_revision=original.revision,
    )

    loaded = StimulusProfileRepository(repository.path).load()
    assert loaded == saved
    assert loaded.revision == 2
    assert loaded.laser_profiles[0].profile_id == "pulse-a"


def test_profile_library_preserves_last_good_value_on_corruption(tmp_path):
    repository = StimulusProfileRepository(tmp_path / "stimulus_profiles.json")
    first = repository.load()
    repository.path.write_text("{broken", encoding="utf-8")

    assert repository.load() == first
    assert "JSONDecodeError" in repository.error


def test_profile_library_uses_optimistic_revision(tmp_path):
    repository = StimulusProfileRepository(tmp_path / "stimulus_profiles.json")
    library = repository.load()
    with pytest.raises(RuntimeError, match="changed"):
        repository.save(library, expected_revision=library.revision + 1)


def test_profile_library_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate tone"):
        StimulusProfileLibrary(tone_profiles=(
            ToneProfile("same", 1, 1000, 10),
            ToneProfile("same", 2, 2000, 20),
        ))
