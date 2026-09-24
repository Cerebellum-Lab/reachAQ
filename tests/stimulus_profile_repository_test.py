import json

import pytest

from tools.acquisition.model.stimulus_profile_repository import (
    PROFILE_SCHEMA_VERSION,
    StimulusProfileLibrary,
    StimulusProfileRepository,
)
from tools.acquisition.model.trial_action import (
    CueIntervalProfile,
    LaserPulseProfile,
    StimulusTriggerProfile,
    ToneProfile,
)


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
            laser_profiles=(LaserPulseProfile("pulse-a", 1, 2.5, 5.0),),
        ),
        expected_revision=original.revision,
    )

    loaded = StimulusProfileRepository(repository.path).load()
    assert loaded == saved
    assert loaded.revision == 2
    assert loaded.laser_profiles[0].profile_id == "pulse-a"
    assert loaded.automatic_shift_profiles[0].policy_id == "default"


def test_profile_library_migrates_schema_one_with_safe_shift_policy(tmp_path):
    path = tmp_path / "stimulus_profiles.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "revision": 3,
        "tone_profiles": [],
        "laser_profiles": [],
    }), encoding="utf-8")

    loaded = StimulusProfileRepository(path).load()

    assert loaded.schema_version == PROFILE_SCHEMA_VERSION
    assert loaded.revision == 3
    assert loaded.automatic_shift_profiles[0].policy_id == "default"


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


def test_a_schema_5_laser_profile_loads_as_its_pulse_train(tmp_path):
    path = tmp_path / "stimulus_profiles.json"
    path.write_text(json.dumps({
        "schema_version": 5,
        "revision": 7,
        "tone_profiles": [],
        "laser_profiles": [{
            "profile_id": "burst_100hz_5s_ch2", "revision": 4, "channel_id": 2,
            "amplitude_volts": 1.0, "pulse_duration_ms": 1.0, "pulse_count": 500,
            "frequency_hz": 100.0, "baseline_ms": 0.0, "post_stim_ms": 0.0,
            "pmt_open_lead_ms": 0.0, "pmt_close_lag_ms": 0.0,
            "trigger_route": "hardware_stim3", "trigger_terminal": "/PXI1Slot4/PXI_Trig2",
            "trigger_pulse_us": 1000, "stim_line": 2,
        }],
    }), encoding="utf-8")

    loaded = StimulusProfileRepository(path).load()

    assert loaded.schema_version == 6
    assert loaded.laser_profiles == (LaserPulseProfile(
        "burst_100hz_5s_ch2", 4, 1.0, 1.0, pulse_count=500, frequency_hz=100.0),)


def test_a_schema_6_laser_profile_with_routing_is_refused(tmp_path):
    path = tmp_path / "stimulus_profiles.json"
    record = StimulusProfileLibrary().to_record()
    record["laser_profiles"] = [dict(
        LaserPulseProfile("p", 1, 1.0, 1.0).to_record(), channel_id=1)]
    path.write_text(json.dumps(record), encoding="utf-8")

    repository = StimulusProfileRepository(path)
    repository.load()

    assert "channel_id" in repository.error


def test_saving_keeps_every_profile_kind(tmp_path):
    # save() rebuilt the library from tones, lasers and shift policies only,
    # so every save erased the cue interval and stimulus trigger profiles.
    repository = StimulusProfileRepository(tmp_path / "stimulus_profiles.json")
    original = repository.load()
    cue = CueIntervalProfile("published", 1, preset="published_4s")
    trigger = StimulusTriggerProfile("first", 1, categories=({
        "category_id": "first_reach", "trigger": "first_reach",
        "label": "First reach", "percentage": 100.0,
    },))

    saved = repository.save(
        StimulusProfileLibrary(
            revision=original.revision,
            tone_profiles=original.tone_profiles,
            cue_interval_profiles=(cue,),
            stimulus_trigger_profiles=(trigger,),
        ),
        expected_revision=original.revision,
    )

    assert saved.cue_interval_profiles == (cue,)
    assert saved.stimulus_trigger_profiles == (trigger,)
    assert StimulusProfileRepository(repository.path).load().cue_interval_profiles == (cue,)


def test_a_profile_summarises_its_train():
    profile = LaserPulseProfile("burst", 1, 1.0, 1.0, pulse_count=500, frequency_hz=100.0)

    assert profile.summary() == "1 V · 500 × 1 ms at 100 Hz · 4.99 s"
