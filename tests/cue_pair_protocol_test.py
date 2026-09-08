import pytest

from autotrainer.core.delay_distribution import (
    MINIMUM_CUE_INTERVAL_MS,
    PUBLISHED_4S_VALUES,
    DelayPreset,
)
from tools.acquisition.model.stimulus_profile_repository import (
    PROFILE_SCHEMA_VERSION,
    StimulusProfileLibrary,
)
from tools.acquisition.model.trial_action import (
    CueIntervalProfile,
    LaserPulseProfile,
    ToneProfile,
    TrialActionCompiler,
    TrialCompileContext,
)
from tools.acquisition.model.trial_protocol_schedule import (
    PROTOCOL_SCHEMA_VERSION,
    TrialProtocolRow,
)


TONE_1 = ToneProfile("tone-1", 1, 5_000, 100)
TONE_2 = ToneProfile("tone-2", 1, 6_000, 100)
PUBLISHED = CueIntervalProfile("published", 1, preset="published_4s")
CUSTOM = CueIntervalProfile(
    "custom", 1, preset="custom", values=(305, 600, 1200), tau_ms=1400
)


def _context(**changes):
    values = dict(
        session_id="session001",
        session_generation=4,
        protocol_id="p",
        protocol_revision=2,
        logical_trial_id=1,
        attempt_id=1,
        session_seed=42,
        animal_base_dcs=(10.0, 20.0, 30.0),
        lane_offsets_dcs={
            "center": (0.0, 0.0, 0.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
        },
    )
    values.update(changes)
    return TrialCompileContext(**values)


def _compiler():
    return TrialActionCompiler(
        tone_profiles={"tone-1": TONE_1, "tone-2": TONE_2},
        laser_profiles={
            "pulse": LaserPulseProfile(
                "pulse", 3, 1, 2.5, 5.0, trigger_terminal="/Dev4/PFI0"
            )
        },
        cue_interval_profiles={"published": PUBLISHED, "custom": CUSTOM},
        dcs_to_motor=lambda values: tuple(value * 2 for value in values),
    )


def _cue_row(trial_id=1, **changes):
    values = {
        "enabled": True,
        "tone_profile_id": "tone-1",
        "tone_phase": "pellet_presentation",
        "cue_tone_profile_id": "tone-2",
        "cue_interval_profile_id": "published",
    }
    values.update(changes)
    return TrialProtocolRow(trial_id=trial_id).with_updates(values)


def test_protocol_schema_version_is_bumped_for_the_cue_pair():
    assert PROTOCOL_SCHEMA_VERSION == 2
    assert PROFILE_SCHEMA_VERSION >= 3


def test_a_row_without_a_cue_tone_is_unchanged():
    row = TrialProtocolRow(trial_id=1)
    assert row.cue_tone_profile_id == ""
    assert row.cue_interval_profile_id == ""
    assert row.cue_interval_fixed_ms == 0
    row.validate()


def test_cue_tone_requires_a_tone_1_profile():
    with pytest.raises(ValueError, match="requires a Tone 1 tone profile"):
        TrialProtocolRow(trial_id=1).with_updates({
            "cue_tone_profile_id": "tone-2",
            "cue_interval_fixed_ms": 900,
        })


def test_cue_tone_requires_an_interval_source():
    with pytest.raises(ValueError, match="requires a cue interval profile"):
        TrialProtocolRow(trial_id=1).with_updates({
            "tone_profile_id": "tone-1",
            "tone_phase": "pellet_presentation",
            "cue_tone_profile_id": "tone-2",
        })


def test_cue_tone_rejects_two_interval_sources():
    with pytest.raises(ValueError, match="not both"):
        _cue_row(cue_interval_fixed_ms=900)


def test_cue_interval_without_a_cue_tone_is_rejected():
    with pytest.raises(ValueError, match="requires a cue tone profile"):
        TrialProtocolRow(trial_id=1).with_updates({"cue_interval_fixed_ms": 900})


def test_tone_1_and_tone_2_must_differ():
    with pytest.raises(ValueError, match="different tone profiles"):
        _cue_row(cue_tone_profile_id="tone-1")


@pytest.mark.parametrize("interval", [MINIMUM_CUE_INTERVAL_MS - 1, 60_001])
def test_fixed_interval_bounds_are_enforced(interval):
    with pytest.raises(ValueError, match="cue_interval_fixed_ms"):
        TrialProtocolRow(trial_id=1).with_updates({
            "tone_profile_id": "tone-1",
            "tone_phase": "pellet_presentation",
            "cue_tone_profile_id": "tone-2",
            "cue_interval_fixed_ms": interval,
        })


def test_compile_resolves_a_fixed_cue_interval_without_a_draw():
    row = _cue_row(cue_interval_profile_id="", cue_interval_fixed_ms=900)
    recipe = _compiler().compile(row, _context())
    assert recipe.cue_tone_profile == TONE_2
    assert recipe.cue_interval_ms == 900
    assert recipe.cue_interval_seed is None
    assert recipe.cue_interval_draw is None
    assert recipe.cue_interval_selection is None


def test_compile_draws_a_cue_interval_from_the_distribution():
    recipe = _compiler().compile(_cue_row(), _context())
    assert recipe.cue_interval_ms in PUBLISHED_4S_VALUES
    assert 0.0 <= recipe.cue_interval_draw < 1.0
    assert recipe.cue_interval_selection["delay_ms"] == recipe.cue_interval_ms
    assert recipe.cue_interval_distribution["preset"] == "published_4s"


def test_cue_interval_draw_is_reproducible_for_the_same_trial():
    first = _compiler().compile(_cue_row(), _context())
    second = _compiler().compile(_cue_row(), _context())
    assert first.cue_interval_seed == second.cue_interval_seed
    assert first.cue_interval_ms == second.cue_interval_ms


def test_cue_interval_draw_is_independent_of_the_stimulus_draw():
    recipe = _compiler().compile(_cue_row(), _context())
    assert recipe.cue_interval_seed != recipe.stimulus_seed
    assert recipe.cue_interval_draw != recipe.stimulus_draw


def test_cue_interval_varies_across_logical_trials():
    compiler = _compiler()
    drawn = {
        compiler.compile(
            _cue_row(trial_id=trial), _context(logical_trial_id=trial)
        ).cue_interval_ms
        for trial in range(1, 40)
    }
    assert len(drawn) > 1
    assert drawn.issubset(set(PUBLISHED_4S_VALUES))


def test_compile_rejects_an_unknown_cue_tone_profile():
    compiler = TrialActionCompiler(
        tone_profiles={"tone-1": TONE_1},
        cue_interval_profiles={"published": PUBLISHED},
        dcs_to_motor=lambda values: values,
    )
    with pytest.raises(ValueError, match="Unknown cue tone profile"):
        compiler.compile(_cue_row(), _context())


def test_compile_rejects_an_unknown_cue_interval_profile():
    with pytest.raises(ValueError, match="Unknown cue interval profile"):
        _compiler().compile(_cue_row(cue_interval_profile_id="missing"), _context())


def test_custom_cue_interval_profile_uses_its_own_support():
    recipe = _compiler().compile(_cue_row(cue_interval_profile_id="custom"), _context())
    assert recipe.cue_interval_ms in (305, 600, 1200)
    assert recipe.cue_interval_distribution["preset"] == "custom"


def test_cue_interval_profile_requires_intervals_for_a_custom_preset():
    with pytest.raises(ValueError, match="requires cue intervals"):
        CueIntervalProfile("empty", 1, preset="custom")


def test_cue_interval_profile_defaults_to_the_published_support():
    assert PUBLISHED.distribution().values == PUBLISHED_4S_VALUES
    assert PUBLISHED.distribution().preset is DelayPreset.PUBLISHED_4S


def test_cue_interval_profile_rejects_bad_identity():
    with pytest.raises(ValueError, match="ID cannot be empty"):
        CueIntervalProfile("", 1)
    with pytest.raises(ValueError, match="revision must be positive"):
        CueIntervalProfile("a", 0)


def test_recipe_record_carries_the_cue_pair():
    record = _compiler().compile(_cue_row(), _context()).to_record()
    assert record["cue_tone_profile"]["profile_id"] == "tone-2"
    assert record["cue_interval_ms"] in PUBLISHED_4S_VALUES
    assert record["cue_interval_distribution"]["source"].startswith("paper_table_4s")
    assert record["cue_interval_selection"]["index"] >= 0


def test_recipe_record_omits_the_cue_pair_when_unused():
    row = TrialProtocolRow(trial_id=1).with_updates({"enabled": True})
    record = _compiler().compile(row, _context()).to_record()
    assert record["cue_tone_profile"] is None
    assert record["cue_interval_ms"] is None
    assert record["cue_interval_distribution"] is None


def test_profile_library_round_trips_cue_interval_profiles():
    library = StimulusProfileLibrary(
        tone_profiles=(TONE_1, TONE_2),
        cue_interval_profiles=(PUBLISHED, CUSTOM),
    )
    restored = StimulusProfileLibrary.from_record(library.to_record())
    assert restored.cue_interval_profiles == (PUBLISHED, CUSTOM)


def test_profile_library_rejects_duplicate_cue_interval_ids():
    with pytest.raises(ValueError, match="Duplicate cue interval profile ID"):
        StimulusProfileLibrary(cue_interval_profiles=(PUBLISHED, PUBLISHED))


def test_profile_library_reads_a_record_without_cue_interval_profiles():
    record = StimulusProfileLibrary(tone_profiles=(TONE_1,)).to_record()
    record.pop("cue_interval_profiles")
    assert StimulusProfileLibrary.from_record(record).cue_interval_profiles == ()
