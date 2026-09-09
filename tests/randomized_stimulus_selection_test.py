import pytest

from tools.acquisition.model.stimulus_profile_repository import (
    PROFILE_SCHEMA_VERSION,
    StimulusProfileLibrary,
)
from tools.acquisition.model.trial_action import (
    CueIntervalProfile,
    LaserPulseProfile,
    StimulusTriggerProfile,
    ToneProfile,
    TrialActionCompiler,
    TrialCompileContext,
)
from tools.acquisition.model.trial_protocol_schedule import (
    StimulusTrigger,
    TrialProtocolRow,
)


TONE_1 = ToneProfile("tone-1", 1, 5_000, 100)
TONE_2 = ToneProfile("tone-2", 1, 6_000, 100)
CUE = CueIntervalProfile("published", 1, preset="published_4s")
LASER = LaserPulseProfile(
    "pulse", 3, 1, 2.5, 5.0, trigger_terminal="/Dev4/PFI0", trigger_pulse_us=1000
)

MIXED = StimulusTriggerProfile(
    "mixed",
    1,
    categories=(
        {
            "category_id": "first_reach",
            "trigger": "first_reach",
            "label": "First reach",
            "percentage": 50.0,
        },
        {
            "category_id": "pre_tone2_100",
            "trigger": "tone_2",
            "label": "100 ms before Tone 2",
            "percentage": 50.0,
            "offset_ms": 100,
        },
    ),
)
FIRST_REACH_ONLY = StimulusTriggerProfile(
    "first_reach_only",
    1,
    categories=(
        {
            "category_id": "first_reach",
            "trigger": "first_reach",
            "percentage": 100.0,
        },
    ),
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
        laser_profiles={"pulse": LASER},
        cue_interval_profiles={"published": CUE},
        stimulus_trigger_profiles={
            "mixed": MIXED,
            "first_reach_only": FIRST_REACH_ONLY,
        },
        dcs_to_motor=lambda values: tuple(value * 2 for value in values),
    )


def _row(trial_id=1, **changes):
    values = {
        "enabled": True,
        "tone_profile_id": "tone-1",
        "tone_phase": "pellet_presentation",
        "cue_tone_profile_id": "tone-2",
        "cue_interval_profile_id": "published",
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "stimulus_assignment": "randomized",
        "stimulus_trigger_profile_id": "mixed",
    }
    values.update(changes)
    return TrialProtocolRow(trial_id=trial_id).with_updates(values)


def test_randomized_requires_a_trigger_profile():
    with pytest.raises(ValueError, match="requires a trigger profile"):
        _row(stimulus_trigger_profile_id="")


def test_randomized_rejects_a_single_trigger():
    with pytest.raises(ValueError, match="leave the single trigger unset"):
        _row(stimulus_trigger="first_reach")


def test_trigger_profile_is_rejected_outside_randomized():
    with pytest.raises(ValueError, match="only to randomized assignment"):
        _row(
            stimulus_assignment="percentage",
            stimulus_trigger="first_reach",
            stimulus_probability_percent=50.0,
        )


def test_disabled_assignment_rejects_a_trigger_profile():
    with pytest.raises(ValueError, match="cannot have a trigger profile"):
        TrialProtocolRow(trial_id=1).with_updates({
            "stimulus_assignment": "disabled",
            "stimulus_trigger_profile_id": "mixed",
        })


def test_randomized_no_longer_collapses_to_a_single_percentage():
    # Before this change RANDOMIZED behaved exactly like PERCENTAGE and could
    # not express more than one trigger.
    recipe = _compiler().compile(_row(), _context())
    assert recipe.stimulus_selected is True
    assert recipe.resolved_stimulus_trigger in {"first_reach", "tone_2"}
    assert recipe.stimulus_trigger_selection is not None


def test_selection_uses_an_independent_seed_stream():
    recipe = _compiler().compile(_row(), _context())
    assert recipe.stimulus_trigger_seed not in {
        recipe.stimulus_seed,
        recipe.cue_interval_seed,
    }
    assert 0.0 <= recipe.stimulus_trigger_draw < 1.0


def test_selection_is_reproducible_for_the_same_trial():
    first = _compiler().compile(_row(), _context())
    second = _compiler().compile(_row(), _context())
    assert first.resolved_stimulus_trigger == second.resolved_stimulus_trigger
    assert first.stimulus_trigger_seed == second.stimulus_trigger_seed


def test_both_categories_are_reachable_across_trials():
    compiler = _compiler()
    drawn = {
        compiler.compile(
            _row(trial_id=trial), _context(logical_trial_id=trial)
        ).resolved_stimulus_trigger
        for trial in range(1, 40)
    }
    assert drawn == {"first_reach", "tone_2"}


def test_offset_is_resolved_from_the_selected_category():
    compiler = _compiler()
    for trial in range(1, 40):
        recipe = compiler.compile(
            _row(trial_id=trial), _context(logical_trial_id=trial)
        )
        if recipe.resolved_stimulus_trigger == "tone_2":
            assert recipe.resolved_trigger_offset_ms == 100
        else:
            assert recipe.resolved_trigger_offset_ms == 0


def test_unselected_trials_resolve_to_no_trigger():
    row = _row(
        stimulus_assignment="percentage",
        stimulus_trigger="first_reach",
        stimulus_trigger_profile_id="",
        stimulus_probability_percent=0.0,
    )
    recipe = _compiler().compile(row, _context())
    assert recipe.stimulus_selected is False
    assert recipe.laser_profile is None


def test_randomized_unselected_trial_clears_the_trigger():
    compiler = _compiler()
    row = _row(stimulus_probability_percent=0.0)
    # Randomized still honours the percentage gate before drawing a category.
    recipe = compiler.compile(row, _context())
    assert recipe.stimulus_selected is False
    assert recipe.resolved_stimulus_trigger == "none"
    assert recipe.resolved_trigger_offset_ms == 0
    assert recipe.stimulus_trigger_selection is None


def test_tone_2_offset_must_fit_inside_the_shortest_cue_interval():
    long_offset = StimulusTriggerProfile(
        "long",
        1,
        categories=(
            {
                "category_id": "pre_tone2",
                "trigger": "tone_2",
                "percentage": 100.0,
                "offset_ms": 400,
            },
        ),
    )
    compiler = TrialActionCompiler(
        tone_profiles={"tone-1": TONE_1, "tone-2": TONE_2},
        laser_profiles={"pulse": LASER},
        cue_interval_profiles={"published": CUE},
        stimulus_trigger_profiles={"long": long_offset},
        dcs_to_motor=lambda values: values,
    )
    # The published 4 s support starts at 305 ms, so a 400 ms lead cannot fire.
    with pytest.raises(ValueError, match="shorter than the minimum cue interval"):
        compiler.compile(_row(stimulus_trigger_profile_id="long"), _context())


def test_tone_2_trigger_requires_a_cue_tone():
    row = _row(cue_tone_profile_id="", cue_interval_profile_id="")
    with pytest.raises(ValueError, match="require a configured cue tone"):
        _compiler().compile(row, _context())


def test_unknown_trigger_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown stimulus trigger profile"):
        _compiler().compile(_row(stimulus_trigger_profile_id="missing"), _context())


def test_offset_is_rejected_for_a_trigger_without_a_lead_time():
    with pytest.raises(ValueError, match="do not take a lead time"):
        StimulusTriggerProfile(
            "bad",
            1,
            categories=(
                {
                    "category_id": "first_reach",
                    "trigger": "first_reach",
                    "percentage": 100.0,
                    "offset_ms": 50,
                },
            ),
        )


def test_profile_rejects_weights_that_do_not_total_100():
    with pytest.raises(ValueError, match="total 60.000%"):
        StimulusTriggerProfile(
            "bad",
            1,
            categories=(
                {
                    "category_id": "first_reach",
                    "trigger": "first_reach",
                    "percentage": 60.0,
                },
            ),
        )


def test_profile_rejects_bad_identity():
    with pytest.raises(ValueError, match="ID cannot be empty"):
        StimulusTriggerProfile("", 1, categories=FIRST_REACH_ONLY.categories)
    with pytest.raises(ValueError, match="revision must be positive"):
        StimulusTriggerProfile("a", 0, categories=FIRST_REACH_ONLY.categories)


def test_recipe_record_carries_the_trigger_selection():
    record = _compiler().compile(_row(), _context()).to_record()
    assert record["resolved_stimulus_trigger"] in {"first_reach", "tone_2"}
    assert record["stimulus_trigger_selection"]["sampling_algorithm"] == (
        "weighted_categorical_with_replacement_v1"
    )


def test_non_randomized_rows_keep_their_declared_trigger():
    row = _row(
        stimulus_assignment="always",
        stimulus_trigger="first_reach",
        stimulus_trigger_profile_id="",
        laser_trigger_route="direct_ni_software",
        laser_profile_id="direct",
    )
    compiler = TrialActionCompiler(
        tone_profiles={"tone-1": TONE_1, "tone-2": TONE_2},
        laser_profiles={
            "direct": LaserPulseProfile(
                "direct", 1, 1, 2.5, 5.0, trigger_route="direct_ni_software"
            )
        },
        cue_interval_profiles={"published": CUE},
        dcs_to_motor=lambda values: values,
    )
    recipe = compiler.compile(row, _context())
    assert recipe.resolved_stimulus_trigger == StimulusTrigger.FIRST_REACH.value
    assert recipe.stimulus_trigger_selection is None


def test_library_round_trips_stimulus_trigger_profiles():
    assert PROFILE_SCHEMA_VERSION == 4
    library = StimulusProfileLibrary(
        tone_profiles=(TONE_1, TONE_2),
        stimulus_trigger_profiles=(MIXED, FIRST_REACH_ONLY),
    )
    restored = StimulusProfileLibrary.from_record(library.to_record())
    assert restored.stimulus_trigger_profiles == (MIXED, FIRST_REACH_ONLY)


def test_library_reads_a_record_without_trigger_profiles():
    record = StimulusProfileLibrary(tone_profiles=(TONE_1,)).to_record()
    record.pop("stimulus_trigger_profiles")
    assert StimulusProfileLibrary.from_record(record).stimulus_trigger_profiles == ()
