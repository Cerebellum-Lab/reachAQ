import pytest

from tools.acquisition.model.trial_protocol_set import (
    MAX_SET_REPEAT,
    ExperimentComposition,
    ExperimentSetEntry,
)


def make_composition(**overrides):
    values = dict(
        experiment_id="day3",
        name="Day 3",
        entries=(
            ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=2),
            ExperimentSetEntry(
                set_id="stim", set_revision=3, shuffle_trials=True, shuffle_seed=7
            ),
        ),
    )
    values.update(overrides)
    return ExperimentComposition(**values)


def test_a_composition_round_trips_through_its_record():
    original = make_composition(description="ABAB day")

    restored = ExperimentComposition.from_record(original.to_record())

    assert restored == original


def test_an_empty_composition_is_allowed_so_it_can_be_built_up():
    assert ExperimentComposition(
        experiment_id="draft", name="Draft", entries=()
    ).entries == ()


def test_an_entry_rejects_a_repeat_below_one():
    with pytest.raises(ValueError, match="repeat must be between"):
        ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=0)


def test_an_entry_rejects_a_repeat_above_the_ceiling():
    with pytest.raises(ValueError, match="repeat must be between"):
        ExperimentSetEntry(
            set_id="baseline", set_revision=1, repeat=MAX_SET_REPEAT + 1
        )


def test_an_entry_rejects_a_non_positive_set_revision():
    with pytest.raises(ValueError, match="revision must be positive"):
        ExperimentSetEntry(set_id="baseline", set_revision=0)


def test_an_entry_normalises_its_set_id():
    entry = ExperimentSetEntry(set_id="  BaseLine ", set_revision=1)

    assert entry.set_id == "baseline"


def test_experiment_id_is_normalised():
    assert make_composition(experiment_id="  Day3  ").experiment_id == "day3"


def test_a_composition_rejects_an_empty_name():
    with pytest.raises(ValueError, match="name cannot be empty"):
        make_composition(name="  ")


def test_a_composition_refuses_an_unknown_schema():
    record = make_composition().to_record()
    record["schema_version"] = 99

    with pytest.raises(ValueError, match="Unsupported experiment schema"):
        ExperimentComposition.from_record(record)
