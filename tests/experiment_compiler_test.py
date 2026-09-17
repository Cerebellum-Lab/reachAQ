import pytest

from tools.acquisition.model.experiment_compiler import compile_experiment
from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    ProtocolScope,
    TrialOverride,
)
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)


def make_library():
    baseline = TrialProtocolSet(
        set_id="baseline",
        name="Baseline",
        revision=1,
        trial_count=2,
        defaults=ProtocolPatch.from_mapping(
            {"enabled": True, "automatic_window_size": 10}
        ),
    )
    stim = TrialProtocolSet(
        set_id="stim",
        name="Stim",
        revision=3,
        trial_count=3,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
        bulk_overrides=(
            ProtocolScope.create("tail", [2, 3], {"automatic_window_size": 40}),
        ),
        trial_overrides=(TrialOverride.create(1, {"automatic_window_size": 99}),),
    )
    return {"baseline": baseline, "stim": stim}


def compose(*entries, **overrides):
    values = dict(experiment_id="day3", name="Day 3", entries=entries)
    values.update(overrides)
    return ExperimentComposition(**values)


def test_entries_concatenate_in_order():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="stim", set_revision=3),
    )

    result = compile_experiment(composition, make_library())

    assert result.document.trial_count == 5
    assert [item.epoch_name for item in result.instances] == ["baseline", "stim"]
    assert result.instances[0].trial_ids == (1, 2)
    assert result.instances[1].trial_ids == (3, 4, 5)


def test_repeat_expands_into_separately_named_instances():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=3)
    )

    result = compile_experiment(composition, make_library())

    assert result.document.trial_count == 6
    assert [item.epoch_name for item in result.instances] == [
        "baseline-1",
        "baseline-2",
        "baseline-3",
    ]


def test_the_same_set_in_two_entries_gets_distinct_epoch_names():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="stim", set_revision=3),
        ExperimentSetEntry(set_id="baseline", set_revision=1),
    )

    result = compile_experiment(composition, make_library())

    names = [item.epoch_name for item in result.instances]
    assert names == ["baseline-1", "stim", "baseline-2"]
    assert len(set(names)) == len(names)


def test_within_set_precedence_survives_compilation():
    composition = compose(ExperimentSetEntry(set_id="stim", set_revision=3))

    result = compile_experiment(composition, make_library())
    rows = result.document.resolve()

    # Trial 1 takes the set's own trial override, which beats the bulk scope.
    assert rows[0].row.automatic_window_size == 99
    # Trials 2 and 3 take the bulk scope, which beats the set defaults.
    assert rows[1].row.automatic_window_size == 40
    assert rows[2].row.automatic_window_size == 40


def test_each_trial_reports_its_set_as_the_source_of_the_set_defaults():
    composition = compose(ExperimentSetEntry(set_id="baseline", set_revision=1))

    result = compile_experiment(composition, make_library())
    sources = dict(result.document.resolve()[0].sources)

    assert sources["enabled"] == "epoch:baseline"


def test_shuffling_is_reproducible_from_the_stored_seed():
    composition = compose(
        ExperimentSetEntry(
            set_id="stim", set_revision=3, shuffle_trials=True, shuffle_seed=11
        )
    )
    library = make_library()

    first = compile_experiment(composition, library)
    second = compile_experiment(composition, library)

    def overridden_trials(result):
        return [item.trial_id for item in result.document.trial_overrides]

    assert overridden_trials(first) == overridden_trials(second)


def test_shuffling_without_a_seed_draws_one_and_returns_it():
    composition = compose(
        ExperimentSetEntry(set_id="stim", set_revision=3, shuffle_trials=True)
    )

    result = compile_experiment(composition, make_library())

    seed = result.composition.entries[0].shuffle_seed
    assert seed is not None
    assert result.instances[0].shuffle_seed == seed


def test_shuffling_permutes_within_the_instance_range_only():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(
            set_id="stim", set_revision=3, shuffle_trials=True, shuffle_seed=5
        ),
    )

    result = compile_experiment(composition, make_library())

    assert result.instances[1].trial_ids == (3, 4, 5)
    for override in result.document.trial_overrides:
        assert 3 <= override.trial_id <= 5


def test_an_unshuffled_instance_keeps_the_sets_own_order():
    composition = compose(ExperimentSetEntry(set_id="stim", set_revision=3))

    result = compile_experiment(composition, make_library())

    # The set's override is on its local trial 1, so unshuffled it lands first.
    assert [item.trial_id for item in result.document.trial_overrides] == [1]


def test_a_revision_mismatch_names_the_offending_entry():
    composition = compose(ExperimentSetEntry(set_id="stim", set_revision=2))

    with pytest.raises(ValueError, match="Entry 1 pins set 'stim' revision 2"):
        compile_experiment(composition, make_library())


def test_an_unknown_set_names_the_offending_entry():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="missing", set_revision=1),
    )

    with pytest.raises(ValueError, match="Entry 2 references unknown set 'missing'"):
        compile_experiment(composition, make_library())


def test_an_empty_experiment_refuses_to_compile():
    with pytest.raises(ValueError, match="no set entries"):
        compile_experiment(compose(), make_library())


def test_the_compiled_document_keeps_the_experiment_identity():
    composition = compose(ExperimentSetEntry(set_id="baseline", set_revision=1))

    result = compile_experiment(composition, make_library())

    assert result.document.protocol_id == "day3"
    assert result.document.name == "Day 3"


def test_the_compiled_document_describes_where_it_came_from():
    composition = compose(ExperimentSetEntry(set_id="stim", set_revision=3))

    result = compile_experiment(composition, make_library())

    assert "day3" in result.document.description
    assert "stim rev 3" in result.document.description


def test_bulk_scopes_are_remapped_into_the_instance_range():
    composition = compose(
        ExperimentSetEntry(set_id="baseline", set_revision=1),
        ExperimentSetEntry(set_id="stim", set_revision=3),
    )

    result = compile_experiment(composition, make_library())

    tail = [
        scope
        for scope in result.document.bulk_overrides
        if scope.name.endswith("tail")
    ]
    assert len(tail) == 1
    # The stim set's local trials 2 and 3 become global 4 and 5.
    assert tail[0].trial_ids == (4, 5)
