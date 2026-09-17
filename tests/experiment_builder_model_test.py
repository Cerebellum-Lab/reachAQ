import pytest

from tools.acquisition.model.trial_protocol_schedule import ProtocolPatch
from tools.acquisition.model.trial_protocol_set import (
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)
from tools.acquisition.model.trial_protocol_set_repository import (
    ExperimentCompositionRepository,
    TrialProtocolSetRepository,
)


def make_set(set_id="baseline"):
    return TrialProtocolSet(
        set_id=set_id,
        name=set_id.title(),
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )


@pytest.fixture
def stocked(app_model, tmp_path):
    sets = TrialProtocolSetRepository(tmp_path / "sets")
    sets.reload()
    sets.save(make_set("baseline"))
    sets.save(make_set("stim"))
    experiments = ExperimentCompositionRepository(tmp_path / "experiments")
    experiments.reload()
    app_model._trial_protocol_set_repository = sets
    app_model._experiment_repository = experiments
    return app_model


def test_with_entries_replaces_the_list():
    composition = ExperimentComposition(experiment_id="day3", name="Day 3")

    updated = composition.with_entries(
        (ExperimentSetEntry(set_id="baseline", set_revision=1),)
    )

    assert composition.entries == ()
    assert updated.entries[0].set_id == "baseline"
    assert updated.experiment_id == "day3"


def test_creating_an_experiment_starts_it_empty(stocked):
    created = stocked.create_experiment("day3", "Day 3")

    assert created.experiment_id == "day3"
    assert created.entries == ()
    assert stocked._experiment_repository.get("day3") is not None


def test_creating_a_duplicate_experiment_is_refused(stocked):
    stocked.create_experiment("day3", "Day 3")

    with pytest.raises(ValueError, match="already exists"):
        stocked.create_experiment("day3", "Day 3 again")


def test_saving_entries_replaces_them_and_bumps_the_revision(stocked):
    created = stocked.create_experiment("day3", "Day 3")

    saved = stocked.save_experiment_entries(
        "day3",
        (
            ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=2),
            ExperimentSetEntry(
                set_id="stim", set_revision=1, shuffle_trials=True
            ),
        ),
    )

    assert [entry.set_id for entry in saved.entries] == ["baseline", "stim"]
    assert saved.entries[0].repeat == 2
    assert saved.entries[1].shuffle_trials is True
    assert saved.revision == created.revision + 1


def test_saving_entries_onto_an_unknown_experiment_is_refused(stocked):
    with pytest.raises(ValueError, match="missing"):
        stocked.save_experiment_entries("missing", ())


def test_an_experiment_built_this_way_compiles(stocked):
    stocked.create_experiment("day3", "Day 3")
    stocked.save_experiment_entries(
        "day3",
        (ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=3),),
    )

    compiled = stocked.compile_and_save_experiment("day3")

    assert compiled.trial_count == 6


def test_entries_can_be_cleared(stocked):
    stocked.create_experiment("day3", "Day 3")
    stocked.save_experiment_entries(
        "day3", (ExperimentSetEntry(set_id="baseline", set_revision=1),)
    )

    saved = stocked.save_experiment_entries("day3", ())

    assert saved.entries == ()
