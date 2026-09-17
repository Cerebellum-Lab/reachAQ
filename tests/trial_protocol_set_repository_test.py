import json

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
        name="Baseline",
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )


def test_a_saved_set_reloads_from_disk(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set())

    reloaded = TrialProtocolSetRepository(tmp_path)
    reloaded.reload()

    assert reloaded.get("baseline").name == "Baseline"


def test_saving_increments_the_revision(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()

    first = repository.save(make_set())
    second = repository.save(make_set())

    assert first.revision == 1
    assert second.revision == 2


def test_library_maps_ids_to_sets(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set("baseline"))
    repository.save(make_set("stim"))

    library = repository.library()

    assert sorted(library) == ["baseline", "stim"]


def test_a_corrupt_file_is_reported_and_does_not_lose_the_cached_set(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set())
    (tmp_path / "baseline.json").write_text("{ not json", encoding="utf-8")

    repository.reload()

    assert repository.errors
    assert repository.get("baseline") is not None


def test_a_filename_that_disagrees_with_its_id_is_refused(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()
    repository.save(make_set())
    record = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    (tmp_path / "wrong-name.json").write_text(json.dumps(record), encoding="utf-8")

    repository.reload()

    assert any(
        "filename must be" in message for message in repository.errors.values()
    )


def test_an_unknown_id_returns_none(tmp_path):
    repository = TrialProtocolSetRepository(tmp_path)
    repository.reload()

    assert repository.get("missing") is None


def test_a_saved_experiment_reloads_from_disk(tmp_path):
    repository = ExperimentCompositionRepository(tmp_path)
    repository.reload()
    repository.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )

    reloaded = ExperimentCompositionRepository(tmp_path)
    reloaded.reload()

    assert reloaded.get("day3").entries[0].set_id == "baseline"


def test_app_model_compiles_an_experiment_into_the_protocol_library(
    app_model, tmp_path
):
    app_model._trial_protocol_set_repository = TrialProtocolSetRepository(
        tmp_path / "sets"
    )
    app_model._trial_protocol_set_repository.reload()
    app_model._trial_protocol_set_repository.save(make_set())
    app_model._experiment_repository = ExperimentCompositionRepository(
        tmp_path / "experiments"
    )
    app_model._experiment_repository.reload()
    app_model._experiment_repository.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=2),
            ),
        )
    )

    saved = app_model.compile_and_save_experiment("day3")

    assert saved.protocol_id == "day3"
    assert saved.trial_count == 4
    assert app_model._trial_protocol_repository.get("day3") is not None


def test_compiling_an_unknown_experiment_is_refused(app_model, tmp_path):
    app_model._experiment_repository = ExperimentCompositionRepository(
        tmp_path / "experiments"
    )
    app_model._experiment_repository.reload()

    try:
        app_model.compile_and_save_experiment("missing")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("compiling an unknown experiment should be refused")


def test_a_drawn_shuffle_seed_is_persisted_so_the_compile_reproduces(
    app_model, tmp_path
):
    sets = TrialProtocolSetRepository(tmp_path / "sets")
    sets.reload()
    sets.save(make_set())
    experiments = ExperimentCompositionRepository(tmp_path / "experiments")
    experiments.reload()
    experiments.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(
                    set_id="baseline", set_revision=1, shuffle_trials=True
                ),
            ),
        )
    )
    app_model._trial_protocol_set_repository = sets
    app_model._experiment_repository = experiments

    app_model.compile_and_save_experiment("day3")

    stored = experiments.get("day3")
    assert stored.entries[0].shuffle_seed is not None
