import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.model.trial_protocol_schedule import ProtocolPatch  # noqa: E402
from tools.acquisition.model.trial_protocol_set import (  # noqa: E402
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)
from tools.acquisition.model.trial_protocol_set_repository import (  # noqa: E402
    ExperimentCompositionRepository,
    TrialProtocolSetRepository,
)
from tools.acquisition.view.protocol_set_sidebar import (  # noqa: E402
    ProtocolSetSidebar,
)


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def stocked_app_model(app_model, tmp_path):
    sets = TrialProtocolSetRepository(tmp_path / "sets")
    sets.reload()
    sets.save(
        TrialProtocolSet(
            set_id="baseline",
            name="Baseline",
            trial_count=2,
            defaults=ProtocolPatch.from_mapping({"enabled": True}),
        )
    )
    experiments = ExperimentCompositionRepository(tmp_path / "experiments")
    experiments.reload()
    experiments.save(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )
    app_model._trial_protocol_set_repository = sets
    app_model._experiment_repository = experiments
    return app_model


def test_the_sidebar_lists_saved_sets(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)

    labels = [
        sidebar.set_list.item(index).text()
        for index in range(sidebar.set_list.count())
    ]

    assert any("Baseline" in label for label in labels)


def test_the_sidebar_lists_saved_experiments(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)

    labels = [
        sidebar.experiment_list.item(index).text()
        for index in range(sidebar.experiment_list.count())
    ]

    assert any("Day 3" in label for label in labels)


def test_selecting_a_set_reports_its_identity(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    seen = []
    sidebar.selection_changed.connect(lambda kind, key: seen.append((kind, key)))

    sidebar.set_list.setCurrentRow(0)

    assert seen[-1] == ("set", "baseline")


def test_selecting_an_experiment_clears_the_set_selection(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.set_list.setCurrentRow(0)

    sidebar.experiment_list.setCurrentRow(0)

    assert sidebar.current_selection() == ("experiment", "day3")


def test_compiling_publishes_the_experiment_as_a_protocol(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.experiment_list.setCurrentRow(0)

    sidebar.compile_selected_experiment()

    assert stocked_app_model._trial_protocol_repository.get("day3") is not None


def test_compiling_reports_what_it_produced(qapp, stocked_app_model):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.experiment_list.setCurrentRow(0)

    sidebar.compile_selected_experiment()

    assert "2 trials" in sidebar.status_label.text()


def test_compiling_with_no_experiment_selected_reports_rather_than_raises(
    qapp, stocked_app_model
):
    sidebar = ProtocolSetSidebar(stocked_app_model)
    sidebar.experiment_list.setCurrentRow(-1)

    sidebar.compile_selected_experiment()

    assert "Select an experiment" in sidebar.status_label.text()


def test_a_compile_failure_is_reported_rather_than_raised(qapp, stocked_app_model):
    # Pin a revision the library does not hold, so the compiler refuses.
    stocked_app_model._experiment_repository.save(
        ExperimentComposition(
            experiment_id="broken",
            name="Broken",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=99),),
        )
    )
    sidebar = ProtocolSetSidebar(stocked_app_model)
    for index in range(sidebar.experiment_list.count()):
        item = sidebar.experiment_list.item(index)
        if item.data(Qt.ItemDataRole.UserRole) == "broken":
            sidebar.experiment_list.setCurrentRow(index)
            break

    sidebar.compile_selected_experiment()

    assert "revision" in sidebar.status_label.text()
