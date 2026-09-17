import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.model.trial_protocol_schedule import ProtocolPatch  # noqa: E402
from tools.acquisition.model.trial_protocol_set import (  # noqa: E402
    ExperimentComposition,
    ExperimentSetEntry,
    TrialProtocolSet,
)
from tools.acquisition.view.experiment_entry_editor import (  # noqa: E402
    ExperimentEntryEditor,
)


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_set(set_id, revision=1):
    return TrialProtocolSet(
        set_id=set_id,
        name=set_id.title(),
        revision=revision,
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )


@pytest.fixture
def editor(qapp):
    widget = ExperimentEntryEditor()
    widget.set_available_sets((make_set("baseline"), make_set("stim", revision=3)))
    return widget


def test_an_editor_with_no_experiment_has_no_rows(editor):
    assert editor.entries() == ()


def test_loading_an_experiment_shows_its_entries(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(set_id="baseline", set_revision=1, repeat=2),
                ExperimentSetEntry(
                    set_id="stim", set_revision=3, shuffle_trials=True
                ),
            ),
        )
    )

    entries = editor.entries()

    assert [item.set_id for item in entries] == ["baseline", "stim"]
    assert entries[0].repeat == 2
    assert entries[1].shuffle_trials is True


def test_adding_an_entry_pins_the_sets_current_revision(editor):
    editor.load(ExperimentComposition(experiment_id="day3", name="Day 3"))

    editor.add_entry("stim")

    entries = editor.entries()
    assert len(entries) == 1
    assert entries[0].set_id == "stim"
    assert entries[0].set_revision == 3


def test_adding_defaults_to_one_repeat_and_no_shuffle(editor):
    editor.load(ExperimentComposition(experiment_id="day3", name="Day 3"))

    editor.add_entry("baseline")

    assert editor.entries()[0].repeat == 1
    assert editor.entries()[0].shuffle_trials is False


def test_removing_the_selected_entry_drops_it(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(set_id="baseline", set_revision=1),
                ExperimentSetEntry(set_id="stim", set_revision=3),
            ),
        )
    )
    editor.table.setCurrentCell(0, 0)

    editor.remove_selected()

    assert [item.set_id for item in editor.entries()] == ["stim"]


def test_moving_an_entry_down_reorders_it(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(set_id="baseline", set_revision=1),
                ExperimentSetEntry(set_id="stim", set_revision=3),
            ),
        )
    )
    editor.table.setCurrentCell(0, 0)

    editor.move_selected(1)

    assert [item.set_id for item in editor.entries()] == ["stim", "baseline"]


def test_moving_an_entry_up_reorders_it(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(
                ExperimentSetEntry(set_id="baseline", set_revision=1),
                ExperimentSetEntry(set_id="stim", set_revision=3),
            ),
        )
    )
    editor.table.setCurrentCell(1, 0)

    editor.move_selected(-1)

    assert [item.set_id for item in editor.entries()] == ["stim", "baseline"]


def test_moving_past_the_end_does_nothing(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )
    editor.table.setCurrentCell(0, 0)

    editor.move_selected(1)

    assert [item.set_id for item in editor.entries()] == ["baseline"]


def test_editing_the_repeat_cell_is_read_back(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )

    editor.table.cellWidget(0, 1).setValue(4)

    assert editor.entries()[0].repeat == 4


def test_toggling_shuffle_is_read_back(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="baseline", set_revision=1),),
        )
    )

    editor.table.cellWidget(0, 2).setChecked(True)

    assert editor.entries()[0].shuffle_trials is True


def test_an_entry_whose_set_has_moved_on_keeps_its_pinned_revision(editor):
    editor.load(
        ExperimentComposition(
            experiment_id="day3",
            name="Day 3",
            entries=(ExperimentSetEntry(set_id="stim", set_revision=1),),
        )
    )

    # The library holds revision 3; the entry was built against 1 and must not
    # be silently repinned by the editor.
    assert editor.entries()[0].set_revision == 1


def test_changes_are_announced(editor):
    editor.load(ExperimentComposition(experiment_id="day3", name="Day 3"))
    seen = []
    editor.changed.connect(lambda: seen.append(1))

    editor.add_entry("baseline")

    assert seen
