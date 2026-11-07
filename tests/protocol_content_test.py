import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox

from tools.acquisition.view.protocol_content import ProtocolContent


class _Event:
    def __init__(self):
        self.callbacks = []

    def __iadd__(self, callback):
        self.callbacks.append(callback)
        return self


class _Model:
    Props = SimpleNamespace(TRIAL_PROTOCOL_STATE="trial_protocol_state")

    def __init__(self):
        self.property_changed = _Event()
        self.updated = []
        # The protocol panel now carries the set and experiment sidebar, so any
        # model it is handed has to answer for those libraries too.
        self.trial_protocol_sets = ()
        self.experiment_compositions = ()
        self.selected_ordered_protocol = None
        self.trial_protocol_state = {
            "selected_protocol": {
                "protocol_id": "test",
                "name": "Test",
                "revision": 1,
            },
            "protocols": ({
                "protocol_id": "test",
                "name": "Test",
                "revision": 1,
                "trial_count": 3,
            },),
            "tone_profiles": ({
                "profile_id": "tone-1",
                "revision": 1,
                "summary": "5000 Hz, 100 ms",
            },),
            "laser_profiles": (),
            "rows": (
                {
                    "trial_id": trial_id,
                    "pellet_behavior": "Standard",
                    "shift_x_mm": 0.0,
                    "shift_y_mm": 0.0,
                    "shift_z_mm": 0.0,
                    "cover": True,
                    "tone": "None",
                    "laser": "None",
                }
                for trial_id in (1, 2, 3)
            ),
            "active_trial_id": 2,
            "completed_trial_ids": (1,),
            "analysis": {},
        }
        self.laser = SimpleNamespace(configuration=SimpleNamespace(channels=(
            SimpleNamespace(channel_id=1, trigger_source="/Dev1/PXI_Trig0", board_stim_line=3),
            SimpleNamespace(channel_id=2, trigger_source="/Dev1/PXI_Trig2", board_stim_line=2),
        )))

    def update_trial_protocol_row(self, trial_id, field, value):
        self.updated.append((trial_id, field, value))
        return True

    def apply_ordered_protocol_values(
        self,
        trial_ids,
        values,
        *,
        scope_kind,
        scope_name="",
        parent_epoch="",
    ):
        self.updated.append((tuple(trial_ids), values, scope_kind, scope_name))
        return {
            "changed_trial_ids": tuple(trial_ids),
            "skipped_trial_ids": (),
        }

    def select_ordered_protocol(self, protocol_id):
        return protocol_id

    def retry_pending_intertrial_analysis(self):
        return True

    def continue_without_pending_intertrial_result(self):
        return True


def test_only_future_protocol_rows_are_editable():
    app = QApplication.instance() or QApplication([])
    model = _Model()
    content = ProtocolContent(model)
    app.processEvents()

    completed = content._table.item(0, 1)
    active = content._table.item(1, 1)
    future = content._table.item(2, 1)

    assert not completed.flags() & Qt.ItemFlag.ItemIsEditable
    assert not active.flags() & Qt.ItemFlag.ItemIsEditable
    assert future.flags() & Qt.ItemFlag.ItemIsEditable
    state_column = next(
        index for index, column in enumerate(content.COLUMNS)
        if column.label == "State"
    )
    assert content._table.item(1, state_column).text() == "Active"
    assert content._table.item(0, state_column).text() == "Completed"
    assert not content._retry_analysis.isVisible()
    content.close()


def test_protocol_table_uses_constrained_editors():
    app = QApplication.instance() or QApplication([])
    content = ProtocolContent(_Model())

    delivery_delegate = content._table.itemDelegateForColumn(2)
    delivery_editor = delivery_delegate.createEditor(content, None, None)
    shift_delegate = content._table.itemDelegateForColumn(5)
    shift_editor = shift_delegate.createEditor(content, None, None)

    assert isinstance(delivery_editor, QComboBox)
    assert isinstance(shift_editor, QDoubleSpinBox)
    tone_column = next(
        index for index, column in enumerate(content.COLUMNS)
        if column.field == "tone_profile_id"
    )
    tone_delegate = content._table.itemDelegateForColumn(tone_column)
    tone_editor = tone_delegate.createEditor(content, None, None)
    assert tone_editor.findData("tone-1") >= 0
    content.close()


def test_bulk_fill_reports_selected_trial_ids():
    app = QApplication.instance() or QApplication([])
    model = _Model()
    content = ProtocolContent(model)
    content._table.selectRow(2)
    content._table.setCurrentCell(2, 2)

    content._fill_selected_field()

    assert model.updated[-1][0] == (3,)
    assert model.updated[-1][2] == "bulk"
    content.close()


def test_analysis_resolution_actions_are_shown_only_when_needed():
    app = QApplication.instance() or QApplication([])
    model = _Model()
    model.trial_protocol_state["analysis"] = {
        "enabled": True,
        "pending_attempts": 1,
        "estimate": "estimating",
        "resolution_required": True,
        "resolution_reason": "analysis worker failed",
    }
    content = ProtocolContent(model)
    content.show()
    app.processEvents()

    assert content._retry_analysis.isVisible()
    assert content._continue_without_analysis.isVisible()
    assert "analysis worker failed" in content._analysis_status.text()
    content.close()


@pytest.fixture
def content():
    QApplication.instance() or QApplication([])
    return ProtocolContent(_Model())


def test_the_default_laser_is_the_first_a_board_stim_action_can_fire(content):
    assert content._laser_channel_options() == (1, 2)
    assert content._default_laser_channel() == 1


def test_the_protocol_table_has_a_laser_column_beside_the_profile(content):
    labels = [column.label for column in content.COLUMNS]

    assert labels.index("Laser") == labels.index("Laser profile") + 1


def test_the_protocol_tab_has_no_new_laser_button(content):
    from PySide6.QtWidgets import QToolButton
    texts = {button.text() for button in content.findChildren(QToolButton)}

    assert "New laser" not in texts
    assert not hasattr(content, "_new_laser_profile")


def test_picking_a_profile_on_a_row_without_a_laser_completes_the_action(content, monkeypatch):
    monkeypatch.setattr(content, "_default_laser_channel", lambda: 2)

    patch = content._laser_patch(
        "laser_profile_id", "pulse", {"laser_trigger_route": "none"})

    assert patch == {
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": 2,
    }


def test_clearing_the_profile_clears_the_whole_laser_action(content):
    patch = content._laser_patch("laser_profile_id", "", {"laser_trigger_route": "hardware_stim3"})

    assert patch == {
        "laser_profile_id": "",
        "laser_phase": "none",
        "laser_trigger_route": "none",
        "laser_channel_id": 0,
    }


def test_filling_a_laser_field_copies_the_rows_whole_laser_action(content):
    source = {
        "laser_profile_id": "pulse",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "direct_ni_software",
        "laser_channel_id": 1,
    }

    assert content._laser_values_from(source, "laser_channel_id") == source
    assert content._laser_values_from(source, "tone_profile_id") is None


def test_the_laser_column_shows_the_laser(content):
    assert content._display_value("laser_channel_id", 0) == "None"
    assert content._display_value("laser_channel_id", 2) == "Laser 2"
