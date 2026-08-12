import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

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
        self.trial_protocol_state = {
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

    def update_trial_protocol_row(self, trial_id, field, value):
        self.updated.append((trial_id, field, value))
        return True

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
    assert content._table.item(1, 8).text() == "Active"
    assert content._table.item(0, 8).text() == "Completed"
    assert not content._retry_analysis.isVisible()
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
