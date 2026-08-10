import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget

from autotrainer.core.configuration import SessionControlConfiguration
from tools.acquisition.view.preferences_content import PreferencesContent


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_counted_trial_outcomes_are_operator_configurable(qapp):
    config = SessionControlConfiguration()
    harness = QWidget()
    harness._app_model = SimpleNamespace(
        set_automatic_protocol_advance_enabled=lambda _enabled: None,
    )
    algorithm = SimpleNamespace(
        active_config=SimpleNamespace(session_control=config),
    )

    group = PreferencesContent._create_session_control_group(
        harness,
        algorithm,
    )
    checkboxes = {
        checkbox.text(): checkbox
        for checkbox in group.findChildren(QCheckBox)
        if checkbox.text()
    }

    assert set(checkboxes) == {
        "Success",
        "Failed reach",
        "Pellet missing",
        "Incomplete trial",
        "Aborted trial",
    }
    assert checkboxes["Failed reach"].isChecked()
    checkboxes["Failed reach"].setChecked(False)
    assert "failure" not in config.counted_trial_outcomes
    checkboxes["Incomplete trial"].setChecked(True)
    assert "incomplete" in config.counted_trial_outcomes

    group.close()
    harness.close()
