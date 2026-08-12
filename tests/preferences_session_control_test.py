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
        set_intertrial_analysis_enabled=lambda enabled: setattr(
            config, "intertrial_analysis_enabled", enabled
        ),
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
        "No reach",
        "Incomplete trial",
        "Aborted trial",
    }
    assert checkboxes["Failed reach"].isChecked()
    checkboxes["Failed reach"].setChecked(False)
    assert "failure" not in config.counted_trial_outcomes
    checkboxes["Incomplete trial"].setChecked(True)
    assert "incomplete" in config.counted_trial_outcomes

    missing_controls = [
        checkbox
        for checkbox in group.findChildren(QCheckBox)
        if checkbox.text() == "Pellet missing"
    ]
    no_reach_controls = [
        checkbox
        for checkbox in group.findChildren(QCheckBox)
        if checkbox.text() == "No reach"
    ]
    assert len(missing_controls) == 2
    assert all(checkbox.isEnabled() for checkbox in missing_controls)
    assert len(no_reach_controls) == 2
    assert sum(checkbox.isEnabled() for checkbox in no_reach_controls) == 1

    group.close()
    harness.close()
