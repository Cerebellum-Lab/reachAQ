"""Log calls in the main window format their arguments."""

import logging
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from reachaq.app.model.app_model import AppModelStatus
from reachaq.app.view.main_window import MainWindow


def test_activation_outside_idle_logs_the_status(caplog):
    window = SimpleNamespace(
        main_content=SimpleNamespace(on_activated=lambda: None),
        _app_model=SimpleNamespace(
            status=AppModelStatus.RUNNING,
            loaded_configuration=object(),
        ),
    )
    caplog.set_level(1)

    MainWindow.on_activated(window)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "not starting acquisition" in message and str(AppModelStatus.RUNNING) in message
        for message in messages
    ), messages
