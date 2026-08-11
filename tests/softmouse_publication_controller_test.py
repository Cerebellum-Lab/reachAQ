import os
import logging
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tools.acquisition.model.app_model_status import SessionRecordingStatus  # noqa: E402
from tools.acquisition.view.softmouse_publication_controller import (  # noqa: E402
    SoftMousePublicationController,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class AppModelStub:
    session_recording_status = SessionRecordingStatus.READY

    def __init__(self):
        self.refreshes = 0

    def refresh_animal_metadata(self):
        self.refreshes += 1


def _wait_for_process(controller, timeout_ms=5000):
    loop = QEventLoop()
    controller.running_changed.connect(
        lambda running: None if running else loop.quit()
    )
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    assert not controller.is_running


def test_successful_publication_refreshes_local_cache(qapp, tmp_path, caplog):
    caplog.set_level(
        logging.INFO,
        logger="tools.acquisition.view.softmouse_publication_controller",
    )
    model = AppModelStub()
    controller = SoftMousePublicationController(
        model,
        program=sys.executable,
        arguments=("-c", "print('Published test snapshot')"),
        working_directory=tmp_path,
    )

    assert controller.start()
    _wait_for_process(controller)

    assert model.refreshes == 1
    assert controller.status == "Published test snapshot; local cache refreshed"
    assert any(
        "SoftMouse manual sync complete" in entry.getMessage()
        for entry in caplog.records
    )


def test_failed_publication_preserves_error_and_does_not_refresh(qapp, tmp_path):
    model = AppModelStub()
    controller = SoftMousePublicationController(
        model,
        program=sys.executable,
        arguments=(
            "-c",
            "import sys; print('publication rejected', file=sys.stderr); sys.exit(1)",
        ),
        working_directory=tmp_path,
    )

    assert controller.start()
    _wait_for_process(controller)

    assert model.refreshes == 0
    assert controller.status == "publication rejected"


def test_publication_is_refused_during_recording(qapp, tmp_path):
    model = AppModelStub()
    model.session_recording_status = SimpleNamespace(value="recording")
    controller = SoftMousePublicationController(
        model,
        program=sys.executable,
        arguments=("-c", "raise SystemExit(99)"),
        working_directory=tmp_path,
    )

    assert not controller.start()
    assert controller.status == "SoftMouse sync is unavailable during recording"
