"""Record asks before it starts when the cameras differ from the active calibration's.

Added, removed or swapped cameras ask "Record anyway?"; Yes acknowledges, then records, and
No records nothing. Calibrated cameras that are only not recording get a passive status-bar
note. Every camera that is not 3D-calibrated carries a "2D only" tag with the reason.
"""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

from reachaq.app.model.app_model import AppModel, SetupDifference  # noqa: E402
from reachaq.app.view import calibration_dialog as dialog_module  # noqa: E402
from reachaq.app.view import camera_content as camera_content_module  # noqa: E402
from reachaq.app.view.behavior_content import BehaviorContent  # noqa: E402
from reachaq.app.view.camera_content import CameraContent  # noqa: E402

TEXT = "Camera setup differs from the active calibration; a calibration looks needed. Record anyway?"
SWAP_REASON = "swapped: calibrated serial 21000001, now 21000099"


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def __call__(self, *args):
        for handler in list(self.handlers):
            handler(*args)


class _AppModel:
    Props = AppModel.Props

    def __init__(self, difference, reasons=None, started=True):
        self.difference = difference
        self.uncalibrated_reasons = dict(reasons or {})
        self.recording_camera_names = ["left", "right"]
        self.protocol_check_notice = None
        self.calls = []
        self._started = started

    def camera_setup_difference(self, selected):
        self.calls.append(("difference", list(selected)))
        return self.difference

    def check_reference_images(self, frames):
        return {}  # the reference-image check is off

    def acknowledge_setup_difference(self):
        self.calls.append("acknowledge")

    def discard_reference_check(self):
        self.calls.append("discard")

    def start_recording(self):
        self.calls.append("start")
        return self._started


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, timeout=0):
        self.messages.append(text)


def _behavior(app_model):
    status_bar = _StatusBar()
    window = SimpleNamespace(statusBar=lambda: status_bar)
    return SimpleNamespace(_app_model=app_model, window=lambda: window), status_bar


def _question(monkeypatch, answer):
    asked = []

    def question(parent, title, text, buttons=None, default=None):
        asked.append((title, text))
        return answer

    monkeypatch.setattr(dialog_module, "QMessageBox", SimpleNamespace(
        question=question, StandardButton=QMessageBox.StandardButton))
    return asked


@pytest.mark.parametrize("difference, detail", [
    (SetupDifference(added=["camera4"], removed=[], swapped=[], uncalibrated_off=[]),
     "Added, not in the calibration: camera4"),
    (SetupDifference(added=[], removed=["camera4"], swapped=[], uncalibrated_off=[]),
     "Removed, calibrated but no longer configured: camera4"),
    (SetupDifference(added=[], removed=[], swapped=["left"], uncalibrated_off=[]),
     "Swapped for another camera body: left, calibrated serial 21000001, now 21000099"),
], ids=["added", "removed", "swapped"])
def test_confirmation_shown_for_added_removed_swapped(monkeypatch, difference, detail):
    asked = _question(monkeypatch, QMessageBox.StandardButton.Yes)
    app_model = _AppModel(difference, {"left": SWAP_REASON})
    behavior, _ = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    (title, text), = asked
    assert title == "Calibration looks needed"
    assert text.startswith(TEXT)
    assert detail in text.splitlines()
    assert app_model.calls[0] == ("difference", ["left", "right"])


def test_subset_shows_only_the_note(monkeypatch):
    asked = _question(monkeypatch, QMessageBox.StandardButton.No)
    app_model = _AppModel(SetupDifference(added=[], removed=[], swapped=[], uncalibrated_off=["right"]))
    behavior, status_bar = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    assert asked == []
    assert app_model.calls[1:] == ["start"]
    assert status_bar.messages == ["Calibrated cameras not recording: right"]


def test_no_returns_without_recording(monkeypatch):
    _question(monkeypatch, QMessageBox.StandardButton.No)
    app_model = _AppModel(SetupDifference(added=["camera4"], removed=[], swapped=[], uncalibrated_off=[]))
    behavior, status_bar = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    assert "start" not in app_model.calls and "acknowledge" not in app_model.calls
    assert status_bar.messages == []


def test_yes_acknowledges_then_records(monkeypatch):
    _question(monkeypatch, QMessageBox.StandardButton.Yes)
    app_model = _AppModel(SetupDifference(added=[], removed=[], swapped=["left"], uncalibrated_off=[]),
                          {"left": SWAP_REASON})
    behavior, _ = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    assert app_model.calls[1:] == ["acknowledge", "start"]


def test_a_matching_setup_asks_nothing(monkeypatch):
    asked = _question(monkeypatch, QMessageBox.StandardButton.No)
    app_model = _AppModel(SetupDifference([], [], [], []))
    behavior, status_bar = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    assert asked == [] and app_model.calls[1:] == ["start"] and status_bar.messages == []


TWO_D_REASON = "too few board views: 4 of 12 needed"


def test_start_summary_names_a_2d_only_camera(monkeypatch):
    # Recording cameras the calibration left 2D only, or does not have (the stimCam), each get
    # a note; a configured camera that is not recording does not.
    _question(monkeypatch, QMessageBox.StandardButton.No)
    app_model = _AppModel(SetupDifference([], [], [], []),
                          {"camera4": TWO_D_REASON, "stimCam": "not in the active calibration",
                           "spare": "not in the active calibration"})
    app_model.recording_camera_names = ["left", "camera4", "stimCam"]
    behavior, status_bar = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    assert app_model.calls[1:] == ["start"]
    assert status_bar.messages == [
        f"2D only: camera4 ({TWO_D_REASON}) | 2D only: stimCam (not in the active calibration)"]

    # With the confirmation, the same lines are in its details, and in the note after Yes.
    asked = _question(monkeypatch, QMessageBox.StandardButton.Yes)
    app_model.difference = SetupDifference(added=["camera9"], removed=[], swapped=[], uncalibrated_off=[])
    BehaviorContent._start_recording(behavior)
    (_, text), = asked
    assert f"2D only: camera4 ({TWO_D_REASON})" in text.splitlines()
    assert "2D only: stimCam (not in the active calibration)" in text.splitlines()
    assert status_bar.messages[-1] == status_bar.messages[0]


def test_start_summary_names_cameras_without_any_calibration(monkeypatch):
    asked = _question(monkeypatch, QMessageBox.StandardButton.No)
    reason = "no active 3D calibration"
    app_model = _AppModel(SetupDifference([], [], [], []), {"left": reason, "right": reason})
    behavior, status_bar = _behavior(app_model)

    BehaviorContent._start_recording(behavior)

    assert asked == [] and app_model.calls[1:] == ["start"]
    assert status_bar.messages == [f"2D only: left ({reason}) | 2D only: right ({reason})"]


class _CaptureModel:
    def __init__(self, name):
        self.name = name
        self.record_mode = None
        self.is_enabled = True
        self.is_recording_enabled = True
        self.is_still_capture_enabled = False
        self.still_image_capture_interval = 1.0
        self.camera_list = []
        self.camera_source = None
        self.shape = (1080, 1440)
        self.property_changed = _Event()


class _TelemetryPanel(QWidget):
    def __init__(self, telemetry, parent=None):
        super().__init__(parent)


def test_2d_only_tag_in_recording_list(monkeypatch):
    monkeypatch.setattr(camera_content_module, "CaptureTelemetryPanel", _TelemetryPanel)
    app_model = _AppModel(SetupDifference([], [], [], []), {"stimCam": "not in the active calibration"})
    app_model.property_changed = _Event()
    app_model.session_telemetry = None
    contents = {name: CameraContent(app_model, _CaptureModel(name)) for name in ("left", "stimCam")}

    def tag(name):
        label = contents[name].camera_view.tag_label
        return label.isVisibleTo(contents[name]), label.text(), label.toolTip()

    assert tag("left")[0] is False
    assert tag("stimCam") == (True, "2D only", "not in the active calibration")

    # A calibration load changes which cameras are 2D only.
    app_model.uncalibrated_reasons = {"left": SWAP_REASON}
    app_model.property_changed(AppModel.Props.CAMERA_CALIBRATION, app_model.uncalibrated_reasons, None)
    QApplication.processEvents()
    assert tag("left") == (True, "2D only", SWAP_REASON)
    assert tag("stimCam")[0] is False
    for content in contents.values():
        content.close()
    assert app_model.property_changed.handlers == []


def test_a_stale_or_stopped_frame_is_no_current_view(monkeypatch):
    import numpy as np

    monkeypatch.setattr(camera_content_module, "CaptureTelemetryPanel", _TelemetryPanel)
    app_model = _AppModel(SetupDifference([], [], [], []))
    app_model.property_changed = _Event()
    app_model.session_telemetry = None
    content = CameraContent(app_model, _CaptureModel("left"))
    now = [100.0]
    monkeypatch.setattr(camera_content_module.time, "monotonic", lambda: now[0])
    try:
        assert content.displayed_frame() is None, "nothing shown yet"
        frame = np.zeros((1080, 1440), np.uint8)
        content.refresh_image(frame, 40.0)
        now[0] += 0.9
        assert content.displayed_frame() is frame
        now[0] += 0.2  # 1.1 s old: the camera stalled
        assert content.displayed_frame() is None

        content.refresh_image(frame, 40.0)
        assert content.displayed_frame() is frame
        content.set_is_capture_active(False)
        QApplication.processEvents()
        assert content.displayed_frame() is None, "cleared when capture stops"
    finally:
        content.close()
