"""A camera's capture source must not change under a capture start.

On christielab10, 2026-11-23, a 3D-calibration start failed in the left
camera's on_prepare_capture with "argument of type 'NoneType' is not
iterable". load_configuration had just published a new camera list. The
panel repopulated its source combo on the GUI thread, and clear() and the
first addItem each emitted a selection, so the model's source was set to
None and then back while the start read it twice.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from reachaq.pyside.capture.QtCaptureView import QCaptureView
from reachaq.video import CaptureCameraAttrs
from reachaq.app.model.video_capture_model import VideoCaptureModel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    return app or QApplication([])


def test_repopulating_the_source_list_selects_nothing(qapp):
    view = QCaptureView()
    first = CaptureCameraAttrs(name="left", url="spinnaker://cam/1")
    second = CaptureCameraAttrs(name="other", url="spinnaker://cam/2")
    view.setCameras([first, second])
    view.setCamera(second)
    selected = []
    view.camera_changed.connect(selected.append)

    view.setCameras([first, second])

    assert selected == []
    assert view._camera.currentData() == second
    assert view._camera_name_label.text() == "other"


def test_a_first_population_names_the_camera(qapp):
    view = QCaptureView()

    view.setCameras([CaptureCameraAttrs(name="left", url="spinnaker://cam/1")])

    assert view._camera_name_label.text() == "left"


def test_a_start_uses_the_source_it_read_first(monkeypatch):
    model = VideoCaptureModel(name="left")
    model._is_enabled = True
    model._update_camera_source(None)
    assert model.camera_source is None

    # The GUI thread swaps the source in between the start's reads.
    def swap_source():
        model._camera_source = CaptureCameraAttrs(name="left", url="spinnaker://cam/1")

    monkeypatch.setattr(model, "_video_reader_initialize", swap_source)

    assert model.on_prepare_capture() is True
    assert model._video_capture is None
