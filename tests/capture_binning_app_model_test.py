import threading
from types import SimpleNamespace

import pytest

from tools.acquisition.model.app_model import AppModel, _camera_configured_metadata


class _Camera:
    def __init__(self, base=4, preset=None, shape=(256, 256)):
        self.base_binning = base
        self.capture_binning = preset
        self.shape = shape
        self.is_enabled = True
        self.is_recording_enabled = True

    @property
    def effective_capture_binning(self):
        return self.base_binning if self.capture_binning is None else self.capture_binning

    @property
    def capture_shape(self):
        k = (self.base_binning // self.effective_capture_binning) if self.base_binning else 1
        return self.shape[0] * k, self.shape[1] * k

    def set_capture_binning(self, value):
        self.capture_binning = value


def _app(*, acquiring=False, starting=False, stopping=False):
    left, right = _Camera(), _Camera()
    saved = []
    app = SimpleNamespace(
        _session_lifecycle_command_lock=threading.Lock(),
        _require_session_ready_for_configuration=lambda _action: None,
        _acquisition=SimpleNamespace(started=acquiring, starting=starting, stopping=stopping),
        _left_camera=left,
        _right_camera=right,
        save_configuration=lambda: saved.append(True),
    )
    app._stereo_cameras = lambda: AppModel._stereo_cameras(app)
    return app, left, right, saved


def test_a_preset_is_set_on_left_and_right_and_saved():
    app, left, right, saved = _app()
    AppModel.set_reach_capture_binning(app, 2)
    assert (left.capture_binning, right.capture_binning) == (2, 2)
    assert saved == [True]
    assert AppModel.reach_capture_binning.fget(app) == 2


def test_the_preset_cannot_change_while_acquiring():
    app, left, _, saved = _app(acquiring=True)
    with pytest.raises(RuntimeError, match="while acquisition is running"):
        AppModel.set_reach_capture_binning(app, 2)
    assert left.capture_binning is None and saved == []


@pytest.mark.parametrize("phase", ("starting", "stopping"))
def test_the_preset_cannot_change_while_acquisition_starts_or_stops(phase):
    # capture_start prepares the cameras before it marks the acquisition started.
    app, left, right, saved = _app(**{phase: True})
    with pytest.raises(RuntimeError, match="while acquisition is running"):
        AppModel.set_reach_capture_binning(app, 2)
    assert (left.capture_binning, right.capture_binning) == (None, None) and saved == []


def test_session_metadata_names_the_capture_and_inference_shapes():
    camera = _Camera(preset=2)
    assert _camera_configured_metadata(camera) == {
        "previewEnabled": True,
        "recordEnabled": True,
        "captureShape": [512, 512],
        "inferenceShape": [256, 256],
        "captureBinning": 2,
    }


def test_the_storage_estimate_uses_the_captured_frame_size():
    camera = _Camera(preset=2)
    camera.active_config = SimpleNamespace(params={"width": 256, "height": 256, "fps": 150})
    app = SimpleNamespace(
        _get_recording_cams=lambda: [camera],
        _nidaq_signal_monitor=SimpleNamespace(hardware_enabled=False, configuration=None),
    )
    # 16 KiB base + 512*512*150*0.25 video + 150*64 timestamps
    assert AppModel._estimate_session_bytes_per_second(app) == 16 * 1024.0 + 9_830_400 + 9_600
