import inspect
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

    @property
    def video_encoder(self):
        return "x264" if self.capture_shape != self.shape else "mp4v"


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


def _differs_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "different binning" in r.getMessage()]


def test_start_warns_when_left_and_right_capture_at_different_binning(caplog):
    app, left, right, _ = _app()
    left.name, right.name = "left", "right"
    right.set_capture_binning(2)
    with caplog.at_level("WARNING"):
        AppModel._warn_if_stereo_capture_binning_differs(app)
    message, = _differs_warnings(caplog)
    assert "left bin 4" in message and "right bin 2" in message


def test_start_does_not_warn_when_both_capture_at_the_same_binning(caplog):
    app, left, right, _ = _app()
    left.name, right.name = "left", "right"
    left.set_capture_binning(2)
    right.set_capture_binning(2)
    with caplog.at_level("WARNING"):
        AppModel._warn_if_stereo_capture_binning_differs(app)
    assert _differs_warnings(caplog) == []


def test_start_does_not_warn_about_a_camera_that_is_not_enabled(caplog):
    app, left, right, _ = _app()
    left.name, right.name = "left", "right"
    right.set_capture_binning(2)
    right.is_enabled = False
    with caplog.at_level("WARNING"):
        AppModel._warn_if_stereo_capture_binning_differs(app)
    assert _differs_warnings(caplog) == []


def test_a_malformed_binning_is_logged_not_raised_so_the_camera_start_can_refuse_it(caplog):
    app, left, right, _ = _app()

    class _Malformed(_Camera):
        @property
        def effective_capture_binning(self):
            raise ValueError("capture_binning='two' is not a number")

    app._right_camera = _Malformed()
    with caplog.at_level("WARNING"):
        AppModel._warn_if_stereo_capture_binning_differs(app)
    assert "could not be compared: capture_binning='two' is not a number" in caplog.text
    assert _differs_warnings(caplog) == []


def test_capture_start_checks_the_stereo_binning_before_it_starts_the_cameras():
    source = inspect.getsource(AppModel.capture_start)
    assert source.index("self._warn_if_stereo_capture_binning_differs()") < source.index(
        "self._start_reach_camera_domains(")


def test_session_metadata_names_the_capture_and_inference_shapes():
    camera = _Camera(preset=2)
    assert _camera_configured_metadata(camera) == {
        "previewEnabled": True,
        "recordEnabled": True,
        "captureShape": [512, 512],
        "inferenceShape": [256, 256],
        "captureBinning": 2,
        "videoEncoder": "x264",
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


def test_a_camera_that_does_not_record_has_no_encoder_in_the_metadata():
    camera = _Camera(preset=2)
    camera.is_recording_enabled = False

    assert _camera_configured_metadata(camera)["videoEncoder"] is None
