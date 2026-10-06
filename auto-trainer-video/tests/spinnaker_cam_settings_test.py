import math

import numpy
import pytest

# See spinnaker_cam_test.py: PySpin is a vendor wheel, present on the rig.
PySpin = pytest.importorskip("PySpin")

from autotrainer.video.camera.camera_base import CameraBase
from autotrainer.video.camera.spinnaker_cam import SpinCam


class _Node:
    """A GenICam node that stores what it is given, like the camera does."""

    def __init__(self, name, value, maximum=None):
        self._name = name
        self.value = value
        self._max = maximum
        self.sets = []

    def GetDisplayName(self):
        return self._name

    def GetAccessMode(self):
        return PySpin.RW

    def GetMax(self):
        return self._max

    def SetValue(self, value):
        self.sets.append(value)
        self.value = value

    def GetValue(self):
        return self.value


class _RoundingExposure(_Node):
    def SetValue(self, value):
        super().SetValue(value + 1)  # 175 requested reads back 176, as on the rig


class _Camera:
    def __init__(self, *, width_max=1232, fps_max=456.1, gain_max=47.99):
        self.ExposureTime = _RoundingExposure("Exposure Time", 0.0, 29999999.0)
        self.AcquisitionFrameRateEnable = _Node("Acquisition Frame Rate Enable", False)
        self.AcquisitionFrameRate = _Node("Acquisition Frame Rate", 30.0, fps_max)
        self.BinningHorizontal = _Node("Binning Horizontal", 4, 4)
        self.BinningVertical = _Node("Binning Vertical", 4, 4)
        self.Width = _Node("Width", 256, width_max)
        self.Height = _Node("Height", 256, 1080)
        self.OffsetX = _Node("Offset X", 0, 1440)
        self.OffsetY = _Node("Offset Y", 0, 1080)
        self.Gain = _Node("Gain", 0.0, gain_max)
        self.GammaEnable = _Node("Gamma Enable", True)
        self.Gamma = _Node("Gamma", 0.7, 4.0)


def _spincam(**overrides):
    cam = SpinCam.__new__(SpinCam)
    CameraBase.__init__(cam, "left")
    cam._camera = None  # nothing for __del__ to release
    cam._exposure = 175
    cam._fps = 150
    cam._horizontal_binning = cam._vertical_binning = 4
    cam._width = cam._height = 256
    cam._offset_x, cam._offset_y = 52, 6
    cam._gain = 1
    cam._gamma = 0.7
    for name, value in overrides.items():
        setattr(cam, name, value)
    return cam


def test_the_live_settings_pass_and_fps_is_set_again_after_the_geometry():
    camera = _Camera()
    _spincam()._apply_settings(camera)
    assert camera.AcquisitionFrameRate.sets == [150, 150]
    assert (camera.Width.value, camera.OffsetX.value, camera.Gain.value) == (256, 52, 1)


def test_a_clamped_width_is_refused_and_named():
    camera = _Camera(width_max=308)
    spin = _spincam(_width=512, _height=512, _horizontal_binning=2, _vertical_binning=2)
    with pytest.raises(RuntimeError, match=r"Width requested 512 applied 308 \(max 308\)"):
        spin._apply_settings(camera)


def test_a_clamped_gain_is_refused():
    with pytest.raises(RuntimeError, match=r"Gain requested 25\.08 dB applied 20\.00 dB"):
        _spincam(_gain=25.08)._apply_settings(_Camera(gain_max=20.0))


def test_a_frame_rate_the_camera_cannot_reach_is_refused():
    with pytest.raises(RuntimeError, match=r"the camera's maximum is 100\.0 fps"):
        _spincam()._apply_settings(_Camera(fps_max=100.0))


class _Image:
    def __init__(self, array, frame_id):
        self._array, self._id = array, frame_id

    def IsIncomplete(self):
        return False

    def GetFrameID(self):
        return self._id

    def GetTimeStamp(self):
        return self._id * 6_666_667

    def GetNDArray(self):
        return self._array

    def Release(self):
        pass


class _Streaming:
    def __init__(self, shape):
        self._shape, self._count = shape, 0

    def GetNextImage(self, _timeout):
        self._count += 1
        return _Image(numpy.zeros(self._shape, numpy.uint8), self._count)


@pytest.mark.parametrize("size,checked", ((256, True), (512, False)))
def test_the_startup_buffer_self_check_runs_only_up_to_256_square(size, checked):
    spin = _spincam(_width=size, _height=size)
    spin._camera = _Streaming((size, size))
    spin._acquisition_started = True
    spin._skip_duplicate_frame_copy = False
    spin._start_frames = []
    spin._current_cam_frame_2_perf_offset = math.nan
    spin._current_cam_frame_2_time_offset = math.nan
    spin._consecutive_late_acquire = 0
    spin._is_primary = True
    try:
        for _ in range(3):
            spin.capture()
        assert (len(spin._start_frames) > 0) is checked
    finally:
        spin._camera = None
