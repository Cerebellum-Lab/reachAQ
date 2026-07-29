import PySpin
import pytest

from autotrainer.video.camera import spinnaker_cam
from autotrainer.video.camera.camera_base import CameraBase
from autotrainer.video.camera.spinnaker_cam import SpinCam


class _EnumEntry:
    def __init__(self, value):
        self._value = value

    def GetSymbolic(self):
        return self._value


class _EnumNode:
    def __init__(self, value):
        self._value = value

    def GetCurrentEntry(self):
        return _EnumEntry(self._value)


class _ValueNode:
    def __init__(self, value):
        self._value = value

    def GetValue(self):
        return self._value


class _IncompleteImage:
    def __init__(self):
        self.released = False

    def IsIncomplete(self):
        return True

    def GetImageStatus(self):
        return 7

    def Release(self):
        self.released = True


class _FailingCamera:
    AcquisitionMode = _EnumNode("Continuous")
    AcquisitionFrameRateEnable = _ValueNode(False)
    TriggerMode = _EnumNode("On")
    TriggerSelector = _EnumNode("FrameStart")
    TriggerSource = _EnumNode("Line3")
    TriggerActivation = _EnumNode("AnyEdge")
    TriggerOverlap = _EnumNode("ReadOut")

    def __init__(self, incomplete_image):
        self._call_count = 0
        self._incomplete_image = incomplete_image

    def GetNextImage(self, _timeout):
        self._call_count += 1
        if self._call_count == 1:
            error = PySpin.SpinnakerException("No frame before polling timeout")
            error.errorcode = PySpin.SPINNAKER_ERR_TIMEOUT
            raise error
        if self._call_count == 2:
            error = PySpin.SpinnakerException("USB transport error")
            error.errorcode = PySpin.SPINNAKER_ERR_IO
            raise error
        return self._incomplete_image


def test_capture_timeout_reports_first_error_incomplete_image_and_trigger_nodes(monkeypatch):
    incomplete_image = _IncompleteImage()
    camera = _FailingCamera(incomplete_image)
    capture = SpinCam.__new__(SpinCam)
    CameraBase.__init__(capture, "right")
    capture._camera = camera
    capture._serial_number = "24152513"
    capture._is_primary = False
    capture._is_secondary = True

    perf_values = iter((0, 0, 0, 0, 0, 0, 0, 0, 5))
    monkeypatch.setattr(spinnaker_cam.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        spinnaker_cam.PySpin,
        "Image_GetImageStatusDescription",
        lambda status: f"status description {status}",
    )

    with pytest.raises(RuntimeError) as exc_info:
        capture._capture()

    message = str(exc_info.value)
    assert "camera=right" in message
    assert "serial=24152513" in message
    assert "role=secondary" in message
    assert "failure_hint=non_timeout_spinnaker_error" in message
    assert "timeout_exceptions=1" in message
    assert (
        "first_spinnaker_exception="
        "[code=-1011 message=No frame before polling timeout]"
    ) in message
    assert (
        "first_non_timeout_spinnaker_exception="
        "[code=-1010 message=USB transport error]"
    ) in message
    assert "first_incomplete_image=[status=7 description=status description 7]" in message
    assert "TriggerMode=On" in message
    assert "TriggerSource=Line3" in message
    assert "TriggerActivation=AnyEdge" in message
    assert incomplete_image.released
    capture._camera = None
