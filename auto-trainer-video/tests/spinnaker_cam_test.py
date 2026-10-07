import math
import pytest

# PySpin is the FLIR Spinnaker SDK: a vendor wheel that is not on PyPI and is not installed in every environment this suite runs in. Skipping is honest - the test genuinely cannot run - and matches the importorskip already used for other optional dependencies in this suite. It is not a statement that Spinnaker is optional on a rig, where it is required.
PySpin = pytest.importorskip("PySpin")

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


class _CompleteImage:
    def IsIncomplete(self):
        return False

    def GetFrameID(self):
        return 41

    def GetTimeStamp(self):
        return 5_000_000_000


class _OneFrameCamera:
    def GetNextImage(self, _timeout):
        return _CompleteImage()


def test_capture_keeps_the_measured_poll_and_arrival_times(monkeypatch):
    capture = SpinCam.__new__(SpinCam)
    CameraBase.__init__(capture, "left")
    capture._camera = _OneFrameCamera()
    capture._is_primary = True
    capture._current_cam_frame_2_perf_offset = math.nan
    capture._current_cam_frame_2_time_offset = math.nan
    capture._consecutive_late_acquire = 0
    # p_timeout, the timeout check, p_before, p_after
    perf_values = iter((100.0, 100.0, 100.010, 100.012))
    monkeypatch.setattr(spinnaker_cam.time, "perf_counter", lambda: next(perf_values))

    try:
        capture._capture()

        assert capture.frame_poll_perf_c == 100.010
        assert capture.frame_arrival_perf_c == 100.012
    finally:
        # SpinCam.__del__ would call EndAcquisition() on this one-method fake.
        capture._camera = None


class _LatchCommand:
    def __init__(self, events, error=None):
        self._events = events
        self._error = error

    def Execute(self):
        self._events.append("execute")
        if self._error is not None:
            raise self._error


class _LatchValue:
    def __init__(self, events):
        self._events = events

    def GetValue(self):
        self._events.append("read")
        return 123_456_789_000


class _NodeMap:
    def __init__(self, nodes):
        self._nodes = nodes

    def GetNode(self, name):
        return self._nodes.get(name)


def _latching_camera(monkeypatch, nodes, *, accessible=True):
    camera = SpinCam.__new__(SpinCam)
    CameraBase.__init__(camera, "left")
    camera._camera = None  # SpinCam.__del__ ends acquisition on a real camera only
    camera._node_map = _NodeMap(nodes)
    # The fakes stand in for the GenApi pointers PySpin would cast the nodes to.
    monkeypatch.setattr(spinnaker_cam.PySpin, "CCommandPtr", lambda node: node)
    monkeypatch.setattr(spinnaker_cam.PySpin, "CIntegerPtr", lambda node: node)
    monkeypatch.setattr(spinnaker_cam.PySpin, "IsAvailable", lambda node: node is not None)
    monkeypatch.setattr(spinnaker_cam.PySpin, "IsWritable", lambda node: accessible)
    monkeypatch.setattr(spinnaker_cam.PySpin, "IsReadable", lambda node: accessible)
    monkeypatch.setattr(SpinCam, "_latch_error_logged", False)
    return camera


def _counting_perf(monkeypatch, events):
    ticks = iter(range(1, 1_000_000))

    def perf():
        value = 1000.0 + next(ticks) * 0.001
        events.append(("perf", value))
        return value

    monkeypatch.setattr(spinnaker_cam.time, "perf_counter", perf)


def test_latch_clock_brackets_the_latch_and_reads_the_value_after_it(monkeypatch):
    events = []
    camera = _latching_camera(monkeypatch, {"TimestampLatch": _LatchCommand(events),
                                            "TimestampLatchValue": _LatchValue(events)})
    _counting_perf(monkeypatch, events)

    latch = camera.latch_clock()

    assert events == [("perf", 1000.001), "execute", ("perf", 1000.002), "read"]
    assert latch == (1000.001, 123_456_789_000, 1000.002)
    assert latch[0] <= latch[2]


@pytest.mark.parametrize("missing", ["TimestampLatch", "TimestampLatchValue", "inaccessible"])
def test_latch_clock_is_none_without_usable_nodes(monkeypatch, missing):
    events = []
    nodes = {"TimestampLatch": _LatchCommand(events), "TimestampLatchValue": _LatchValue(events)}
    nodes.pop(missing, None)
    camera = _latching_camera(monkeypatch, nodes, accessible=missing != "inaccessible")

    assert camera.latch_clock() is None
    assert events == []


def test_latch_clock_is_none_and_logs_once_when_the_camera_raises(monkeypatch, caplog):
    events = []
    error = PySpin.SpinnakerException("USB transport error")
    camera = _latching_camera(monkeypatch, {"TimestampLatch": _LatchCommand(events, error),
                                            "TimestampLatchValue": _LatchValue(events)})

    with caplog.at_level("DEBUG", logger=spinnaker_cam.logger.name):
        first = camera.latch_clock()
        second = camera.latch_clock()

    assert first is None and second is None
    assert events == ["execute", "execute"]
    assert len([record for record in caplog.records
                if record.name == spinnaker_cam.logger.name]) == 1
