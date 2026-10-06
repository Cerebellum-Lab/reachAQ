import time
from multiprocessing import Queue, Value, Array

from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.video import CaptureAttrs, CaptureCameraAttrs, VideoCapture


def wait_for_status(status: Value, expected: CaptureProcessStatus, timeout: int):
    start_ns = time.perf_counter_ns()
    elapsed = 0

    while status.value != expected and elapsed < timeout:
        time.sleep(0.05)
        elapsed = (time.perf_counter_ns() - start_ns) / 1e9

    return elapsed <= timeout


def test_video_capture_invalid_camera(video_capture):

    assert video_capture._status.value == CaptureProcessStatus.INITIALIZED

    assert len(video_capture._errors.value.decode()) == 0

    video_capture.start()

    assert wait_for_status(video_capture._status, CaptureProcessStatus.FAILED, timeout=4)

    assert len(video_capture._errors.value.decode()) > 0


def test_video_capture_model_requires_a_real_first_frame(video_capture_model):
    video_capture_model._video_status.value = CaptureProcessStatus.RUNNING
    video_capture_model._video_frame_index.value = -1
    video_capture_model._errors.value = b"serial=24152513 TriggerSource=Line3"

    assert video_capture_model.wait_for_first_frame(timeout=0.001) is False
    assert "did not deliver a frame" in video_capture_model.last_error
    assert "first capture error: serial=24152513 TriggerSource=Line3" in video_capture_model.last_error

    video_capture_model._video_frame_index.value = 0
    assert video_capture_model.wait_for_first_frame(timeout=0.001) is True


def test_a_capture_that_failed_hands_its_error_text_to_the_waiting_caller(video_capture_model):
    # The app model waits for (RUNNING, FAILED) and then raises last_error; a
    # refused camera setting must reach the operator, not a generic message.
    refusal = "<left> camera did not accept its settings: Width requested 512 applied 308 (max 308)"
    video_capture_model._video_status.value = CaptureProcessStatus.FAILED
    video_capture_model._errors.value = refusal.encode()

    assert video_capture_model.wait_for_capture_status(
        (CaptureProcessStatus.RUNNING, CaptureProcessStatus.FAILED), timeout=0.5) is True
    assert refusal in video_capture_model.last_error


def test_a_running_capture_leaves_last_error_alone(video_capture_model):
    video_capture_model._video_status.value = CaptureProcessStatus.RUNNING
    video_capture_model._errors.value = b"left over from an earlier run"

    assert video_capture_model.wait_for_capture_status(
        (CaptureProcessStatus.RUNNING, CaptureProcessStatus.FAILED), timeout=0.5) is True
    assert not video_capture_model.last_error


class _WriteRecorder:
    """Stands in for a shared Value or Array; notes each write in a log its partner shares."""

    def __init__(self, label: str, log: list, size: int = 512):
        self._label = label
        self._log = log
        self._size = size
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        self._value = new
        self._log.append(self._label)

    def __len__(self):
        return self._size


def test_a_capture_error_text_is_written_before_the_status_turns_failed():
    # The parent polls the status every millisecond and reads the error text the
    # moment it sees FAILED. Text written after FAILED can be missed, which shows
    # the generic "did not become ready" message instead of the camera's refusal.
    # Both live in shared memory in another process, so the order is the contract.
    writes = []
    status = _WriteRecorder("status", writes)
    errors = _WriteRecorder("errors", writes)
    capture = VideoCapture(CaptureAttrs(
        command_queue=Queue(), status=status, image_queue=None, frame=Value("q", 0),
        camera=CaptureCameraAttrs(name="test", url="spinnaker://000000"), errors=errors))
    writes.clear()  # the constructor's own INITIALIZED write is not under test

    capture._set_error("camera did not accept its settings")

    assert writes == ["errors", "status"]
    assert status.value == CaptureProcessStatus.FAILED
    assert errors.value == b"camera did not accept its settings"
