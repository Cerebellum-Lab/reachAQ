import time
from multiprocessing import Queue, Value

import h5py
import pytest

from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.latency import latency_stream_path
from autotrainer.video import (
    CaptureAttrs,
    CaptureCameraAttrs,
    CaptureCommandKind,
    VideoCapture,
    VideoRecordMode,
    VideoRecordProperties,
)


def _wait(status, expected, timeout):
    deadline = time.monotonic() + timeout
    while status.value != expected and time.monotonic() < deadline:
        time.sleep(0.05)
    return status.value == expected


@pytest.mark.functional
def test_a_recording_leaves_camera_and_recorder_latency_streams(project_info):
    commands = Queue()
    status = Value("i", CaptureProcessStatus.UNKNOWN)
    attrs = CaptureAttrs(
        command_queue=commands, status=status, image_queue=None,
        camera=CaptureCameraAttrs(name="lat", url="random://0"),
        frame=Value("i", 0), errors=None,
    )
    record = VideoRecordProperties(project_info=project_info,
                                   record_mode=VideoRecordMode.TRIGGER,
                                   video_rotate_interval=0)
    process = VideoCapture(attrs, record_properties=record, project_info=project_info)
    process.start()
    try:
        assert _wait(status, CaptureProcessStatus.RUNNING, 10)
        commands.put((CaptureCommandKind.ENABLE_CAPTURE, None))
        commands.put((CaptureCommandKind.ENABLE_RECORDING, None))
        time.sleep(1.5)
        commands.put((CaptureCommandKind.DISABLE_RECORDING, None))
        time.sleep(1.0)
    finally:
        commands.put((CaptureCommandKind.TERMINATE, None))
        assert _wait(status, CaptureProcessStatus.TERMINATED, 10)
        process.join(10)

    with h5py.File(latency_stream_path(project_info, "camera_lat"), "r") as store:
        frames = store["frames"][:]
        batches = store["record_batches"][:]
    assert len(frames) > 10
    assert (frames["arrival_perf"] >= frames["poll_perf"]).all()
    assert (frames["capture_return_perf"] >= frames["arrival_perf"]).all()
    assert len(batches) >= 1 and not batches["lost"].any()
    with h5py.File(latency_stream_path(project_info, "record_lat"), "r") as store:
        assert len(store["record_writes"]) == len(batches)


@pytest.mark.functional
def test_a_disable_before_any_recording_does_not_cut_the_next_stream_short(project_info):
    commands = Queue()
    status = Value("i", CaptureProcessStatus.UNKNOWN)
    attrs = CaptureAttrs(
        command_queue=commands, status=status, image_queue=None,
        camera=CaptureCameraAttrs(name="lat", url="random://0"),
        frame=Value("i", 0), errors=None,
    )
    record = VideoRecordProperties(project_info=project_info,
                                   record_mode=VideoRecordMode.TRIGGER,
                                   video_rotate_interval=0)
    process = VideoCapture(attrs, record_properties=record, project_info=project_info)
    process.start()
    try:
        assert _wait(status, CaptureProcessStatus.RUNNING, 10)
        commands.put((CaptureCommandKind.ENABLE_CAPTURE, None))
        # The app sends this to every camera on entering a calibration step,
        # with no stream open: it must not leave a close request behind.
        commands.put((CaptureCommandKind.DISABLE_RECORDING, None))
        commands.put((CaptureCommandKind.ENABLE_RECORDING, None))
        time.sleep(1.5)
        commands.put((CaptureCommandKind.DISABLE_RECORDING, None))
        time.sleep(1.0)
    finally:
        commands.put((CaptureCommandKind.TERMINATE, None))
        assert _wait(status, CaptureProcessStatus.TERMINATED, 10)
        process.join(10)

    with h5py.File(latency_stream_path(project_info, "camera_lat"), "r") as store:
        frames = store["frames"][:]
    # About 45 frames in 1.5 s of recording at 30 fps; a stale request left one.
    assert len(frames) > 20


def _closed_within(path, timeout):
    """True once the stream's writer has closed it: writer_stats is the last attribute written."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with h5py.File(path, "r") as store:
                if "writer_stats" in store.attrs:
                    return True
        except OSError:
            pass  # still held open for writing
        time.sleep(0.05)
    return False


def _recording_process(project_info):
    commands = Queue()
    status = Value("i", CaptureProcessStatus.UNKNOWN)
    attrs = CaptureAttrs(
        command_queue=commands, status=status, image_queue=None,
        camera=CaptureCameraAttrs(name="lat", url="random://0"),
        frame=Value("i", 0), errors=None,
    )
    record = VideoRecordProperties(project_info=project_info,
                                   record_mode=VideoRecordMode.TRIGGER,
                                   video_rotate_interval=0)
    return commands, status, VideoCapture(attrs, record_properties=record, project_info=project_info)


@pytest.mark.functional
def test_a_stop_that_lands_while_capture_is_disabled_still_closes_the_stream(project_info):
    commands, status, process = _recording_process(project_info)
    process.start()
    try:
        assert _wait(status, CaptureProcessStatus.RUNNING, 10)
        commands.put((CaptureCommandKind.ENABLE_CAPTURE, None))
        commands.put((CaptureCommandKind.ENABLE_RECORDING, None))
        assert _wait(status, CaptureProcessStatus.RECORDING, 10)
        time.sleep(0.5)
        # Capture off first: the loop is idle when the stop's close request
        # arrives, so no frame can ever run the close.
        commands.put((CaptureCommandKind.DISABLE_CAPTURE, None))
        commands.put((CaptureCommandKind.DISABLE_RECORDING, None))
        assert _closed_within(latency_stream_path(project_info, "camera_lat"), 3.0), \
            "the stream was still open 3 s after the stop; it would close only at TERMINATE"
        # The idle close must not cost the recording its frames (about 0.5 s of them).
        with h5py.File(latency_stream_path(project_info, "camera_lat"), "r") as store:
            assert len(store["frames"]) > 0
    finally:
        commands.put((CaptureCommandKind.TERMINATE, None))
        assert _wait(status, CaptureProcessStatus.TERMINATED, 10)
        process.join(10)


@pytest.mark.functional
def test_the_apps_stop_order_closes_the_stream_before_the_process_ends(project_info):
    # The app sends DISABLE_RECORDING (behavior_algorithm.end_capture_session) and
    # then DISABLE_CAPTURE (the status change). Whether or not a frame slips in
    # between, the stream must be closed well before TERMINATE.
    commands, status, process = _recording_process(project_info)
    process.start()
    try:
        assert _wait(status, CaptureProcessStatus.RUNNING, 10)
        commands.put((CaptureCommandKind.ENABLE_CAPTURE, None))
        commands.put((CaptureCommandKind.ENABLE_RECORDING, None))
        assert _wait(status, CaptureProcessStatus.RECORDING, 10)
        time.sleep(0.5)
        commands.put((CaptureCommandKind.DISABLE_RECORDING, None))
        commands.put((CaptureCommandKind.DISABLE_CAPTURE, None))
        assert _closed_within(latency_stream_path(project_info, "camera_lat"), 3.0)
        # The idle close must not cost the recording its frames (about 0.5 s of them).
        with h5py.File(latency_stream_path(project_info, "camera_lat"), "r") as store:
            assert len(store["frames"]) > 0
    finally:
        commands.put((CaptureCommandKind.TERMINATE, None))
        assert _wait(status, CaptureProcessStatus.TERMINATED, 10)
        process.join(10)
