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
