import ctypes
import multiprocessing
from multiprocessing import Queue, Value, Array

import pytest

from reachaq.core import CameraId
from reachaq.video import (
    CaptureCameraAttrs,
    CaptureAttrs,
    VideoCapture,
    VideoRecordMode,
)
from reachaq.core.capture import CaptureProcessStatus
from reachaq.app.model.video_capture_model import VideoCaptureModel


@pytest.fixture
def video_capture():
    cmd_queue = Queue()
    image_queue = Queue()
    status = Value("i", CaptureProcessStatus.UNKNOWN)
    frame = Value(ctypes.c_int64, 0)
    errors = Array(ctypes.c_char, bytes(512))

    record_stop_sema = multiprocessing.Semaphore(0)

    camera = CaptureCameraAttrs(name="test", url="spinnaker://000000")

    attrs = CaptureAttrs(
        command_queue=cmd_queue,
        status=status,
        image_queue=image_queue,
        camera=camera,
        frame=frame,
        errors=errors,
        record_stop_sema=record_stop_sema,
    )

    process = VideoCapture(attrs)
    try:
        yield process
    finally:
        process.terminate()
        process.join(3)


@pytest.fixture
def video_capture_model(project_info, user_pref) -> VideoCaptureModel:  # noqa
    model = VideoCaptureModel(
        name="test_cam1",
        preferences=user_pref,
        camera_index=0,
        msg_queue=multiprocessing.Queue(),
        synced_cam_recording=multiprocessing.Value(ctypes.c_bool, False),
        synced_cam_frame_index=multiprocessing.Value(ctypes.c_int64, 0),
        record_stop_sema=multiprocessing.Semaphore(0),
    )
    model.project = project_info
    conf = model.save_configuration()
    conf.id = CameraId.Left
    conf.is_enabled = True
    conf.is_record_enabled = True
    conf.record_mode = VideoRecordMode.TRIGGER
    conf.params["fps"] = 30
    model.load_configuration(conf)  # don't forget
    try:
        yield model  # noqa
    finally:
        model.on_close()


@pytest.fixture
def video_capture_model2(video_capture_model, user_pref) -> VideoCaptureModel:  # noqa
    model = VideoCaptureModel(
        name="test_cam1",
        preferences=user_pref,
        camera_index=1,
        msg_queue=video_capture_model._msg_queue,
        synced_cam_recording=video_capture_model._synced_cam_recording,
        synced_cam_frame_index=video_capture_model._synced_cam_frame_index,
        record_stop_sema=video_capture_model._record_stop_sema,
    )
    model.project = video_capture_model.project  # use same
    conf = model.save_configuration()
    conf.id = CameraId.Right
    conf.is_enabled = True
    conf.is_record_enabled = True
    conf.record_mode = VideoRecordMode.TRIGGER
    model.load_configuration(conf)  # don't forget
    try:
        yield model  # noqa
    finally:
        model.on_close()
