import queue
import threading
from pathlib import Path

import cv2
import numpy
import pytest

from autotrainer.core import SystemStatusMessageKind
from autotrainer.core.project import ProjectInfo
from autotrainer.video import VideoRecord, VideoRecordMode, VideoRecordProperties
from autotrainer.video import video_record


@pytest.fixture
def writer_spy(monkeypatch):
    """Wrap the real cv2.VideoWriter and note how each one is opened and fed."""
    real = cv2.VideoWriter
    opened = []

    def spy(path, fourcc, fps, size, isColor=True):
        writer = real(path, fourcc, fps, size, isColor)
        record = {"is_color": isColor, "shapes": []}
        opened.append(record)

        class Writer:
            def isOpened(self):
                return writer.isOpened()

            def write(self, frame):
                record["shapes"].append(numpy.shape(frame))
                writer.write(frame)

            def release(self):
                writer.release()

        return Writer()

    monkeypatch.setattr(video_record.cv2, "VideoWriter", spy)
    return opened


def _gradient_frames(count, rows=64, cols=64, channels=None):
    # Smooth content, so mp4v's loss stays well under a grey level; noise would not.
    frames = []
    for idx in range(count):
        ramp = (numpy.arange(cols)[None, :] * 2 + numpy.arange(rows)[:, None] + idx) % 200 + 20
        frame = ramp.astype(numpy.uint8)
        if channels is not None:
            frame = numpy.stack([frame, 255 - frame, numpy.full_like(frame, 90)], axis=2)[:, :, :channels]
        frames.append(frame)
    return frames


def _recorder(project_info, name, rows=64, cols=64, msg_queue=None):
    input_queue = queue.Queue()
    stopped = threading.Semaphore(0)
    recorder = VideoRecord(
        VideoRecordProperties(project_info=project_info, name=name, frame_size=(cols, rows), fps=30,
                              record_mode=VideoRecordMode.TRIGGER, video_rotate_interval=0),
        input_queue,
        record_stop_sema=stopped,
        msg_queue=msg_queue,
    )
    return recorder, input_queue, stopped


def _videos(project_info, name):
    return sorted(Path(project_info.root).rglob(f"*{name}*.{ProjectInfo.video_write_ext}"))


def _record(project_info, frames, name, batches=1):
    rows, cols = frames[0].shape[:2]
    recorder, input_queue, stopped = _recorder(project_info, name, rows, cols)
    recorder.start()
    try:
        size = -(-len(frames) // batches)
        for start in range(0, len(frames), size):
            input_queue.put([(idx, frames[idx], idx / 30.0, idx / 30.0)
                             for idx in range(start, min(start + size, len(frames)))])
        input_queue.put([])  # end of recording
        assert stopped.acquire(timeout=10), "recorder never closed the video"
    finally:
        recorder.cancel()
        recorder.join(5)
    videos = _videos(project_info, name)
    assert len(videos) == 1, videos
    return videos[0]


def _decode(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames.append(frame)


def test_a_grayscale_camera_is_written_to_a_grayscale_writer(project_info, writer_spy):
    frames = _gradient_frames(20)

    video = _record(project_info, frames, "graycam")

    assert [w["is_color"] for w in writer_spy] == [False]
    assert set(writer_spy[0]["shapes"]) == {(64, 64)}  # no 3-channel copy
    decoded = _decode(video)
    assert len(decoded) == 20
    assert numpy.abs(decoded[5][:, :, 0].astype(int) - frames[5].astype(int)).mean() < 2


def test_a_single_channel_frame_is_written_as_grayscale(project_info, writer_spy):
    frames = [frame[:, :, None] for frame in _gradient_frames(10)]

    _record(project_info, frames, "onechan")

    assert [w["is_color"] for w in writer_spy] == [False]
    assert set(writer_spy[0]["shapes"]) == {(64, 64)}


def test_the_playback_cameras_strided_channel_view_is_recorded(project_info, writer_spy):
    # PlaybackCam hands over frame[:, :, 1] of a decoded BGR frame: a non-contiguous view.
    bgr = _gradient_frames(10, channels=3)
    frames = [frame[:, :, 1] for frame in bgr]

    video = _record(project_info, frames, "playback")

    assert [w["is_color"] for w in writer_spy] == [False]
    assert len(_decode(video)) == 10


def test_a_colour_source_is_still_recorded_in_colour(project_info, writer_spy):
    frames = _gradient_frames(10, channels=3)

    video = _record(project_info, frames, "colourcam")

    assert [w["is_color"] for w in writer_spy] == [True]
    decoded = _decode(video)
    assert len(decoded) == 10
    # The channels were different on the way in and must still differ on the way out.
    assert numpy.abs(decoded[3][:, :, 0].astype(int) - decoded[3][:, :, 1].astype(int)).mean() > 20


def test_mono_filler_frames_in_a_colour_recording_are_kept(project_info, writer_spy):
    # The capture loop fills missed camera frames with mono zeros. OpenCV drops a
    # frame whose channel count does not match the writer, silently, so a filler
    # must be converted or every later frame shifts against its timestamp row.
    colour = _gradient_frames(6, channels=3)
    frames = colour[:3] + [numpy.zeros((64, 64), numpy.uint8)] * 2 + colour[3:]

    video = _record(project_info, frames, "mixedcolour")

    assert [w["is_color"] for w in writer_spy] == [True]
    assert set(writer_spy[0]["shapes"]) == {(64, 64, 3)}
    assert len(_decode(video)) == len(frames)


def test_a_colour_frame_in_a_grayscale_recording_is_kept(project_info, writer_spy):
    frames = _gradient_frames(3) + _gradient_frames(2, channels=3) + _gradient_frames(3)

    video = _record(project_info, frames, "mixedgray")

    assert [w["is_color"] for w in writer_spy] == [False]
    assert set(writer_spy[0]["shapes"]) == {(64, 64)}
    assert len(_decode(video)) == len(frames)


def _closed_message(msg_queue):
    while True:
        kind, payload = msg_queue.get(timeout=5)
        if kind == SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED:
            return payload


def test_a_frame_with_an_unrecordable_channel_count_is_refused_and_reported(project_info, writer_spy):
    msg_queue = queue.Queue()
    recorder, input_queue, stopped = _recorder(project_info, "fourchan", msg_queue=msg_queue)
    recorder.start()
    try:
        input_queue.put([(idx, numpy.zeros((64, 64, 4), numpy.uint8), 0.0, 0.0) for idx in range(5)])
        input_queue.put([])
        assert stopped.acquire(timeout=10)
    finally:
        recorder.cancel()
        recorder.join(5)

    assert writer_spy == []  # never opened for it
    _cam_idx, written, _project, _generation, diagnostics = _closed_message(msg_queue)
    assert written == 0
    assert diagnostics["errorCount"] >= 1
    assert "4-channel" in diagnostics["firstError"]


def test_a_frame_of_the_wrong_size_is_refused_and_reported(project_info, writer_spy):
    msg_queue = queue.Queue()
    recorder, input_queue, stopped = _recorder(project_info, "wrongsize", msg_queue=msg_queue)
    recorder.start()
    try:
        input_queue.put([(idx, numpy.zeros((32, 64), numpy.uint8), 0.0, 0.0) for idx in range(5)])
        input_queue.put([])
        assert stopped.acquire(timeout=10)
    finally:
        recorder.cancel()
        recorder.join(5)

    assert writer_spy == []
    _cam_idx, written, _project, _generation, diagnostics = _closed_message(msg_queue)
    assert written == 0
    assert "does not match the recording size 64x64" in diagnostics["firstError"]


def test_a_writer_that_cannot_open_is_retried_once_per_batch_and_reported(project_info, monkeypatch):
    constructed = []

    class Unopenable:
        def isOpened(self):
            return False

        def release(self):
            pass

    def failing_writer(*args, **kwargs):
        constructed.append(args[0])
        return Unopenable()

    monkeypatch.setattr(video_record.cv2, "VideoWriter", failing_writer)
    msg_queue = queue.Queue()
    recorder, input_queue, stopped = _recorder(project_info, "unopenable", msg_queue=msg_queue)
    recorder.start()
    try:
        for batch in range(3):
            input_queue.put([(batch * 10 + idx, _gradient_frames(1)[0], 0.0, 0.0) for idx in range(10)])
        input_queue.put([])
        assert stopped.acquire(timeout=10)
        assert recorder.is_alive()  # the failure is reported, not fatal
    finally:
        recorder.cancel()
        recorder.join(5)

    assert len(constructed) == 3  # once per batch, not once per frame
    _cam_idx, written, _project, _generation, diagnostics = _closed_message(msg_queue)
    assert written == 0
    assert diagnostics["errorCount"] == 3
    assert "Failed open" in diagnostics["firstError"]


def test_closing_a_recording_that_got_no_frames_leaves_no_stale_file(project_info, writer_spy):
    recorder, _input_queue, _stopped = _recorder(project_info, "idle")
    recorder._prepare_writers()
    recorder._close_writers()

    assert recorder._video_file is None
    assert recorder._video_writer is None
    assert recorder._video_timestamp_file is None
    assert writer_spy == []  # nothing was opened without a frame to write
    assert _videos(project_info, "idle") == []
