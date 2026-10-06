import queue
import threading
from pathlib import Path

import cv2
import numpy
import pytest

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


def _record(project_info, frames, name):
    rows, cols = frames[0].shape[:2]
    input_queue = queue.Queue()
    stopped = threading.Semaphore(0)
    recorder = VideoRecord(
        VideoRecordProperties(project_info=project_info, name=name, frame_size=(cols, rows), fps=30,
                              record_mode=VideoRecordMode.TRIGGER, video_rotate_interval=0),
        input_queue,
        record_stop_sema=stopped,
    )
    recorder.start()
    try:
        input_queue.put([(idx, frame, idx / 30.0, idx / 30.0) for idx, frame in enumerate(frames)])
        input_queue.put([])  # end of recording
        assert stopped.acquire(timeout=10), "recorder never closed the video"
    finally:
        recorder.cancel()
        recorder.join(5)
    videos = sorted(Path(project_info.root).rglob(f"*{name}*.mp4"))
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


def test_a_colour_source_is_still_recorded_in_colour(project_info, writer_spy):
    frames = _gradient_frames(10, channels=3)

    video = _record(project_info, frames, "colourcam")

    assert [w["is_color"] for w in writer_spy] == [True]
    decoded = _decode(video)
    assert len(decoded) == 10
    # The channels were different on the way in and must still differ on the way out.
    assert numpy.abs(decoded[3][:, :, 0].astype(int) - decoded[3][:, :, 1].astype(int)).mean() > 20


def test_closing_a_recording_that_got_no_frames_leaves_no_stale_file(project_info, writer_spy):
    recorder = VideoRecord(
        VideoRecordProperties(project_info=project_info, name="idle", frame_size=(64, 64), fps=30,
                              record_mode=VideoRecordMode.TRIGGER, video_rotate_interval=0),
        queue.Queue(),
    )
    recorder._prepare_writers()
    recorder._close_writers()

    assert recorder._video_file is None
    assert recorder._video_writer is None
    assert writer_spy == []  # nothing was opened without a frame to write
