"""End-to-end check that playback cameras stay frame-aligned.

The unit tests cover the barrier mechanism. This covers the thing that actually
matters: two real VideoCapture processes, playing real video files, keeping the
same frame index as each other for the whole run.

That property is what live 3D triangulation rests on. It pairs by frame id, so
a constant skew produces wrong 3D underneath a correct-looking 2D overlay -
which nothing downstream detects.

No camera hardware, no GPU: playback cameras need neither.
"""

import ctypes
import logging
import time
from multiprocessing import Array, Queue, Value
from pathlib import Path

import cv2
import numpy
import pytest

from autotrainer.core import clear_queue
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.core.project import ProjectInfo
from autotrainer.video import (
    CaptureAttrs,
    CaptureCameraAttrs,
    CaptureCommandKind,
    VideoCapture,
)
from autotrainer.video.video_record import VideoRecordMode, VideoRecordProperties

logger = logging.getLogger(__name__)

FPS = 60
FRAME_COUNT = 600  # 10s of video, far longer than the test runs
WIDTH = HEIGHT = 64

#: How long one camera is held back before being enabled, standing in for the
#: real application's sequential per-camera startup.
STAGGER_SECONDS = 0.4

#: At 60 FPS the stagger is ~24 frames. The barrier should hold skew to a couple
#: of frames; scheduling noise on a loaded CI box makes anything tighter flaky.
MAX_SYNCED_SKEW_FRAMES = 4


def _write_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened(), f"could not open a writer for {path}"
    try:
        for index in range(FRAME_COUNT):
            frame = numpy.full((HEIGHT, WIDTH, 3), index % 256, dtype=numpy.uint8)
            writer.write(frame)
    finally:
        writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    directory = tmp_path_factory.mktemp("playback_sync")
    return [_write_video(directory / f"cam{index}.mp4") for index in range(2)]


class _Camera:
    """One VideoCapture process plus the shared values needed to drive it."""

    def __init__(self, name: str, video: Path, barrier, project_root: Path):
        self.name = name
        self.command_queue = Queue()
        self.status = Value("i", CaptureProcessStatus.UNKNOWN)
        self.frame = Value("i", -1)
        # A real buffer, not None: the capture process reports failures through
        # it, and a None here turns any such report into a crash that hides the
        # original cause.
        self.errors = Array(ctypes.c_char, 512)
        self.attrs = CaptureAttrs(
            command_queue=self.command_queue,
            status=self.status,
            # No preview queue. Frame index comes from shared memory, and an
            # unconsumed multiprocessing.Queue keeps its feeder thread alive,
            # which stops the child from ever exiting.
            image_queue=None,
            frame=self.frame,
            camera=CaptureCameraAttrs(name=name, url=f"playback://{video.as_posix()}"),
            errors=self.errors,
            playback_start_barrier=barrier,
            playback_start_timeout=20.0,
        )
        # Nothing here records, but the record thread still starts, and it reads
        # its project from the record properties rather than from VideoCapture.
        # Without one it exits immediately, and its death tears down the capture
        # loop before a single frame is taken.
        project = ProjectInfo(root=str(project_root), device_id="playback-sync-test")
        self.process = VideoCapture(
            self.attrs,
            VideoRecordProperties(
                record_mode=VideoRecordMode.NONE, name=name, project_info=project
            ),
            project_info=project,
        )

    def start(self):
        self.process.start()

    def enable(self):
        self.command_queue.put((CaptureCommandKind.ENABLE_CAPTURE, None))

    def stop(self):
        try:
            self.command_queue.put((CaptureCommandKind.TERMINATE, None))
        except Exception:
            pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        clear_queue(self.command_queue)


def _wait_for_status(status, expected, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if status.value == expected:
            return True
        time.sleep(0.02)
    return False


def _run(videos, project_root: Path, *, use_barrier: bool):
    """Start two playback cameras with a stagger; return the observed frame skews."""
    barrier = get_mp_ctx().Barrier(2) if use_barrier else None
    cameras = [
        _Camera(f"cam{index}", videos[index], barrier, project_root)
        for index in range(2)
    ]
    skews = []
    try:
        for camera in cameras:
            camera.start()
        for camera in cameras:
            assert _wait_for_status(camera.status, CaptureProcessStatus.RUNNING, 25), \
                f"{camera.name} never reached RUNNING"

        # Enable one, wait, then the other. This is what the application does:
        # each camera gets its own enable, and they do not arrive together.
        cameras[0].enable()
        time.sleep(STAGGER_SECONDS)
        cameras[1].enable()

        # Let both settle into steady state before sampling.
        time.sleep(0.6)
        for _ in range(12):
            first, second = cameras[0].frame.value, cameras[1].frame.value
            if first >= 0 and second >= 0:
                skews.append(abs(first - second))
            time.sleep(0.05)
    finally:
        for camera in cameras:
            camera.stop()

    assert skews, "neither camera reported a frame index"
    return skews


@pytest.mark.functional
def test_barrier_keeps_staggered_playback_cameras_frame_aligned(videos, tmp_path):
    """The property live 3D depends on: same frame index on both cameras."""
    skews = _run(videos, tmp_path, use_barrier=True)

    worst = max(skews)
    logger.info("barrier skews: %s", skews)
    assert worst <= MAX_SYNCED_SKEW_FRAMES, (
        f"playback cameras drifted {worst} frames apart despite the start barrier; "
        f"samples={skews}"
    )


@pytest.mark.functional
def test_without_the_barrier_the_stagger_survives(videos, tmp_path):
    """Control: proves the barrier is doing the work, not the test setup.

    Without it, the delay between the two ENABLE_CAPTURE commands becomes a
    permanent frame offset, because each camera sets its own capture start on
    its own first frame.
    """
    skews = _run(videos, tmp_path, use_barrier=False)

    worst = max(skews)
    logger.info("unsynced skews: %s", skews)
    expected = STAGGER_SECONDS * FPS
    assert worst > MAX_SYNCED_SKEW_FRAMES, (
        f"expected roughly {expected:.0f} frames of skew without the barrier but saw "
        f"{worst}; if this passes, the barrier test above proves nothing"
    )
