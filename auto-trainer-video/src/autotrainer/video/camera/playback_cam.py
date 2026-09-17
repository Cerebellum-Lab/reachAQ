import time
import urllib.parse
from typing import Tuple

import cv2
import numpy

from . camera_base import CameraBase
from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)


class PlaybackCam(CameraBase):

    def __init__(self, file_name, name: str = ""):
        super().__init__(name)
        self._file_name = file_name
        self._file_name = urllib.parse.unquote(file_name)
        self._video_capture = None
        self._make_precise_timestamps = True
        # make_precise_timestamps:
        # used to bypass/workaround analyze code only being able to handle very precise timestamps
        self._video_frame_count = -1

    def init(self):
        vc = self._video_capture = cv2.VideoCapture(self._file_name)
        self.fps = vc.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30
        self.width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._video_frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.notice("init with fps=%s W=%s H=%s", self.fps, self.width, self.height)

    @property
    def frame_count(self) -> int:
        """How many frames the file holds, or -1 before init()."""
        return self._video_frame_count

    def seek(self, frame: int) -> int:
        """Jump playback to `frame`, returning where it actually landed.

        Clamped to the file rather than refused: the caller is a navigation
        control working from an event index, and an index that runs slightly
        past the end of a re-encoded clip should land on the last frame rather
        than do nothing.

        The frame counter moves with it and the pacing clock is re-based, so
        capture() does not then try to catch up on the frames it skipped - it
        paces against elapsed time since capture started, and seeking forward
        would otherwise make every subsequent frame look overdue and stream out
        as fast as the file can be decoded.
        """
        if self._video_capture is None:
            raise RuntimeError("seek before init()")
        target = max(0, int(frame))
        if self._video_frame_count > 0:
            target = min(target, self._video_frame_count - 1)
        self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._frame_count = target
        # Re-base so "now" corresponds to the frame just seeked to.
        self._capture_start = time.time_ns() - int(target * 1e9 / self._fps)
        logger.notice("%s: seek to frame %s", self._file_name, target)
        return target

    def capture(self) -> Tuple[numpy.ndarray, int]:
        ret, frame = self._video_capture.read()
        if not ret:
            if self._frame_count == self._video_frame_count:
                self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logger.notice("%s: loop-back to begin file", self._file_name)
                self._frame_count = 0
                ret, frame = self._video_capture.read()
            if not ret:
                raise RuntimeError(f"capture failed on {self._video_capture}")
        max_diff = 0.05 / self._fps
        while True:
            now = time.time_ns()
            delta = self._frame_count / self._fps - 1e-9 * (now - self._capture_start)
            if delta < max_diff:
                break
            time.sleep(0.5 * delta)
        super().capture()
        if self._make_precise_timestamps:
            new_last_when = self._capture_start + self._frame_count * int(1e9 / self._fps)
            self._last_when = new_last_when
        return frame[:, :, 1], self._last_when
