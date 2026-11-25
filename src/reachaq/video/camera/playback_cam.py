import time
import urllib.parse
from typing import Tuple

import cv2
import numpy

from . camera_base import CameraBase
from reachaq.core.logging import get_verbose_logger

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
