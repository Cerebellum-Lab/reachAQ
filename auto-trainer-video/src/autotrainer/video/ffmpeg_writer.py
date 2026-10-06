"""H.264 recording through an ffmpeg process, with cv2.VideoWriter's interface.

OpenCV's mp4v writer encodes on one thread. At the 1024x1024 capture preset
on a lit rig scene (+24 dB of gain noise) it managed 136 fps per stream with
two cameras in parallel, below the 150 fps the cameras deliver, so the
recorder queue filled. libx264 at the ultrafast preset and CRF 18 encoded the
same frames at 250-276 fps per stream, at 15-19 Mbit/s and a mean error of
0.8-0.9 grey levels (christielab10, 2026-10-06).
"""
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy

from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)

X264_OUTPUT_ARGS = (
    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
    # Two cameras share the machine with capture and pose; six threads each
    # gave the throughput above without taking every core.
    "-threads", "6",
    "-pix_fmt", "yuv420p",
)

# Long enough for ffmpeg to encode what is still in the pipe and write the
# file index at the end of a recording.
_CLOSE_TIMEOUT_S = 120


def ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg")


class FfmpegX264Writer:
    """Pipes raw frames to `ffmpeg -c:v libx264`; mono or BGR, fixed size."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int], is_color: bool):
        executable = ffmpeg_executable()
        if executable is None:
            raise RuntimeError("ffmpeg is not installed; it is needed to record H.264 video")
        width, height = size
        self._path = path
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [
                executable, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24" if is_color else "gray",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                *X264_OUTPUT_ARGS, path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoWriter's name
        return self._process.poll() is None

    def write(self, frame: numpy.ndarray) -> None:
        # A C-contiguous copy only when the frame is a view (e.g. one channel
        # of a decoded frame); the flat byte view avoids a second copy.
        self._process.stdin.write(memoryview(numpy.ascontiguousarray(frame)).cast("B"))

    def release(self) -> None:
        process = self._process
        try:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass  # ffmpeg already exited; its return code says why
            try:
                returncode = process.wait(timeout=_CLOSE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError(f"ffmpeg did not finish {self._path} within {_CLOSE_TIMEOUT_S} s")
            if returncode != 0:
                self._stderr.seek(0)
                detail = self._stderr.read().decode(errors="replace").strip()[-500:]
                raise RuntimeError(f"ffmpeg exited with {returncode} writing {self._path}: {detail}")
        finally:
            self._stderr.close()
