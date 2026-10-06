"""H.264 recording through an ffmpeg process, with cv2.VideoWriter's interface.

OpenCV's mp4v writer encodes on one thread. At the 1024x1024 capture preset
on a lit rig scene (+24 dB of gain noise) it managed 136 fps per stream with
two cameras in parallel, below the 150 fps the cameras deliver, so the
recorder queue filled. libx264 at the ultrafast preset and CRF 18 encoded the
same frames at 250-276 fps per stream, at 15-19 Mbit/s and a mean error of
0.8-0.9 grey levels (christielab10, 2026-10-06).
"""
import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

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

# The encoder runs below capture and live pose: it has throughput to spare at
# every preset, they have a deadline. Priority alone does not protect live
# pose at 1024x1024, though: with x264 free to use every core, 59% of frames
# were posed live, with or without it. See EFFICIENCY_CPUS_FILE.
ENCODER_NICENESS = 10

# Hybrid Intel processors (Alder Lake and later) list their efficiency cores
# here. On christielab10 (i9-12900: CPUs 0-15 are the 8 performance cores'
# hyperthreads, 16-23 the 8 efficiency cores) x264 running on the
# performance cores cut live pose at 1024x1024 to 59% of frames (5.89 ms
# inference); confined to the efficiency cores it kept up with both cameras
# and pose recovered to 87%, the same as the 256 base with no H.264 at all
# (85%) (2026-10-06).
EFFICIENCY_CPUS_FILE = "/sys/devices/cpu_atom/cpus"

# Long enough for ffmpeg to encode what is still in the pipe and write the
# file index at the end of a recording.
_CLOSE_TIMEOUT_S = 120


def ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg")


@functools.lru_cache(maxsize=None)
def x264_available() -> bool:
    """Whether the installed ffmpeg can encode with libx264 (checked once per process)."""
    executable = ffmpeg_executable()
    if executable is None:
        return False
    try:
        encoders = subprocess.run([executable, "-hide_banner", "-encoders"],
                                  capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return " libx264 " in encoders


def efficiency_cpus() -> Optional[FrozenSet[int]]:
    """The efficiency cores this process may use, or None if the processor has none.

    The kernel writes the list as ranges ("16-23", "0,2-3"); anything else
    reads as no efficiency cores, so the encoder is simply not confined.
    """
    try:
        text = Path(EFFICIENCY_CPUS_FILE).read_text().strip()
        cpus = set()
        for part in text.split(","):
            first, _, last = part.partition("-")
            cpus.update(range(int(first), int(last or first) + 1))
        cpus &= os.sched_getaffinity(0)
    except (OSError, ValueError, AttributeError):  # no such file, unparsable, or not Linux
        return None
    return frozenset(cpus) or None


class FfmpegX264Writer:
    """Pipes raw frames to `ffmpeg -c:v libx264`; mono or BGR, fixed size."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int], is_color: bool):
        executable = ffmpeg_executable()
        if executable is None:
            raise RuntimeError("ffmpeg is not installed; it is needed to record H.264 video")
        width, height = size
        self._path = path
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = self._start(executable, width, height, fps, is_color, path)
        except BaseException:
            self._stderr.close()
            raise

    def _start(self, executable, width, height, fps, is_color, path) -> subprocess.Popen:
        process = subprocess.Popen(
            [
                executable, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24" if is_color else "gray",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                *X264_OUTPUT_ARGS, path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            # Its own session: the capture process ignores a terminal's Ctrl+C,
            # and ffmpeg must not end the recording on one either.
            start_new_session=True,
        )
        try:
            os.setpriority(os.PRIO_PROCESS, process.pid, ENCODER_NICENESS)
        except (AttributeError, OSError):  # not POSIX, or not permitted: run at normal priority
            logger.warning("could not lower the ffmpeg encoder's priority for %s", path)
        cpus = efficiency_cpus()
        if cpus:
            # Set before the first frame is written: ffmpeg opens the encoder,
            # and so starts x264's threads, only once a frame arrives on stdin,
            # and those threads inherit this mask.
            try:
                os.sched_setaffinity(process.pid, cpus)
            except OSError:
                logger.warning("could not confine the ffmpeg encoder to the efficiency cores for %s", path)
        return process

    def _stderr_tail(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()[-500:]

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoWriter's name
        return self._process.poll() is None

    def write(self, frame: numpy.ndarray) -> None:
        # A C-contiguous copy only when the frame is a view (e.g. one channel
        # of a decoded frame); the flat byte view avoids a second copy.
        try:
            self._process.stdin.write(memoryview(numpy.ascontiguousarray(frame)).cast("B"))
        except OSError as err:  # BrokenPipeError: ffmpeg has exited
            try:
                returncode = self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                returncode = None
            raise RuntimeError(
                f"ffmpeg stopped (exit {returncode}) while recording {self._path}: {self._stderr_tail()}"
            ) from err

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
                # Terminate first: ffmpeg then still writes the file index, so
                # what was encoded stays readable. Kill only if that hangs too.
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError(f"ffmpeg did not finish {self._path} within {_CLOSE_TIMEOUT_S} s")
            if returncode != 0:
                raise RuntimeError(f"ffmpeg exited with {returncode} writing {self._path}: {self._stderr_tail()}")
        finally:
            self._stderr.close()
