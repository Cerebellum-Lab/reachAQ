from __future__ import annotations

import math
import multiprocessing
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from multiprocessing import synchronize
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Optional, Tuple, TextIO

import cv2
import numpy

from autotrainer.core import ProjectInfo, ProjectInterval, SystemStatusMessageKind
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.logging import get_verbose_logger

from .ffmpeg_writer import FfmpegX264Writer

logger = get_verbose_logger(__name__)

VIDEO_ENCODERS = ("mp4v", "x264")


class VideoRecordMode(IntEnum):
    NONE = -1
    CONTINUOUS = 0
    TRIGGER = 1
    START_CONTINUOUS = 2


@dataclass
class VideoRecordProperties:

    project_info: Optional[ProjectInfo] = None
    """Information to determine file names and directories."""

    name: str = "camera"
    """Name used as part of video file names and image capture directory."""

    frame_size: Tuple[int, int] = (0, 0)
    """Expected shape (W, H) of video frames. Not required for image-only capture."""

    fps: int = 30
    """Expected FPS of video feed.  Not required for image-only capture."""

    record_mode: VideoRecordMode = VideoRecordMode.NONE
    """Continuous or triggered mode for video and image capture. NONE to disabled video recording."""

    video_rotate_interval: int = -1
    """Interval in seconds to rotate the video file.  0 to never rotate. Negative to disabled video recording."""

    image_interval: float = 0
    """Interval in seconds to capture images.  Values <= 0 disable image capture."""

    record_generation: Optional[Synchronized[int]] = None
    """Shared application session generation for stale-callback rejection."""

    encoder: str = "mp4v"
    """Video encoder: "mp4v" (OpenCV, MPEG-4 Part 2) or "x264" (H.264 through an ffmpeg
    process, fast enough for the larger capture presets; see ffmpeg_writer)."""

    queue_batch_size = 60
    """Number of frames to batch for passing between capture and record queues."""

    def should_record(self, is_triggered: bool, *, is_from_start: bool = False) -> bool:
        project = self.project_info
        any_active = (
            project is not None and project.is_valid()
            and (self.video_rotate_interval >= 0 or self.image_interval > 0)
        )
        logger.debug("should_record: vri=%s ii=%s any_active=%s is_from_start=%s",
                     self.video_rotate_interval, self.image_interval, any_active, is_from_start)
        if is_from_start:
            return any_active and self.record_mode == VideoRecordMode.START_CONTINUOUS
        if self.record_mode == VideoRecordMode.CONTINUOUS:
            return any_active
        if self.record_mode == VideoRecordMode.TRIGGER:
            return is_triggered and any_active
        return False


class VideoRecord(Thread):
    def __init__(
        self,
        properties: VideoRecordProperties,
        input_queue: Queue,
        *,
        cam_idx: int = -1,
        record_stop_sema: Optional[SemaphoreType] = None,
        msg_queue: Optional[multiprocessing.Queue] = None,
    ):
        super().__init__(name=properties.name, daemon=True)
        self._cam_idx = cam_idx
        self._project_info = properties.project_info
        self._prepared_project: Optional[ProjectInfo] = None
        self._record_generation = properties.record_generation
        self._prepared_generation = 0
        self._name = properties.name
        self._width = properties.frame_size[0]
        self._height = properties.frame_size[1]
        self._fps = properties.fps
        self._record_mode = properties.record_mode
        self._video_rotate_interval = properties.video_rotate_interval
        if properties.encoder not in VIDEO_ENCODERS:
            raise ValueError(f"unknown video encoder {properties.encoder!r}; expected one of {VIDEO_ENCODERS}")
        self._encoder = properties.encoder
        self._image_interval = properties.image_interval

        self._input_queue: Queue = input_queue
        self._record_stop_sema = record_stop_sema
        self._msg_queue = msg_queue

        self._is_running = True

        self._is_video_enabled = self._video_rotate_interval >= 0
        self._video_writer = None
        self._video_is_color = False
        self._video_file = None
        self._video_timestamp_file: Optional[TextIO] = None

        self._image_location: Optional[Path] = None
        self._image_name: Optional[str] = None
        self._last_image_perf_now = time.perf_counter()

        self._interval_mode = ProjectInterval.NONE
        self._interval_reference = -1
        self._first_frame_id = -1
        self._first_frame_when = math.inf
        self._first_frame_time = math.inf
        self._first_frame_perf_c = math.inf
        self._first_writer_error = ""
        self._writer_error_count = 0

    def _record_writer_error(self, error: BaseException) -> None:
        self._writer_error_count += 1
        if not self._first_writer_error:
            self._first_writer_error = (
                f"{error.__class__.__name__}: "
                f"{str(error) or error.__class__.__name__}"
            )

    def _take_writer_diagnostics(self) -> dict:
        diagnostics = {
            "firstError": self._first_writer_error or None,
            "errorCount": int(self._writer_error_count),
        }
        self._first_writer_error = ""
        self._writer_error_count = 0
        return diagnostics

    @property
    def first_frame_id(self) -> int:
        return self._first_frame_id

    @first_frame_id.setter
    def first_frame_id(self, value: int):
        self._first_frame_id = value

    @property
    def first_frame_time(self):
        return self._first_frame_time

    @first_frame_time.setter
    def first_frame_time(self, value):
        self._first_frame_time = value

    def run(self):
        logger.notice("%s: running", self)
        try:
            self._run()
        except Exception as err:
            self._record_writer_error(err)
            logger.exception("%s: Error during run: %s", self, err)
        try:
            self._close_writers()
        except Exception as err:
            self._record_writer_error(err)
            logger.exception("%s: Error closing writers: %s", self, err)

    def _run(self) -> None:
        input_q = self._input_queue

        project = self._project_info
        if project is None or not project.is_valid():
            logger.error("video recording and image capture can not proceed without value project information")
            return

        if self._record_mode in {VideoRecordMode.START_CONTINUOUS, VideoRecordMode.CONTINUOUS}:
            logger.verbose("Forcing interval HOUR")
            self._interval_mode = ProjectInterval.HOUR
            if self._record_mode == VideoRecordMode.START_CONTINUOUS:
                self._prepare_writers()

        prev_perf_now = prev_frame_when = None
        tot_written = 0
        consecutive_failures = 0
        record_stop_sema = self._record_stop_sema
        msg_queue = self._msg_queue

        fps = self._fps

        while self._is_running:

            try:
                queue_list = input_q.get(timeout=0.1)
            except Empty:
                continue
            input_q.task_done()  # always !

            try:
                # if frame is None or when is None:
                if len(queue_list) == 0:
                    # Indicator for trigger disabled
                    try:
                        self._close_writers()
                    except Exception as err:
                        self._record_writer_error(err)
                        logger.exception(
                            "%s: Error closing session writers: %s",
                            self,
                            err,
                        )
                    closed_frames_written = tot_written
                    writer_diagnostics = self._take_writer_diagnostics()
                    logger.info("Closed video file: tot frames written: %s ; last_perf_now=%s",
                                closed_frames_written, prev_perf_now)
                    if record_stop_sema is not None:
                        record_stop_sema.release()
                        logger.verbose("released record_stop_sema: %s", record_stop_sema)
                    if msg_queue is not None:
                        # allows main process to know when it can merge the cameras timestamp files.
                        msg_queue.put((
                            SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED, (
                                self._cam_idx, closed_frames_written, self._prepared_project,
                                self._prepared_generation,
                                writer_diagnostics,
                        )))
                    tot_written = 0
                    continue

                for frame_id, frame, frame_when, frame_perf_now in queue_list:
                    # reconstructing frame_time (based on first frame start ~time):
                    estimated_frame_rel_t = (frame_id - self._first_frame_id) / fps
                    frame_time = self._first_frame_time + estimated_frame_rel_t

                    if self._is_video_enabled:
                        if self._video_file is None:
                            # If triggered, may not be configured yet for this batch
                            prev_perf_now = prev_frame_when = None
                            self._prepare_writers()

                        if self._video_file is not None:
                            channels = self._frame_channels(frame)
                            if self._video_writer is None:
                                # Opened on the first frame, when its channel count is known.
                                self._open_video_writer(is_color=channels == 3)
                            self._video_writer.write(self._fit_to_writer(frame, channels))
                            tot_written += 1

                        vid_ts_file = self._video_timestamp_file
                        if vid_ts_file is not None:
                            d2 = self._fps  # currently keeping in timestamps.txt file for eventual back-compat
                            vid_ts_file.write(f"{frame_time}, {d2}, {frame_when}, {frame_perf_now}, {frame_id}\n")
                            prev_perf_now = frame_perf_now

                    if 0 < self._image_interval <= frame_perf_now - self._last_image_perf_now:
                        img_loc, img_name = self._image_location, self._image_name
                        if img_loc is None:
                            self._prepare_writers()
                            img_loc, img_name = self._image_location, self._image_name
                        if img_loc is not None and img_name is not None:
                            self._last_image_perf_now = frame_perf_now
                            when_str = datetime.fromtimestamp(frame_time).strftime("%Y%m%d_%H%M%S_%f")
                            when_str = when_str[:-3]  # only keep 3 digits precision (milliseconds)
                            cv2.imwrite(img_loc.joinpath(img_name.format(when=when_str)),
                                        frame)

            except Exception as err:
                self._record_writer_error(err)
                if consecutive_failures < 5:
                    logger.exception("%s: loop error: %s", self, err)
                consecutive_failures += 1

            try:
                self._check_writers()
            except Exception as err:
                self._record_writer_error(err)
                if consecutive_failures < 5:
                    logger.exception("%s: check writers error: %s", self, err)
                consecutive_failures += 1
            else:
                consecutive_failures = 0

        logger.notice("%s: main loop exited", self)

    def cancel(self):
        self._is_running = False

    def _check_writers(self):
        if self._interval_mode != ProjectInterval.NONE:
            timestamp = datetime.now()
            needs_update = timestamp.hour != self._interval_reference \
                if self._interval_mode == ProjectInterval.HOUR \
                else timestamp.minute != self._interval_reference

            if needs_update:
                self._prepare_writers()

    def _prepare_writers(self):
        logger.debug("preparing writers...")
        now = datetime.now()
        project = self._project_info
        if project is None or not project.is_valid():
            logger.warning("Cannot prepare writers with None project_info or not valid: %s", project)
            return
        self._interval_reference = project.get_interval(self._interval_mode, when=now)
        project = project.to_local_value()  # ensure it doesn't change for below
        self._prepared_generation = (
            0
            if self._record_generation is None
            else int(self._record_generation.value)
        )
        self._prepare_video_writer(project)
        self._prepare_image_capture(project)
        self._prepared_project = project

    def _close_writers(self):
        logger.spam("closing writers...")
        self._close_image_writer()
        self._close_video_writer()

    def _prepare_image_capture(self, project: ProjectInfo):
        logger.debug("preparing image capture")
        self._close_image_writer()
        if self._image_interval > 0:
            self._image_location, self._image_name = (
                project.get_image_capture_path(self._name, interval=self._interval_mode,
                                               when=datetime.now()))
            logger.debug(f"<{self.name}>: image capture to {self._image_location}")

    def _close_image_writer(self):
        self._image_location = None
        self._image_name = None

    def _prepare_video_writer(self, project: ProjectInfo):
        self._close_video_writer()
        if not self._is_video_enabled:
            logger.verbose("_prepare_video_writer but _is_video_enabled False")
            return

        video_file, timestamp_file, _ = project.get_video_path(
            self._name, interval=self._interval_mode, allow_overwrite=True)
        logger.notice("<%s>: video record to %s", self.name, video_file)

        Path(video_file).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._video_timestamp_file = open(timestamp_file, "w")
        except IOError as err:
            raise RuntimeError(f"Failed open {timestamp_file} for writing: {err}")
        # The video writer itself is opened on the first frame (_open_video_writer),
        # once the frames' channel count is known.
        self._video_file = video_file

    def _open_video_writer(self, *, is_color: bool):
        size = (self._width, self._height)
        if self._encoder == "x264":
            vid_writer = FfmpegX264Writer(self._video_file, self._fps, size, is_color)
        else:
            # Grayscale frames, which every reach camera delivers, go to a grayscale
            # writer: expanding them to three channels first cost 2.2 ms per
            # 1024x1024 frame and roughly halved mp4v throughput (christielab10,
            # 2026-10-06).
            vid_writer = cv2.VideoWriter(
                self._video_file, cv2.VideoWriter_fourcc(*'mp4v'), self._fps, size, isColor=is_color)  # noqa
        if not vid_writer.isOpened():
            raise RuntimeError(f"Failed open {self._video_file} for writing")
        self._video_writer = vid_writer
        self._video_is_color = is_color

    def _frame_channels(self, frame: numpy.ndarray) -> int:
        """1 for mono ((H, W) or (H, W, 1)), 3 for colour, at the recording's size; else refused.

        A writer drops a frame it cannot take without any error, so a wrong
        channel count or size would silently shift every later frame against
        its timestamp row. Refusing it records a writer error instead.
        """
        shape = numpy.shape(frame)
        channels = 1 if len(shape) == 2 else shape[2] if len(shape) == 3 else 0
        if channels not in (1, 3):
            raise ValueError(f"cannot record a {channels}-channel frame of shape {shape}")
        if tuple(shape[:2]) != (self._height, self._width):
            raise ValueError(f"frame of shape {shape} does not match the recording size "
                             f"{self._width}x{self._height}")
        return channels

    def _fit_to_writer(self, frame: numpy.ndarray, channels: int) -> numpy.ndarray:
        # The capture loop fills missed frames with mono zeros, so a colour
        # recording also receives mono frames (and could, the other way round).
        # Convert each frame to the writer's channel count rather than let the
        # writer drop it.
        mono = frame if numpy.ndim(frame) == 2 else frame[:, :, 0] if channels == 1 else None
        if self._video_is_color:
            return frame if channels == 3 else cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        return mono if mono is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _close_video_writer(self):
        # Cleared before releasing: an ffmpeg writer can raise on release (it
        # reports a failed encode then), and the recorder must still be left
        # closed, with the timestamp file flushed, for the next recording.
        vid_writer, self._video_writer = self._video_writer, None
        video_file, self._video_file = self._video_file, None
        try:
            if vid_writer is not None:
                vid_writer.release()
                logger.debug("Released %s", video_file)
        finally:
            vid_ts_file = self._video_timestamp_file
            if vid_ts_file is not None:
                self._video_timestamp_file = None
                vid_ts_file.flush()
                vid_ts_file.close()
