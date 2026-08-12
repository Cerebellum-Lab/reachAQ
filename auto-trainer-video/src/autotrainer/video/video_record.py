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

logger = get_verbose_logger(__name__)


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
        self._image_interval = properties.image_interval

        self._input_queue: Queue = input_queue
        self._record_stop_sema = record_stop_sema
        self._msg_queue = msg_queue

        self._is_running = True

        self._is_video_enabled = self._video_rotate_interval >= 0
        self._video_writer = None
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
            logger.exception("%s: Error during run: %s", self, err)
        self._close_writers()

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
                    self._close_writers()
                    closed_frames_written = tot_written
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
                        )))
                    tot_written = 0
                    continue

                for frame_id, frame, frame_when, frame_perf_now in queue_list:
                    # reconstructing frame_time (based on first frame start ~time):
                    estimated_frame_rel_t = (frame_id - self._first_frame_id) / fps
                    frame_time = self._first_frame_time + estimated_frame_rel_t

                    if self._is_video_enabled:
                        vid_writer = self._video_writer
                        if vid_writer is None:
                            # If triggered, may not be configured yet for this batch
                            prev_perf_now = prev_frame_when = None
                            self._prepare_writers()
                            vid_writer = self._video_writer

                        if vid_writer is not None:
                            if len(numpy.shape(frame)) < 3 or numpy.shape(frame)[2] == 1:
                                vid_writer.write(numpy.tile(frame[:, :, numpy.newaxis], (1, 1, 3)))
                            else:
                                vid_writer.write(frame)
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
                if consecutive_failures < 5:
                    logger.exception("%s: loop error: %s", self, err)
                consecutive_failures += 1

            try:
                self._check_writers()
            except Exception as err:
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
        vid_writer = cv2.VideoWriter(
            video_file, cv2.VideoWriter_fourcc(*'mp4v'), self._fps, (self._width, self._height))  # noqa
        if not vid_writer.isOpened():
            raise RuntimeError(f"Failed open {video_file} for writing")
        try:
            self._video_timestamp_file = open(timestamp_file, "w")
        except IOError as err:
            vid_writer.release()
            raise RuntimeError(f"Failed open {timestamp_file} for writing: {err}")
        self._video_file = video_file
        self._video_writer = vid_writer

    def _close_video_writer(self):
        vid_writer = self._video_writer
        if vid_writer is not None:
            vid_writer.release()
            logger.debug("Released %s", self._video_file)
            self._video_writer = None
            self._video_file = None

        vid_ts_file = self._video_timestamp_file
        if vid_ts_file is not None:
            vid_ts_file.flush()
            vid_ts_file.close()
            self._video_timestamp_file = None
