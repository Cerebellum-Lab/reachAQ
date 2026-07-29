from __future__ import annotations

import logging.config
import multiprocessing
import math
import queue
import signal
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from multiprocessing import Process
from multiprocessing.synchronize import Semaphore as SemaphoreType
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray, SynchronizedString
from typing import Callable, Dict, Union, Optional, List, Tuple

import numpy

from autotrainer.core import FixedArrayMultiQueue, FixedArrayQueue, ProjectInfo, SystemStatusMessageKind, get_perf_now
from autotrainer.core.logging import (
    get_verbose_logger,
    install_log_exception_hook,
    log_hardware_initialization,
    make_log_dict_config,
    set_logger_level,
    setup_logging,
)
from autotrainer.core.frame_index import FrameIndexCategory
from autotrainer.core.fixed_array_queue import BufferResult
from autotrainer.core.video_detection import PresenceDetectionAttrs
from autotrainer.video.video_detection import VideoDetection
from autotrainer.core.capture import CaptureProcessStatus
from .camera.camera_base import CameraBase

from .video_manager import VideoManager
from .video_record import VideoRecord, VideoRecordProperties, VideoRecordMode

logger = get_verbose_logger(__name__)


# Always keep that extra more nbr of frames for cameras recording sync purpose:
_PREBUFFER_SYNC_KEEP = 3
assert _PREBUFFER_SYNC_KEEP > 0


class CaptureCommandKind(IntEnum):
    """Commands accepted by VideoCaptureProcess through the command Queue"""

    TERMINATE = 1
    """Fully terminate the Process"""

    ENABLE_CAPTURE = 2
    """Enable image capture"""

    DISABLE_CAPTURE = 3
    """Disable image capture"""

    ENABLE_RECORDING = 4
    """Enable image recording (requires capture enabled)"""

    DISABLE_RECORDING = 5
    """Disable image recording"""

    SET_LOGGER_LEVEL = 6
    """Set a logger log level"""


@dataclass
class CaptureCameraAttrs:
    """
    Camera attributes for VideoCapture configuration

    Attributes
    ----------
    name: str
        the name to use for the Process and in log messages
    url: str
        the camera URL for the capture process
    """
    name: str
    url: str


@dataclass
class CaptureInferenceAttrs:
    """
    Inference attributes for VideoCapture configuration

    Attributes
    ----------
    queue: Queue
        A queue to push frames for pellet
    index: int
        The index of this camera in the queue
    """
    queue: FixedArrayMultiQueue
    index: int


@dataclass
class CaptureAttrs:
    """
    Attributes for VideoCapture configuration
    """

    command_queue: multiprocessing.Queue
    """Input queue for submitting commands to the capture process"""

    status: Synchronized[int]
    """Flag for status of the capture process - value is read-only to callers"""

    image_queue: Optional[Union[queue.Queue, FixedArrayQueue]]
    """Queue for camera frame output"""

    frame: Synchronized[int]  # actually unused
    """Current frame index - value is read-only to callers"""

    camera: CaptureCameraAttrs
    """Camera attributes for the capture process"""

    errors: SynchronizedString
    """Shared string value for communicating errors - value is read-only to callers"""

    watchdog_perf_c: Optional[Synchronized[float]] = None
    """Watchdog shared perf counter"""

    synced_cam_record_enabled: Optional[Synchronized[bool]] = None
    """Shared bool value for communicating recording enabled from primary cam to secondary cam(s)"""

    synced_cam_frame_index: Optional[Synchronized[int]] = None
    """Multiprocessing int value for communicating synced frame index between cameras"""

    inference: Optional[CaptureInferenceAttrs] = None
    """Inference attributes for the capture process (optional)"""

    fps_image_queue: Optional[float] = 15
    """Desired write FPS for the image_queue, if None then default to capture FPS"""

    presence_detection_attrs: Optional[PresenceDetectionAttrs] = None
    """Optional Presence detection"""

    is_primary: bool = False

    msg_queue: Optional[multiprocessing.Queue] = None
    # optional msg queue to send messages to main process from camera process

    record_prebuffer_duration: float = 0

    camera_index: int = -1

    record_stop_sema: Optional[SemaphoreType] = None
    """Use by record thread to signal when it has closed its video file,
    so that offline reader thread can know when it can open the files for offline processing"""


class VideoCapture(Process):
    """
    Process-based class for video capture and recording.

    VideoCapture runs as a separate process for video capture and recording.
    An optional image queue can be provided to receive frames from the camera.
    A separate, optional "network/process" queue can be provided to feed inference models or any other process.
    """

    def __init__(
        self,
        attrs: CaptureAttrs,
        record_properties: Optional[VideoRecordProperties] = None,
        project_info: Optional[ProjectInfo] = None,
    ):
        logger.debug("project_info=%s", project_info)
        log_dict_config = make_log_dict_config()
        super().__init__(
            name=attrs.camera.name,
            target=self._do_run,
            kwargs=dict(log_dict_config=log_dict_config),
            daemon=True,
        )

        self._attrs = attrs
        self._name = attrs.camera.name
        self._camera_url = attrs.camera.url

        self._project_info = project_info
        self._command_queue = attrs.command_queue
        self._command_thread: Optional[threading.Thread] = None
        self._camera_idx = attrs.camera_index
        self._status = attrs.status
        self._image_queue: Optional[Union[queue.Queue, FixedArrayQueue]] = attrs.image_queue
        self._image_queue_frame_delay = None if attrs.fps_image_queue is None else 1 / attrs.fps_image_queue

        self._is_record_active = False

        if record_properties is None:
            record_properties = VideoRecordProperties(record_mode=VideoRecordMode.NONE)
        self._record_properties = record_properties
        self._is_record_active = record_properties.should_record(False, is_from_start=True)
        self._record_batch_size = record_properties.queue_batch_size

        self._errors = attrs.errors

        self._is_running = True
        self._is_capturing = False
        self._camera = None
        self._record: Optional[VideoRecord] = None
        self._record_queue: Optional[queue.Queue] = None
        self._record_queue_list: List[
                  # frame_id , frame, frame_when, frame_perf
            Tuple[int, numpy.ndarray, float, float]] = []

        self._detection_attrs = attrs.presence_detection_attrs
        self._video_detection: Optional[VideoDetection] = None

        self._command_handlers: Dict[CaptureCommandKind, Callable] = {
            CaptureCommandKind.TERMINATE: self._user_terminate,
            CaptureCommandKind.ENABLE_CAPTURE: self._begin_capture,
            CaptureCommandKind.DISABLE_CAPTURE: self._end_capture,
            CaptureCommandKind.ENABLE_RECORDING: self._enable_record,
            CaptureCommandKind.DISABLE_RECORDING: self._disable_record,
            CaptureCommandKind.SET_LOGGER_LEVEL: set_logger_level,
        }

        self._set_status(CaptureProcessStatus.INITIALIZED)

    def _do_run(self, log_dict_config: Optional[Dict]):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if log_dict_config is None:
            setup_logging(logger_level=logging.DEBUG)
        else:
            logging.config.dictConfig(log_dict_config)
            install_log_exception_hook()

        logger.info("%s: started running ; name=%s cam_index=%s primary=%s log_dict=%s",
                    self, self._attrs.camera.name, self._camera_idx, self._attrs.is_primary,
                    log_dict_config)
        log_hardware_initialization(
            logger,
            "START | camera child process | name=%s index=%s primary=%s",
            self._name,
            self._camera_idx,
            self._attrs.is_primary,
        )
        if not self._prepare_to_run():
            return

        try:
            self._run_capture_loop(self._camera)
        except BaseException as err:
            logger.exception("Fatal error: %s", err)
            save_err = str(err)
        else:
            save_err = None
        self._terminate_capture_loop(save_err)

    def _set_status(self, status: CaptureProcessStatus):
        self._status.value = status
        # only relaying start/stop recording equivalent in run_capture_loop method
        # msg_q = self._attrs.msg_queue
        # if self._attrs.is_primary and msg_q is not None:
        #     msg_q.put(status)

    def _set_error(self, error: str):
        logger.error("set_error: %s", error)
        self._set_status(CaptureProcessStatus.FAILED)
        if self._errors:
            self._errors.value = f"{error}"[:len(self._errors)].encode()

    def _prepare_to_run(self) -> bool:
        logger.info("<%s> process started: %s", self._name, self._attrs.inference)
        project = self._project_info
        try:
            if self._camera_url is None:
                self._set_error("camera_url not specified")
                return False

            try:
                camera_started = time.perf_counter()
                log_hardware_initialization(
                    logger,
                    "START | camera backend initialization | name=%s url=%s",
                    self._name,
                    self._camera_url,
                )
                camera = self._camera = VideoManager.create_camera(self._camera_url, self._name)
                if camera is None:
                    raise RuntimeError("VideoManager returned None")
            except BaseException as err:
                log_hardware_initialization(
                    logger,
                    "FAILED | camera backend initialization | name=%s elapsed=%.3fs error=%s",
                    self._name,
                    time.perf_counter() - camera_started,
                    str(err) or err.__class__.__name__,
                    level=logging.ERROR,
                )
                self._set_error(f"Could not create camera {self._name}: {err}")
                return False

            log_hardware_initialization(
                logger,
                "READY | camera backend initialization | name=%s type=%s width=%s height=%s fps=%s elapsed=%.3fs",
                self._name,
                camera.__class__.__name__,
                camera.width,
                camera.height,
                camera.fps,
                time.perf_counter() - camera_started,
            )
            prepare_started = time.perf_counter()
            log_hardware_initialization(logger, "START | camera prepare_capture | name=%s", self._name)
            camera.prepare_capture()
            log_hardware_initialization(
                logger,
                "READY | camera prepare_capture | name=%s elapsed=%.3fs",
                self._name,
                time.perf_counter() - prepare_started,
            )

            rec_queue = queue.Queue(maxsize=128)
            # NB: we put batch per batch (of self._record_batch_size) into the queue,
            # so this makes actually space for 128 * record_batch_size frames,
            # which is at the moment 60, so that's actually more than 45s at 150 FPS.
            self._record_queue = rec_queue
            self._record_properties.name = self._name
            self._record_properties.frame_size = (camera.width, camera.height)
            self._record_properties.fps = camera.fps

            vid_rec = self._record = VideoRecord(
                self._record_properties, rec_queue,
                record_stop_sema=self._attrs.record_stop_sema,
                msg_queue=self._attrs.msg_queue,
                cam_idx=self._camera_idx,
            )
            vid_rec.start()

            det_attrs = self._attrs.presence_detection_attrs
            if project is None or det_attrs is None:
                self._video_detection = None
            else:
                vid_det = self._video_detection = VideoDetection(project, det_attrs)
                vid_det.start()

            logger.verbose("%s: video_detection: %s", self._name, self._video_detection)

            # only start command handler thread after others before, given it's checking alive of them.
            thread = self._command_thread = threading.Thread(
                target=self._command_handler, daemon=True, name="CommandHandler")
            thread.start()

            log_hardware_initialization(
                logger,
                "READY | camera child process | name=%s recorder_alive=%s detection_enabled=%s",
                self._name,
                vid_rec.is_alive(),
                self._video_detection is not None,
            )

            return True
        except Exception as err:
            logger.exception("%s: Error during prepare to run: %s", self, err)
            log_hardware_initialization(
                logger,
                "FAILED | camera child process | name=%s error=%s",
                self._name,
                str(err) or err.__class__.__name__,
                level=logging.ERROR,
            )
            self._set_error(str(err))
            return False

    def _command_handler(self):
        while True:
            vid_rec = self._record
            if vid_rec is not None and not vid_rec.is_alive():
                logger.warning("VideoRecord not alive, terminating")
                self._user_terminate()
                self._set_error("video_record thread dead")
                break
            vid_det = self._video_detection
            if vid_det is not None and not vid_det.is_alive():
                logger.warning("Video Detection not alive, terminating")
                self._user_terminate()
                self._set_error("video_detection thread dead")
                break
            try:
                raw = self._command_queue.get(timeout=1)
            except queue.Empty:
                continue
            if raw is None:
                break
            try:
                cmd, context = raw
                self._handle_command(cmd, context)
            except Exception as err:
                logger.exception("Failure executing cmd %s: %s", raw, err)

    def _run_capture_loop(self, camera: CameraBase) -> None:
        log_cam_frame_info_delay_frame_count = camera.fps * 5
        fault_count = 0
        cnt_net_q_put = 0
        is_record_active = False
        cam_frame_id = -1  # expected first frame id of any new acquisition is 0.
        count_frames_received = 0
        when_secs = math.nan
        attrs = self._attrs
        is_primary = attrs.is_primary
        frame_period = 1 / camera.fps
        prim_cam_record_enabled = attrs.synced_cam_record_enabled
        prim_cam_synced_frame_idx = attrs.synced_cam_frame_index
        synced_frame_idx: Optional[int] = None
        msg_q = attrs.msg_queue  # message queue to main process
        record_start_stop_frame_idx: Optional[int] = None
        last_recorded_frame_id = -1
        last_recorded_frame_perf = math.nan
        next_t_image_q = time.perf_counter()
        img_q = self._image_queue  # image queue to main/GUI process view
        record_q_list = self._record_queue_list
        record_q = self._record_queue  # record queue to file
        if record_q is None:
            raise RuntimeError("supposed be created")
        rec_q_put = partial(record_q.put, timeout=3)
        # using a ~small timeout on record_q put, to prevent deadlock if queue is full, given it has a maxsize.
        # but the size is quite large + we write batch of frames, and command handler is supposed to detect possible
        # exit of the record thread, and then trigger a full exit of the capture process.
        # So this is extra protection to not hang the capture/main thread.
        cam_capture = camera.capture
        inference = attrs.inference
        if inference is None:
            net_q = net_q_idx = net_q_put = None
        else:
            net_q = inference.queue
            net_q_put = net_q.put
            net_q_idx = inference.index
            # although is same than self._camera_idx
        image_queue_delay = self._image_queue_frame_delay
        empty_frame = numpy.zeros((camera.height, camera.width), dtype=numpy.uint8)
        vid_detection = self._video_detection
        p_prev_watchdog = -math.inf
        if attrs.watchdog_perf_c is not None:
            def set_watchdog(value):
                nonlocal p_prev_watchdog
                if value - p_prev_watchdog >= 0.2:
                    p_prev_watchdog = attrs.watchdog_perf_c.value = value
        else:
            def set_watchdog(_):
                """Void set_watchdog without shared value"""
        #
        frames_prebuffer_list: List[Tuple[numpy.ndarray, Union[int, float], float, float, int]] = []
        #                           frame, frame_when, time, perf_now, frame-idx
        def update_frames_prebuffer(f, fw, t, p, fidx):
            idx = len(frames_prebuffer_list) - 1
            while True:
                # keep always more, for sync purpose,
                # Allowing to ensure that the primary and secondary cams have always the same start frame
                # in common available for use for a start/stop recording.
                if idx <= _PREBUFFER_SYNC_KEEP:
                    break
                if frame_perf_c - frames_prebuffer_list[idx][3] >= attrs.record_prebuffer_duration:
                    del frames_prebuffer_list[:idx - _PREBUFFER_SYNC_KEEP]  # keep more !
                    break
                idx -= 1
            frames_prebuffer_list.append((f, fw, t, p, fidx))

        # primary_acquire/release:
        # Ensure sync, or normal, cameras will use the same frame index for the start frame of all recordings.
        def primary_acquire(frame_idx, enabled):
            nonlocal synced_frame_idx
            if synced_frame_idx is not None:
                return
            if not is_primary and prim_cam_record_enabled is not None:
                # non-primary cams get synced via secondary_acquire()
                return
            target_idx = frame_idx
            if enabled:
                idx = len(frames_prebuffer_list) - 1
                while idx >= 0:
                    frame_p = frames_prebuffer_list[idx][3]
                    if math.isfinite(frame_p):
                        if perf_now - frame_p >= attrs.record_prebuffer_duration:
                            logger.debug("will use frames_prebuffer_list at %s for synced_frame_idx", idx)
                            break
                        target_idx = frame_idx  # keep the most valid one
                    idx -= 1
                if len(frames_prebuffer_list) > 0:
                    idx = idx if idx >= 0 else 0
                    # keep frame_idx if it was drop in prebuffer:
                    if math.isfinite(frames_prebuffer_list[idx][2]):  # frame_when, could check frame_perf as well
                        target_idx = frames_prebuffer_list[idx][-1]

            synced_frame_idx = target_idx
            if prim_cam_record_enabled is not None:
                with prim_cam_record_enabled:  # acquire lock
                    prim_cam_synced_frame_idx.value = target_idx
                    prim_cam_record_enabled.value = enabled
            logger.verbose("Set target rec_enabled=%s frame_id=%s ; frame_idx=%s",
                           enabled, target_idx, frame_idx)

        def secondary_acquire():
            nonlocal synced_frame_idx
            if prim_cam_record_enabled is None:
                return
            assert prim_cam_synced_frame_idx is not None
            if not is_primary:
                with prim_cam_record_enabled:
                    synced_frame_idx = prim_cam_synced_frame_idx.value
                logger.verbose("got synced frame_id=%s ; cur frame_id=%s",
                               synced_frame_idx, cam_frame_id)

        def perform_stop_recording(force: bool=False):
            nonlocal record_start_stop_frame_idx, synced_frame_idx, record_q_list
            nonlocal cnt_net_q_put, last_recorded_frame_id, last_recorded_frame_perf
            if force:
                synced_frame_idx = cam_frame_id
            if cam_frame_id >= synced_frame_idx:
                record_start_stop_frame_idx = None
            #
            idx = len(record_q_list) - 1
            logger.debug("checking record_q_list for synced_frame: %s",
                         tuple(t[0] for t in record_q_list))
            while idx >= 0:
                if record_q_list[idx][0] > synced_frame_idx:
                    if idx == 0 or record_q_list[idx - 1][0] <= synced_frame_idx:
                        break
                idx -= 1
            if idx >= 0:
                logger.debug("cutted record_q_list at %s, len=%s", idx, len(record_q_list))
                del record_q_list[idx:]
            final_frame_id = last_recorded_frame_id
            final_frame_perf = last_recorded_frame_perf
            if len(record_q_list) > 0:
                final_frame_id = record_q_list[-1][0]
                final_frame_perf = record_q_list[-1][3]
                rec_q_put(record_q_list)
                record_q_list = self._record_queue_list = []
            if cam_frame_id <= synced_frame_idx:
                # don't miss this one too
                rec_q_put([(cam_frame_id, frame, when, frame_perf_c)])
                final_frame_id = cam_frame_id
                final_frame_perf = frame_perf_c
            rec_q_put([])  # empty list is mark for EOR for recorder thread

            synced_frame_idx = None  # don't forget now.

            if net_q is not None:
                # we might eventually have written some extra frame(s) vs the other camera(s) used in
                # the net_q, so this pad_to_batch_size :
                net_q.pad_to_batch_size(net_q_idx, empty_frame, cnt_net_q_put, timeout=5)
                # required: must set back to 0 given will now be same in all cams,
                # and also aligned with frames_per_camera_per_batch
                cnt_net_q_put = 0
                # now
                logger.info(
                    "sending EOF_RECORDING batch frame indices to signify eof recording. "
                    "last frame_id: %s when=%.4f perf=%.4f",
                    cam_frame_id, when_secs, frame_perf_c)
                net_q.put_frame_index_category(empty_frame, FrameIndexCategory.EOF_RECORDING,
                                               cam_idx=net_q_idx, timeout=5)

            self._set_status(CaptureProcessStatus.RUNNING)

            if msg_q is not None:
                msg_q.put((SystemStatusMessageKind.CAMERA_STATUS_CHANGE,
                           (self._camera_idx, CaptureProcessStatus.RUNNING,
                            final_frame_perf, final_frame_id)))

        logger.notice("starting capture loop ..")
        self._set_status(CaptureProcessStatus.RUNNING)
        if msg_q is not None:
            logger.verbose("cam%s: sending running", self._camera_idx)
            msg_q.put((SystemStatusMessageKind.CAMERA_STATUS_CHANGE, (self._camera_idx, CaptureProcessStatus.RUNNING)))

        frame_perf_c = math.nan
        frame_time = time.time()
        save_err = None

        while True:

            if not self._is_running:
                if record_start_stop_frame_idx is not None:
                    synced_frame_idx = cam_frame_id
                    perform_stop_recording(force=True)
                break

            prev_frame_when_secs = when_secs
            prev_frame_id = cam_frame_id
            prev_frame_perf_now = frame_perf_c
            prev_frame_time = frame_time

            if fault_count > 5:
                logger.critical("Too many capture loop processing errors ; giving up")
                if record_start_stop_frame_idx is not None:
                    synced_frame_idx = cam_frame_id
                    perform_stop_recording(force=True)
                self._end_capture()
                self._user_terminate()
                break

            if is_primary or prim_cam_record_enabled is None:
                # for primary cam or non-synced cam(s):
                is_record_active = self._is_record_active
                # secondary cam reads this value from the primary cam shared flag below.

            perf_now = time.perf_counter()
            set_watchdog(perf_now)

            if not self._is_capturing:
                if record_start_stop_frame_idx is not None:
                    synced_frame_idx = cam_frame_id  # ensure immediate stop
                    perform_stop_recording(force=True)
                time.sleep(0.001)
                frame_time = time.time()
                continue

            try:
                # this eventually set/unset recording enabled on the primary cam, or on non-synced cam(s):
                # + 1 because next frame will have that frame_id
                if is_record_active and record_start_stop_frame_idx is None and synced_frame_idx is None:
                    # prev_frame_id is normally a valid acquired frame,
                    # which should always allow to have one valid for start-recording, even in case of heavy drops.
                    primary_acquire(prev_frame_id, enabled=True)
                elif not is_record_active and record_start_stop_frame_idx is not None and synced_frame_idx is None:
                    primary_acquire(prev_frame_id + 1, enabled=False)

                # camera capture:
                frame, when = cam_capture()
                if frame is None:
                    save_err = "camera capture returned None"
                    logger.error("Failed to capture a frame (frame = None) ; prev_frame_id=%s",
                                 prev_frame_id)
                    fault_count += 1
                    continue

                perf_now = time.perf_counter()

                cam_frame_id = camera.frame_id
                # NB: the frame 'when' can be in different clock than what we can assume,
                # using time.time() and .perf_counter() for precision :
                frame_perf_c = camera.frame_perf_c
                frame_time = camera.frame_unix_time
                count_frames_received += 1
                frame_late_delay = perf_now - frame_perf_c
                #
                count_missed_frames = cam_frame_id - (1 + prev_frame_id)
                when_secs = when / 1e9
                if count_missed_frames > 0:
                    # insert blank/black missed frames:
                    frames_prebuffer_list.extend(
                        # frame, frame_when, frame_time, frame_perf_c, frame_id
                        (empty_frame, math.nan, prev_frame_time + idx1 * frame_period, math.nan, missed_frame_id)
                        for idx1, missed_frame_id in enumerate(
                            range(prev_frame_id + 1, prev_frame_id + 1 + count_missed_frames),
                            start=1,
                        )
                    )
                # perf_frame_dropped = (
                #     0
                #     if not math.isfinite(prev_frame_perf_now)
                #     else int((frame_perf_now - prev_frame_perf_now - 1.5 * frame_normal_delay) / frame_normal_delay)
                # )
                if (
                    inference is not None
                    and (
                        count_missed_frames > 0
                        # or frame_late_delay > 0.15  ~arbitrary
                    )
                ):
                    effective_frame_dropped = cam_frame_id - prev_frame_id - 1
                    logger.warning("frame_id=%s (prev=%s) detected frame dropped=%s "
                                   "diff_when=%.3f prev_when=%.3f frame_when=%.3f "
                                   "diff_perf=%.3f prev_perf=%.3f frame_perf=%.3f",
                                    cam_frame_id, prev_frame_id,
                                   effective_frame_dropped, when_secs - prev_frame_when_secs,
                                   prev_frame_when_secs, when_secs,
                                   frame_perf_c - prev_frame_perf_now, prev_frame_perf_now, frame_perf_c)

                if count_frames_received == 1:
                    self._record.first_frame_time = frame_time
                    logger.success("captured first frame id=%s ; cam_when=%.4f cam_perf_c=%.4f",
                                   cam_frame_id, when_secs, frame_perf_c)
                elif inference is not None and (cam_frame_id % log_cam_frame_info_delay_frame_count == 0 or cam_frame_id < 300):
                    if (
                        cam_frame_id % log_cam_frame_info_delay_frame_count == 0
                        or (cam_frame_id < 300 and cam_frame_id % 64 == 0)
                        or (cam_frame_id < 64 and cam_frame_id % 16 == 0)
                        or (cam_frame_id < 32 and cam_frame_id % 4 == 0)
                        or cam_frame_id < 4
                    ):
                        logger.debug("got frame_id=%s frame_when=%.4f frame_perf=%.4f delay=%.4f",
                                     cam_frame_id, when_secs, frame_perf_c, frame_late_delay)

                if img_q is not None:
                    # image queue goes to GUI video reader frame, currently FixedArrayQueue
                    if perf_now >= next_t_image_q:
                        if image_queue_delay is not None:
                            next_t_image_q = perf_now + image_queue_delay
                        if len(numpy.shape(frame)) < 3:
                            img_q.put(frame)
                        else:
                            img_q.put(frame[:, :, 0])

                if prim_cam_record_enabled is not None and not is_primary:
                    # for secondary synced cams we don't have other choice than to read
                    # the primary cam recording enabled shared flag on each frame read:
                    is_record_active = prim_cam_record_enabled.value
                    # other possibility is to use a synchronized/acked message from primary to all secondary cams.

                # record queue goes to video save to disk/file
                if is_record_active and record_start_stop_frame_idx is None:
                    secondary_acquire()
                    if synced_frame_idx is not None and cam_frame_id >= synced_frame_idx:
                        record_start_stop_frame_idx = synced_frame_idx
                        synced_frame_idx = None
                    if record_start_stop_frame_idx is not None:
                        record_start_stop_frame_idx: int
                        cut_idx = len(frames_prebuffer_list) - 1
                        # use the frame_id to be sure:
                        while cut_idx >= 0:
                            if frames_prebuffer_list[cut_idx][-1] < record_start_stop_frame_idx:
                                logger.debug(
                                    "cutting frames_prebuffer_list by %s ; frame_id=%s len(prebuff)=%s",
                                    cut_idx, cam_frame_id, len(frames_prebuffer_list),
                                )
                                del frames_prebuffer_list[:cut_idx + 1]
                                break
                            cut_idx -= 1
                        #
                        first_frame_id = (
                            cam_frame_id if len(frames_prebuffer_list) == 0
                            else frames_prebuffer_list[0][-1]
                        )
                        first_frame_when = (
                            when
                            if len(frames_prebuffer_list) == 0
                            else frames_prebuffer_list[0][1]
                        )
                        first_frame_time = (
                            frame_time
                            if len(frames_prebuffer_list) == 0
                            else frames_prebuffer_list[0][2]
                        )
                        first_frame_p_now = (
                            frame_perf_c
                            if len(frames_prebuffer_list) == 0
                            else frames_prebuffer_list[0][3]
                        )
                        logger.notice(
                            "Starting record with frame_idx=%s cam_frame_id=%s perf_now=%.4f when=%.4f ; prebuffer_cnt=%s",
                            record_start_stop_frame_idx, first_frame_id,
                            first_frame_p_now, first_frame_when / 1e9,
                            len(frames_prebuffer_list),
                        )
                        #
                        self._record.first_frame_id = first_frame_id
                        self._record.first_frame_time = first_frame_time
                        #
                        if len(frames_prebuffer_list) > 0:
                            rec_q_put(
                                [(fix, f, fw, p) for f, fw, _, p, fix in frames_prebuffer_list]
                            )
                            last_recorded_frame_id = frames_prebuffer_list[-1][-1]
                            last_recorded_frame_perf = frames_prebuffer_list[-1][3]
                            frames_prebuffer_list = []  # thread safety, don't use .clear()
                        rec_q_put([(cam_frame_id, frame, when, frame_perf_c)])  # thread queue
                        last_recorded_frame_id = cam_frame_id
                        last_recorded_frame_perf = frame_perf_c
                        record_q_list = (
                            self._record_queue_list
                        ) = []  # ensure we (re)start clean

                        #
                        self._set_status(CaptureProcessStatus.RECORDING)
                        #
                        if msg_q is not None:
                            msg_q.put((SystemStatusMessageKind.CAMERA_STATUS_CHANGE,
                                       (self._camera_idx, CaptureProcessStatus.RECORDING,
                                        first_frame_p_now, first_frame_when, first_frame_time)))

                elif not is_record_active and record_start_stop_frame_idx is not None:
                    secondary_acquire()
                    if synced_frame_idx is not None:
                        if cam_frame_id < synced_frame_idx:
                            # still put
                            record_q_list.append(
                                (cam_frame_id, frame, when, frame_perf_c)
                            )
                        else:
                            perform_stop_recording()

                elif record_start_stop_frame_idx is not None:
                    # normal recording case in progress
                    if count_missed_frames > 0:
                        record_q_list.extend(
                            # frame-id, frame, frame_when, frame_perf_c
                            (prev_frame_id + idx1, empty_frame, math.nan, prev_frame_perf_now + idx1 * frame_period)
                            for idx1 in range(1, count_missed_frames + 1)
                        )
                    record_q_list.append((cam_frame_id, frame, when, frame_perf_c))
                    last_recorded_frame_id = cam_frame_id
                    last_recorded_frame_perf = frame_perf_c
                    if len(record_q_list) >= self._record_batch_size:
                        rec_q_put(record_q_list)
                        record_q_list = self._record_queue_list = []

                if net_q_put is not None:
                    # network queue goes to processing/inference
                    frame_idx_cat = (
                        FrameIndexCategory.ONLINE_NO_RECORDING if record_start_stop_frame_idx is None
                        else cam_frame_id - record_start_stop_frame_idx
                    )
                    if net_q_put(frame, net_q_idx, frame_idx_cat, block=False) == BufferResult.Ok:
                        cnt_net_q_put += 1

                if vid_detection is not None:
                    vid_detection.update_frame(when, frame, frame_perf_c)

                # if not (is_record_active and record_start_stop_frame_idx is not None) and attrs.record_prebuffer_duration > 0:
                update_frames_prebuffer(frame, when, frame_time, frame_perf_c, cam_frame_id)

            except Exception as err:
                logger.exception("Error during capture loop: %s", err)
                if save_err is not None:
                    save_err = str(err)
                fault_count += 1
            else:
                save_err = None
                fault_count = 0

        # end while self._is_running
        if save_err is not None:
            self._set_error(save_err)

    def _terminate_capture_loop(self, error: Optional[str]):
        camera = self._camera
        try:
            logger.info(f"<{self._name}> capture loop ended")

            if camera is not None:
                camera.end_capture()

            vid_rec = self._record
            if vid_rec is not None:
                vid_rec.cancel()
                logger.debug("joining record thread")
                vid_rec.join()
                self._record = None

            video_detection = self._video_detection
            if video_detection is not None:
                video_detection.cancel()
                logger.debug("joining video-detection thread")
                video_detection.join()
                self._video_detection = None

            logger.debug("joining command thread")
            thread = self._command_thread
            if thread is not None:
                if thread.is_alive():
                    self._command_queue.put(None)
                thread.join()

        except Exception as err:
            logger.exception("%s: terminate capture loop error: %s", self, err)
            if error is None:  # keep orig error
                error = str(err)
        # set error if not already failed:
        is_failed = CaptureProcessStatus(self._status.value) == CaptureProcessStatus.FAILED
        if error is not None and not is_failed:
            self._set_error(error)
        else:
            # set terminated if not already failed:
            if not is_failed:
                self._set_status(CaptureProcessStatus.TERMINATED)
        logger.debug("exiting")

    def _handle_command(self, cmd: CaptureCommandKind, context: object):
        logger.info("executing %s", cmd)
        handler = self._command_handlers.get(cmd)
        if handler is None:
            logger.warning("No handler for command %s", cmd)
        else:
            if context is None:
                args = ()
                kwargs = None
            else:
                args, kwargs = context
                args: Tuple
                kwargs: Optional[Dict]
            handler(*args) if kwargs is None else handler(*args, **kwargs)
            logger.debug("status: capturing=%s recording=%s", self._is_capturing, self._is_record_active)

    def _user_terminate(self):
        self._is_running = False

    def _begin_capture(self):
        self._is_capturing = True

    def _end_capture(self):
        self._is_capturing = False

    def _enable_record(self, *, is_from_start: bool=False):
        self._is_record_active = self._record_properties.should_record(True, is_from_start=is_from_start)
        logger.verbose("_enable_record: is_record_active=%s", self._is_record_active)

    def _disable_record(self, *, is_triggered: Optional[bool]=False, is_from_start: bool=False):
        entry_is_triggered = is_triggered
        if is_triggered is None:
            is_triggered = self._is_record_active
        self._is_record_active = self._record_properties.should_record(is_triggered, is_from_start=is_from_start)
        logger.verbose("_disable_record(is_triggered=%s, is_from_start=%s): is_record_active=%s",
                       entry_is_triggered, is_from_start, self._is_record_active)
