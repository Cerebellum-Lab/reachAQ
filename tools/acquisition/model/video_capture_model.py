import ctypes
import logging
import math
import multiprocessing
import os
import pathlib
import time
from multiprocessing.context import BaseContext
from multiprocessing.sharedctypes import (
    SynchronizedString,
    Synchronized,
)
from multiprocessing.synchronize import Semaphore as SemaphoreType
from typing import Optional, List, Tuple, Dict, Any, Iterable, Union
from threading import Event
import urllib.parse

import numpy
from numpy import ndarray

from autotrainer.core import clear_queue, FixedArrayQueue, FixedArrayMultiQueue, ObservableObject, \
    CameraConfiguration, CameraId, NotificationCenter, TriggerNotification, Notification, get_verbose_logger, \
    get_perf_now
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.core.logging import log_hardware_initialization
from autotrainer.core.project import ProjectInfo, ProjectDependentProtocol
from autotrainer.core.video_detection import PresenceDetectionAttrs
from autotrainer.video import VideoCapture, VideoRecordProperties, VideoRecordMode, VideoManager, \
    VideoReader, CaptureCommandKind, CaptureCameraAttrs, CaptureInferenceAttrs, CaptureAttrs
from autotrainer.core.capture import CaptureProcessStatus

from tools.acquisition.model.user_preferences import UserPreferences

logger = get_verbose_logger(__name__)


def create_camera_list(*, include_hardware: bool = False):
    cameras = list()

    cameras.append(CaptureCameraAttrs(name="Random Image", url="random://0?width=300&height=200"))

    loc = pathlib.Path(__file__).parent.resolve().parents[2].joinpath("cameras.txt")

    if os.path.isfile(loc):
        file = open(loc, "r")
        lines = file.readlines()
        file.close()
        for line in lines:
            parts = line.split(",")
            if len(parts) == 2:
                _append_unique_camera(cameras, parts[0].strip(), parts[1].strip())

    if include_hardware:
        started = time.perf_counter()
        log_hardware_initialization(logger, "START | camera discovery | backend=spinnaker")
        try:
            spin_serials = tuple(VideoManager.list_spin_cameras())
        except Exception as exc:
            log_hardware_initialization(
                logger,
                "FAILED | camera discovery | backend=spinnaker elapsed=%.3fs error=%s",
                time.perf_counter() - started,
                str(exc) or exc.__class__.__name__,
                level=logging.ERROR,
            )
            raise
        log_hardware_initialization(
            logger,
            "READY | camera discovery | backend=spinnaker count=%d elapsed=%.3fs serials=%s",
            len(spin_serials),
            time.perf_counter() - started,
            spin_serials,
        )
        for serial in spin_serials:
            _append_unique_camera(cameras, f"Spinnaker {serial}", f"spinnaker://{serial}")

        if tuple(pathlib.Path("/dev").glob("video*")):
            started = time.perf_counter()
            log_hardware_initialization(logger, "START | camera discovery | backend=opencv-usb")
            try:
                usb_indices = tuple(VideoManager.list_usb_cameras())
            except Exception as exc:
                log_hardware_initialization(
                    logger,
                    "FAILED | camera discovery | backend=opencv-usb elapsed=%.3fs error=%s",
                    time.perf_counter() - started,
                    str(exc) or exc.__class__.__name__,
                    level=logging.ERROR,
                )
                raise
            log_hardware_initialization(
                logger,
                "READY | camera discovery | backend=opencv-usb count=%d elapsed=%.3fs indices=%s",
                len(usb_indices),
                time.perf_counter() - started,
                usb_indices,
            )
            for index in usb_indices:
                _append_unique_camera(cameras, f"USB Camera {index}", f"opencv://{index}")

    return cameras


def _append_unique_camera(cameras: List[CaptureCameraAttrs], name: str, url: str) -> None:
    if not name or not url:
        return
    if any(camera.url == url for camera in cameras):
        return
    candidate = name
    suffix = 1
    existing_names = {camera.name for camera in cameras}
    while candidate in existing_names:
        candidate = f"{name} ({suffix})"
        suffix += 1
    cameras.append(CaptureCameraAttrs(name=candidate, url=url))


def _is_spinnaker_source(camera: CaptureCameraAttrs) -> bool:
    return camera_source_binding_key(camera.url).startswith("spinnaker://")


def camera_source_binding_key(url: str) -> str:
    """Return the physical binding identity, excluding capture parameters."""
    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme.lower() == "spinnaker":
        return f"spinnaker://{parsed.hostname or parsed.netloc}"
    return url or ""


class VideoCaptureModel(ObservableObject, ProjectDependentProtocol):

    CAMERA_PROP = "camera"
    CAMERA_LIST_PROP = "camera_list"
    IS_ENABLED_PROP = "is_enabled"
    IS_RECORDING_ENABLED_PROP = "is_recording_enabled"
    RECORD_MODE_PROP = "record_mode"
    IS_STILL_CAPTURE_ENABLED_PROP = "is_still_capture_enabled"
    STILL_IMAGE_CAPTURE_INTERVAL_PROP = "still_image_capture_interval"
    SHAPE_PROP = "shape"
    TEXT_OVERLAY_PROP = "text_overlay"
    TEXT_OVERLAY_COLOR_PROP = "text_overlay_color"
    DISPLAY_DOTS_DETECTION_PROP = "display_dots_detection"

    def __init__(
        self,
        name: str,
        preferences: UserPreferences = None,
        camera_index: int = -1,
        *,
        cam_id: CameraId = CameraId.Left,
        mp_ctx: Optional[BaseContext] = None,
        msg_queue: Optional[multiprocessing.Queue] = None,
        presence_detection: Optional[PresenceDetectionAttrs] = None,
        synced_cam_recording: Optional[Synchronized] = None,
        synced_cam_frame_index: Optional[Synchronized] = None,
        record_stop_sema: Optional[SemaphoreType] = None,
    ):
        super().__init__()

        if mp_ctx is None:
            mp_ctx = get_mp_ctx()

        self._text_overlay: Optional[str] = None
        self._text_color: Optional[str] = "yellow"

        self._id = cam_id

        self._name = name
        self._preferences = preferences
        self._camera_index = camera_index
        self._presence_detection = presence_detection
        self._synced_cam_recording = synced_cam_recording
        self._synced_cam_frame_index = synced_cam_frame_index
        self._record_stop_sema = record_stop_sema

        self._camera_source: CaptureCameraAttrs = CaptureCameraAttrs(name="", url="")
        self._camera_properties = {}

        self._video_capture: Optional[VideoCapture] = None
        self._video_reader: Optional[VideoReader] = None
        self._video_reader_reset_event = None
        self._video_reader_stop_event = None

        self._msg_queue = msg_queue  # for sending "status" message(s) to main process
        self._video_command_queue = mp_ctx.Queue(maxsize=64)
        self._video_status = mp_ctx.Value(ctypes.c_int, CaptureProcessStatus.UNKNOWN)
        self._video_frame_index = mp_ctx.Value(ctypes.c_int64, -1)  # actually unused
        self._video_image_queue: Optional[FixedArrayQueue] = None
        self._errors: SynchronizedString = mp_ctx.Array(ctypes.c_char, bytes(512))
        self._watchdog_capture_perf_c = mp_ctx.Value(ctypes.c_double, math.nan)
        self._shape = None

        self._cur_conf: CameraConfiguration = CameraConfiguration()
        self._is_enabled = True
        self._is_primary = False

        self._record_mode = VideoRecordMode.CONTINUOUS
        self._is_recording_enabled = False
        self._record_rotate_interval = 3600
        self._is_still_capture_enabled = False
        self._still_image_capture_interval = 0.0
        self._display_dots_detection = True

        self._display_update_fcn = None

        self._frame_count = 0
        self._start = 0
        self._fps = 0  # NB: this is NOT the camera acquire FPS

        self._last_error = None

        self._camera_list = create_camera_list()

        self._is_trace_enabled = True

        self._project: Optional[ProjectInfo] = None

        NotificationCenter.default_center().add_observer(TriggerNotification.CAPTURE_ID, self._on_trigger)

        self._update_camera_source(self._camera_list[0])

    @property
    def camera_index(self):
        """Unique camera index"""
        return self._camera_index

    @property
    def camera_id(self) -> CameraId:
        return self._id

    @property
    def video_status(self) -> CaptureProcessStatus:
        return CaptureProcessStatus(self._video_status.value)

    @property
    def watchdog_capture_perf_c(self) -> float:
        proc = self._video_capture
        if proc is None:
            return math.nan
        return self._watchdog_capture_perf_c.value

    @property
    def capture_process_status(self) -> CaptureProcessStatus:
        return CaptureProcessStatus(self._video_status.value)

    @property
    def project(self) -> ProjectInfo:
        return self._project

    @project.setter
    def project(self, value: ProjectInfo):
        self._project = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def camera_list(self) -> List[CaptureCameraAttrs]:
        return self._camera_list

    def refresh_camera_list(self, camera_list: Optional[Iterable[CaptureCameraAttrs]] = None) -> int:
        old_list = self._camera_list
        refreshed = list(create_camera_list(include_hardware=True) if camera_list is None else camera_list)
        current_source = self._camera_source
        # Spinnaker serial bindings belong to CameraConfiguration. Discovery still
        # verifies every serial, but the editor must not offer another camera's
        # serial as an interchangeable source.
        refreshed = [camera for camera in refreshed if not _is_spinnaker_source(camera)]
        if (
            current_source is not None
            and current_source.url
            and (
                _is_spinnaker_source(current_source)
                or not any(camera.url == current_source.url for camera in refreshed)
            )
        ):
            refreshed.insert(0, current_source)
        self._camera_list = refreshed
        self.property_changed(self.CAMERA_LIST_PROP, self._camera_list, old_list)
        if current_source is not None:
            matching_source = next(
                (camera for camera in refreshed if camera.url == current_source.url),
                None,
            )
            if matching_source is not None and matching_source != current_source:
                previous_source = self._camera_source
                self._camera_source = matching_source
                self.property_changed(self.CAMERA_PROP, matching_source, previous_source)
        return len(self._camera_list)

    @property
    def camera_source(self) -> CaptureCameraAttrs:
        return self._camera_source

    @camera_source.setter
    def camera_source(self, value: CaptureCameraAttrs):
        if self._camera_source == value:
            return
        old_value = self._camera_source
        self._update_camera_source(value)
        self.property_changed(self.CAMERA_PROP, value, old_value)

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, value: bool):
        prev, self._is_enabled = self._is_enabled, value
        self._on_property_changed(self.IS_ENABLED_PROP, value, prev)

    @property
    def is_recording_enabled(self) -> bool:
        return self._is_recording_enabled

    @is_recording_enabled.setter
    def is_recording_enabled(self, value: bool):
        prev, self._is_recording_enabled = self._is_recording_enabled, value
        self._on_property_changed(self.IS_RECORDING_ENABLED_PROP, value, prev)

    @property
    def record_mode(self) -> VideoRecordMode:
        return self._record_mode

    @record_mode.setter
    def record_mode(self, value: VideoRecordMode):
        prev, self._record_mode = self._record_mode, value
        self._on_property_changed(self.RECORD_MODE_PROP, value, prev)

    @property
    def is_still_capture_enabled(self) -> bool:
        return self._is_still_capture_enabled

    @is_still_capture_enabled.setter
    def is_still_capture_enabled(self, value: bool):
        prev, self._is_still_capture_enabled = self._is_still_capture_enabled, value
        self._on_property_changed(self.IS_STILL_CAPTURE_ENABLED_PROP, value, prev)

    @property
    def still_image_capture_interval(self) -> float:
        return self._still_image_capture_interval

    @still_image_capture_interval.setter
    def still_image_capture_interval(self, value: float):
        prev, self._still_image_capture_interval = self._still_image_capture_interval, value
        self._on_property_changed(self.STILL_IMAGE_CAPTURE_INTERVAL_PROP, value, prev)

    @property
    def is_primary(self) -> bool:
        return self._is_primary

    def set_runtime_primary(self, is_primary: bool) -> None:
        self._is_primary = is_primary

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Be aware this is the shape as seen by the cameras ndarray frames which is row x col not width x height.
        """
        if "height" in self._camera_properties and "width" in self._camera_properties:
            rows = int(self._camera_properties["height"])
            cols = int(self._camera_properties["width"])
            if rows > 0 and cols > 0:
                return rows, cols

        return 0, 0

    @shape.setter
    def shape(self, value):
        prev, self._shape = self._shape, value
        self._on_property_changed(self.SHAPE_PROP, value, prev)

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def presence_detection(self) -> PresenceDetectionAttrs:
        return self._presence_detection

    @property
    def display_dots_detection(self):
        return self._display_dots_detection

    @display_dots_detection.setter
    def display_dots_detection(self, value):
        prev, self._display_dots_detection = self._display_dots_detection, value
        self._on_property_changed(self.DISPLAY_DOTS_DETECTION_PROP, value, prev)

    @property
    def text_overlay(self) -> Optional[str]:
        return self._text_overlay

    @text_overlay.setter
    def text_overlay(self, value: Optional[str]):
        prev, self._text_overlay = self._text_overlay, value
        self._on_property_changed(self.TEXT_OVERLAY_PROP, value, prev)

    def set_text_overlay(self, value: Optional[str], *, color: str = "yellow"):
        self._text_color = color
        self.property_changed(self.TEXT_OVERLAY_COLOR_PROP, color, None)
        self.text_overlay = value

    @property
    def is_trace_enabled(self) -> bool:
        return self._is_trace_enabled

    @is_trace_enabled.setter
    def is_trace_enabled(self, value: bool):
        self._is_trace_enabled = value

    def set_display_fcn(self, display_fcn):
        self._display_update_fcn = display_fcn

    def refresh_image(self, data: ndarray):
        if self._frame_count == 0:
            self._start = time.perf_counter_ns()

        self._frame_count += 1

        if self._frame_count % 1500 == 0:
            self._fps = 1e9 * self._frame_count / (time.perf_counter_ns() - self._start)
            self._trace(f"display fps: {int(self._fps)}")

        if self._display_update_fcn is not None and self._video_capture is not None:
            self._display_update_fcn(data, self._fps)

    def on_prepare_capture(
        self,
        network_queue: Optional[FixedArrayMultiQueue] = None,
        *,
        inference_index: Optional[int] = None,
    ) -> bool:
        self._last_error = None
        if not self._is_enabled:
            return True
        self._frame_count = 0

        # before everything below, particularly video_reader
        self._video_image_queue = None if self._shape is None else FixedArrayQueue(
            3,
            self._shape,
            name="video_q",
            mp_ctx=get_mp_ctx(),
        )

        self._video_reader_initialize()

        if self._camera_source is not None:
            if "?" in self._camera_source.url:
                url = self._camera_source.url + f"&name={self._name}"
            else:
                url = self._camera_source.url + f"?name={self._name}"

            camera = CaptureCameraAttrs(name=self._name, url=url)

            inference = None if network_queue is None else CaptureInferenceAttrs(
                queue=network_queue,
                index=self._camera_index if inference_index is None else inference_index,
            )

            capture_attrs = CaptureAttrs(
                command_queue=self._video_command_queue,
                status=self._video_status,
                image_queue=self._video_image_queue,
                fps_image_queue=15 if self._preferences is None else self._preferences.live_feed_refresh_rate,
                frame=self._video_frame_index,
                camera=camera,
                camera_index=self._camera_index,
                inference=inference,
                errors=self._errors,
                presence_detection_attrs=self._presence_detection,
                is_primary=self._is_primary,
                msg_queue=self._msg_queue,
                record_prebuffer_duration=self._cur_conf.record_prebuffer_duration,
                watchdog_perf_c=self._watchdog_capture_perf_c,
                synced_cam_record_enabled=self._synced_cam_recording,
                synced_cam_frame_index=self._synced_cam_frame_index,
                record_stop_sema=self._record_stop_sema,
            )

            rotate_interval = self._record_rotate_interval if self._is_recording_enabled else -1
            image_interval = self._still_image_capture_interval if self._is_still_capture_enabled else 0
            record_properties = VideoRecordProperties(project_info=self._project, record_mode=self.record_mode,
                                                      video_rotate_interval=rotate_interval,
                                                      image_interval=image_interval)
            # Leave the watchdog counter unset (nan) until the capture child writes its first
            # real timestamp; the monitor ignores nan. Seeding a live perf_counter here makes the
            # counter age in real time while the child is still spawning, which on slow spawn-based
            # startup (notably macOS) can exceed the timeout and raise a false "element(s) timedout"
            # before capture has even begun.
            self._watchdog_capture_perf_c.value = math.nan
            vid_capture = self._video_capture = VideoCapture(capture_attrs, record_properties,
                                                             project_info=self._project)
            vid_capture.start()

        return True

    def on_capture_start(self):
        if not self._is_enabled:
            logger.warning("%s: on_capture_start called but disabled", self._name)
            return
        self._send_command(CaptureCommandKind.ENABLE_CAPTURE)

    def on_capture_notify_end(self):
        self._send_command(CaptureCommandKind.DISABLE_CAPTURE)

    def on_capture_stop(self):
        self._video_reader_teardown()

        video_capture = self._video_capture
        if video_capture is not None and video_capture.is_alive():
            self._send_command(CaptureCommandKind.TERMINATE)
            if self.wait_for_capture_status(
                    (CaptureProcessStatus.TERMINATED, CaptureProcessStatus.FAILED),
                    timeout=5):
                logger.debug(f"<{self._name}> video capture terminate acknowledged")
            else:
                logger.error(f"<{self._name}> did not receive process terminates status")

        if video_capture is not None:
            logger.debug("waiting for process termination")
            video_capture.join(5)
            if video_capture.is_alive():
                logger.warning("capture not exited yet, terminating..")
                video_capture.terminate()
                video_capture.join()
            logger.info("process exited: exitcode=%s", video_capture.exitcode)
            self._video_capture = None

        # NB: clearing video cmd queue having waited & joined the capture process is best.
        video_cmd_q = self._video_command_queue
        clear_queue(video_cmd_q, name="video_cmd_queue")

        self._video_image_queue = None
        # video_image_queue is our FixedArrayQueue which cannot be "cleared" by another thread than the
        # one consuming it. We anyway recreate a new one for each new capture.

    def on_close(self):
        logger.debug("closing %s", self.name)
        self.on_capture_stop()
        NotificationCenter.default_center().remove_observer(TriggerNotification.CAPTURE_ID, self._on_trigger)

    @property
    def active_config(self) -> CameraConfiguration:
        return self._cur_conf

    def load_configuration(self, conf: CameraConfiguration):
        self._id = conf.id
        self._name = conf.name if conf.id == CameraId.Camera3 and conf.name else str(conf.id)
        self._cur_conf = conf  # keeping config on self too
        self.is_enabled = conf.is_enabled
        self.is_recording_enabled = conf.is_record_enabled
        self.record_mode = VideoRecordMode(conf.record_mode)
        self.is_still_capture_enabled = conf.is_still_image_capture_enabled
        self.still_image_capture_interval = conf.still_image_capture_interval
        raw_primary = conf.params.get("primary") or "false"
        self._is_primary = raw_primary.lower() in {"yes", "true", "1", "on"} if isinstance(raw_primary, str) else raw_primary is True

        url = f"{conf.scheme}://{conf.host}"

        if conf.port > 0:
            url += f":{conf.port}"

        if len(conf.path) > 0:
            # Reminder: we url-decoded, if it was encoded, in CameraConfiguration.__post_init__
            # so here we have to quote with safe=() :
            encoded_path = urllib.parse.quote(conf.path, safe=())
            url += f"/{encoded_path}"

        if len(conf.params) > 0:
            url += "?" + urllib.parse.urlencode(conf.params)

        logger.debug("%s: built url=%s", conf.name, url)

        configured_spinnaker_source = (
            conf.id in CameraId.reach_camera_ids()
            and conf.scheme.lower() == "spinnaker"
        )
        if configured_spinnaker_source:
            self._camera_list = [
                camera
                for camera in self._camera_list
                if not _is_spinnaker_source(camera)
            ]
        existing = [] if configured_spinnaker_source else list(
            filter(lambda m: m.url == url, self._camera_list)
        )

        name = self._name or "<unnamed>"

        if len(existing) == 0:
            non_duplicate = name
            idx = 1
            same_name = list(filter(lambda m: m.name == non_duplicate, self._camera_list))
            while len(same_name) > 0:
                non_duplicate = f"{name} ({idx})"
                same_name = list(filter(lambda m: m.name == non_duplicate, self._camera_list))
                idx += 1
            source = CaptureCameraAttrs(name=non_duplicate, url=url)
            self._camera_list.insert(0, source)
            self.property_changed(self.CAMERA_LIST_PROP, self._camera_list, self._camera_list)
        else:
            source = existing[0]

        self.camera_source = source

    def save_configuration(self) -> CameraConfiguration:
        parsed, params = VideoManager.parse_params(self._camera_source.url)
        params: Dict[str, Any]
        for key, value in params.items():
            try:
                val = float(value)
                if abs(int(val) - val) < 2.0 * float(numpy.finfo(float).eps):
                    val = int(val)
                value = val
            except (ValueError, TypeError):
                if str(value).lower() == "true":
                    value = True
                elif str(value).lower() == "false":
                    value = False
            params[key] = value
        # undo the %-encode which happened in self.load_configuration():
        path = urllib.parse.unquote(parsed.path)[1:]  # [1:] for strip of first leading "/"

        conf = self._cur_conf

        return CameraConfiguration(id=self._id, name=self._name, is_enabled=self._is_enabled,
                                   is_record_enabled=self._is_recording_enabled,
                                   record_mode=self._record_mode.value,
                                   is_still_image_capture_enabled=self._is_still_capture_enabled,
                                   still_image_capture_interval=self.still_image_capture_interval,
                                   scheme=parsed.scheme, host=parsed.hostname or "", port=parsed.port or 0, path=path,
                                   params=params, record_prebuffer_duration=conf.record_prebuffer_duration)

    def wait_for_capture_status(
        self,
        expected: Union[CaptureProcessStatus, Iterable[CaptureProcessStatus]],
        *,
        timeout: float,
    ) -> bool:
        if isinstance(expected, CaptureProcessStatus):
            expected = (expected,)
        wait_status = set(expected)
        perf_timeout = time.perf_counter() + timeout
        logger.debug("<%s> waiting for %s acknowledgement", self._name, wait_status)
        while (cur_status := CaptureProcessStatus(self._video_status.value)) not in wait_status:
            if time.perf_counter() > perf_timeout:
                self._last_error = self._errors.value.decode()
                logger.error("<%s> failed to receive %s acknowledgement ; current=%s",
                             self._name, expected, cur_status)
                return False
            time.sleep(0.001)
        return True

    def on_trigger_recording(self, record: bool, *, is_triggered: bool=False, is_from_start: bool=False):
        if record:
            self._send_command(CaptureCommandKind.ENABLE_RECORDING, is_from_start=is_from_start)
        else:
            self._send_command(CaptureCommandKind.DISABLE_RECORDING, is_triggered=is_triggered, is_from_start=is_from_start)

    def _on_trigger(self, notification: Notification):
        if self._video_capture is not None:
            self.on_trigger_recording(notification.context)

    def _update_camera_source(self, cam: CaptureCameraAttrs):
        if cam is None or len(cam.url) == 0:
            self._camera_source = None
            self._camera_properties = dict()
            self._video_image_queue = None
            return

        value = cam.url

        if "&name=" not in value:
            if "?" in value:
                value = value + f"&name={self._name}"
            else:
                value = value + f"?name={self._name}"

        parsed, properties = VideoManager.parse_params(value)

        self.shape = None

        if "height" in properties and "width" in properties:
            width = int(properties["height"])
            height = int(properties["width"])
            if width > 0 and height > 0:
                self.shape = (width, height)
            else:
                logger.error("Invalid shape: %s", (width, height))
        else:
            self.shape = (300, 200)

        self._camera_source = cam

        self._camera_properties = properties

        self._trace(str(self._camera_properties))

    def _video_reader_initialize(self):
        if self._video_reader is None and self._video_image_queue is not None:
            self._video_reader_stop_event = Event()
            self._video_reader_reset_event = Event()
            self._video_reader = VideoReader(self._name, self._video_image_queue, self.refresh_image,
                                             stop_event=self._video_reader_stop_event,
                                             reset_event=self._video_reader_reset_event)
            self._video_reader.start()

    def _video_reader_teardown(self):
        reader = self._video_reader
        if reader is not None:
            self._video_reader_stop_event.set()
            logger.debug("joining video reader")
            reader.join(5)
            logger.debug("%s: joined video_reader", self._name)
            self._video_reader = None

    def _send_command(self, cmd: CaptureCommandKind, *args, **kwargs):
        if self._video_command_queue is not None:
            data = (args, kwargs) if (args and kwargs) else (args, None) if args else ((), kwargs) if kwargs else None
            self._video_command_queue.put((cmd, data))
        else:
            logger.warning("%s: _send_command: %s but video command queue is None", self._name, cmd)

    def _trace(self, message: str):
        if self._is_trace_enabled:
            logger.debug(f"<{self._name}> {message}")
