import csv
import ctypes
import dataclasses
import enum
import json
import logging
import math
import os
import pickle
import queue
import shlex
import subprocess
import threading
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Union, ClassVar, Protocol, Tuple

import pandas
import yaml

from autotrainer.api import ApiSystemStatus, ApiDetectorKind, ApiProjectStatus, \
    ApiDetectorStatus, ApiTunnelDeviceStatus, ApiPelletDeviceStatus, ApiTrainingMode, \
    ApiSystemConfiguration, ApiApplicationMode, ApiCommand, ApiCommandRequestErrorKind
from autotrainer.api.api_system_status import ApiBehaviorStatus, ApiReachStatus

from autotrainer.core import (
    ObservableObject,
    EventManager,
    SystemMessageHandler,
    SystemConfiguration,
    CameraId,
    PersistenceConfiguration,
    HardwareConfiguration,
    Notification,
    NotificationCenter,
    TriggerNotification,
    SystemStatusMessageKind,
    SensorAnalysis,
    Offset3DTuple,
    get_perf_now,
)
from autotrainer.core import AnimalSubject, FixedArrayMultiQueue
from autotrainer.core.configuration.behavior_configuration import CageCleaningConfig
from autotrainer.core.configuration.json_compat import SystemConfigurationJSONEncoder
from autotrainer.core.interfaces import RecordingEndingReason, CaptureAnalysisResult
from autotrainer.core.project import ProjectInfo, ProjectDependentProtocol
from autotrainer.core.configuration import SystemConfigurationDumper, DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.multiproc import no_op_timer
from autotrainer.core.logging import get_verbose_logger, set_log_location
from autotrainer.core.multiproc import get_mp_ctx, make_daemon_timer, DaemonTimer
from autotrainer.core.pose_elements import SceneElement
from autotrainer.core.project.project_info import DATE_TIME_FORMAT
from autotrainer.core.video_detection import PresenceDetectionAttrs

from autotrainer.inference import (
    PoseAlgorithm,
    InferenceStatus,
    PoseResponse,
    calibration_FLIR,
    DlcPoseModel,
)
from autotrainer.inference.analysis import IntersessionResponse
from autotrainer.inference.config import load_calib_stereo_params
from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME

from autotrainer.core.capture import CaptureProcessStatus

from autotrainer.behavior.behavior_algorithm import BehaviorAlgoProps, BehaviorAlgoStatus
from autotrainer.behavior import IntersessionState, BehaviorAlgorithm, TrainingMode, InferenceProtocol, SystemMachine, \
    IntersessionMachine

from autotrainer.training import TrainingPlan, TrainingPhase, PlanRepository, PlanInfo, LoadProgressResult

from autotrainer.api import (
    RpcService,
    ApiCommandRequest,
    ApiCommandRequestResponse,
    ApiCommandRequestResult,
    ApiEventKind,
)

from tools.acquisition.model.app_model_status import AppModelStatus
from tools.autotrainer_version import __version__ as app_version
from tools.acquisition.model.helpers import get_config_location
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.behavior_model import BehaviorModel
from tools.acquisition.model.user_preferences import UserPreferences, get_default_animals_location
from tools.acquisition.model.video_capture_model import VideoCaptureModel

logger = get_verbose_logger(__name__)

# allow be patched from tests
_daily_timer = make_daemon_timer


def _failed_camera_template(name: str, error: str):
    return f"Failed to start capture process for camera {name}:\n\t{error}\nPlease check all connections and settings."


class InvalidTargetAppModelStatus(Exception):
    """When trying to switch to invalid, or disabled, target app model status"""


def app_status_is_target_status_valid(self: AppModelStatus, target: AppModelStatus):
    return target in _app_model_status_valid_targets.get(self, ())


def app_status_to_api_app_mode(self: AppModelStatus) -> ApiApplicationMode:
    return _app_model_status_2_api_app_mode[self]


def app_status_to_behavior_algo_status(self: AppModelStatus) -> Optional[BehaviorAlgoStatus]:
    return _to_behavior_algo_status.get(self, None)


_app_model_status_2_api_app_mode = {
    AppModelStatus.IDLE: ApiApplicationMode.IDLE,
    AppModelStatus.ACQUIRING: ApiApplicationMode.RUNNING,
    AppModelStatus.CALIBRATION_3D: ApiApplicationMode.CALIBRATION_3D,
    AppModelStatus.CALIBRATION_DCS: ApiApplicationMode.CALIBRATION_DCS,
    AppModelStatus.ANIMAL_IN_DEVICE: ApiApplicationMode.IN_DEVICE,
    AppModelStatus.ANIMAL_IN_TRAINING: ApiApplicationMode.IN_TRAINING,
}

_app_model_status_valid_targets = {
    AppModelStatus.IDLE: {
        AppModelStatus.ACQUIRING,
        AppModelStatus.CALIBRATION_3D,
        AppModelStatus.ANIMAL_IN_DEVICE,
        AppModelStatus.ANIMAL_IN_TRAINING,
    },
    AppModelStatus.ACQUIRING: {
        AppModelStatus.IDLE,
        AppModelStatus.CALIBRATION_DCS,
        AppModelStatus.ANIMAL_IN_DEVICE,
        AppModelStatus.ANIMAL_IN_TRAINING,
    },
    AppModelStatus.ANIMAL_IN_DEVICE: {
        AppModelStatus.IDLE,
        AppModelStatus.ACQUIRING,
        AppModelStatus.ANIMAL_IN_TRAINING,
    },
    AppModelStatus.ANIMAL_IN_TRAINING: {
        AppModelStatus.ANIMAL_IN_DEVICE,
        AppModelStatus.ACQUIRING,
        AppModelStatus.IDLE,
    },
    AppModelStatus.CALIBRATION_3D: {AppModelStatus.IDLE},
    AppModelStatus.CALIBRATION_DCS: {AppModelStatus.ACQUIRING, AppModelStatus.IDLE},
}

_to_behavior_algo_status = {
    AppModelStatus.IDLE: BehaviorAlgoStatus.IDLE,
    AppModelStatus.ACQUIRING: BehaviorAlgoStatus.ACQUIRING,
    AppModelStatus.ANIMAL_IN_DEVICE: BehaviorAlgoStatus.ANIMAL_IN_DEVICE,
    AppModelStatus.ANIMAL_IN_TRAINING: BehaviorAlgoStatus.ANIMAL_IN_TRAINING,
}


def training_mode_to_api_training_mode(mode: TrainingMode) -> ApiTrainingMode:
    try:
        member = getattr(ApiTrainingMode, mode.name)
        assert isinstance(member, ApiTrainingMode)
        return member
    except AttributeError:
        return ApiTrainingMode.UNDEFINED


class TrainingPlanDeserializedEvent(Protocol):
    def __call__(self, plan: TrainingPlan, result: LoadProgressResult, *, force_update: bool):
        """Emitted when we deserialize a plan"""


class AppModelEvents:
    # NB: events "definition/typehint" are forwarded *into* AppModel below,
    # so we *assign* them here (=), but we'll typehint (:) in AppModel.

    configuration_loaded_event = Callable[[SystemConfiguration], None]
    on_error = Callable[[str, str], None]
    training_plan_deserialized = TrainingPlanDeserializedEvent

    current_day_changed = Callable[[date], None]


class WatchdogItems(str, enum.Enum):

    DEVICE_READER = "device_reader"
    DEVICE_WRITER = "device_writer"
    POSE_PROCESS = "pose_process"
    POSE_DATA_MONITOR_PROC = "pose_data_monitor_proc"


class AppModel(ObservableObject):
    status_file_path: ClassVar[Path] = Path("~/.config/Colorado/autotrainer_running_status.env")

    configuration_loaded_event: AppModelEvents.configuration_loaded_event
    on_error: AppModelEvents.on_error

    training_plan_deserialized: AppModelEvents.training_plan_deserialized

    current_day_changed: AppModelEvents.current_day_changed

    class Props(str, enum.Enum):

        STATUS = "status"
        ACQUISITION_RUNNING = "acquisition_running"  # False / True

        ANIMALS = "animals"
        SELECTED_ANIMAL = "selected_animal"
        OUTPUT_LOCATION = "output_location"
        ANIMAL_NAME = "animal_name"
        NOTES = "notes"
        TRAINING_MODE = 'training_mode'
        TRAINING_PLAN = "training_plan"
        TRAINING_PLANS = 'training_plans'
        TRAINING_PHASE = "training_plan.current_phase"
        TRAINING_PLAN_PROP = 'training_plan_prop'
        TRAINING_PHASE_PROP = 'training_phase_prop'

    def __init__(
            self,
            preferences: UserPreferences,
            *,
            config_file: Optional[Path] = None,
            calib_dir: Optional[Path] = None,
            sensor_analysis: Optional[SensorAnalysis] = None,
            inference_model: Optional[InferenceProtocol] = None,
            system_message_handler: Optional[SystemMessageHandler] = None,
            system_machine: Optional[SystemMachine] = None,
    ):
        event_names = tuple(filter(lambda n: not n.startswith('_'), dir(AppModelEvents)))
        super().__init__(event_names)

        self._app_version = app_version

        # self._app_lock = threading.RLock()  using BehaviorAlgo lock

        def log_on_error(title, msg):
            logger.error("%s: %s", title, msg)

        self.on_error += log_on_error

        # using a shared process manager,
        # this allows to put shared values, created via the manager, to any multiprocess shared queue, notably.
        mp_ctx = get_mp_ctx()
        self._mp_manager = mp_ctx.Manager()

        # otherwise (new) shared values can only be inherited from newly spawned sub-process(es) and not from already
        # existing sub-process(es).

        self._status = AppModelStatus.IDLE
        self._start_count = 0

        self._preferences = preferences
        self._loaded_configuration: Optional[SystemConfiguration] = None
        self._loaded_config_dir_path = Path()

        self._output_location = PersistenceConfiguration.get_default_output_path().as_posix()
        self._project_info: Optional[ProjectInfo] = None
        self._animal_name = ""
        self._notes = ""
        self._left_camera = self._right_camera = None

        self._timer_daily: DaemonTimer = _daily_timer(0, self._on_daily_timer)
        self._current_day: Optional[date] = None
        self._log_file_path: Optional[Path] = None

        self.set_log_location()

        self._plan_repo = PlanRepository()
        self._training_mode = TrainingMode.MANUAL
        self._training_plan: Optional[TrainingPlan] = None
        self._training_plan_animal: Optional[AnimalSubject] = None
        self._acquisition_starting = False
        self._acquisition_started = False
        self._acquisition_stopping = False
        self._reload_plans_needed = False
        self._prev_diamond_coord: Offset3DTuple = Offset3DTuple(math.nan, math.nan, math.nan)
        self._prev_raw_diamond_coord: Offset3DTuple = Offset3DTuple(math.nan, math.nan, math.nan)
        self._prev_valid_diamond_perf_c: float = -math.inf
        self._check_diamond_coord_enabled = True
        self._report_bad_diamond_coord_error = False
        self._warned_bad_diamond_coord = False
        self._triggered_bad_diamond_coord = False
        self._p_start_capture = -math.inf
        self._p_inference_live_begin = -math.inf

        self._event_manager = EventManager.default()

        # not sure this should better be in SystemMachine or BehaviorAlgo or BehaviorModel or eventually HardwareModel ?
        # although here it's also working, so keeping for now.
        proc_msg_queue = self._multiproc_msg_queue = mp_ctx.Queue()
        self._handle_proc_msg_thread = threading.Thread(
            target=self._handle_proc_msg_queue, name="handle_proc_msg_queue", daemon=True)
        self._handle_proc_msg_thread.start()
        # end not sure

        # this is used to sync the start record frame of both cameras:
        self._cams_record_enabled = mp_ctx.Value(ctypes.c_bool, False)
        self._cams_synced_frame_index = mp_ctx.Value(ctypes.c_int64, -1)

        self._record_stop_sema = mp_ctx.Semaphore(0)
        # and this is used to notify the end of recording from the 2 camera video_record threads to the offline one,
        # so that the later doesn't try to open the video files, before they are finished written to and closed.
        # Preventing the opencv lib to emit warning on stderr.

        self._left_camera = VideoCaptureModel(
            "left", self._preferences, 0,
            msg_queue=proc_msg_queue, cam_id=CameraId.Left,
            synced_cam_frame_index=self._cams_synced_frame_index,
            synced_cam_recording=self._cams_record_enabled,
            record_stop_sema=self._record_stop_sema,
        )

        self._right_camera = VideoCaptureModel(
            "right",
            self._preferences,
            1,
            msg_queue=proc_msg_queue,
            cam_id=CameraId.Right,
            synced_cam_frame_index=self._cams_synced_frame_index,
            synced_cam_recording=self._cams_record_enabled,
            record_stop_sema=self._record_stop_sema,
        )

        self._top_camera_presence_detection = PresenceDetectionAttrs()
        self._top_camera = VideoCaptureModel("web", self._preferences, -1,
                                             presence_detection=self._top_camera_presence_detection,
                                             msg_queue=None,  # not interested to webcam status for now.
                                             cam_id=CameraId.Web)

        self._cameras = [  # must respect camera_idx/inference_index order
            self._left_camera,
            self._right_camera,
            self._top_camera,
        ]
        self._camera_by_id = {
            camera.camera_id: camera
            for camera in self._cameras
        }
        self._reach_cameras = (
            self._left_camera,
            self._right_camera,
        )

        self._system_message_queue = queue.Queue()  # only dedicated to CAN bus messages reading/handling

        if sensor_analysis is None:
            sensor_analysis = SensorAnalysis(
                topcam_presence=self._top_camera_presence_detection
            )
        analysis = self._analysis = sensor_analysis
        del sensor_analysis  # using "analysis" instead
        #
        if system_message_handler is None:
            system_message_handler = SystemMessageHandler(self._system_message_queue, sensor_analysis=analysis)
        self._system_message_handler = system_message_handler
        assert self._system_message_handler.analysis is analysis, \
            "something very wrong: sensor_analysis different in system_message_handler"
        self._system_message_handler.start()

        self._hardware = HardwareModel(self._system_message_handler, sensor_analysis=analysis)
        self._laser = LaserModel()
        self._nidaq_signal_monitor = NidaqSignalMonitorModel()

        self._inference_queue = None

        self._pose_algorithm: PoseAlgorithm = None
        self._inference: InferenceModel = None  # noqa. needed before reload_calib
        self.reload_calib(calib_dir)
        #
        if inference_model is None:
            inference_model = InferenceModel(
                self._pose_algorithm,
                calib_dir=calib_dir,
                mp_manager=self._mp_manager,
                record_stop_sema=self._record_stop_sema,
            )
        inference = self._inference =  inference_model
        #

        self._training_plans: List[PlanInfo] = []
        self._training_plan_by_plan_id: Dict[str, PlanInfo] = {}
        self._plans_by_path: Dict[Path, Dict[str, Any]] = {}

        behavior_model = self._behavior = BehaviorModel(
            self._system_message_handler, self._analysis, self._hardware, self._inference,
            topcam_presence=self._top_camera_presence_detection,
            system_machine=system_machine,
        )
        system_machine = behavior_model.system_machine  # ensure same

        self._models: List[ProjectDependentProtocol] = [
            self._left_camera,
            self._right_camera,
            self._top_camera,
            self._inference,
            self._behavior,
            self._nidaq_signal_monitor,
        ]

        self._animals: List[AnimalSubject] = []
        self._animal_by_id: Dict[str, AnimalSubject] = {}

        self._selected_animal: Optional[AnimalSubject] = None
        self._attached_plan: Optional[TrainingPlan] = None
        self._attached_phase: Optional[TrainingPhase] = None
        self._attached_animal: Optional[AnimalSubject] = None

        self._rpc_service: Optional[RpcService] = None

        self._hardware.property_changed += self._on_hardware_property_changed
        inference.property_changed += self._on_inference_property_changed
        inference.pose_response_ready += self._on_pose_response_ready
        inference.detection_result_ready += self._on_detection_result_ready

        preferences.property_changed += self._on_preferences_property_changed

        algo = behavior_model.algorithm
        algo.property_changed += self._on_behavior_algo_property_changed
        algo.session_starting_before_record_start += self._on_session_starting_before_record_start
        algo.session_capture_ending += self._on_session_capture_ended
        # algo.session_ending += self._on_session_ending

        intersession = system_machine.intersession
        intersession.events.property_changed += self._on_intersession_property_changed

        pellet_m = system_machine.pellet
        pellet_m.events.pellet_loaded += self._on_pellet_loaded
        pellet_m.events.pellet_sent += self._on_pellet_sent

        analysis.watchdog_monitor.property_changed += self._on_watchdog_property_changed
        analysis.autoclamp_evasion_detector.property_changed += self._on_autoclamp_evasion_property_changed

        self._timer_one_minute_repeat = no_op_timer

        def one_minute_timer_handle_and_reschedule():
            self._update_led_color()
            self._send_api_system_status()
            delay = 60
            timer = self._timer_one_minute_repeat = make_daemon_timer(delay, one_minute_timer_handle_and_reschedule)
            timer.start()
            logger.verbose("Scheduled send_system_status in %.1f seconds", delay)

        one_minute_timer_handle_and_reschedule()

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_daily_timer(self):
        logger.notice("Daily timer triggered")
        self._timer_daily.cancel()  # in case of
        prev_day = self._current_day
        assert prev_day is not None
        prj = self._project_info
        assert prj is not None  # should never be None
        # if prj is None:
        #     prj = self.make_project_info()
        new_day = prev_day + timedelta(days=1)
        now = datetime.now()
        today = now.date()
        if new_day != today:
            logger.warning("detected time change: %s vs %s", new_day, today)
            new_day = today
        today_midnight = datetime.combine(new_day, datetime.min.time())
        #
        new_log_path = prj.get_log_file_path(today_midnight)
        self.set_log_location(new_log_path)
        #
        self._current_day = new_day
        #
        delay = (today_midnight + timedelta(days=1) - now).total_seconds()
        # delay = 30  # uncomment for manual testing purpose
        timer = self._timer_daily = _daily_timer(delay, self._on_daily_timer)
        timer.start()
        logger.verbose("Created new daily timer in %.1f seconds ; today=%s", delay, new_day)
        self.current_day_changed(new_day)

    def check_max_pellet_loaded(self):
        return None

    @property
    def app_lock(self) -> threading.RLock:
        return self._behavior.algorithm.thread_lock

    @property
    def acquisition_started(self):
        return self._acquisition_started

    def check_target_status_valid(self, target: AppModelStatus):
        current_status = self._status
        if target != current_status:
            valid = app_status_is_target_status_valid(current_status, target)
            if not valid:
                raise InvalidTargetAppModelStatus(f"New status {target} not valid for current status {current_status}")
        if target == AppModelStatus.ANIMAL_IN_TRAINING:
            dcs_cfg = self._behavior.algorithm.diamond_triangle_config
            valid_dcs = dcs_cfg is not None and dcs_cfg.fully_valid
            if not valid_dcs:
                raise InvalidTargetAppModelStatus("Cannot change to ANIMAL_IN_TRAINING without valid DCS")

    def is_target_status_valid(self, target: AppModelStatus) -> bool:
        try:
            self.check_target_status_valid(target)
        except InvalidTargetAppModelStatus:
            return False
        return True

    @property
    def status(self) -> AppModelStatus:
        return self._status

    @status.setter
    def status(self, status: AppModelStatus):
        prev = self._status
        if status == prev:
            return
        self.check_target_status_valid(status)
        self._status = status
        algo_status = app_status_to_behavior_algo_status(status)
        if algo_status is not None:
            self._behavior.algorithm.status = algo_status
        self.property_changed(self.Props.STATUS, status, prev)
        is_from_start = status in {AppModelStatus.ACQUIRING, AppModelStatus.IDLE}
        for cam in self._cameras:
            if status == AppModelStatus.ANIMAL_IN_DEVICE:
                is_from_start = cam != self._top_camera
            # NB: using is_triggered=None to ensure same state is kept in process side,
            # see: VideoRecord._disable_record()
            cam.on_trigger_recording(False, is_triggered=None, is_from_start=is_from_start)
            # kind of strangely, this can actually start the recording on the camera,
            # if it's continous mode and is_from_start is not True, or else it was already recording.
        if status in {AppModelStatus.IDLE, AppModelStatus.CALIBRATION_3D, AppModelStatus.CALIBRATION_DCS}:
            self._analysis.stop()
        else:
            self._analysis.restart()
        # reload training plans:
        self.reload_training_plans()
        if status == AppModelStatus.ANIMAL_IN_TRAINING:
            # NB: need to be after set of algo_status
            # self._behavior.system_machine.pellet.send_pellet()
            # send_pellet most likely not needed anymore. since pellet_machine takes care of it.
            self._hardware.open_tunnel_gate()
        else:
            self._hardware.close_tunnel_gate()
        #
        status_file_path = self.status_file_path.expanduser()
        status_file_path.parent.mkdir(parents=True, exist_ok=True)
        if status == AppModelStatus.IDLE:
            status_file_path.unlink(missing_ok=True)
        else:
            with status_file_path.open("w") as fh:
                print(f"status={status.value!r}", file=fh)

    def reload_calib(self, calib_dir: Optional[Path]):
        calib_src_dir = (
            Path(f"~/Autotrainer/{DEFAULT_3D_CALIB_DIR_NAME}") if calib_dir is None
            else calib_dir
        ).expanduser()
        logger.info("loading calib from %s", calib_src_dir)
        if calib_src_dir.exists():
            stereo_params = load_calib_stereo_params(
                calib_src_dir.joinpath('camera_matrix', 'stereo_params.pickle')
            )
            metadata_path = calib_src_dir.joinpath('calibration_userset.yaml')
            with metadata_path.open() as fh:
                calib_metadata = yaml.safe_load(fh)
            square_size, _, _, _ = calibration_FLIR.get_calibration_info(calib_src_dir.as_posix())
            cam_names = calibration_FLIR.get_video_list(calib_src_dir.as_posix())
            path_offsets = calib_src_dir.joinpath(DEFAULT_CAM_OFFSET_FILE_NAME)
            with open(path_offsets, "rb") as fh:
                cam_offsets = pickle.load(fh)
        else:
            stereo_params = None
            calib_metadata = None
            square_size = None
            cam_names = None
            cam_offsets = None
            logger.warning("calib_src_dir=%r does not exist", calib_src_dir.as_posix())

        pose_algo = PoseAlgorithm(
            stereo_params=stereo_params,
            calib_metadata=calib_metadata,
            cam_names=cam_names,
            square_size=square_size,
            cam_offsets=cam_offsets,
        )
        inference = self._inference
        if inference is not None:
            pose_algo.initialize(inference.pose_parts)
            inference.pose_algorithm = pose_algo
        self._pose_algorithm = pose_algo

    def _identify_primary_main_cam_idx(self):
        for idx, cam in enumerate(self._cameras):
            if cam.is_primary:
                logger.debug("using primary cam_idx=%s", idx)
                return idx
        logger.verbose(
            "No primary camera defined, using camera-0 as main one: %s",
            self._cameras[0],
        )
        return 0

    def _merge_camera_timestamp_files(self, project: ProjectInfo, cams: Tuple[VideoCaptureModel]):
        # assert len(cams) == 2
        timing_path = project.get_frame_timing_path()
        data = []
        prim_cam = cams[0]  # primary
        main_fps = prim_cam.active_config.params.get('fps', math.nan)
        logger.info("Merging camera timestamps files for trial%03d into %s (fps=%s)",
                    project.session, timing_path, main_fps)
        txt_files = []
        if not isinstance(main_fps, (int, float)) or main_fps == 0 or not math.isfinite(main_fps):
            logger.error("skipping invalid camera config fps=%s. cam=%s", main_fps, prim_cam.name)
            return
        frame_duration = 1 / main_fps
        utc_when = None
        ts_file_fields = ("frame_time", "fps", "frame_when", "frame_perf", "frame_id")
        for cam in cams:
            _, ts_filename, *_ = project.get_video_path(cam.name, allow_overwrite=True)
            ts_filename = Path(ts_filename)
            df = pandas.read_csv(ts_filename, names=ts_file_fields, sep=",", skipinitialspace=True)
            data.append(df)
            txt_files.append(ts_filename)
            logger.debug("cam%s: df=%s", cam.camera_index, df)
        #
        df_main_cam = data[0]
        if len(df_main_cam) == 0:
            logger.warning("_merge_camera_timestamp_files: empty df for main cam")
            return
        main_cam_first_frame_id = df_main_cam["frame_id"][0]
        for idx_df, df in enumerate(data[1:], start=1):
            df: pandas.DataFrame
            if len(df) == 0:
                logger.warning("_merge_camera_timestamp_files: empty df for cam-%s", cams[idx_df].name)
                df = df_main_cam.copy()
                df["frame_when"] = df["frame_perf"] = df["frame_time"] = math.nan
            else:
                # align on same first frame_id than primary cam:
                cur_first_frame_id = df["frame_id"][0]
                if cur_first_frame_id < main_cam_first_frame_id:
                    df = df.tail(-(main_cam_first_frame_id - cur_first_frame_id)).reset_index(drop=True)
            data[idx_df] = df
        #
        df_second_cam = data[1]
        r0 = df_main_cam[:1]
        start_frame_id = r0['frame_id'][0]
        first_frame_utc_when = r0['frame_time'][0]
        logger.debug("start_frame_id=%s (utc_when=%s)", start_frame_id, first_frame_utc_when)
        timing_csv_fields = [
            'frame_id',
            'frame_when',
            'frame_present_primary',
            'frame_present_secondary',
            'utc_when',
        ]
        with timing_path.open("w") as fh:
            dw = csv.DictWriter(fh, timing_csv_fields)
            dw.writeheader()
            # nb: only using main_cam as reference:
            for idx, frame_id in enumerate(range(start_frame_id, start_frame_id + len(df_main_cam))):
                # expected_frame_id = start_frame_id + idx
                utc_when = first_frame_utc_when + idx * frame_duration
                frame_when = df_main_cam['frame_when'][idx]
                # protect in case of secondary camera recorded less frames:
                if idx >= len(df_second_cam):
                    second_frame_when = math.nan
                else:
                    second_frame_when = df_second_cam['frame_when'][idx]
                d = dict(
                    frame_id=frame_id,
                    frame_when=frame_when if math.isfinite(frame_when) else "",  # could keep the math.nan otherwise
                    frame_present_primary=1 if math.isfinite(frame_when) else 0,
                    frame_present_secondary=1 if math.isfinite(second_frame_when) else 0,
                    utc_when=utc_when,
                )
                dw.writerow(d)
        logger.info("Written %s entries into %s", len(df_main_cam), timing_path)
        self._remove_timestamps_txt_files(project)

    def _get_monitored_cams(self):
        cams = []  # put primary first
        monitored_cams = self._reach_cameras
        for cam in monitored_cams:
            if cam.is_primary:
                cams.append(cam)
                break
        for cam in monitored_cams:
            if not cam.is_primary:
                cams.append(cam)
        monitored_cams = tuple(cams)
        return monitored_cams

    def _handle_proc_msg_queue(self):
        proc_msg_q = self._multiproc_msg_queue
        logger.info("handle_proc_msg_queue now running")
        cams_closed_finished = {}
        while True:
            raw = proc_msg_q.get()
            if raw is None:
                break
            try:
                self._handle_proc_msg(raw, cams_closed_finished=cams_closed_finished)
            except Exception as err:
                logger.exception("Error handling message %s: %s ; continuing", raw, err)
        # end while True
        logger.info("handle_proc_msg_queue exiting")

    def _handle_proc_msg(self, raw, *, cams_closed_finished):
        args = ()
        kwargs = None
        # unpack args/kwargs from raw:
        if isinstance(raw, tuple):
            if len(raw) < 1:
                logger.warning("Invalid status msg: %r", raw)
                return
            cmd = raw[0]
            if len(raw) > 1:
                args = raw[1]
                if len(raw) > 2:
                    kwargs = raw[2]
                    if len(raw) > 3:
                        logger.warning("Unhandled extra args to status msg: %r", raw[3:])
        else:
            cmd = raw
            args = raw
            kwargs = None
        extra_info = (args, kwargs) if logger.isEnabledFor(logging.DEBUG) else "NA"
        logger.verbose("Handling %s ; data=%s", cmd, extra_info)
        algo = self._behavior.algorithm
        if cmd == SystemStatusMessageKind.CAMERA_STATUS_CHANGE:
            cam_idx, new_status, *r_args = args
            if cam_idx == self._identify_primary_main_cam_idx():
                if new_status == CaptureProcessStatus.RECORDING:
                    first_frame_perf, first_frame_when, first_frame_time, *r_args = r_args
                    p_now = first_frame_perf
                    project = self._project_info
                    if project is not None:
                        project.start_record_timestamp = first_frame_time
                    logger.info("received RECORDING: frame-0: time=%.3f perf_c=%.3f now=%.3f")
                else:
                    p_now = get_perf_now()
                algo.set_capture_status(new_status, perf_now=p_now)
            else:
                logger.verbose("not handling non-primary camera status, cam_idx=%s status=%s",
                               cam_idx, new_status)
        elif cmd == SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED:
            cam_idx, frames_written, project, *r_args = args
            if project is None:
                return
            monitored_cams = self._get_monitored_cams()
            monitored_cam_indices = tuple(cam.camera_index for cam in monitored_cams)
            if cam_idx in monitored_cam_indices:
                cams_closed_finished[cam_idx] = (project, frames_written)
                if all(cam.camera_index in cams_closed_finished for cam in monitored_cams):
                    project = cams_closed_finished[0][0]  # always take the primary cam passed project as ref.
                    self._merge_camera_timestamp_files(project, monitored_cams)
                    cams_closed_finished.clear()  # now clear
        else:
            logger.warning("unhandled command: %s raw=%s", cmd, raw)

    @property
    def preferences(self) -> UserPreferences:
        return self._preferences

    @property
    def loaded_configuration(self) -> Optional[SystemConfiguration]:
        return self._loaded_configuration

    @property
    def project(self) -> Optional[ProjectInfo]:
        return self._project_info

    @project.setter
    def project(self, project: Optional[ProjectInfo]):
        self._project_info = project
        for model in self._models:
            model.project = project
        self._analysis.project_info = project
        self._event_manager.post_event_content(
            ApiEventKind.projectChanged,
            data=None if project is None else dict(
                root=project.root,
                device_id=project.device_id,
                day=project.get_day_path()[1],
                session=project.session,
            ))

    @property
    def left_camera(self):
        return self._left_camera

    @property
    def right_camera(self):
        return self._right_camera

    @property
    def top_camera(self):
        return self._top_camera

    @property
    def cameras(self) -> Tuple[VideoCaptureModel, ...]:
        return tuple(self._cameras)

    @property
    def reach_cameras(self) -> Tuple[VideoCaptureModel, ...]:
        return self._reach_cameras

    def get_camera_model(self, camera_id: CameraId) -> Optional[VideoCaptureModel]:
        return self._camera_by_id.get(camera_id)

    @property
    def top_camera_presence_detection(self):
        return self._behavior.algorithm.top_camera_presence_detection

    @property
    def behavior(self) -> BehaviorModel:
        return self._behavior

    @property
    def analysis(self):
        return self._analysis

    @property
    def inference(self) -> InferenceModel:
        return self._inference

    @property
    def hardware(self) -> HardwareModel:
        return self._hardware

    @property
    def laser(self) -> LaserModel:
        return self._laser

    @property
    def nidaq_signal_monitor(self) -> NidaqSignalMonitorModel:
        return self._nidaq_signal_monitor

    @property
    def message_handler(self) -> SystemMessageHandler:
        return self._system_message_handler

    @property
    def animals(self) -> List[AnimalSubject]:
        return self._animals

    @animals.setter
    def animals(self, value: List[AnimalSubject]):
        prev, self._animals = self._animals, value
        self._on_property_changed(self.Props.ANIMALS, value, prev)

    def get_animal_by_id(self, animal_id) -> Optional[AnimalSubject]:
        for animal in self._animals:
            if animal.id == animal_id:
                return animal
        return None

    def get_days_before_cage_clean(self) -> int:
        pref = self._preferences
        cfg = self._behavior.algorithm.active_config.cage_cleaning
        return (
            cfg.clean_days_interval
            - (date.today() - pref.cage_clean_previous_day).days
        )

    @property
    def selected_animal(self) -> Optional[AnimalSubject]:
        return self._selected_animal

    @selected_animal.setter
    def selected_animal(self, animal: Optional[AnimalSubject]):
        prev, self._selected_animal = self._selected_animal, animal
        if animal == prev:
            return
        self._detach_training_plan()  # always
        logger.debug("updating animal to %s (prev=%s)", animal, prev)
        if prev is not None:
            self._save_animal_metadata(prev, sender="selected_animal_detach")  # in case of
        self.property_changed(self.Props.ANIMAL_NAME, *(
            ("(none)", "(none)") if animal is None
            else (animal.name, self.animal_name)
        ))
        algo = self._behavior.algorithm
        self._behavior.system_machine.shift_xyz_handler.reset()

        if animal is None:
            self.training_plan = None
            algo.reset_selected_animal_counts(None)
        else:
            logger.debug("animal pellet=%s is_dcs=%s",
                         (animal.pellet_x, animal.pellet_y, animal.pellet_z), animal.is_pellet_dcs)
            diamond_cfg = algo.diamond_triangle_config
            if diamond_cfg is None or not diamond_cfg.fully_valid:
                self.on_error("Notice", "Animal Send Pos reset to 0 due to not fully valid diamond-triangle config")
                animal.is_pellet_dcs = False
                animal.pellet_x = animal.pellet_y = animal.pellet_z = 0
            algo.reset_selected_animal_counts(animal)
            if self._training_mode == TrainingMode.MANUAL:
                # only set animal base position if manual training mode
                self._set_animal_base_positions(animal)
            else:
                self.training_plan = self.get_training_plan_by_id(animal.training.current_protocol)
        self._preferences.selected_animal = "" if animal is None else animal.name
        analysis = self._behavior.analysis
        analysis.autoclamp_evasion_detector.pellets_consumed = (
            0 if animal is None
            else animal.autoclamp_evasion_pellets_consumed
        )
        self._on_property_changed(self.Props.SELECTED_ANIMAL, animal, prev)
        self._event_manager.post_event_content(
            ApiEventKind.animalSelected, None if animal is None else animal.to_api_status())
        logger.success("Switched to animal %s", animal)

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, mode: TrainingMode):
        prev, self._training_mode = self._training_mode, mode
        if prev == mode:
            return
        if mode == TrainingMode.MANUAL:
            self._detach_training_plan()
        else:
            animal = self._selected_animal
            if animal is None:  # animal might be not active/created yet
                self._detach_training_plan()
            else:
                attached = self._attached_plan
                if attached is not None:
                    is_auto = mode == TrainingMode.AUTOMATIC
                    logger.info("Updating plan is_automatic to %s", is_auto)
                    attached.is_automatic = is_auto
                else:
                    # this will also attach to it (given current mode != manual):
                    self.training_plan = self.get_training_plan_by_id(animal.training.current_protocol)
        self._on_property_changed(self.Props.TRAINING_MODE, mode, prev)
        self._event_manager.post_event_content(
            ApiEventKind.trainingModeChanged, dict(training_mode=mode))

    @property
    def attached_plan(self) -> Optional[TrainingPlan]:
        return self._attached_plan

    @property
    def training_plan(self) -> Optional[TrainingPlan]:
        return self._training_plan

    @training_plan.setter
    def training_plan(self, plan: Optional[TrainingPlan]):
        self.set_training_plan(plan)

    def set_training_plan(self, plan: Optional[TrainingPlan], *, force_update: bool = False):
        animal = self._selected_animal
        prev, self._training_plan = self._training_plan, plan
        if prev == plan and self._training_plan_animal == animal:
            return
        self._training_plan_animal = animal
        if animal is not None:
            self._detach_training_plan()
            new_plan_id = None if plan is None else plan.plan_id
            prev_plan_id, animal.training.current_protocol = animal.training.current_protocol, new_plan_id
            logger.debug("training_plan attach: animal prev_plan=%s new=%s", prev_plan_id, new_plan_id)
            if new_plan_id != prev_plan_id:
                self._save_animal_metadata(animal, sender="animal_current_plan_changed")
        if plan is None:
            self._detach_training_plan()  # always
        elif animal is not None:
            if self._training_mode != TrainingMode.MANUAL:
                if self._attach_training_plan(plan, force_update=force_update) is False:
                    return
        self._on_property_changed(self.Props.TRAINING_PLAN, plan, prev)
        self._event_manager.post_event_content(
            ApiEventKind.trainingPlanLoad, {'training_plan_id': None if plan is None else plan.plan_id})

    @property
    def output_location(self) -> str:
        return self._output_location

    @output_location.setter
    def output_location(self, value: str):
        if self._output_location == value and self._project_info is not None:
            return
        old_value = self._output_location
        self._output_location = value
        self.property_changed(self.Props.OUTPUT_LOCATION, value, old_value)
        new_prj = self.make_project_info()
        self.project = new_prj
        logger.success("Set new project to %s", dataclasses.asdict(new_prj))
        log_path = new_prj.get_log_file_path(auto_new=False)
        self.set_log_location(log_path)

    @property
    def animal_name(self) -> str:
        animal = self._selected_animal
        return "(none)" if animal is None else animal.name

    @property
    def notes(self) -> str:
        return self._notes

    @notes.setter
    def notes(self, value: str):
        prev, self._notes = self._notes, value
        self._on_property_changed(self.Props.NOTES, value, prev)

    @property
    def rpc_service(self) -> Optional[RpcService]:
        return self._rpc_service

    @rpc_service.setter
    def rpc_service(self, value: Optional[RpcService]):
        prev, self._rpc_service = self._rpc_service, value
        if value == prev:
            return
        logger.info("Updating rpc service to %s", value)
        if prev is not None:
            prev.command_request_delegate = None
        if value is not None:
            value.command_request_delegate = self._handle_rpc_service_command

    @property
    def training_plans(self) -> List[PlanInfo]:
        return self._training_plans

    @training_plans.setter
    def training_plans(self, value: List[PlanInfo]):
        self._training_plans = value
        self._training_plan_by_plan_id = {
            plan.plan_id: plan
            for idx, plan in enumerate(self._training_plans)
        }
        self._detach_training_plan()  # always
        animal = self._selected_animal
        if animal is not None:
            plan_id = animal.training.current_protocol
            if plan_id is not None:
                logger.info("reattaching %s to animal", plan_id)
                self.training_plan = self.get_training_plan_by_id(animal.training.current_protocol)
        self.property_changed(self.Props.TRAINING_PLANS, value, None)

    @property
    def check_diamond_coord_enabled(self):
        return self._check_diamond_coord_enabled

    @check_diamond_coord_enabled.setter
    def check_diamond_coord_enabled(self, value):
        self._check_diamond_coord_enabled = value

    def get_training_plan_by_id(self, plan_id: Optional[str]) -> Optional[TrainingPlan]:
        if plan_id is None:
            return None
        return self._plan_repo.get_plan(plan_id)

    def _attach_training_plan(self, plan: TrainingPlan, *, force_update: bool = False) -> Optional[bool]:
        """Returns False if attach not done"""
        algo = self._behavior.algorithm
        animal = self._selected_animal
        if animal is None:
            # if animal not created yet
            return
        assert isinstance(animal, AnimalSubject)
        attached = self._attached_plan
        if attached is not None:
            if attached.plan_id == plan.plan_id and animal == self._attached_animal:
                logger.verbose("Plan %s already attached", plan.plan_id)
                return
            self._detach_training_plan()
        prog = animal.training.get_plan_progress(plan.plan_id)
        # if prog is None:
        #     logger.debug("plan first use, using plan.serialize_progress")
        #     prog = plan.serialize_progress()
        # do we ?
        if prog is not None:
            logger.debug("%s: deserializing plan progress: %s", animal, prog)
            load_result = plan.deserialize_progress(prog, force_update=force_update)
            self.training_plan_deserialized(plan, load_result, force_update=force_update)
            if load_result != LoadProgressResult.OK:
                return False

        is_auto = self._training_mode == TrainingMode.AUTOMATIC
        logger.success("Animal %s: attaching auto=%s to plan %s (%s) ..",
                       animal.name, is_auto, plan.plan_id, hex(id(plan)))
        plan.is_automatic = is_auto
        pellet_dev = tunnel_dev = self._hardware
        plan.attach(algo, pellet_dev, tunnel_dev)
        self._attached_plan = plan
        self._attached_animal = animal
        plan.property_changed += self._on_training_plan_property_changed  # first, to be sure get everything
        plan.progress_updated += self._on_training_plan_progress_updated
        self._attach_training_phase(plan.current_phase)
        plan.resume()

    def _attach_training_phase(self, phase: Optional[TrainingPhase]):
        self._detach_training_phase()  # always
        self._attached_phase = phase
        if phase is not None:
            phase.property_changed += self._on_training_phase_property_changed
            self._event_manager.post_event_content(
                ApiEventKind.trainingPhaseEnter, {'training_phase_id': phase.phase_id})

    def _detach_training_plan(self):
        plan = self._attached_plan
        if plan is None:
            return
        self._detach_training_phase()
        plan.property_changed -= self._on_training_plan_property_changed
        plan.progress_updated -= self._on_training_plan_progress_updated
        animal = self._attached_animal
        assert isinstance(animal, AnimalSubject)
        logger.notice("%s: detaching from plan %s (%s)", animal.name, plan.plan_id, hex(id(plan)))
        plan.detach()
        self._attached_plan = None
        self._attached_animal = None
        prog = plan.serialize_progress()
        animal.training.set_plan_progress(plan.plan_id, prog)
        self._save_animal_metadata(animal, sender="detach_plan")

    def _detach_training_phase(self):
        phase = self._attached_phase
        if phase is not None:
            phase.property_changed -= self._on_training_phase_property_changed
            self._attached_phase = None
            self._event_manager.post_event_content(
                ApiEventKind.trainingPhaseExit, {'training_phase_id': phase.phase_id})

    #

    def add_animal(self, name: str, select: bool = False) -> Optional[AnimalSubject]:
        if not name:
            if select:
                self.selected_animal = None
            return None

        matching_animals = [x for x in self._animals if x.name == name]

        if len(matching_animals) == 0:
            logger.info("Adding new animal name=%s", name)
            animal = AnimalSubject(name=name)
            self._save_animal_metadata(animal, sender="add_animal")
            self._event_manager.post_event_content(
                ApiEventKind.animalCreated, animal.to_api_status(),
            )

            # Ensure property change events for listeners
            animals = self._animals
            animals.append(animal)
            self._animals = None
            self.animals = animals
        else:
            animal = matching_animals[0]

        if select:
            self.selected_animal = animal
        return animal

    def make_project_info(self) -> ProjectInfo:
        left = None if self._left_camera is None else self._left_camera.name
        right = None if self._right_camera is None else self._right_camera.name
        return ProjectInfo(
            root=self._output_location,
            device_id=self._preferences.serial_number,
            ensure_exists=True,
            camera_1=left,
            camera_2=right,
            mp_manager=self._mp_manager,  # required,
            # so to have shared values that can be put to multiprocess queue.
            # The active ProjectInfo must effectively be shared across all processes/threads.
            # and/but some of the sub-processes are started early (and kept alive after),
            # so using mp manager allows to put this ProjectInfo instance, along the shared values created via this
            # manager, to any of these already alive sub-processes, via a multiprocess.Queue().put() call/transfer.
        )

    # keep previous name temporarily:
    def on_capture_start(self, *args, **kwargs):
        warnings.warn(f"{self.__class__.__name__}.on_capture_start is renamed to capture_start. please update",
                      PendingDeprecationWarning, stacklevel=2)
        return self.capture_start(**kwargs)

    def on_capture_stop(self, *args, **kwargs):
        warnings.warn(f"{self.__class__.__name__}.on_capture_stop is renamed to capture_stop. please update",
                      PendingDeprecationWarning, stacklevel=2)
        return self.capture_stop(**kwargs)

    def capture_start(
        self,
        *,
        target_status: AppModelStatus = AppModelStatus.ACQUIRING,
        wait_connected: bool = True,
    ) -> bool:
        """Request to start the acquisition"""
        if target_status == AppModelStatus.IDLE:
            raise ValueError("AppModelStatus.IDLE not accepted as target for capture_start")
        with self.app_lock:
            before_status = self._status
            if target_status == before_status:
                logger.verbose("AppModelStatus already %s", before_status)
                return True
            if self._acquisition_started:
                if self.is_target_status_valid(target_status):
                    self.status = target_status
                    return True
                self.on_error("AppModelStatus change error",
                              f"Target status {target_status} not valid for source status {before_status}")
                return False
            if self._acquisition_starting:
                logger.warning("Acquisition already starting")
                return False
            self._acquisition_starting = True
            self._start_count += 1
            is_first_start = self._start_count == 1

        algo = self._behavior.algorithm
        analysis = self._analysis

        # first:
        self._behavior.system_machine.intersession.reset_to_idle()
        # to ensure clear state on start, previous segmentation/detection could have fails,
        # and left behind their context.

        # also:
        project_info = self.project = self.make_project_info()

        algo.reload_diamond_triangle_config()

        self._behavior.on_prepare_capture()  # might be better at the end...

        self._inference_queue = None

        if self._inference.is_enabled:
            shape_1 = self._left_camera.shape
            shape_2 = self._right_camera.shape
            if shape_1 == shape_2:
                self._inference_queue = FixedArrayMultiQueue(
                    # live queue does not need/require a lot of "depth" == total nbr of batches that can sit
                    # in the ring-buffer-queue at the same time.
                    # Now only using a "depth" of 1 frame batches,
                    # this should makes less delay / be more reactive in live inference results,
                    1,
                    2,
                    3,
                    shape=shape_1,
                    name="inference_q",
                    mp_ctx=get_mp_ctx(),
                )
            else:
                logger.warning("inference disabled: left and right camera frame sizes do not match")
        else:
            self._inference_queue = None

        #
        synced_cameras = self._reach_cameras  # normally/usually left cam is primary
        did_start = True

        # 1) prepare synced primary camera(s)
        if did_start:
            for camera in synced_cameras:
                if camera.is_primary and camera.is_enabled:
                    logger.info("Preparing capture on %s", camera.name)
                    did_start = camera.on_prepare_capture(self._inference_queue)
                    if not did_start:
                        self.on_error("Camera Process Failed",
                                      _failed_camera_template(camera.name, camera.last_error))
                        break
                    # 1.1) wait it's running or failed
                    if (
                        not camera.wait_for_capture_status((CaptureProcessStatus.RUNNING, CaptureProcessStatus.FAILED), timeout=5)
                    ) or camera.video_status != CaptureProcessStatus.RUNNING:
                        did_start = False
                        self.on_error("Camera start failed", _failed_camera_template(camera.name, camera.last_error))
                        break

        # 2) prepare synced non-primary camera(s)
        if did_start:
            time.sleep(0.5)
            for camera in synced_cameras:
                if not camera.is_primary and camera.is_enabled:
                    logger.info("Preparing capture on %s", camera.name)
                    did_start = camera.on_prepare_capture(self._inference_queue)
                    if not did_start:
                        self.on_error("Camera Process Failed",
                                      _failed_camera_template(camera.name, camera.last_error))
                        break

        # 3) wait all synced cameras are running
        if did_start:
            p_before = time.perf_counter()
            p_timeout = p_before + 10
            for camera in synced_cameras:
                p_now = time.perf_counter()
                if not camera.is_primary and camera.is_enabled:
                    if (not camera.wait_for_capture_status((CaptureProcessStatus.RUNNING, CaptureProcessStatus.FAILED), timeout=p_timeout - p_now)
                        or camera.video_status != CaptureProcessStatus.RUNNING):
                        did_start = False
                        self.on_error("Camera start failed", _failed_camera_template(camera.name, camera.last_error))
                        break
                    logger.verbose("%s now running", camera.name)

        # 4) trigger enable capture on synced cameras
        if did_start:
            # 4.1) first on non-primary
            for camera in synced_cameras:
                if not camera.is_primary and camera.is_enabled:
                    logger.info("Starting capture on %s", camera.name)
                    camera.on_capture_start()
            # small delay to ensure not-primary cam(s) are waiting on primary:
            time.sleep(0.5)
            # 4.2) then on primary
            for camera in synced_cameras:
                if camera.is_primary and camera.is_enabled:
                    logger.info("Starting capture on %s", camera.name)
                    camera.on_capture_start()

        # sleep, relatively a bit, to give more time to synced cameras to start together
        time.sleep(1.5)

        # 5) remaining non-synced camera(s)
        camera = self._top_camera
        if did_start and camera.is_enabled:
            logger.info("Preparing capture on %s", camera.name)
            did_start = camera.on_prepare_capture()
            if not did_start:
                self.on_error("Camera Process Failed",
                              _failed_camera_template(camera.name, camera.last_error))
            else:
                if (
                    not camera.wait_for_capture_status((CaptureProcessStatus.RUNNING, CaptureProcessStatus.FAILED), timeout=5)
                    or camera.video_status != CaptureProcessStatus.RUNNING
                ):
                    did_start = False
                    self.on_error("Camera start failed", _failed_camera_template(camera.name, camera.last_error))
                else:
                    camera.on_capture_start()

        if not did_start:
            logger.error("failed to start all subprocesses")
            self.capture_stop(force=True)
            return False

        # Connect "hardware" (motors/steppers/etc..) after cameras are setup/running,
        # so that any movement pre-applied should be visible on camera(s).
        logger.debug("connecting hardware ...")
        hard = self._hardware
        hard.connect(self._system_message_handler.input_queue)
        # hard.set_auto_correct_motor_drift(algo.auto_correct_motors_drift)  # disabled
        if wait_connected:
            timeout = 3
            # full establishement of connection to/from device should be very fast actually, but not immediate,
            # so this timeout.
            p_end = time.perf_counter() + timeout
            while True:
                for tok in hard.pending_tokens:
                    p0 = time.perf_counter()
                    try:
                        hard.wait_pending_command_acked(tok, timeout=timeout)
                    except Exception as err:
                        logger.error("pending token %s not acked: %s", tok, err)
                        self.capture_stop(force=True)
                        return False
                    timeout -= time.perf_counter() - p0
                break
            while True:
                if hard.connected:
                    break
                if time.perf_counter() > p_end:
                    logger.error("timeout waiting hardware connected")
                    self.capture_stop(force=True)
                    return False
                time.sleep(0.05)
        logger.info("finished connecting hardware")
        #
        watchdog_mon_register = self._analysis.watchdog_monitor.register_watchdog
        watchdog_mon_register(WatchdogItems.DEVICE_READER, lambda: self._hardware.watchdog_reader_perf_c)
        watchdog_mon_register(WatchdogItems.DEVICE_WRITER, lambda: self._hardware.watchdog_writer_perf_c)

        for cam in self._cameras:
            if cam.is_enabled:
                watchdog_mon_register(f"camera.{cam.name}", lambda cam=cam: cam.watchdog_capture_perf_c)

        if __debug__ and os.getenv("_AUTOTRAINER_TEST_WATCHDOG") == "1":
            def fake_watchdog(t_end):
                return t_end
            watchdog_mon_register("test-watchdog", lambda t=time.perf_counter() + 180: fake_watchdog(t))

        # we always be/go at home on acquisition start, so:
        self._behavior.system_machine.pellet.move_home(force=True)

        # once cameras successfully started:
        self._save_project_metadata(project_info, when=datetime.now(), session=None, caller="capture_start")
        #
        # Start inference & hardware AFTER cameras started, so we can see the initial eventual motor move.
        if self._inference.is_enabled:
            logger.info("Starting inference ..")
            if not self._inference.start(self._inference_queue):
                logger.error("Inference did not start; stopping capture")
                self.capture_stop(force=True)
                return False
            watchdog_mon_register(WatchdogItems.POSE_DATA_MONITOR_PROC,
                                  lambda: self._inference.watchdog_monitor_data_proc_perf_c)

        if not algo.algo_paused:
            analysis.restart()

        animal = self._selected_animal
        plan = (
            None if (animal is None or animal.training.current_protocol is None)
            else self.get_training_plan_by_id(animal.training.current_protocol)
        )
        if self._training_mode == TrainingMode.MANUAL or animal is None:
            # logger.notice("training mode is MANUAL or animal is none")
            # forcing manual so:
            self.training_mode = TrainingMode.MANUAL
        else:
            self.training_plan = plan

        if animal is not None:
            self._set_animal_base_positions(animal)

        self._acquisition_started = True
        self.status = target_status
        self.property_changed(self.Props.ACQUISITION_RUNNING, True, False)
        self._event_manager.post_event_content(
            ApiEventKind.applicationModeChanged,
            dict(mode=app_status_to_api_app_mode(target_status))
        )

        self.check_max_pellet_loaded()

        return True

    def capture_stop(self, force: bool = False):
        logger.debug("AppModel.capture_stop")
        with self.app_lock:
            if not self._acquisition_started and not force:
                logger.verbose("acquisition not running")
                return
            if self._acquisition_stopping:
                logger.verbose("acquisition already stopping")
                return
            self._acquisition_stopping = True
            before_status = self._status
        # always remove status-file on stop:
        status_file_path = self.status_file_path.expanduser()
        status_file_path.unlink(missing_ok=True)
        try:
            self._capture_stop()
        finally:
            # always:
            # must be set before try reload training plans, given checked in it
            self._acquisition_started = False
            self._acquisition_stopping = False
            self._acquisition_starting = False
            analysis = self._analysis
            analysis.project_info = None
            self.status = AppModelStatus.IDLE
            if before_status is AppModelStatus.IDLE:
                # force:
                self.property_changed(self.Props.STATUS, AppModelStatus.IDLE, None)
            if self._reload_plans_needed:
                self._reload_plans_needed = False
                self.reload_training_plans(refresh=True)
            self._event_manager.post_event_content(
                ApiEventKind.applicationModeChanged,
                dict(mode=app_status_to_api_app_mode(AppModelStatus.IDLE))
            )
            self.property_changed(self.Props.ACQUISITION_RUNNING, False, True)

    def _capture_stop(self):

        tok = self._hardware.set_color_led(0, 0, 0)
        self._hardware.wait_pending_command_acked(tok, timeout=2, raise_on_timeout=False)

        self._detach_training_plan()  # always

        watchdog_mon_unregister = self._analysis.watchdog_monitor.unregister_watchdog
        for item in WatchdogItems:
            watchdog_mon_unregister(item)

        try:
            self._laser.close_all_shutters()
        except Exception as err:
            logger.exception("Failed to close laser shutters during capture stop: %s", err)
        self._inference.stop()
        self._hardware.disconnect()

        for camera in self._cameras:
            watchdog_mon_unregister(f"camera.{camera.name}")

            if not camera.is_primary:
                logger.verbose("notifying end to %s", camera.name)
                camera.on_capture_notify_end()

        time.sleep(0.01)

        for camera in self._cameras:
            if camera.is_primary:
                logger.verbose("notifying end to %s", camera.name)
                camera.on_capture_notify_end()

        for camera in self._cameras:
            if not camera.is_primary:
                logger.verbose("stopping capture to %s", camera.name)
                camera.on_capture_stop()

        for camera in self._cameras:
            if camera.is_primary:
                logger.verbose("stopping capture to %s", camera.name)
                camera.on_capture_stop()

    def set_log_location(self, location: Optional[Path] = None):
        if location is None:
            prj = self._project_info
            if prj is None:
                prj = self.make_project_info()
            location = prj.get_log_file_path()
        prev_loc = self._log_file_path
        if location == prev_loc:
            return
        logger.info("switching to log location %s", location)
        set_log_location(location)
        self._log_file_path = location
        if prev_loc is not None:
            logger.success("Switched from %s to %s ; app_version=%s", prev_loc, location, self._app_version)

    def get_config_location(self, location: Optional[str] = None) -> Path:
        return get_config_location(self._preferences, location)

    @classmethod
    def get_config_from_location(cls, location: Path):
        if location.exists():
            logger.info("using configuration from %r", location.as_posix())
            configuration = SystemConfiguration.load_yaml_file(location, save_backup=True)
        else:
            logger.info("using default configuration")
            configuration = SystemConfiguration()
            configuration.save_file(location, as_yaml=True)
        return configuration

    def load_configuration(self, location: Optional[Path] = None):
        if location is None:
            location = self.get_config_location()

        configuration: SystemConfiguration = self.get_config_from_location(location)

        prebuffer_duration = 0

        frame_rate = None
        if (left_cam_cfg := configuration.get_camera(CameraId.Left)) is not None:
            prebuffer_duration = left_cam_cfg.record_prebuffer_duration
            frame_rate = left_cam_cfg.params.get("fps")

        pose_algo = self._pose_algorithm
        pose_algo.frame_rate = frame_rate
        # also reset it to inference:
        self._inference.pose_algorithm = pose_algo
        # which force a sync to pose-result process.
        self._behavior.system_machine.intersession.frame_rate = frame_rate

        if (right_cam_cfg := configuration.get_camera(CameraId.Right)) is not None:
            prebuffer_duration = max(prebuffer_duration, right_cam_cfg.record_prebuffer_duration)
            if right_cam_cfg.record_prebuffer_duration != prebuffer_duration:
                logger.warning("left & right cameras don't have same record_prebuffer_duration: %s vs %s ; using max",
                               right_cam_cfg.record_prebuffer_duration, prebuffer_duration)
            right_cam_cfg.record_prebuffer_duration = prebuffer_duration

        if left_cam_cfg is not None:
            self._left_camera.load_configuration(left_cam_cfg)

        if right_cam_cfg is not None:
            self._right_camera.load_configuration(right_cam_cfg)

        if (camera := configuration.get_camera(CameraId.Web)) is not None:
            self._top_camera.load_configuration(camera)

        if prebuffer_duration > 0:
            prebuffer_scale = os.getenv("AUTOTRAINER_PREBUFFER_SCALE")
            if prebuffer_scale is not None:
                prebuffer_duration *= float(prebuffer_scale)
                logger.notice("Using AUTOTRAINER_PREBUFFER_SCALE=%s ; prebuffer_duration -> %.3f",
                              prebuffer_scale, prebuffer_duration)

        logger.verbose("Will use algo record_prebuffer_duration=%.1f seconds", prebuffer_duration)
        self._behavior.algorithm.record_prebuffer_duration = prebuffer_duration

        self._hardware.load_config(configuration.hardware)
        self.inference.load_configuration(configuration.inference)
        self.laser.load_configuration(configuration.laser)
        self.nidaq_signal_monitor.load_configuration(configuration.nidaq_stream)
        self.behavior.load_configuration(configuration.behavior)

        self._analysis.watchdog_monitor.config = configuration.watchdog

        self._loaded_configuration = configuration
        self._loaded_config_dir_path = location.parent.resolve()

        # only at the end:
        self.output_location = configuration.persistence.output_location

        self.reload_training_plans(reraise_on_error=True)

        # and:
        self._load_animals()

        self.configuration_loaded_event(configuration)

        return True

    def reload_training_plans(self, *, refresh: bool = False, reraise_on_error: bool = False):
        if self._acquisition_started or self._status != AppModelStatus.IDLE:
            logger.notice("delaying reload training plans given acquisition started(%s) or status not idle: %s",
                          self._acquisition_started, self._status)
            self._reload_plans_needed = True
            return
        try:
            plan_infos = self._plan_repo.get_plans(refresh=refresh)
        except Exception as err:
            logger.exception("Could not load plans: %s", err)
            if reraise_on_error:
                raise RuntimeError(f"Could not load training plans: {err}") from None
            self.on_error("Reload training protocols error",
                          f"Could not reload plans:\n\n"
                          f"{err}\n\nPrevious plans are retained.")
            return
        self._training_plans = plan_infos

    def save_configuration(self):
        if self._loaded_configuration is None:
            # do not save if loaded_config is still None, which signify the load configuration failed,
            # so we won't overwrite the (currently) bad user config file with one having all defaults.
            return
        loc = self._preferences.configuration_location
        logger.info("Saving configuration to %s", loc)
        conf = self._create_configuration()
        conf.save_default(loc)

    def on_activated(self):
        """Must be called at start"""
        with self.app_lock:
            self._on_activated()

    def _on_activated(self):
        logger.notice("Activated app_model with version %s", app_version)
        logger.debug("start cmdline=%s", shlex.join(sys.argv))
        logger.debug("start env:\n%s", '\n'.join(f"{k}={v!r}" for k, v in os.environ.items()))
        # getoutput doesn't raise if pstree isn't installed, for instance.
        pstree_output = subprocess.getoutput(f"pstree -a -l -p -t -s -S -u -U {os.getpid()}")
        logger.debug("start pstree:\n%s", pstree_output)
        #
        now = datetime.now()
        today = self._current_day = now.date()
        delay = (
            datetime.combine(today, datetime.min.time()) + timedelta(days=1, seconds=1)
            - now
        ).total_seconds()
        prev = self._timer_daily
        prev.cancel()
        # delay = 45  # uncomment for manual testing
        timer = self._timer_daily = _daily_timer(delay, self._on_daily_timer)
        timer.start()
        logger.notice("Created daily timer in %.1f seconds ; today=%s", delay, today)
        self._analysis.start()

    def on_close(self):
        logger.debug("AppModel.on_close")

        for timer in (
                self._timer_one_minute_repeat,
                self._timer_daily,
        ):
            logger.debug("stopping timer %s", timer)
            timer.cancel()

        self._analysis.stop()

        # ensure go back to IDLE mode + stop cameras & inference & analysis + hardware disconnect :
        self.capture_stop()

        try:
            self._laser.close()
        except Exception as err:
            logger.exception("Failed to close laser controller: %s", err)

        try:
            self._nidaq_signal_monitor.close()
        except Exception as err:
            logger.exception("Failed to close NI-DAQ signal monitor: %s", err)

        if self._inference is not None:
            # fully terminate inference, which keeps a background process alive between different stop/start
            self._inference.terminate()

        self._behavior.system_machine.cancel_timers()

        # also fully close cameras now:
        for camera in self._cameras:
            camera.on_close()

        # Now stop the multi-proc messages handler thread:
        logger.debug("Putting None to process messages thread")
        self._multiproc_msg_queue.put(None)
        logger.debug("Joining process messages thread")
        self._handle_proc_msg_thread.join(5)
        if self._handle_proc_msg_thread.is_alive():
            logger.warning("Handle process messages thread still alive ; closing queue")
        # self._multiproc_msg_queue.close()
        # do not close to allow multiple on_close() calls.

        # at this point all background processes are normally stopped and fully joined,
        # there only remains the system message handler thread:
        self._system_message_handler.request_terminate()
        # request terminate post a stop message on the thread input queue,
        # given the queue is ordered, if there are any other unprocessed message(s) in it at this point,
        # then these will be naturally processed before the stop message is processed.
        self._system_message_handler.wait_terminated()

        # somehow if many AppModel are created (like in test cases), this makes the ones following an on_close on any
        # of them to fails hardly. MP manager looks be a singleton per python process so it might be smth related.
        # commenting to prevent this bad effect for now.
        # TODO: could investigate to see if can close it or not, might be at cli main() level is where to do
        # mp_mgr = self._mp_manager
        # logger.debug("shutting down multiprocess manager %s", mp_mgr)
        # mp_mgr.shutdown()
        # mp_mgr.join()

        self._preferences.save()
        self.save_configuration()

    def _load_animals(self):
        animals = []
        animals_dir_path = Path(self._preferences.animal_location)

        if animals_dir_path.is_dir():
            files = list(animals_dir_path.glob("*.json"))
            animals: Dict[Path, AnimalSubject] = {
                path: animal
                for path, animal in (
                    (path, AnimalSubject.from_file(path))
                    for path in files
                )
                if animal is not None
            }

            for path, animal in animals.items():
                prev_day_date = animal.pellet_counts_day_date
                prev_counts = animal.pellet_counts_day
                animal.check_today_date()
                if animal.pellet_counts_day_date != prev_day_date:
                    logger.info("Reset animal day count to 0 given saved before today: %s ; prev counts=%s",
                                prev_day_date, prev_counts)
                    animal.to_file(path)
            animals = sorted(animals.values(), key=lambda a: a.name)

        pref_animal = self._preferences.selected_animal
        for animal in animals:
            if pref_animal == animal.name:
                self.selected_animal = animal
                break

        self.animals = animals

    def _update_status_text_overlay(self):
        parts = []
        cur_inf_status = self._inference.status
        is_running = cur_inf_status in {InferenceStatus.live, InferenceStatus.intersession}
        if not is_running:
            parts.append(f"Inference: {cur_inf_status}")
        cur_inter_state = self._behavior.system_machine.intersession.state
        if cur_inter_state != IntersessionState.idle:
            parts.append(f"Intersession: {cur_inter_state}")
        self._left_camera.text_overlay = None if len(parts) == 0 else "\n".join(parts)

    def _set_animal_base_positions(self, animal: AnimalSubject):
        xyz = Offset3DTuple(animal.pellet_x, animal.pellet_y, animal.pellet_z)
        logger.verbose("Setting animal base positions and sending to %s is_pellet_dcs=%s",
                       xyz.humanize(n_digits=1), animal.is_pellet_dcs)
        algo = self._behavior.algorithm
        diamond_cfg = algo.diamond_triangle_config
        if diamond_cfg is None and animal.is_pellet_dcs:
            logger.warning("loaded animal with pellet DCS, but no diamond-triangle config, forcing to 0")
            animal.is_pellet_dcs = False
            animal.pellet_x = animal.pellet_y = animal.pellet_z = 0
            self._save_animal_metadata(animal, backup_previous=True, sender="selected_animal")
        if animal.is_pellet_dcs:
            assert diamond_cfg is not None
            _xyz = xyz
            xyz = diamond_cfg.diamond_to_motor(xyz)
            logger.verbose("converted %s to %s", _xyz.humanize(), xyz.humanize())
        else:
            if diamond_cfg is not None:
                assert not animal.is_pellet_dcs
                save_xyz = diamond_cfg.motor_to_diamond(xyz)
                logger.notice("Converting animal pellet XYZ to DCS: %s -> %s",
                              xyz.humanize(), save_xyz.humanize())
                animal.pellet_x = save_xyz.x
                animal.pellet_y = save_xyz.y
                animal.pellet_z = save_xyz.z
                animal.is_pellet_dcs = True
                self._save_animal_metadata(animal, backup_previous=True, sender="selected_animal")
        hardware = self._hardware
        if not hardware.connected:
            logger.notice("Not setting animal base positions on hardware given not connected (yet?)")
        else:
            hardware.set_x(xyz.x)
            hardware.set_y(xyz.y)
            hardware.set_z(xyz.z)
            # don't :
            #   pellet_m = self.behavior.system_machine.pellet
            #   pellet_m.send_pellet(force=True)
            # yet, it will be done by pellet-machine automatically if/when status goes to animal-in-training

    def _refresh_cage_clean_data(self):
        return None

    def _on_preferences_property_changed(self, name: str, value, old_value):
        prefs = UserPreferences
        if name == prefs.SELECTED_ANIMAL:
            for animal in self._animals:
                if animal.name == value:
                    self.selected_animal = animal
                    break
        elif name == prefs.PELLET_LOAD_COUNT_TOTAL:
            self.check_max_pellet_loaded()
        elif name == prefs.CAGE_CLEAN_PREVIOUS_DAY:
            self._refresh_cage_clean_data()

    def _update_led_color(self):
        color = self._behavior.get_led_color()
        cur_led = self._hardware.color_led
        if cur_led is None or color != (cur_led.red, cur_led.green, cur_led.blue):
            self._hardware.set_color_led(*color)

    def _on_watchdog_property_changed(self, name, value, old_value):
        wd_mon = self._analysis.watchdog_monitor
        if name == wd_mon.IS_ENGAGED:
            engaged_d = wd_mon.engaged_watchdogs
            if value and not old_value:  # watchd.is_engaged:
                self.on_error("Watchdog timeout",
                              f"Fatal error: element(s) timedout:\n\n"
                              f"{engaged_d}\n\n"
                              f"Application shall be restarted fully.")

    def _on_autoclamp_evasion_property_changed(self, name, value, _):
        det = self._analysis.autoclamp_evasion_detector
        animal = self._selected_animal
        if name == det.PELLETS_CONSUMED:
            if animal is not None and value != animal.autoclamp_evasion_pellets_consumed:
                animal.autoclamp_evasion_pellets_consumed = value
                self._save_animal_metadata(animal, sender="autoclamp_evasion_pellets_consumed")

    def _on_intersession_property_changed(self, name, value, _):
        if name == IntersessionMachine.Properties.STATE_PROPERTY:
            self._update_status_text_overlay()

    def _on_session_starting_before_record_start(self):
        drained = 0
        while self._record_stop_sema.acquire(block=False):
            drained += 1
        if drained:
            logger.verbose("drained record_stop_sema by %s", drained)

    def _on_session_capture_ended(self, reason: RecordingEndingReason):
        project = self._project_info
        if project is not None:
            self._save_project_metadata(project, caller="session_capture_ended")
        # self._behavior.system_machine.on_session_capture_ended(reason)
        # NB: had to keep in system_machine for many tests.
        # is ok as long as the project isn't modified in system_machine.on_session_capture_ended()

    def _remove_timestamps_txt_files(self, project: ProjectInfo):
        removed = []
        for cam in self._cameras:
            _, ts_file, _ = project.get_video_path(cam.name, allow_overwrite=True)
            ts_file = Path(ts_file)
            if ts_file.exists():
                ts_file.unlink(missing_ok=True)
                removed.append(ts_file)
        logger.debug("%s: removed timestamps txt files: %s", project.short_id, removed)

    def _on_behavior_algo_property_changed(self, name: str, value, old_value):
        props = BehaviorAlgoProps
        animal = self._selected_animal
        #
        if name == props.DIAMOND_TRIANGLE_CONFIG:
            self._hardware.set_diamond_triangle_config(value)

        elif name == props.PELLET_SHIFT_Y_LIMIT:
            if animal is None:
                return
            prev, animal.target_y_limit = animal.target_y_limit, value
            if prev != value:
                self._save_animal_metadata(animal, sender="pellet_shift_y_limit")
                self._event_manager.post_event_content(
                    ApiEventKind.animalUpdated, animal.to_api_status())

        elif name == props.CAGE_CLEAN_CONFIG:
            self._refresh_cage_clean_data()

        elif name == props.HEAD_FIXATION_ENABLED:
            det = self._behavior.analysis.autoclamp_evasion_detector
            if value and not old_value:
                det.pellets_consumed = 0
                # we monitor the detector pellets_consumed property,
                # and update it in selected animal.
            det.autoclamp_enabled = value

    def _on_hardware_property_changed(self, name: str, value, _):
        animal = self._selected_animal
        hard = self._hardware
        if animal is not None and name in {hard.SET_X, hard.SET_Y, hard.SET_Z}:
            # only when manual:
            if self._training_mode != TrainingMode.MANUAL:
                return
            coord = name[-1]
            coord_idx = "xyz".index(coord)
            # prevent NaN if hardware has not yet reported any send_x :
            pos = hard.last_set_position or Offset3DTuple.get_nan()
            t = list(pos)
            t[coord_idx] = value
            if any((math.isnan(v) or v is None) for v in t):
                logger.verbose("hardware set_xyz has NaN/None still: %s", t)
                return
            changed = False
            xyz = Offset3DTuple(*t)
            cfg = self._behavior.algorithm.diamond_triangle_config
            if cfg is None:
                changed |= animal.is_pellet_dcs
                animal.is_pellet_dcs = False
            else:
                changed |= not animal.is_pellet_dcs
                animal.is_pellet_dcs = True
                xyz = cfg.motor_to_diamond(xyz)
            pellet_dcs_changed = changed
            # only update same animal coordinate,
            # we are supposing the all same axis in the 2 coordinate system are parallel :
            if coord == 'x':
                prev, animal.pellet_x = animal.pellet_x, xyz.x
                new = xyz.x
            elif coord == 'y':
                prev, animal.pellet_y = animal.pellet_y, xyz.y
                new = xyz.y
            else:
                assert coord == 'z'
                prev, animal.pellet_z = animal.pellet_z, xyz.z
                new = xyz.z
            changed |= new != prev
            if changed:
                self._save_animal_metadata(animal, sender=f"hardware_{name}", backup_previous=pellet_dcs_changed)

    def _on_inference_property_changed(self, name: str, value, _):
        if name == InferenceModel.STATUS:
            new_is_live = value == InferenceStatus.live
            if new_is_live:
                self._p_inference_live_begin = time.perf_counter()

            if new_is_live or value == InferenceStatus.intersession:
                self._analysis.watchdog_monitor.register_watchdog(
                    WatchdogItems.POSE_PROCESS, lambda: self._inference.watchdog_pose_process_perf_c)
            else:
                self._analysis.watchdog_monitor.unregister_watchdog(WatchdogItems.POSE_PROCESS)

            left_cam = self._left_camera
            left_cam.display_dots_detection = new_is_live
            self._right_camera.display_dots_detection = new_is_live
            self._update_status_text_overlay()
        elif name == InferenceModel.MODEL_LOCATION:
            if value:
                try:
                    DlcPoseModel.pre_validate(value)
                except Exception as err:
                    self.on_error("DlcPoseModel pre_validate failed",
                                  f"\nModel at {value} failed pre-validate:\n\n{err}")

    def _on_pose_response_ready(self, response: PoseResponse):
        # TODO: move to behavior algo or analysis (as BaseDetector subclass)
        if not self._check_diamond_coord_enabled or self._behavior.algorithm.algo_paused:
            return
        cfg = self._behavior.algorithm.diamond_triangle_config
        if cfg is None:
            # nothing we can do
            return
        # maybe todo: make these configurable:
        min_check_delay = 5  # seconds ; if no valid check/measure within this delay -> error + emergency
        delay_inference_begin = 3  # seconds ; wait inference started for that duration before consider min_check_delay
        max_dist_diff = 5  # mm ; if distance between obtained & expected above that -> invalid measure
        #
        loc3d = response.locations_3d.get(SceneElement.Diamond)
        raw3d = response.raw_loc_3d.get(SceneElement.Diamond)
        if loc3d is None or raw3d is None:
            return
        self._prev_diamond_coord = loc3d
        diff = loc3d - cfg.diamond_coord
        raw_diff = raw3d - cfg.raw_diamond_coord
        p_now = time.perf_counter()
        if diff.distance > max_dist_diff or raw_diff.distance > max_dist_diff:
            if not self._warned_bad_diamond_coord:
                logger.warning("Diamond coordinate invalid: %s ; dist=%.2f raw=%.2f ; pose=%s",
                               loc3d.humanize(n_digits=2), diff.distance, raw_diff.distance, response)
                self._warned_bad_diamond_coord = True
        else:
            self._prev_valid_diamond_perf_c = p_now
            self._warned_bad_diamond_coord = False
            self._triggered_bad_diamond_coord = False
        #
        if (p_now - self._p_inference_live_begin > delay_inference_begin
                and p_now - self._prev_valid_diamond_perf_c > min_check_delay
        ):
            if not self._triggered_bad_diamond_coord:
                self._triggered_bad_diamond_coord = True
                if self._report_bad_diamond_coord_error:
                    self.on_error("Diamond not detected or invalid position",
                                  "Could not ensure valid diamond position for too long.\n\n"
                                  "Please re-execute a diamond-triangle calibration via menu Tools -> Calibrate Coordinate System\n\n"
                                  "Automatic pause is disabled in reachAQ, so acquisition was not paused automatically."
                                  )
                else:
                    logger.error("Bad diamond coord check: distance=%.2f ; %s vs %s",
                                 diff.distance,
                                 loc3d.humanize(), cfg.diamond_coord.humanize())

    def _on_detection_result_ready(self, project: ProjectInfo, result: IntersessionResponse):
        animal = self._selected_animal
        if animal is None:
            return
        # NB: instead of reacting to inference.detection_result_ready event,
        # we could eventually sub-depend on system_machine._on_detection_result_ready cb handler,
        # and simply assign from the behavior algo instance pellets counts .. to be sure to be in sync with it.
        day_changed = animal.check_today_date()  # 1st
        day_counts = animal.pellet_counts_day
        total_counts = animal.pellet_counts_total
        #
        # NB2: presented count is handled via pellet-sent event.
        day_counts.success_reaches += result.successful_reaches
        total_counts.success_reaches += result.successful_reaches
        day_counts.consumed += result.food_consumed
        total_counts.consumed += result.food_consumed
        day_counts.reaches += result.total_reaches
        total_counts.reaches += result.total_reaches
        if day_changed or result.successful_reaches or result.food_consumed or result.total_reaches:
            self._save_animal_metadata(animal, sender="detection_result_ready")
            self._event_manager.post_event_content(
                ApiEventKind.animalUpdated, animal.to_api_status())

    def _on_training_plan_property_changed(self, name, value, _):
        logger.debug("plan prop: %s -> %s", name, value)
        if name == "current_phase":
            if value is not None:
                assert isinstance(value, TrainingPhase)
            self._attach_training_phase(value)
            self.property_changed(self.Props.TRAINING_PHASE, value, _)
        else:
            self.property_changed(self.Props.TRAINING_PLAN_PROP, (name, value), _)

    def _on_training_plan_progress_updated(self):
        plan = self._attached_plan
        logger.debug("plan %s progress updated", plan.plan_id)
        prog = plan.serialize_progress()
        animal = self._attached_animal
        assert animal is not None
        animal.training.set_plan_progress(plan.plan_id, prog)
        self._save_animal_metadata(animal, sender="plan-progress-updated")
        self.property_changed(self.Props.TRAINING_PLAN_PROP, None, None)
        self._event_manager.post_event_content(
            ApiEventKind.trainingProgressUpdate, dict(training_phase_id=plan.current_phase.phase_id))

    def _on_training_phase_property_changed(self, name, value, _):
        logger.debug("phase prop: %s -> %s", name, value)
        self.property_changed(self.Props.TRAINING_PHASE_PROP, (name, value), _)

    def _save_animal_metadata(self, animal: AnimalSubject, *, backup_previous: bool = False, sender: str = "na"):
        prev_animals = self._animals  # in case _animals content is copied, we reset it to current animal
        for idx, prev_animal in enumerate(prev_animals):
            if prev_animal.id == animal.id:
                prev_animals[idx] = animal
                break
        animal.check_today_date()  # in case of
        dst = Path(self._preferences.animal_location).joinpath(f"{animal.name}.json")
        logger.verbose("Saving %s to %s ; sender=%s", animal, dst, sender)
        if backup_previous and dst.exists():
            now = datetime.now()
            dst.with_suffix(f'.{now.strftime(DATE_TIME_FORMAT)}.json.bak').write_bytes(dst.read_bytes())
        animal.to_file(dst)

    def _create_configuration(self) -> SystemConfiguration:
        hardware_configuration = HardwareConfiguration(
            tunnel_identifier="CAN",
            pellet_identifier="CAN",
            tunnel_headfix_enabled=False,
        )

        cameras = []
        for camera in self._cameras:
            cameras.append(camera.save_configuration())

        configuration = SystemConfiguration(cameras=cameras,
                                            hardware=hardware_configuration,
                                            inference=self._inference.save_configuration(),
                                            laser=self._laser.save_configuration(),
                                            nidaq_stream=self._nidaq_signal_monitor.save_configuration(),
                                            behavior=self._behavior.save_configuration(),
                                            persistence=PersistenceConfiguration(output_location=self.output_location))

        return configuration

    def _save_project_metadata(self, project_info: ProjectInfo, *,
                               when: Optional[datetime] = None, session: Optional[int] = -1,
                               caller: str="NA"):
        """Save the given project_info metadata, if session is None : it's main/global metadata"""
        when = when if when is not None else project_info.when
        file_name = project_info.get_metadata_file(session, when)
        logger.verbose(
            "Saving metadata to %s ; caller=%s when=%s sess=%s prj=%s",
            file_name, caller, when, session, project_info,
        )
        if session is not None and session < 0:
            session = project_info.session  # ensure use this one
        self._save_metadata(project_info, when, file_name, session)

    def _save_metadata(self, project: ProjectInfo, when: datetime, file_name: str, session: Optional[int] = -1):
        when_as_utc = when.astimezone(timezone.utc)
        info: Dict[str, Any] = {
            "date": when.strftime("%Y%m%d_%H%M%S"),
            "created": when.timestamp(),
            "createdUtc": when_as_utc.timestamp(),  # same than created
            "start_record_timestamp": project.start_record_timestamp,
            "serialNumber": self._preferences.serial_number or "",
            "appVersion": self._app_version,
            "animalName": self.animal_name,
            "notes": self.notes or "",
            "session": session,
            "t_pellet_delivered": project.t_pellet_delivered,
            "t_pellet_presented": project.t_pellet_presented,
            "configuration": None,
        }

        configuration = self._create_configuration()

        out = info.copy()
        out["configuration"] = asdict(configuration)
        with open(file_name + ".json", "w") as file:
            json.dump(out, file, cls=SystemConfigurationJSONEncoder)

        out = info.copy()
        out["configuration"] = configuration
        with open(file_name + ".yaml", "w") as file:
            yaml.dump(out, file, Dumper=SystemConfigurationDumper, sort_keys=False)

    #

    def _handle_rpc_service_command(self, request: ApiCommandRequest) -> ApiCommandRequestResponse:
        logger.notice("RPC command: %s ; nonce=%s", request.command, request.nonce)
        logger.debug("RPC cmd=%s custom=%s data=%s", request.command, request.custom_command, request.data)
        data = None
        error_message = None
        error_code = ApiCommandRequestErrorKind.NONE
        try:
            rsp = self.__handle_rpc_service_command(request)
        except Exception as err:
            logger.exception("RPC command %s exception: %s", request.command, err)
            result = ApiCommandRequestResult.EXCEPTION
            error_code = ApiCommandRequestErrorKind.COMMAND_ERROR
            error_message = f"Exception executing {request.command}: {type(err)} -> {err}"
        else:
            if isinstance(rsp, ApiCommandRequestResponse):
                return dataclasses.replace(rsp, command=request.command, nonce=request.nonce)
            if isinstance(rsp, ApiCommandRequestResult):
                result = rsp
            elif any(map(lambda x: rsp is x,
                         (None, True, False))):  # to not have to create/return it for all possible request
                if rsp is not False:
                    result = ApiCommandRequestResult.SUCCESS
                else:
                    result = ApiCommandRequestResult.FAILED
                    error_code = ApiCommandRequestErrorKind.COMMAND_ERROR
                    error_message = f"Command request {request.command} failed"
            else:
                result = ApiCommandRequestResult.SUCCESS
                data = rsp

        return ApiCommandRequestResponse(
            nonce=request.nonce,
            command=request.command,
            result=result,
            data=data,
            error_code=error_code,
            error_message=error_message,
        )

    def _handle_rpc_async_command(self, request: ApiCommandRequest, func) -> ApiCommandRequestResult:
        def execute():
            try:
                res = func()
            except BaseException as err:
                logger.exception("Failure during async execution of RPC command %s: %s", request.command, err)
                res = None
                has_err = err
            else:
                has_err = None
            rpc = self._rpc_service
            if rpc is None:
                # service gone
                return
            if has_err is not None:
                result = ApiCommandRequestResult.EXCEPTION
                error_code = ApiCommandRequestErrorKind.SYSTEM_ERROR
                error_message = f"{has_err}"
            else:
                if res is None:
                    res = True
                if res is True:
                    result = ApiCommandRequestResult.SUCCESS
                    error_code = ApiCommandRequestErrorKind.NONE
                    error_message = None
                else:
                    result = ApiCommandRequestResult.FAILED
                    error_code = ApiCommandRequestErrorKind.COMMAND_ERROR
                    error_message = f"{request.command} failed (result=False)"
            #
            message = ApiCommandRequestResponse(
                result=result,
                command=request.command,
                nonce=request.nonce,
                error_code=error_code,
                error_message=error_message,
            )
            rpc.send_command_result(message)

        #
        th = threading.Thread(target=execute, daemon=True, name=f"Handle-{func}")
        th.start()
        return ApiCommandRequestResult.PENDING_WITH_NOTIFICATION

    def __handle_rpc_service_command(self, request: ApiCommandRequest) -> Optional[
        Union[bool, ApiCommandRequestResponse, ApiCommandRequestResult, Any]]:
        cmd = request.command
        rsp = None  # let caller handle it
        if cmd == ApiCommand.START_ACQUISITION:
            return self._handle_rpc_async_command(request, self.capture_start)

        elif cmd == ApiCommand.STOP_ACQUISITION:
            return self._handle_rpc_async_command(request, self.capture_stop)

        elif cmd == ApiCommand.EMERGENCY_STOP:
            return ApiCommandRequestResponse(
                result=ApiCommandRequestResult.FAILED,
                nonce=request.nonce,
                command=cmd,
                error_code=ApiCommandRequestErrorKind.COMMAND_ERROR,
                error_message="Emergency stop is disabled in reachAQ; use STOP_ACQUISITION for controlled shutdown.",
            )

        elif cmd == ApiCommand.EMERGENCY_RESUME:
            return ApiCommandRequestResponse(
                result=ApiCommandRequestResult.FAILED,
                nonce=request.nonce,
                command=cmd,
                error_code=ApiCommandRequestErrorKind.COMMAND_ERROR,
                error_message="Emergency resume is disabled in reachAQ; no emergency pause state is maintained.",
            )

        elif cmd == ApiCommand.USER_DEFINED:
            logger.verbose("TODO")

        elif cmd == ApiCommand.GET_CONFIGURATION:
            project_info = self._project_info
            if project_info is None:
                raise RuntimeError("No current project info")
            prefs = self._preferences
            return ApiSystemConfiguration(
                application_version=self._app_version,
                device_id=project_info.device_id,
                configuration_location=self._loaded_config_dir_path.as_posix(),
                data_location=self._output_location,
                animal_location=prefs.animal_location,
                log_location=prefs.log_location,
                inference_model=self._inference.model_location,
            )

        elif cmd == ApiCommand.GET_STATUS:
            return self._make_api_system_status_payload()

        elif cmd == ApiCommand.NONE:
            pass

        else:
            rsp = ApiCommandRequestResponse(
                result=ApiCommandRequestResult.UNRECOGNIZED,
                nonce=request.nonce,
                command=cmd,
                error_code=ApiCommandRequestErrorKind.COMMAND_ERROR,
                error_message=f"Unknown/Unhandled request command: {cmd!r}"
            )
        return rsp

    def _send_api_system_status(self):
        system_status = self._make_api_system_status_payload()
        self._event_manager.post_event_content(
            kind=ApiEventKind.systemStatus,
            data=dataclasses.asdict(system_status),
        )

    def _make_api_system_status_payload(self) -> ApiSystemStatus:
        hard = self._hardware
        algo = self._behavior.algorithm
        analysis = self._behavior.analysis
        tunnel_headfix_enabled = hard.tunnel_headfix_enabled
        magnet_intensity = hard.head_magnet_intensity
        project = self._project_info
        if project is None:
            project = self.make_project_info()
        if not tunnel_headfix_enabled:
            magnet_intensity = math.nan
        if magnet_intensity is None:
            magnet_intensity = math.nan
        misplaced_mon = analysis.pellet_misplaced_monitor
        animal = self._selected_animal

        detectors = [
            ApiDetectorStatus(
                detector_id=ApiDetectorKind.pelletMisplaced,
                is_enabled=misplaced_mon.running,
                is_active=misplaced_mon.is_engaged,
            ),
        ]

        alarms = []

        dcs_pos_xyz = hard.last_dcs_position
        dcs_send_xyz = hard.last_dcs_set_position
        if dcs_pos_xyz is None or any(map(math.isnan, dcs_pos_xyz)):
            dcs_pos_xyz = Offset3DTuple.get_nan()
        if dcs_send_xyz is None or any(map(math.isnan, dcs_send_xyz)):
            dcs_send_xyz = Offset3DTuple.get_nan()

        if animal is None:
            reach_status = ApiReachStatus()
        else:
            animal_count = animal.pellet_counts_total
            reach_status = ApiReachStatus(
                pellets_presented=animal_count.presented,
                pellets_consumed=animal_count.consumed,
                reaches=animal_count.reaches,
                successful_reaches=animal_count.success_reaches,
            )

        system_status = ApiSystemStatus(
            application_mode=app_status_to_api_app_mode(self._status),
            training_mode=training_mode_to_api_training_mode(self._training_mode),
            animal=None if animal is None else animal.to_api_status(),
            project=ApiProjectStatus(
                day_path=project.get_day_path()[0],
                session_index=project.session,
            ),
            detectors=detectors,
            alarms=alarms,
            pellet_device=ApiPelletDeviceStatus(
                dcs_x=dcs_pos_xyz.x,
                dcs_y=dcs_pos_xyz.y,
                dcs_z=dcs_pos_xyz.z,
                dcs_send_x=dcs_send_xyz.x,
                dcs_send_y=dcs_send_xyz.y,
                dcs_send_z=dcs_send_xyz.z,
                load_arm=hard.load_arm_position,
                barrier_arm=hard.cover_arm_position,
            ),
            tunnel_device=ApiTunnelDeviceStatus(
                magnet_intensity=magnet_intensity,
                gate_open=hard.tunnel_gate_open_status if tunnel_headfix_enabled else False,
            ),
            behavior=ApiBehaviorStatus(
                baseline_magnet_intensity=algo.baseline_intensity if tunnel_headfix_enabled else math.nan,
                reaches=reach_status,
            )
        )
        return system_status

    #

    # pellet machine events

    def _on_pellet_loaded(self):
        prefs = self._preferences
        prefs.pellet_load_count_total += 1
        prefs.pellet_load_count_day += 1
        prefs.save()  # always
        #
        self.check_max_pellet_loaded()

    def _on_pellet_sent(self, *, perf_c: Optional[float]=None):
        selected = self._selected_animal
        status = self._status
        recent = self._behavior.algorithm.pellet_recently_seen
        algo = self._behavior.algorithm
        logger.debug("on_pellet_sent: status=%s pellet_recently_seen=%s sel=%s",
                     status, recent, selected)
        if (
            selected is not None
            and recent
            and status == AppModelStatus.ANIMAL_IN_TRAINING
        ):
            algo.increase_pellets_presented(1)
            # now recopy the values from algo:
            selected.pellet_counts_day.presented = algo.pellets_presented_day
            selected.pellet_counts_total.presented = algo.pellets_presented_total
            self._save_animal_metadata(selected, sender="on_pellet_sent")  # always
