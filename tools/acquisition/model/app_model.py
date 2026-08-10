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
import re
import shlex
import shutil
import subprocess
import threading
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Union, ClassVar, Protocol, Tuple

import pandas
import numpy as np
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
    CameraConfiguration,
    PersistenceConfiguration,
    HardwareConfiguration,
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    Notification,
    NotificationCenter,
    TriggerNotification,
    SystemStatusMessageKind,
    Offset3DTuple,
    get_perf_now,
)
from autotrainer.core import AnimalSubject, FixedArrayMultiQueue
from autotrainer.core.configuration.json_compat import SystemConfigurationJSONEncoder
from autotrainer.core.interfaces import RecordingEndingReason, CaptureAnalysisResult
from autotrainer.core.project import ProjectInfo, ProjectDependentProtocol
from autotrainer.core.configuration import SystemConfigurationDumper, DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.multiproc import no_op_timer
from autotrainer.core.logging import (
    get_verbose_logger,
    log_hardware_initialization,
    register_fatal_exception_callback,
    set_log_location,
    unregister_fatal_exception_callback,
)
from autotrainer.core.analysis import ReachAnalysis
from autotrainer.core.multiproc import get_mp_ctx, make_daemon_timer, DaemonTimer
from autotrainer.core.pose_elements import SceneElement
from autotrainer.core.project.project_info import DATE_TIME_FORMAT
from autotrainer.video import VideoRecordMode

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
from autotrainer.behavior import (
    AttemptAssignmentPolicy,
    BehaviorAlgorithm,
    InferenceProtocol,
    IntersessionMachine,
    IntersessionState,
    PelletTrialLedger,
    RetrySettingsPolicy,
    SystemMachine,
    TrialAccountingConfiguration,
    TrialCountBasis,
    TrialOutcome,
)

from autotrainer.training import TrainingPlan, TrainingPhase, PlanRepository, PlanInfo, LoadProgressResult

from autotrainer.api import (
    RpcService,
    ApiCommandRequest,
    ApiCommandRequestResponse,
    ApiCommandRequestResult,
    ApiEventKind,
)

from tools.acquisition.model.app_model_status import AppModelStatus, SessionRecordingStatus
from tools.autotrainer_version import __version__ as app_version
from tools.acquisition.model.helpers import get_config_location
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.laser_model import LaserModel
from autotrainer.device import CanTransportConfiguration
from tools.acquisition.model.hardware_scan import HardwareScanEntry, scan_can_adapters, scan_gpus
from tools.acquisition.model.nidaq_discovery import device_name_from_channel, discover_nidaq_devices
from tools.acquisition.model.nidaq_channel_plan import build_nidaq_acquisition_configuration
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.nidaq_timing import (
    remap_nidaq_physical_channel,
    resolve_nidaq_device_aliases,
)
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.session_stop_policy import (
    SessionStopConfiguration,
    SessionStopDecision,
    SessionStopEvaluation,
    SessionStopPolicy,
    SessionStopReason,
)
from tools.acquisition.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
    SubsystemStatus,
    SubsystemStatusRegistry,
)
from tools.acquisition.model.trial_protocol_runner import TrialProtocolRunner
from tools.acquisition.model.behavior_model import BehaviorModel
from tools.acquisition.model.user_preferences import UserPreferences, get_default_animals_location
from tools.acquisition.model.video_capture_model import (
    VideoCaptureModel,
    camera_source_binding_key,
    create_camera_list,
)

logger = get_verbose_logger(__name__)


def _metadata_without_nonfinite_numbers(value):
    """Replace NaN/Infinity recursively so JSON and YAML stay portable."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            key: _metadata_without_nonfinite_numbers(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _metadata_without_nonfinite_numbers(item)
            for item in value
        ]
    return value

# allow be patched from tests
_daily_timer = make_daemon_timer

_RANDOM_CAMERA_DEFAULT_PARAMS = {
    "width": 300,
    "height": 200,
    "fps": 30,
}


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
    AppModelStatus.RUNNING: ApiApplicationMode.RUNNING,
    AppModelStatus.CALIBRATION_3D: ApiApplicationMode.CALIBRATION_3D,
    AppModelStatus.CALIBRATION_DCS: ApiApplicationMode.CALIBRATION_DCS,
}

_app_model_status_valid_targets = {
    AppModelStatus.IDLE: {
        AppModelStatus.RUNNING,
        AppModelStatus.CALIBRATION_3D,
    },
    AppModelStatus.RUNNING: {
        AppModelStatus.IDLE,
        AppModelStatus.CALIBRATION_DCS,
    },
    AppModelStatus.CALIBRATION_3D: {AppModelStatus.IDLE},
    AppModelStatus.CALIBRATION_DCS: {AppModelStatus.RUNNING, AppModelStatus.IDLE},
}

_to_behavior_algo_status = {
    AppModelStatus.IDLE: BehaviorAlgoStatus.IDLE,
    AppModelStatus.RUNNING: BehaviorAlgoStatus.RUNNING,
}


def protocol_state_to_api_training_mode(
    plan: Optional[TrainingPlan],
    *,
    automatic_advance: bool,
) -> ApiTrainingMode:
    """Translate the simplified protocol state for the legacy status API."""
    name = (
        "MANUAL"
        if plan is None
        else "AUTOMATIC" if automatic_advance else "MANUAL_WITH_PROTOCOL"
    )
    return getattr(ApiTrainingMode, name, ApiTrainingMode.UNDEFINED)


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
        TRAINING_PLAN = "training_plan"
        TRAINING_PLANS = 'training_plans'
        TRAINING_PHASE = "training_plan.current_phase"
        TRAINING_PLAN_PROP = 'training_plan_prop'
        TRAINING_PHASE_PROP = 'training_phase_prop'
        HARDWARE_SCAN_RESULTS = "hardware_scan_results"
        SESSION_RECORDING_STATUS = "session_recording_status"
        SUBSYSTEM_STATUSES = "subsystem_statuses"
        RECORDING_BLOCKERS = "recording_blockers"

    def __init__(
            self,
            preferences: UserPreferences,
            *,
            config_file: Optional[Path] = None,
            calib_dir: Optional[Path] = None,
            analysis: Optional[ReachAnalysis] = None,
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
        self._loaded_configuration_has_runtime_override = False
        self._runtime_live_inference_override: Optional[bool] = None
        self._nidaq_ports = NidaqPortConfiguration()
        self._hardware_scan_results: Dict[str, HardwareScanEntry] = {}
        self._subsystem_status_registry = SubsystemStatusRegistry()
        self._subsystem_status_lock = threading.RLock()

        self._output_location = PersistenceConfiguration.get_default_output_path().as_posix()
        self._project_info: Optional[ProjectInfo] = None
        self._animal_name = ""
        self._notes = ""
        self._left_camera = self._right_camera = self._stim_camera = None
        self._reach_cameras: Tuple[VideoCaptureModel, ...] = ()

        self._timer_daily: DaemonTimer = _daily_timer(0, self._on_daily_timer)
        self._current_day: Optional[date] = None
        self._log_file_path: Optional[Path] = None

        self.set_log_location()

        self._plan_repo = PlanRepository()
        self._training_plan: Optional[TrainingPlan] = None
        self._training_plan_animal: Optional[AnimalSubject] = None
        self._acquisition_starting = False
        self._acquisition_started = False
        self._acquisition_stopping = False
        self._session_recording_status = SessionRecordingStatus.READY
        self._session_analysis_finished = True
        self._session_analysis_started_perf: Optional[float] = None
        self._session_analysis_duration_seconds: Optional[float] = None
        self._pending_session_end_perf: Optional[float] = None
        self._session_boundary: Optional[SessionBoundary] = None
        self._session_hardware_status_at_record: Optional[dict] = None
        self._session_data_complete = True
        self._session_data_errors: Tuple[str, ...] = ()
        self._session_enabled_sources: Tuple[dict, ...] = ()
        self._trial_ledger: Optional[PelletTrialLedger] = None
        self._session_stop_policy: Optional[SessionStopPolicy] = None
        self._session_stop_evaluation: Optional[SessionStopEvaluation] = None
        self._recording_ending_reason = RecordingEndingReason.NA
        self._automatic_stop_timer = no_op_timer
        self._stop_drain_timer = no_op_timer
        self._aborting_project: Optional[ProjectInfo] = None
        self._aborted_session_ids = set()
        self._abort_had_recording_started = False
        self._abort_cleanup_timer = no_op_timer
        self._record_start_timer = no_op_timer
        self._closing_event = threading.Event()
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

        # this is used to sync the start record frame of all reach cameras:
        self._cams_record_enabled = mp_ctx.Value(ctypes.c_bool, False)
        self._cams_synced_frame_index = mp_ctx.Value(ctypes.c_int64, -1)
        self._cams_record_start_perf = mp_ctx.Value(ctypes.c_double, math.nan)

        self._record_stop_sema = mp_ctx.Semaphore(0)
        # and this is used to notify the end of recording from the reach-camera video_record threads to the offline one,
        # so that the later doesn't try to open the video files, before they are finished written to and closed.

        self._reach_cameras = tuple(
            self._make_reach_camera_model(camera_id, camera_index)
            for camera_index, camera_id in enumerate((CameraId.Left, CameraId.Right))
        )
        self._left_camera = self._reach_cameras[0]
        self._right_camera = self._reach_cameras[1]

        self._cameras = list(self._reach_cameras)
        self._camera_by_id = {
            camera.camera_id: camera
            for camera in self._cameras
        }
        for camera in self._cameras:
            self._subsystem_status_registry.ensure(
                SubsystemId.camera(camera.name),
                state=SubsystemState.DISABLED,
                reason="camera disabled",
            )
        for subsystem_id in SubsystemId:
            self._subsystem_status_registry.ensure(
                subsystem_id,
                state=SubsystemState.DISABLED,
            )

        self._system_message_queue = queue.Queue()  # only dedicated to CAN bus messages reading/handling

        if analysis is None:
            analysis = ReachAnalysis()
        self._analysis = analysis
        #
        if system_message_handler is None:
            system_message_handler = SystemMessageHandler(self._system_message_queue)
        self._system_message_handler = system_message_handler
        self._system_message_handler.start()

        self._hardware = HardwareModel(self._system_message_handler)
        self._laser = LaserModel()
        self._nidaq_signal_monitor = NidaqSignalMonitorModel()
        self._session_data_recorder = SessionDataRecorder(
            self._nidaq_signal_monitor,
            self._laser,
            system_message_handler=self._system_message_handler,
            hardware_model=self._hardware,
        )

        self._inference_queue = None
        self._inference_cameras: Tuple[VideoCaptureModel, ...] = ()

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
            system_machine=system_machine,
        )
        system_machine = behavior_model.system_machine  # ensure same

        self._models: List[ProjectDependentProtocol] = [
            *self._reach_cameras,
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
        self._protocol_runner = TrialProtocolRunner(
            on_protocol_complete=self._on_trial_protocol_complete,
        )

        self._rpc_service: Optional[RpcService] = None

        self._hardware.property_changed += self._on_hardware_property_changed
        self._nidaq_signal_monitor.property_changed += (
            self._on_nidaq_monitor_property_changed
        )
        self._laser.property_changed += self._on_laser_runtime_property_changed
        inference.property_changed += self._on_inference_property_changed
        inference.pose_response_ready += self._on_pose_response_ready
        inference.detection_result_ready += self._on_detection_result_ready

        preferences.property_changed += self._on_preferences_property_changed

        algo = behavior_model.algorithm
        algo.property_changed += self._on_behavior_algo_property_changed
        algo.session_starting_before_record_start += self._on_session_starting_before_record_start
        algo.session_capture_ending += self._on_session_capture_ended
        algo.session_ending += self._on_session_ending

        intersession = system_machine.intersession
        intersession.events.property_changed += self._on_intersession_property_changed

        pellet_m = system_machine.pellet
        pellet_m.events.pellet_loading += self._on_pellet_loading_for_trial
        pellet_m.events.pellet_cycle_completed += self._on_pellet_cycle_completed
        pellet_m.events.pellet_sending += self._on_pellet_sending
        pellet_m.events.pellet_sent += self._on_pellet_sent

        analysis.watchdog_monitor.property_changed += self._on_watchdog_property_changed

        # Establish the default project before publishing startup status. This
        # gives the event-file plugin a valid destination and also prevents a
        # configuration using the default output root from reopening the log.
        self.project = self.make_project_info()

        self._timer_one_minute_repeat = no_op_timer

        def one_minute_timer_handle_and_reschedule():
            if self._closing_event.is_set():
                return
            self._update_led_color()
            self._send_api_system_status()
            if self._closing_event.is_set():
                return
            delay = 60
            timer = self._timer_one_minute_repeat = make_daemon_timer(delay, one_minute_timer_handle_and_reschedule)
            timer.start()
            logger.verbose("Scheduled send_system_status in %.1f seconds", delay)

        one_minute_timer_handle_and_reschedule()
        register_fatal_exception_callback(self._on_fatal_exception)

    def _make_reach_camera_model(self, camera_id: CameraId, camera_index: int) -> VideoCaptureModel:
        return VideoCaptureModel(
            str(camera_id),
            self._preferences,
            camera_index,
            msg_queue=self._multiproc_msg_queue,
            cam_id=camera_id,
            synced_cam_frame_index=self._cams_synced_frame_index,
            synced_cam_recording=self._cams_record_enabled,
            record_start_perf=self._cams_record_start_perf,
            record_stop_sema=self._record_stop_sema,
        )

    @staticmethod
    def _configured_reach_camera_ids(configuration: SystemConfiguration) -> Tuple[CameraId, ...]:
        reach_camera_ids = set(CameraId.reach_camera_ids())
        configured_ids: List[CameraId] = []
        for camera_config in configuration.cameras:
            camera_id = camera_config.id
            if camera_id in reach_camera_ids and camera_id not in configured_ids:
                configured_ids.append(camera_id)
        return tuple(configured_ids)

    def _refresh_camera_collections(self) -> None:
        self._left_camera = next(
            (camera for camera in self._reach_cameras if camera.camera_id == CameraId.Left),
            None,
        )
        self._right_camera = next(
            (camera for camera in self._reach_cameras if camera.camera_id == CameraId.Right),
            None,
        )
        self._stim_camera = next(
            (camera for camera in self._reach_cameras if camera.camera_id == CameraId.Camera3),
            None,
        )
        self._cameras = list(self._reach_cameras)
        self._camera_by_id = {
            camera.camera_id: camera
            for camera in self._cameras
        }
        self._models = [
            *self._reach_cameras,
            self._inference,
            self._behavior,
            self._nidaq_signal_monitor,
        ]
        if self._project_info is not None:
            for camera in self._reach_cameras:
                camera.project = self._project_info

    def _sync_reach_cameras_to_configuration(self, configuration: SystemConfiguration) -> None:
        configured_camera_ids = self._configured_reach_camera_ids(configuration)
        current_by_id = {
            camera.camera_id: camera
            for camera in self._reach_cameras
        }
        next_cameras: List[VideoCaptureModel] = []
        for camera_index, camera_id in enumerate(configured_camera_ids):
            camera = current_by_id.pop(camera_id, None)
            if camera is None or camera.camera_index != camera_index:
                if camera is not None:
                    camera.on_close()
                camera = self._make_reach_camera_model(camera_id, camera_index)
            next_cameras.append(camera)

        for removed_camera in current_by_id.values():
            removed_camera.on_close()

        self._reach_cameras = tuple(next_cameras)
        self._refresh_camera_collections()

    @staticmethod
    def _ensure_optional_stim_camera(configuration: SystemConfiguration) -> None:
        """Add the standard third camera without enabling or probing it."""
        configured_stim_camera = configuration.get_camera(CameraId.Camera3)
        if configured_stim_camera is not None:
            configured_stim_camera.name = "stimCam"
            return
        configuration.cameras.append(
            CameraConfiguration(
                id=CameraId.Camera3,
                name="stimCam",
                is_enabled=False,
                is_record_enabled=False,
                record_prebuffer_duration=0,
                scheme="random",
                params={**_RANDOM_CAMERA_DEFAULT_PARAMS, "primary": "no"},
            )
        )
        configuration._camera_map = {}

    @staticmethod
    def _make_random_camera_config(camera_id: CameraId) -> CameraConfiguration:
        camera_config = CameraConfiguration(
            id=camera_id,
            name=str(camera_id),
            is_enabled=True,
            is_record_enabled=False,
            record_prebuffer_duration=0,
            scheme="random",
            params=dict(_RANDOM_CAMERA_DEFAULT_PARAMS),
        )
        camera_config.params["primary"] = "yes" if camera_id == CameraId.Left else "no"
        return camera_config

    @classmethod
    def _configure_camera_as_random(cls, camera_config: CameraConfiguration) -> None:
        camera_config.scheme = "random"
        camera_config.host = ""
        camera_config.port = 0
        camera_config.path = ""

        params = dict(camera_config.params)
        for key, value in _RANDOM_CAMERA_DEFAULT_PARAMS.items():
            params.setdefault(key, value)
        if camera_config.id == CameraId.Left:
            params["primary"] = "yes"
        elif camera_config.id in CameraId.reach_camera_ids():
            params.setdefault("primary", "no")
        camera_config.params = params

    @classmethod
    def _apply_random_camera_override(cls, configuration: SystemConfiguration) -> None:
        reach_ids = set(CameraId.reach_camera_ids())
        reach_configs_by_id = {
            camera_config.id: camera_config
            for camera_config in configuration.cameras
            if camera_config.id in reach_ids
        }

        if len(reach_configs_by_id) == 0:
            for camera_id in (CameraId.Left, CameraId.Right):
                camera_config = cls._make_random_camera_config(camera_id)
                configuration.cameras.append(camera_config)
                reach_configs_by_id[camera_id] = camera_config

        should_enable_default_reach_cameras = not any(
            config.is_enabled for config in reach_configs_by_id.values()
        )
        for camera_id in (CameraId.Left, CameraId.Right):
            camera_config = reach_configs_by_id.get(camera_id)
            if camera_config is None:
                continue
            if should_enable_default_reach_cameras:
                camera_config.is_enabled = True

        for camera_config in configuration.cameras:
            if camera_config.id in reach_ids:
                cls._configure_camera_as_random(camera_config)

        configuration._camera_map = {}

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

    @property
    def app_lock(self) -> threading.RLock:
        return self._behavior.algorithm.thread_lock

    @property
    def acquisition_started(self):
        return self._acquisition_started

    @property
    def session_recording_status(self) -> SessionRecordingStatus:
        return self._session_recording_status

    def _set_session_recording_status(self, status: SessionRecordingStatus) -> None:
        previous = self._session_recording_status
        if status == previous:
            return
        self._session_recording_status = status
        logger.info("session recording status: %s -> %s", previous.value, status.value)
        self.property_changed(self.Props.SESSION_RECORDING_STATUS, status, previous)
        self.property_changed(
            self.Props.RECORDING_BLOCKERS,
            self.recording_blockers,
            None,
        )

    def start_recording(self) -> bool:
        if not self._acquisition_started or self._status == AppModelStatus.IDLE:
            self.on_error("Recording unavailable", "Set System Mode to Running before recording.")
            return False
        recording_reach_cams = tuple(
            camera
            for camera in self._get_monitored_cams()
            if camera.is_recording_enabled
        )
        if not recording_reach_cams:
            self.on_error(
                "Recording unavailable",
                "Enable recording for at least one reach camera.",
            )
            return False
        if self._session_recording_status != SessionRecordingStatus.READY:
            logger.warning("start_recording refused while %s", self._session_recording_status.value)
            return False
        blockers = tuple(
            blocker
            for blocker in self.recording_blockers
            if blocker != "System Mode is not running"
        )
        if blockers:
            self.on_error(
                "Recording unavailable",
                "Required session streams are not ready:\n"
                + "\n".join(f"- {blocker}" for blocker in blockers),
            )
            return False
        self._session_analysis_finished = False
        self._session_analysis_started_perf = None
        self._session_analysis_duration_seconds = None
        self._set_subsystem_status(
            SubsystemId.OFFLINE_ANALYSIS,
            SubsystemState.DISABLED,
            reason="recording active; no stopped session pending",
        )
        self._pending_session_end_perf = None
        self._session_boundary = None
        self._session_data_complete = True
        self._session_data_errors = ()
        self._session_enabled_sources = ()
        self._session_hardware_status_at_record = (
            self._subsystem_status_registry.snapshot()
        )
        self._abort_had_recording_started = False
        self._set_session_recording_status(SessionRecordingStatus.ARMING)
        project = self._project_info
        if project is None:
            self._set_session_recording_status(SessionRecordingStatus.READY)
            return False
        self._session_data_recorder.arm(
            project,
            source_manifest=self._build_session_source_manifest(),
        )
        try:
            started = self._behavior.algorithm.start_session(reason="manual_record")
        except Exception as err:
            logger.exception("manual recording start failed: %s", err)
            self._session_data_recorder.abort()
            self._set_session_recording_status(SessionRecordingStatus.READY)
            self.on_error("Recording failed", str(err))
            return False
        if not started:
            self._session_data_recorder.abort()
            self._set_session_recording_status(SessionRecordingStatus.READY)
            return False
        self._aborted_session_ids.discard(project.short_id)
        if self._session_recording_status == SessionRecordingStatus.ARMING:
            self._record_start_timer.cancel()
            self._record_start_timer = make_daemon_timer(
                10.0,
                self._record_start_timed_out,
            )
            self._record_start_timer.start()
        return True

    def _build_session_source_manifest(self) -> Tuple[dict, ...]:
        def runtime_state(subsystem_id) -> str:
            status = self._subsystem_status_registry.get(subsystem_id)
            return "unknown" if status is None else status.state.value

        sources = [
            {
                "id": f"camera.{camera.name}",
                "kind": "camera",
                "binding": (
                    ""
                    if camera.camera_source is None
                    else camera.camera_source.url
                ),
                "path": f"video/{camera.name}",
                "runtimeState": runtime_state(SubsystemId.camera(camera.name)),
            }
            for camera in self._get_recording_cams()
        ]
        if self._inference.is_enabled:
            sources.append({
                "id": "pose",
                "kind": "pose",
                "path": "pose",
                "runtimeState": runtime_state(SubsystemId.LIVE_INFERENCE),
            })
        if (
            self._nidaq_signal_monitor.hardware_enabled
            and self._nidaq_signal_monitor.configuration.is_enabled
        ):
            sources.extend(
                {
                    "id": f"nidaq.{channel.name}",
                    "kind": f"nidaq_{channel.kind}",
                    "binding": channel.physical_channel,
                    "path": "streams/nidaq.h5",
                    "runtimeState": runtime_state(SubsystemId.NIDAQ_STREAM),
                }
                for channel in self._nidaq_signal_monitor.configuration.channels
            )
        if self._hardware.requires_connection:
            sources.append({
                "id": "device",
                "kind": "decoded_can_and_device_events",
                "path": "streams/device.csv",
                "runtimeState": runtime_state(SubsystemId.CAN_PELLET),
            })
        if self._laser.configuration.backend != "disabled":
            sources.append({
                "id": "laser_outputs",
                "kind": "laser_commands_and_states",
                "path": "streams/laser.csv",
                "runtimeState": runtime_state(SubsystemId.LASER),
            })
        sources.append({
            "id": "session_logs",
            "kind": "logs",
            "path": "logs/session.log",
            "runtimeState": runtime_state(SubsystemId.SESSION_LOGS),
        })
        sources.append({
            "id": "trials",
            "kind": "pellet_trial_ledger",
            "path": "streams/trials.jsonl",
            "runtimeState": "ready",
        })
        return tuple(sources)

    def _record_start_timed_out(self) -> None:
        if self._session_recording_status != SessionRecordingStatus.ARMING:
            return
        logger.error("Timed out waiting for the primary camera to begin recording")
        self.on_error(
            "Recording failed",
            "The primary camera did not begin recording within 10 seconds. "
            "The partial session will be aborted.",
        )
        self.abort_recording()

    def _start_automatic_stop_policy(self, start_perf_time: float) -> None:
        policy = self._session_stop_policy
        if policy is None:
            return
        policy.start(start_perf_time)
        duration = policy.configuration.duration_seconds
        if duration is not None:
            self._automatic_stop_timer.cancel()
            self._automatic_stop_timer = make_daemon_timer(
                duration,
                self._evaluate_automatic_stop_policy,
            )
            self._automatic_stop_timer.start()

    def _cancel_automatic_stop_timers(self) -> None:
        for timer in (self._automatic_stop_timer, self._stop_drain_timer):
            if not timer.finished.is_set():
                timer.cancel()
        self._automatic_stop_timer = no_op_timer
        self._stop_drain_timer = no_op_timer

    def _evaluate_automatic_stop_policy(
        self,
        *,
        protocol_complete: bool = False,
    ) -> Optional[SessionStopEvaluation]:
        policy = self._session_stop_policy
        if (
            policy is None
            or not policy.is_started
            or self._session_recording_status is not SessionRecordingStatus.RECORDING
        ):
            return None
        ledger = self._trial_ledger
        evaluation = policy.evaluate(
            get_perf_now(),
            trial_count=0 if ledger is None else ledger.count(),
            protocol_complete=protocol_complete,
            trial_active=(ledger is not None and ledger.active_attempt is not None),
        )
        self._session_stop_evaluation = evaluation
        if evaluation.decision is SessionStopDecision.NONE:
            return evaluation
        if evaluation.decision is SessionStopDecision.FINISH_ACTIVE_TRIAL:
            self._behavior.algorithm.pellet_automation_stop_requested = True
            if self._stop_drain_timer is no_op_timer:
                self._stop_drain_timer = make_daemon_timer(
                    policy.configuration.drain_timeout_seconds,
                    self._evaluate_automatic_stop_policy,
                )
                self._stop_drain_timer.start()
            logger.info(
                "automatic session stop requested; finishing active trial: %s",
                evaluation.reason,
            )
            return evaluation
        if evaluation.decision is SessionStopDecision.TIMEOUT_ERROR:
            now_perf = get_perf_now()
            if ledger is not None and ledger.active_attempt is not None:
                ledger.finalize(
                    TrialOutcome.INCOMPLETE,
                    now_perf,
                    time.time(),
                    error=(
                        "active trial did not finish within the configured "
                        "stop drain timeout"
                    ),
                )
            message = (
                "Active pellet trial did not finish within "
                f"{policy.configuration.drain_timeout_seconds:g} seconds"
            )
            logger.error(message)
            self.on_error("Automatic recording stop timeout", message)
        self._stop_recording_with_reason(evaluation.reason, evaluation.decision)
        return evaluation

    def _stop_recording_with_reason(
        self,
        reason: Optional[SessionStopReason],
        decision: SessionStopDecision = SessionStopDecision.STOP,
    ) -> bool:
        ending_reason = {
            SessionStopReason.DURATION_LIMIT: RecordingEndingReason.DURATION_LIMIT,
            SessionStopReason.TRIAL_LIMIT: RecordingEndingReason.TRIAL_LIMIT,
            SessionStopReason.PROTOCOL_COMPLETE: RecordingEndingReason.PROTOCOL_COMPLETE,
        }.get(reason, RecordingEndingReason.STOP_DRAIN_TIMEOUT)
        if decision is SessionStopDecision.TIMEOUT_ERROR:
            ending_reason = RecordingEndingReason.STOP_DRAIN_TIMEOUT
        return self._stop_recording(ending_reason)

    def _stop_recording(self, reason: RecordingEndingReason) -> bool:
        if self._session_recording_status != SessionRecordingStatus.RECORDING:
            logger.warning(
                "stop_recording refused while %s",
                self._session_recording_status.value,
            )
            return False
        self._cancel_automatic_stop_timers()
        self._session_analysis_finished = False
        self._set_session_recording_status(SessionRecordingStatus.STOPPING)
        stopped = self._behavior.algorithm.end_capture_session(reason=reason)
        if not stopped:
            self._set_session_recording_status(SessionRecordingStatus.RECORDING)
        return bool(stopped)

    def stop_recording(self) -> bool:
        return self._stop_recording(RecordingEndingReason.MANUAL_STOP)

    def abort_recording(self) -> bool:
        previous_status = self._session_recording_status
        if previous_status not in {
            SessionRecordingStatus.ARMING,
            SessionRecordingStatus.RECORDING,
        }:
            logger.warning("abort_recording refused while %s", self._session_recording_status.value)
            return False
        project = self._project_info
        if project is None:
            return False
        self._aborting_project = project.to_local_value()
        self._aborted_session_ids.add(self._aborting_project.short_id)
        self._session_data_recorder.abort()
        self._cancel_automatic_stop_timers()
        self._pending_session_end_perf = None
        self._record_start_timer.cancel()
        self._record_start_timer = no_op_timer
        self._set_session_recording_status(SessionRecordingStatus.ABORTING)
        stopped = self._behavior.algorithm.end_capture_session(
            reason=RecordingEndingReason.MANUAL_ABORT,
        )
        if not stopped:
            self._set_session_recording_status(previous_status)
            self._aborting_project = None
            return False
        if previous_status == SessionRecordingStatus.ARMING:
            self._abort_cleanup_timer.cancel()
            self._abort_cleanup_timer = make_daemon_timer(
                1.0,
                self._finish_abort_if_never_started,
            )
            self._abort_cleanup_timer.start()
        return True

    def _finish_abort_if_never_started(self) -> None:
        if (
            self._session_recording_status == SessionRecordingStatus.ABORTING
            and not self._abort_had_recording_started
            and all(
                camera.video_status != CaptureProcessStatus.RECORDING
                for camera in self._get_recording_cams()
            )
        ):
            self._finish_abort_recording()

    def check_target_status_valid(self, target: AppModelStatus):
        current_status = self._status
        if (
            target != current_status
            and self._session_recording_status != SessionRecordingStatus.READY
        ):
            raise InvalidTargetAppModelStatus(
                "System Mode cannot change while a recording is being captured or analyzed"
            )
        if target != current_status:
            valid = app_status_is_target_status_valid(current_status, target)
            if not valid:
                raise InvalidTargetAppModelStatus(f"New status {target} not valid for current status {current_status}")

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
        is_from_start = status in {AppModelStatus.RUNNING, AppModelStatus.IDLE}
        for cam in self._cameras:
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
        ordered_cameras = self._ordered_reach_cameras(enabled_only=True) or self._ordered_reach_cameras()
        if len(ordered_cameras) > 0:
            primary = ordered_cameras[0]
            logger.debug("using primary cam_idx=%s", primary.camera_index)
            return primary.camera_index
        logger.verbose(
            "No primary camera defined, using camera-0 as main one: %s",
            self._cameras[0],
        )
        return 0

    def _identify_session_reference_cam_idx(self):
        for camera in self._ordered_reach_cameras(enabled_only=True):
            if camera.is_recording_enabled:
                return camera.camera_index
        return self._identify_primary_main_cam_idx()

    def _ordered_reach_cameras(self, *, enabled_only: bool = False) -> Tuple[VideoCaptureModel, ...]:
        cameras = [
            camera for camera in self._reach_cameras
            if not enabled_only or camera.is_enabled
        ]
        if len(cameras) == 0:
            return ()
        primary = next((camera for camera in cameras if camera.is_primary), cameras[0])
        return (primary, *(camera for camera in cameras if camera is not primary))

    def _ensure_reach_primary_camera(self) -> None:
        enabled_cameras = [
            camera for camera in self._reach_cameras
            if camera.is_enabled
        ]
        if len(enabled_cameras) == 1:
            enabled_cameras[0].set_runtime_primary(True)
            return
        for camera in enabled_cameras:
            camera.set_runtime_primary(camera.configured_is_primary)

    @staticmethod
    def _camera_timing_field_name(camera: VideoCaptureModel) -> str:
        name = re.sub(r"[^0-9A-Za-z]+", "_", camera.name).strip("_").lower()
        return name or f"camera_{camera.camera_index}"

    def _merge_camera_timestamp_files(self, project: ProjectInfo, cams: Tuple[VideoCaptureModel]):
        if len(cams) == 0:
            logger.warning("_merge_camera_timestamp_files called without cameras")
            return
        timing_path = project.get_frame_timing_path()
        data = []
        prim_cam = cams[0]  # primary
        main_fps = prim_cam.active_config.params.get('fps', math.nan)
        logger.info("Merging camera timestamp files for session%03d into %s (fps=%s)",
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
        boundary = self._session_boundary
        if (
            boundary is not None
            and boundary.session_id == project.short_id
            and int(main_cam_first_frame_id) != boundary.primary_frame_id
        ):
            raise RuntimeError(
                "Merged primary camera timing begins at frame "
                f"{int(main_cam_first_frame_id)}, expected canonical frame "
                f"{boundary.primary_frame_id}"
            )
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
        r0 = df_main_cam[:1]
        start_frame_id = r0['frame_id'][0]
        first_frame_utc_when = r0['frame_time'][0]
        logger.debug("start_frame_id=%s (utc_when=%s)", start_frame_id, first_frame_utc_when)
        camera_field_names = []
        used_camera_field_names = set()
        for cam in cams:
            base_name = self._camera_timing_field_name(cam)
            name = base_name
            suffix = 1
            while name in used_camera_field_names:
                suffix += 1
                name = f"{base_name}_{suffix}"
            used_camera_field_names.add(name)
            camera_field_names.append(name)
        timing_csv_fields = [
            'frame_id',
            'frame_when',
            'frame_present_primary',
            'frame_present_secondary',
            *(
                field
                for camera_field_name in camera_field_names
                for field in (f"frame_when_{camera_field_name}", f"frame_present_{camera_field_name}")
            ),
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
                second_frame_when = data[1]['frame_when'][idx] if len(data) > 1 and idx < len(data[1]) else math.nan
                d = dict(
                    frame_id=frame_id,
                    frame_when=frame_when if math.isfinite(frame_when) else "",  # could keep the math.nan otherwise
                    frame_present_primary=1 if math.isfinite(frame_when) else 0,
                    frame_present_secondary=1 if math.isfinite(second_frame_when) else 0,
                    utc_when=utc_when,
                )
                for cam, df, camera_field_name in zip(cams, data, camera_field_names):
                    if idx >= len(df):
                        camera_frame_when = math.nan
                    else:
                        camera_frame_when = df['frame_when'][idx]
                    d[f"frame_when_{camera_field_name}"] = (
                        camera_frame_when if math.isfinite(camera_frame_when) else ""
                    )
                    d[f"frame_present_{camera_field_name}"] = 1 if math.isfinite(camera_frame_when) else 0
                dw.writerow(d)
        logger.info("Written %s entries into %s", len(df_main_cam), timing_path)
        self._remove_timestamps_txt_files(project, cams)

    def _get_monitored_cams(self):
        return self._ordered_reach_cameras(enabled_only=True)

    def _get_recording_cams(self) -> Tuple[VideoCaptureModel, ...]:
        """Return every enabled camera expected to acknowledge this session."""
        return tuple(
            camera for camera in self._get_monitored_cams()
            if camera.is_recording_enabled
        )

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
            reference_cam_idx = (
                self._identify_session_reference_cam_idx()
                if self._session_recording_status
                is not SessionRecordingStatus.READY
                else self._identify_primary_main_cam_idx()
            )
            if cam_idx == reference_cam_idx:
                if new_status == CaptureProcessStatus.RECORDING:
                    first_frame_perf, first_frame_when, first_frame_time, *r_args = r_args
                    p_now = first_frame_perf
                    project = self._project_info
                    if project is not None:
                        project.start_record_timestamp = first_frame_time
                        primary = next(
                            camera
                            for camera in self._cameras
                            if camera.camera_index == cam_idx
                        )
                        first_frame_id = (
                            int(r_args[0])
                            if r_args
                            else int(self._cams_synced_frame_index.value)
                        )
                        self._session_boundary = SessionBoundary(
                            session_id=project.short_id,
                            primary_camera=primary.name,
                            primary_frame_id=first_frame_id,
                            start_perf_time=float(first_frame_perf),
                            start_wall_time=float(first_frame_time),
                            camera_when=float(first_frame_when),
                        )
                    self._session_data_recorder.commit_start(
                        first_frame_perf,
                        first_frame_time,
                        boundary=self._session_boundary,
                    )
                    self._record_start_timer.cancel()
                    self._record_start_timer = no_op_timer
                    self._abort_had_recording_started = True
                    logger.info(
                        "received RECORDING: frame-0 time=%.3f perf_c=%.3f now=%.3f",
                        first_frame_time,
                        first_frame_perf,
                        get_perf_now(),
                    )
                    if self._session_recording_status == SessionRecordingStatus.ARMING:
                        self._set_session_recording_status(SessionRecordingStatus.RECORDING)
                        self._start_automatic_stop_policy(first_frame_perf)
                else:
                    p_now = get_perf_now()
                algo.set_capture_status(new_status, perf_now=p_now)
                if new_status == CaptureProcessStatus.RUNNING:
                    if self._session_recording_status == SessionRecordingStatus.STOPPING:
                        self._pending_session_end_perf = (
                            r_args[0] if r_args else get_perf_now()
                        )
                    elif (
                        self._session_recording_status == SessionRecordingStatus.ABORTING
                        and not self._abort_had_recording_started
                    ):
                        self._finish_abort_recording()
            else:
                logger.verbose("not handling non-primary camera status, cam_idx=%s status=%s",
                               cam_idx, new_status)
        elif cmd == SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED:
            cam_idx, frames_written, project, *r_args = args
            if project is None:
                return
            recording_cams = self._get_recording_cams()
            recording_cam_indices = tuple(cam.camera_index for cam in recording_cams)
            if cam_idx in recording_cam_indices:
                cams_closed_finished[cam_idx] = (project, frames_written)
                if all(cam.camera_index in cams_closed_finished for cam in recording_cams):
                    project = cams_closed_finished[recording_cams[0].camera_index][0]
                    session_dir = Path(project.get_session_path().location)
                    for camera in recording_cams:
                        camera_project, camera_frames = cams_closed_finished[
                            camera.camera_index
                        ]
                        video_path, _, _ = camera_project.get_video_path(
                            camera.name,
                            allow_overwrite=True,
                        )
                        self._session_data_recorder.set_source_result(
                            f"camera.{camera.name}",
                            sample_count=camera_frames,
                            path=Path(video_path).relative_to(
                                session_dir
                            ).as_posix(),
                        )
                    monitored_cams = tuple(
                        camera for camera in recording_cams
                        if camera in self._reach_cameras
                    )
                    try:
                        self._merge_camera_timestamp_files(project, monitored_cams)
                    except Exception as err:
                        logger.exception("Failed to merge camera timestamps: %s", err)
                        self.on_error("Camera timestamp merge failed", str(err))
                    cams_closed_finished.clear()  # now clear
                    wait_pose_closed = getattr(
                        type(self._inference),
                        "wait_session_pose_closed",
                        None,
                    )
                    if (
                        wait_pose_closed is not None
                        and self._abort_had_recording_started
                        and not wait_pose_closed(
                            self._inference,
                            project,
                            timeout=10.0,
                        )
                    ):
                        message = (
                            "Timed out waiting for live pose files to close for "
                            f"{project.short_id}; continuing writer finalization."
                        )
                        logger.error(message)
                        self.on_error("Pose writer close timeout", message)
                        self._session_data_recorder.set_source_result(
                            "pose",
                            failure=message,
                        )
                    if self._session_recording_status == SessionRecordingStatus.ABORTING:
                        self._finish_abort_recording()
                    elif self._session_recording_status == SessionRecordingStatus.STOPPING:
                        self._complete_stopped_recording(project)
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
    def stim_camera(self):
        return self._stim_camera

    @property
    def cameras(self) -> Tuple[VideoCaptureModel, ...]:
        return tuple(self._cameras)

    @property
    def reach_cameras(self) -> Tuple[VideoCaptureModel, ...]:
        return self._reach_cameras

    @property
    def inference_cameras(self) -> Tuple[VideoCaptureModel, ...]:
        return self._inference_cameras

    def get_camera_model(self, camera_id: CameraId) -> Optional[VideoCaptureModel]:
        return self._camera_by_id.get(camera_id)

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
    def nidaq_ports(self) -> NidaqPortConfiguration:
        return self._nidaq_ports

    @property
    def hardware_scan_results(self) -> Dict[str, HardwareScanEntry]:
        return dict(self._hardware_scan_results)

    @property
    def subsystem_statuses(self) -> Dict[str, SubsystemStatus]:
        return dict(self._subsystem_status_registry.statuses)

    @property
    def recording_blockers(self) -> Tuple[str, ...]:
        blockers = list(self._subsystem_status_registry.recording_blockers())
        if self._protocol_runner.protocol_complete:
            blockers.append("Selected protocol is complete")
        if self._session_recording_status is not SessionRecordingStatus.READY:
            blockers.append(
                f"recording state: {self._session_recording_status.value}"
            )
        if not self._acquisition_started:
            blockers.append("System Mode is not running")
        return tuple(blockers)

    def _set_subsystem_status(
        self,
        subsystem_id,
        state: SubsystemState,
        *,
        reason: str = "",
        error: str = "",
        required_for_recording: Optional[bool] = None,
        generation: Optional[int] = None,
    ) -> SubsystemStatus:
        with self._subsystem_status_lock:
            previous_statuses = self._subsystem_status_registry.statuses
            status, _ = self._subsystem_status_registry.transition(
                subsystem_id,
                state,
                reason=reason,
                error=error,
                required_for_recording=required_for_recording,
                generation=generation,
            )
            current_statuses = self._subsystem_status_registry.statuses
        self.property_changed(
            self.Props.SUBSYSTEM_STATUSES,
            current_statuses,
            previous_statuses,
        )
        self.property_changed(
            self.Props.RECORDING_BLOCKERS,
            self.recording_blockers,
            None,
        )
        return status

    def _begin_subsystem_start(
        self,
        subsystem_id,
        *,
        required_for_recording: Optional[bool] = None,
        reason: str = "",
    ) -> int:
        with self._subsystem_status_lock:
            previous_statuses = self._subsystem_status_registry.statuses
            status, _ = self._subsystem_status_registry.begin_retry(
                subsystem_id,
                required_for_recording=required_for_recording,
                reason=reason,
            )
            current_statuses = self._subsystem_status_registry.statuses
        self.property_changed(
            self.Props.SUBSYSTEM_STATUSES,
            current_statuses,
            previous_statuses,
        )
        return status.generation

    def _configure_subsystem_intent(
        self,
        configuration: SystemConfiguration,
    ) -> None:
        enabled_reach = tuple(
            camera for camera in self._reach_cameras if camera.is_enabled
        )
        multi_reach = len(enabled_reach) > 1
        for camera in self._cameras:
            required = bool(
                camera.is_enabled
                and (
                    camera.is_recording_enabled
                    or (camera in self._reach_cameras and multi_reach)
                )
            )
            self._set_subsystem_status(
                SubsystemId.camera(camera.name),
                (
                    SubsystemState.STOPPED
                    if camera.is_enabled
                    else SubsystemState.DISABLED
                ),
                reason=(
                    "configured; not started"
                    if camera.is_enabled
                    else "camera disabled"
                ),
                required_for_recording=required,
            )
        self._set_subsystem_status(
            SubsystemId.REACH_SYNCHRONIZATION,
            (
                SubsystemState.STOPPED
                if enabled_reach
                else SubsystemState.DISABLED
            ),
            reason="configured; not validated" if enabled_reach else "no reach cameras",
            required_for_recording=bool(
                multi_reach
                or any(camera.is_recording_enabled for camera in enabled_reach)
            ),
        )
        intent = (
            (
                SubsystemId.LIVE_INFERENCE,
                configuration.inference.is_enabled,
                configuration.inference.is_enabled,
            ),
            (
                SubsystemId.CAN_PELLET,
                self._hardware.requires_connection,
                self._hardware.requires_connection,
            ),
            (
                SubsystemId.NIDAQ_STREAM,
                configuration.hardware.nidaq_enabled
                and configuration.nidaq_stream.is_enabled,
                configuration.hardware.nidaq_enabled
                and configuration.nidaq_stream.is_enabled,
            ),
            (
                SubsystemId.LASER,
                configuration.laser.backend != "disabled",
                configuration.laser.backend != "disabled",
            ),
        )
        for subsystem_id, enabled, required in intent:
            self._set_subsystem_status(
                subsystem_id,
                SubsystemState.STOPPED if enabled else SubsystemState.DISABLED,
                reason="configured; not started" if enabled else "disabled",
                required_for_recording=required,
            )
        self._set_subsystem_status(
            SubsystemId.SESSION_LOGS,
            SubsystemState.STOPPED,
            reason="not validated",
            required_for_recording=True,
        )
        self._set_subsystem_status(
            SubsystemId.OFFLINE_ANALYSIS,
            SubsystemState.DISABLED,
            reason="no stopped session pending",
            required_for_recording=False,
        )

    @property
    def configured_nidaq_device_names(self) -> Tuple[str, ...]:
        names: List[str] = []

        def add_channel(channel_name: Optional[str]) -> None:
            device_name = device_name_from_channel(channel_name)
            if device_name and device_name not in names:
                names.append(device_name)

        if self._nidaq_ports.device_name:
            names.append(self._nidaq_ports.device_name)
        for field in dataclasses.fields(self._nidaq_ports):
            if field.name != "device_name":
                value = getattr(self._nidaq_ports, field.name)
                if isinstance(value, str):
                    add_channel(value)
        for channel in self._nidaq_signal_monitor.configuration.channels:
            add_channel(channel.physical_channel)
        for channel in self._laser.configuration.channels:
            for physical_channel in (
                channel.analog_output,
                channel.diode_input,
                channel.shutter_output,
                channel.command_copy_input,
            ):
                add_channel(physical_channel)
        return tuple(names)

    def refresh_hardware_bindings(self) -> str:
        if self._acquisition_started or self._status != AppModelStatus.IDLE:
            raise RuntimeError("Hardware refresh is only available while acquisition is idle")

        scan_started = time.perf_counter()
        details: List[str] = []
        warnings_list: List[str] = []
        scan_results: Dict[str, HardwareScanEntry] = {}
        try:
            can_transport = CanTransportConfiguration.from_environment()
            selected_can_interface = can_transport.channel
            selected_can_backend = can_transport.kind.value
        except Exception:
            logger.exception("CAN transport selection is invalid")
            selected_can_interface = None
            selected_can_backend = None
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hardware-scan")
        camera_future = executor.submit(create_camera_list, include_hardware=True)
        nidaq_future = executor.submit(discover_nidaq_devices)
        can_adapter_future = executor.submit(
            scan_can_adapters,
            selected_interface=selected_can_interface,
            selected_backend=selected_can_backend,
        )
        gpu_future = executor.submit(scan_gpus)

        try:
            camera_sources = camera_future.result()
            source_bindings = {
                camera_source_binding_key(source.url)
                for source in camera_sources
            }
            for camera in self._cameras:
                camera.refresh_camera_list(camera_sources)
            missing_enabled_cameras = [
                camera.name
                for camera in self._cameras
                if (
                    camera.is_enabled
                    and camera.camera_source is not None
                    and camera_source_binding_key(camera.camera_source.url) not in source_bindings
                )
            ]
            details.append(f"cameras {len(camera_sources)} source(s)")
            source_names = tuple(
                source.name or source.url
                for source in camera_sources
                if source.name or source.url
            )
            camera_info = f"✓ {len(camera_sources)} camera source(s)"
            if source_names:
                camera_info += "\n" + "\n".join(f"→ {name}" for name in source_names)
            if missing_enabled_cameras:
                missing_text = ", ".join(missing_enabled_cameras)
                warnings_list.append(f"configured camera(s) not currently discovered: {missing_text}")
                camera_info += f"\n! missing: {missing_text}"
            scan_results["cameras"] = HardwareScanEntry(
                camera_info,
                "warning" if missing_enabled_cameras or not camera_sources else "ok",
            )
        except Exception as exc:
            logger.exception("Hardware refresh camera scan failed")
            error_text = str(exc) or exc.__class__.__name__
            warnings_list.append(f"camera scan failed: {error_text}")
            scan_results["cameras"] = HardwareScanEntry(f"scan failed: {error_text}", "error")

        try:
            nidaq_devices, nidaq_error = nidaq_future.result()
            details.append(f"NI-DAQ {len(nidaq_devices)} device(s)")
            discovered_nidaq_names = {device.name for device in nidaq_devices}
            nidaq_info = f"✓ {len(nidaq_devices)} NI-DAQ card(s)"
            if nidaq_devices:
                nidaq_info += "\n" + "\n".join(
                    self._nidaq_device_summary(device)
                    for device in nidaq_devices
                )
            nidaq_state = "ok" if nidaq_devices else "warning"
            if nidaq_error:
                warnings_list.append(nidaq_error)
                nidaq_info += f"\n! {nidaq_error}"
                nidaq_state = "error"
            missing_nidaq_names = tuple(
                name
                for name in self.configured_nidaq_device_names
                if name not in discovered_nidaq_names
            )
            if self._hardware.nidaq_enabled and missing_nidaq_names:
                missing_text = ", ".join(missing_nidaq_names)
                warnings_list.append(f"configured NI-DAQ device(s) not discovered: {missing_text}")
                nidaq_info += f"\n! missing: {missing_text}"
                if nidaq_state != "error":
                    nidaq_state = "warning"
            elif self._hardware.nidaq_enabled and not nidaq_devices:
                warnings_list.append("no NI-DAQ devices discovered while NI-DAQ is enabled")
            scan_results["nidaq"] = HardwareScanEntry(nidaq_info, nidaq_state)
        except Exception as exc:
            logger.exception("Hardware refresh NI-DAQ scan failed")
            error_text = str(exc) or exc.__class__.__name__
            warnings_list.append(f"NI-DAQ scan failed: {error_text}")
            scan_results["nidaq"] = HardwareScanEntry(f"scan failed: {error_text}", "error")

        try:
            can_adapter_entry = can_adapter_future.result()
        except Exception as exc:
            logger.exception("Hardware refresh CAN adapter scan failed")
            error_text = str(exc) or exc.__class__.__name__
            can_adapter_entry = HardwareScanEntry(f"adapter scan failed: {error_text}", "error")
            warnings_list.append(f"CAN adapter scan failed: {error_text}")
        finally:
            executor.shutdown(wait=True)
        scan_results["can"] = can_adapter_entry
        details.append(can_adapter_entry.info.splitlines()[0])

        try:
            gpu_entry = gpu_future.result()
        except Exception as exc:
            logger.exception("Hardware refresh GPU scan failed")
            gpu_entry = HardwareScanEntry(f"! GPU scan failed\n→ {str(exc) or exc.__class__.__name__}", "error")
            warnings_list.append(f"GPU scan failed: {str(exc) or exc.__class__.__name__}")
        scan_results["gpu"] = gpu_entry
        details.append(gpu_entry.info.splitlines()[0])

        pellet_entry = self._initialize_pellet_controller_for_refresh()
        scan_results["pellet"] = pellet_entry
        details.append(pellet_entry.info.splitlines()[0])
        if pellet_entry.state == "error":
            warnings_list.append(
                f"pellet controller initialization failed: "
                f"{pellet_entry.info.splitlines()[-1].lstrip('→ ').strip()}"
            )

        laser_configuration = self._laser.configuration
        if laser_configuration.backend == "disabled":
            details.append("laser not in use")
            scan_results["laser"] = HardwareScanEntry("not probed; backend disabled", "disabled")
        elif self._laser.is_connected:
            details.append(f"laser {laser_configuration.backend} connected")
            scan_results["laser"] = HardwareScanEntry(
                f"not probed; {laser_configuration.backend} connected",
                "ok",
            )
        else:
            details.append(f"laser {laser_configuration.backend} configured")
            warnings_list.append(f"laser backend {laser_configuration.backend} is configured but not connected")
            scan_results["laser"] = HardwareScanEntry(
                f"not probed; {laser_configuration.backend} configured but not connected",
                "warning",
            )

        previous_scan_results = self._hardware_scan_results
        self._hardware_scan_results = scan_results
        self._on_property_changed(
            self.Props.HARDWARE_SCAN_RESULTS,
            dict(scan_results),
            previous_scan_results,
        )

        message = (
            f"Hardware refresh completed in {time.perf_counter() - scan_started:.2f}s: "
            + ", ".join(details)
        )
        if warnings_list:
            warning_text = "; ".join(warnings_list)
            log_hardware_initialization(
                logger,
                "WARNING | hardware refresh | %s; warnings: %s",
                message,
                warning_text,
                level=logging.WARNING,
            )
            return f"{message}; warnings: {warning_text}"
        log_hardware_initialization(logger, "READY | hardware refresh | %s", message)
        return message

    def _ensure_pellet_controller_connected(self) -> None:
        if self._hardware.connected:
            return
        self._hardware.connect(self._system_message_handler.input_queue)

    def _initialize_pellet_controller_for_refresh(self) -> HardwareScanEntry:
        hardware = self._hardware
        if not hardware.can_enabled or not hardware.pellet_controller_enabled:
            return HardwareScanEntry("– not in use", "disabled")
        try:
            self._ensure_pellet_controller_connected()
            if not hardware.connected:
                raise RuntimeError("controller did not report a connected session")
        except Exception as exc:
            error_text = str(exc) or exc.__class__.__name__
            logger.exception("Pellet controller initialization during hardware refresh failed")
            try:
                hardware.safety_shutdown(
                    f"startup hardware refresh failure: {error_text}",
                    wait=True,
                )
            except Exception:
                logger.exception("Pellet controller cleanup after refresh failure failed")
            return HardwareScanEntry(
                f"! controller connection failed\n→ {error_text}",
                "error",
            )

        version = hardware.pellet_version
        version_text = f" · firmware {version}" if version else ""
        return HardwareScanEntry(
            f"✓ controller session connected{version_text}",
            "ok",
        )

    @staticmethod
    def _nidaq_device_summary(device) -> str:
        identity = f"→ {device.name}"
        if device.product_type:
            identity += f" · {device.product_type}"
        if device.product_number is not None:
            identity += f" · #{device.product_number}"
        return identity

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
            self._set_animal_base_positions(animal)
            self.training_plan = self.get_training_plan_by_id(
                animal.training.current_protocol
            )
        self._preferences.selected_animal = "" if animal is None else animal.name
        self._on_property_changed(self.Props.SELECTED_ANIMAL, animal, prev)
        self._event_manager.post_event_content(
            ApiEventKind.animalSelected, None if animal is None else animal.to_api_status())
        logger.success("Switched to animal %s", animal)

    @property
    def attached_plan(self) -> Optional[TrainingPlan]:
        return self._attached_plan

    def _derived_api_training_mode(self) -> ApiTrainingMode:
        return protocol_state_to_api_training_mode(
            self._attached_plan,
            automatic_advance=(
                self._behavior.algorithm.active_config.session_control
                .automatic_protocol_advance_enabled
            ),
        )

    def set_automatic_protocol_advance_enabled(self, enabled: bool) -> None:
        """Apply protocol advancement consistently to config and live runner."""
        enabled = bool(enabled)
        session_control = self._behavior.algorithm.active_config.session_control
        session_control.automatic_protocol_advance_enabled = enabled
        self._protocol_runner.automatic_advance = enabled
        if self._attached_plan is not None:
            self._attached_plan.is_automatic = enabled
            self._on_property_changed(
                self.Props.TRAINING_PLAN_PROP,
                self._attached_plan,
                self._attached_plan,
            )
        self._event_manager.post_event_content(
            ApiEventKind.trainingModeChanged,
            dict(training_mode=self._derived_api_training_mode()),
        )

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
            if self._attach_training_plan(plan, force_update=force_update) is False:
                return
        self._on_property_changed(self.Props.TRAINING_PLAN, plan, prev)
        self._event_manager.post_event_content(
            ApiEventKind.trainingPlanLoad, {'training_plan_id': None if plan is None else plan.plan_id})
        self._event_manager.post_event_content(
            ApiEventKind.trainingModeChanged,
            dict(training_mode=self._derived_api_training_mode()),
        )

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

        session_control = algo.active_config.session_control
        is_auto = session_control.automatic_protocol_advance_enabled
        logger.success("Animal %s: attaching auto=%s to plan %s (%s) ..",
                       animal.name, is_auto, plan.plan_id, hex(id(plan)))
        plan.is_automatic = is_auto
        pellet_dev = self._hardware
        self._protocol_runner.automatic_advance = is_auto
        self._protocol_runner.attach(plan, algo, pellet_dev)
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
        if self._protocol_runner.plan is plan:
            self._protocol_runner.detach()
        else:
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
        camera_names = tuple(
            camera.name
            for camera in self._reach_cameras
            if camera.is_enabled and camera.is_recording_enabled
        )
        return ProjectInfo(
            root=self._output_location,
            device_id=self._preferences.serial_number,
            ensure_exists=True,
            camera_names=camera_names,
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

    def _prepare_camera_domain(
        self,
        camera: VideoCaptureModel,
        inference_queue,
        inference_index: Optional[int],
    ) -> bool:
        subsystem_id = SubsystemId.camera(camera.name)
        generation = self._begin_subsystem_start(
            subsystem_id,
            reason="starting camera process",
        )
        started = time.perf_counter()
        try:
            did_start = camera.on_prepare_capture(
                inference_queue,
                inference_index=inference_index,
            )
            if not did_start:
                raise RuntimeError(camera.last_error or "camera process did not start")
            if (
                not camera.wait_for_capture_status(
                    (CaptureProcessStatus.RUNNING, CaptureProcessStatus.FAILED),
                    timeout=5,
                )
                or camera.video_status != CaptureProcessStatus.RUNNING
            ):
                raise RuntimeError(camera.last_error or "camera process did not become ready")
        except Exception as exc:
            try:
                camera.on_capture_stop()
            except Exception:
                logger.exception("Failed to clean up camera %s", camera.name)
            self._set_subsystem_status(
                subsystem_id,
                SubsystemState.FAILED,
                error=str(exc) or exc.__class__.__name__,
                generation=generation,
            )
            logger.exception("Camera %s initialization failed", camera.name)
            return False
        log_hardware_initialization(
            logger,
            "READY | camera process | name=%s status=%s elapsed=%.3fs",
            camera.name,
            camera.video_status.name,
            time.perf_counter() - started,
        )
        return True

    def _enable_camera_capture_domain(
        self,
        camera: VideoCaptureModel,
        generation: int,
        *,
        wait_for_first_frame: bool = True,
    ) -> bool:
        subsystem_id = SubsystemId.camera(camera.name)
        try:
            camera.on_capture_start()
            if not wait_for_first_frame:
                return True
        except Exception as exc:
            try:
                camera.on_capture_stop()
            except Exception:
                logger.exception("Failed to stop camera %s after capture failure", camera.name)
            self._set_subsystem_status(
                subsystem_id,
                SubsystemState.FAILED,
                error=str(exc) or exc.__class__.__name__,
                generation=generation,
            )
            logger.exception("Camera %s capture failed", camera.name)
            return False
        return self._wait_camera_first_frame_domain(camera, generation)

    def _wait_camera_first_frame_domain(
        self,
        camera: VideoCaptureModel,
        generation: int,
    ) -> bool:
        subsystem_id = SubsystemId.camera(camera.name)
        try:
            if not camera.wait_for_first_frame(timeout=5):
                raise RuntimeError(camera.last_error or "camera delivered no frame")
        except Exception as exc:
            try:
                camera.on_capture_stop()
            except Exception:
                logger.exception("Failed to stop camera %s after capture failure", camera.name)
            self._set_subsystem_status(
                subsystem_id,
                SubsystemState.FAILED,
                error=str(exc) or exc.__class__.__name__,
                generation=generation,
            )
            logger.exception("Camera %s first-frame validation failed", camera.name)
            return False
        self._set_subsystem_status(
            subsystem_id,
            SubsystemState.READY,
            reason=f"first frame {camera.last_captured_frame_index}",
            generation=generation,
        )
        return True

    def _start_reach_camera_domains(
        self,
        inference_camera_indices,
    ) -> bool:
        cameras = self._ordered_reach_cameras(enabled_only=True)
        if not cameras:
            self._set_subsystem_status(
                SubsystemId.REACH_SYNCHRONIZATION,
                SubsystemState.DISABLED,
                reason="no enabled reach cameras",
            )
            return False
        configured_primaries = tuple(
            camera for camera in cameras if camera.is_primary
        )
        topology_error = (
            None
            if len(cameras) == 1 or len(configured_primaries) == 1
            else (
                "multiple enabled reach cameras require exactly one primary; "
                f"found {len(configured_primaries)}"
            )
        )
        primary = cameras[0]
        prepared = {}
        generation = self._begin_subsystem_start(
            SubsystemId.REACH_SYNCHRONIZATION,
            reason="validating reach camera topology",
        )
        primary_index = inference_camera_indices.get(primary)
        primary_ready = self._prepare_camera_domain(
            primary,
            self._inference_queue if primary_index is not None else None,
            primary_index,
        )
        if not primary_ready:
            for secondary in cameras[1:]:
                self._set_subsystem_status(
                    SubsystemId.camera(secondary.name),
                    SubsystemState.BLOCKED,
                    reason="primary trigger unavailable",
                )
            self._set_subsystem_status(
                SubsystemId.REACH_SYNCHRONIZATION,
                SubsystemState.FAILED,
                error=f"primary camera {primary.name} unavailable",
                generation=generation,
            )
            return False
        prepared[primary] = self._subsystem_status_registry.get(
            SubsystemId.camera(primary.name)
        ).generation

        for secondary in cameras[1:]:
            time.sleep(0.5)
            inference_index = inference_camera_indices.get(secondary)
            if self._prepare_camera_domain(
                secondary,
                self._inference_queue if inference_index is not None else None,
                inference_index,
            ):
                prepared[secondary] = self._subsystem_status_registry.get(
                    SubsystemId.camera(secondary.name)
                ).generation

        # Triggered secondaries must be armed before the primary emits edges.
        for secondary in cameras[1:]:
            if secondary in prepared and not self._enable_camera_capture_domain(
                    secondary,
                    prepared[secondary],
                    wait_for_first_frame=False,
            ):
                prepared.pop(secondary)
        primary_capture_started = self._enable_camera_capture_domain(
            primary,
            prepared[primary],
            wait_for_first_frame=False,
        )
        primary_capture_ready = (
            primary_capture_started
            and self._wait_camera_first_frame_domain(primary, prepared[primary])
        )
        for secondary in cameras[1:]:
            if secondary in prepared:
                self._wait_camera_first_frame_domain(
                    secondary,
                    prepared[secondary],
                )
        all_ready = topology_error is None and primary_capture_ready and all(
            (
                status := self._subsystem_status_registry.get(
                    SubsystemId.camera(camera.name)
                )
            ) is not None
            and status.is_ready
            for camera in cameras
        )
        self._set_subsystem_status(
            SubsystemId.REACH_SYNCHRONIZATION,
            SubsystemState.READY if all_ready else SubsystemState.FAILED,
            reason=(
                "standalone camera ready"
                if all_ready and len(cameras) == 1
                else "all enabled reach cameras synchronized"
                if all_ready
                else topology_error or "one or more enabled reach cameras unavailable"
            ),
            error="" if all_ready else topology_error or "reach synchronization incomplete",
            generation=generation,
        )
        return all_ready

    def _start_can_domain(self, *, wait_connected: bool) -> bool:
        hard = self._hardware
        if not hard.requires_connection:
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.DISABLED,
                reason="CAN/pellet controller disabled",
            )
            return False
        generation = self._begin_subsystem_start(
            SubsystemId.CAN_PELLET,
            reason="connecting CAN/pellet controller",
        )
        started = time.perf_counter()
        try:
            self._ensure_pellet_controller_connected()
            if wait_connected:
                deadline = time.perf_counter() + 3.0
                for token in tuple(hard.pending_tokens):
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for pending CAN command")
                    hard.wait_pending_command_acked(token, timeout=remaining)
                while not hard.connected:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("timed out waiting for CAN hardware connection")
                    time.sleep(0.05)
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.exception("CAN/pellet initialization failed")
            try:
                hard.safety_shutdown(
                    f"CAN/pellet initialization failure: {error}",
                    wait=True,
                )
            except Exception:
                logger.exception("CAN safety shutdown failed")
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.FAILED,
                error=error,
                generation=generation,
            )
            return False
        self._set_subsystem_status(
            SubsystemId.CAN_PELLET,
            SubsystemState.READY,
            reason="CAN/pellet controller connected",
            generation=generation,
        )
        log_hardware_initialization(
            logger,
            "READY | CAN/pellet controller | connected=%s elapsed=%.3fs",
            hard.connected,
            time.perf_counter() - started,
        )
        return True

    def _start_nidaq_domain(self, *, timeout: float = 12.0) -> bool:
        monitor = self._nidaq_signal_monitor
        if not (monitor.hardware_enabled and monitor.configuration.is_enabled):
            self._set_subsystem_status(
                SubsystemId.NIDAQ_STREAM,
                SubsystemState.DISABLED,
                reason="NI-DAQ stream disabled",
            )
            return False
        generation = self._begin_subsystem_start(
            SubsystemId.NIDAQ_STREAM,
            reason="starting synchronized NI-DAQ tasks",
        )
        try:
            if not monitor.start():
                raise RuntimeError(
                    monitor.error_message or "NI-DAQ stream did not start"
                )
            deadline = time.perf_counter() + timeout
            while monitor.is_starting and time.perf_counter() < deadline:
                time.sleep(0.02)
            if not monitor.is_running:
                raise RuntimeError(
                    monitor.error_message
                    or f"NI-DAQ stream was not ready within {timeout:g} seconds"
                )
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.exception("NI-DAQ initialization failed")
            monitor.stop()
            self._set_subsystem_status(
                SubsystemId.NIDAQ_STREAM,
                SubsystemState.FAILED,
                error=error,
                generation=generation,
            )
            return False
        timing_plan = monitor.timing_plan
        if (
            timing_plan is not None
            and timing_plan.resolved_mode == "independent"
            and len(timing_plan.task_start_order) > 1
        ):
            self._set_subsystem_status(
                SubsystemId.NIDAQ_STREAM,
                SubsystemState.BLOCKED,
                reason=(
                    "NI-DAQ diagnostic preview is running with independent device "
                    "clocks; deterministic session recording is unavailable"
                ),
                generation=generation,
            )
            return False
        self._set_subsystem_status(
            SubsystemId.NIDAQ_STREAM,
            SubsystemState.READY,
            reason="synchronized NI-DAQ tasks running",
            generation=generation,
        )
        return True

    def _start_laser_domain(self) -> bool:
        configuration = self._laser.configuration
        if configuration.backend == "disabled":
            self._set_subsystem_status(
                SubsystemId.LASER,
                SubsystemState.DISABLED,
                reason="laser disabled",
            )
            return False
        generation = self._begin_subsystem_start(
            SubsystemId.LASER,
            reason="opening laser controller",
        )
        try:
            runtime_configuration = configuration
            if (
                configuration.backend == "nidaq"
                and self._nidaq_ports.device_identities
            ):
                aliases = self._nidaq_signal_monitor.runtime_device_aliases
                if not aliases:
                    devices, discovery_error = discover_nidaq_devices()
                    if discovery_error:
                        raise RuntimeError(discovery_error)
                    aliases = resolve_nidaq_device_aliases(
                        self._nidaq_ports.device_identities,
                        devices,
                    )
                runtime_configuration = self._remap_laser_configuration(
                    configuration,
                    aliases,
                )
            feedback_reader = (
                self._read_nidaq_feedback_channel
                if (
                    configuration.backend == "nidaq"
                    and self._nidaq_signal_monitor.is_running
                )
                else None
            )
            self._laser.load_configuration(
                runtime_configuration,
                feedback_reader=feedback_reader,
                persisted_configuration=configuration,
            )
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.exception("Laser initialization failed")
            try:
                self._laser.close()
            except Exception:
                logger.exception("Laser cleanup failed")
            self._set_subsystem_status(
                SubsystemId.LASER,
                SubsystemState.FAILED,
                error=error,
                generation=generation,
            )
            return False
        self._set_subsystem_status(
            SubsystemId.LASER,
            SubsystemState.READY,
            reason="laser controller connected",
            generation=generation,
        )
        return True

    @staticmethod
    def _remap_laser_configuration(configuration, aliases):
        def remap(value):
            return (
                None
                if value is None
                else remap_nidaq_physical_channel(value, aliases)
            )

        return dataclasses.replace(
            configuration,
            channels=tuple(
                dataclasses.replace(
                    channel,
                    analog_output=remap(channel.analog_output),
                    diode_input=remap(channel.diode_input),
                    shutter_output=remap(channel.shutter_output),
                    auxiliary_output=remap(channel.auxiliary_output),
                    command_copy_input=remap(channel.command_copy_input),
                    trigger_source=remap(channel.trigger_source),
                    trigger_output=remap(channel.trigger_output),
                    timing_trigger_output=remap(
                        channel.timing_trigger_output
                    ),
                )
                for channel in configuration.channels
            ),
            pmt_shutter_output=remap(configuration.pmt_shutter_output),
            trigger_listener_inputs=tuple(
                remap(value)
                for value in configuration.trigger_listener_inputs
            ),
        )

    def _read_nidaq_feedback_channel(self, physical_channel: str) -> float:
        monitor = self._nidaq_signal_monitor
        if not monitor.is_running:
            raise RuntimeError("shared NI-DAQ input stream is not running")
        configuration = monitor.configuration
        channel_index = next(
            (
                index
                for index, channel in enumerate(configuration.channels)
                if remap_nidaq_physical_channel(
                    channel.physical_channel,
                    monitor.runtime_device_aliases,
                ) == physical_channel
            ),
            None,
        )
        if channel_index is None:
            raise KeyError(
                f"NI-DAQ input channel is not in the acquisition plan: {physical_channel}"
            )
        ring = monitor.sample_ring
        scratch = np.empty(
            (max(1, len(ring.channel_names)), 1),
            dtype=np.float32,
        )
        for _ in range(5):
            end_sample_index = ring.current_end_sample_index()
            if end_sample_index <= 0:
                break
            read = ring.copy_since(end_sample_index - 1, scratch)
            if read is not None and read.sample_count == 1:
                return float(scratch[channel_index, 0])
        raise RuntimeError(
            f"No current NI-DAQ feedback sample is available for {physical_channel}"
        )

    def _start_inference_domain(
        self,
        *,
        reach_synchronization_ready: bool,
        preflight_error: Optional[str],
    ) -> bool:
        if not self._inference.is_enabled:
            self._set_subsystem_status(
                SubsystemId.LIVE_INFERENCE,
                SubsystemState.DISABLED,
                reason="live inference disabled",
            )
            return False
        if preflight_error:
            return False
        if not reach_synchronization_ready:
            self._set_subsystem_status(
                SubsystemId.LIVE_INFERENCE,
                SubsystemState.BLOCKED,
                reason="enabled reach camera synchronization is not ready",
            )
            return False
        generation = self._begin_subsystem_start(
            SubsystemId.LIVE_INFERENCE,
            reason="starting live inference",
        )
        try:
            if not self._inference.start(self._inference_queue):
                raise RuntimeError("live inference process did not start")
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.exception("Live inference initialization failed")
            try:
                self._inference.stop()
            except Exception:
                logger.exception("Live inference cleanup failed")
            self._set_subsystem_status(
                SubsystemId.LIVE_INFERENCE,
                SubsystemState.FAILED,
                error=error,
                generation=generation,
            )
            return False
        self._set_subsystem_status(
            SubsystemId.LIVE_INFERENCE,
            SubsystemState.READY,
            reason="live inference running",
            generation=generation,
        )
        return True

    def _validate_session_logs_domain(self) -> bool:
        generation = self._begin_subsystem_start(
            SubsystemId.SESSION_LOGS,
            reason="validating session output location",
        )
        try:
            output_location = Path(self._output_location).expanduser()
            output_location.mkdir(parents=True, exist_ok=True)
            if not os.access(output_location, os.W_OK):
                raise PermissionError(f"session output is not writable: {output_location}")
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            logger.exception("Session output validation failed")
            self._set_subsystem_status(
                SubsystemId.SESSION_LOGS,
                SubsystemState.FAILED,
                error=error,
                generation=generation,
            )
            return False
        self._set_subsystem_status(
            SubsystemId.SESSION_LOGS,
            SubsystemState.READY,
            reason=f"session output writable: {output_location}",
            generation=generation,
        )
        return True

    def _log_acquisition_startup_summary(self) -> None:
        statuses = self._subsystem_status_registry.statuses
        for subsystem_id, status in statuses.items():
            detail = status.error or status.reason or "no detail"
            log_hardware_initialization(
                logger,
                "SUMMARY | %s | state=%s detail=%s",
                subsystem_id,
                status.state.value,
                detail,
            )

        failed_cameras = tuple(
            subsystem_id
            for subsystem_id, status in statuses.items()
            if subsystem_id.startswith("camera.")
            and status.state in {SubsystemState.FAILED, SubsystemState.BLOCKED}
        )
        nidaq_status = statuses.get(SubsystemId.NIDAQ_STREAM.value)
        if (
            failed_cameras
            and nidaq_status is not None
            and nidaq_status.state in {SubsystemState.FAILED, SubsystemState.BLOCKED}
        ):
            log_hardware_initialization(
                logger,
                "CORRELATION | camera failure(s)=%s and NI-DAQ=%s occurred "
                "during the same startup. They may share a physical power, "
                "timing, trigger, or ground dependency; NI-DAQ software "
                "initialization does not trigger the cameras.",
                ",".join(failed_cameras),
                nidaq_status.state.value,
                level=logging.WARNING,
            )

    def retry_failed_subsystems(self) -> str:
        """Retry failed acquisition domains without disturbing healthy domains."""
        if not self._acquisition_started:
            raise RuntimeError("Subsystem retry requires System Mode to be running")
        if self._session_recording_status is not SessionRecordingStatus.READY:
            raise RuntimeError("Subsystem retry is unavailable during recording or analysis")

        failed = {
            subsystem_id
            for subsystem_id, status in self._subsystem_status_registry.statuses.items()
            if status.state in {SubsystemState.FAILED, SubsystemState.BLOCKED}
        }
        retried = []
        reach_failed = any(
            subsystem_id == SubsystemId.REACH_SYNCHRONIZATION.value
            or subsystem_id.startswith("camera.")
            and any(
                subsystem_id == SubsystemId.camera(camera.name)
                for camera in self._reach_cameras
            )
            for subsystem_id in failed
        )
        if reach_failed:
            if self._retry_reach_camera_domains():
                retried.append("reach cameras")
        if SubsystemId.CAN_PELLET.value in failed:
            if self._start_can_domain(wait_connected=True):
                retried.append("CAN/pellet")
        if SubsystemId.NIDAQ_STREAM.value in failed:
            if self._start_nidaq_domain():
                retried.append("NI-DAQ")
        if SubsystemId.LASER.value in failed:
            if self._start_laser_domain():
                retried.append("laser")
        if SubsystemId.SESSION_LOGS.value in failed:
            if self._validate_session_logs_domain():
                retried.append("session output")

        synchronization = self._subsystem_status_registry.get(
            SubsystemId.REACH_SYNCHRONIZATION
        )
        inference = self._subsystem_status_registry.get(SubsystemId.LIVE_INFERENCE)
        if (
            self._inference.is_enabled
            and synchronization is not None
            and synchronization.is_ready
            and (inference is None or not inference.is_ready)
            and self._start_inference_domain(
                reach_synchronization_ready=True,
                preflight_error=None,
            )
        ):
            retried.append("live inference")
        remaining = self._subsystem_status_registry.recording_blockers()
        if remaining:
            return (
                f"Retried {', '.join(retried) if retried else 'failed subsystems'}; "
                f"Record remains blocked: {'; '.join(remaining)}"
            )
        return (
            f"Ready after retry: {', '.join(retried)}"
            if retried
            else "All required subsystems are already ready"
        )

    def _retry_reach_camera_domains(self) -> bool:
        cameras = self._ordered_reach_cameras(enabled_only=True)
        if not cameras:
            return False
        primary = cameras[0]
        primary_status = self._subsystem_status_registry.get(
            SubsystemId.camera(primary.name)
        )
        inference_indices = {
            camera: index
            for index, camera in enumerate(self._inference_cameras)
        }
        if primary_status is None or not primary_status.is_ready:
            # Secondaries depend on the primary trigger, so this is the one retry
            # case where the complete reach-camera synchronization domain restarts.
            for camera in cameras:
                try:
                    camera.on_capture_stop()
                except Exception:
                    logger.exception("Failed to reset reach camera %s", camera.name)
            return self._start_reach_camera_domains(inference_indices)

        generation = self._begin_subsystem_start(
            SubsystemId.REACH_SYNCHRONIZATION,
            reason="retrying unavailable secondary cameras",
        )
        for secondary in cameras[1:]:
            status = self._subsystem_status_registry.get(
                SubsystemId.camera(secondary.name)
            )
            if status is not None and status.is_ready:
                continue
            inference_index = inference_indices.get(secondary)
            if not self._prepare_camera_domain(
                secondary,
                self._inference_queue if inference_index is not None else None,
                inference_index,
            ):
                continue
            camera_generation = self._subsystem_status_registry.get(
                SubsystemId.camera(secondary.name)
            ).generation
            if self._enable_camera_capture_domain(
                secondary,
                camera_generation,
                wait_for_first_frame=False,
            ):
                self._wait_camera_first_frame_domain(
                    secondary,
                    camera_generation,
                )
        all_ready = all(
            (
                status := self._subsystem_status_registry.get(
                    SubsystemId.camera(camera.name)
                )
            ) is not None
            and status.is_ready
            for camera in cameras
        )
        self._set_subsystem_status(
            SubsystemId.REACH_SYNCHRONIZATION,
            SubsystemState.READY if all_ready else SubsystemState.FAILED,
            reason=(
                "all enabled reach cameras synchronized"
                if all_ready
                else "one or more enabled reach cameras unavailable"
            ),
            error="" if all_ready else "reach synchronization incomplete",
            generation=generation,
        )
        return all_ready

    def capture_start(
        self,
        *,
        target_status: AppModelStatus = AppModelStatus.RUNNING,
        wait_connected: bool = True,
    ) -> bool:
        """Request to start the acquisition"""
        hardware_init_started = time.perf_counter()
        log_hardware_initialization(
            logger,
            "START | acquisition hardware | target_status=%s wait_connected=%s",
            target_status.value,
            wait_connected,
        )
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

        inference_preflight_error = None
        if self._inference.is_enabled:
            runtime_check = getattr(self._inference, "check_live_inference_runtime", None)
            if callable(runtime_check):
                gpu_status = runtime_check()
                if not gpu_status.is_available:
                    inference_preflight_error = (
                        "Live inference cannot start because a compatible TensorFlow GPU runtime was not found. "
                        "Cameras and independent hardware will continue without live inference. "
                        "Disable Live inference in Preferences or launch with --no-live-inference "
                        "to suppress this failure.\n\n"
                        f"{gpu_status.error}"
                    )
                    logger.error(inference_preflight_error)
                    self._set_subsystem_status(
                        SubsystemId.LIVE_INFERENCE,
                        SubsystemState.FAILED,
                        error=inference_preflight_error,
                    )

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

        self._ensure_reach_primary_camera()
        self._inference_queue = None
        self._inference_cameras = ()
        inference_camera_indices = {}

        if self._inference.is_enabled and inference_preflight_error is None:
            inference_cameras = self._ordered_reach_cameras(enabled_only=True)
            inference_error = None
            if len(inference_cameras) < 2:
                inference_error = (
                    "Live inference requires at least two enabled reach cameras; "
                    f"found {len(inference_cameras)}."
                )
            else:
                shape = inference_cameras[0].shape
                mismatched_cameras = [
                    camera.name
                    for camera in inference_cameras
                    if camera.shape != shape
                ]
                if len(mismatched_cameras) > 0:
                    inference_error = (
                        "Live inference requires all enabled reach cameras to use the same frame size; "
                        f"mismatched cameras: {', '.join(mismatched_cameras)}."
                    )
            if inference_error is None:
                self._inference_queue = FixedArrayMultiQueue(
                    # live queue does not need/require a lot of "depth" == total nbr of batches that can sit
                    # in the ring-buffer-queue at the same time.
                    # Now only using a "depth" of 1 frame batches,
                    # this should makes less delay / be more reactive in live inference results,
                    1,
                    len(inference_cameras),
                    3,
                    shape=shape,
                    primary=0,
                    name="inference_q",
                    mp_ctx=get_mp_ctx(),
                )
                inference_camera_indices = {
                    camera: camera_index
                    for camera_index, camera in enumerate(inference_cameras)
                }
                self._inference_cameras = inference_cameras
            else:
                logger.error(inference_error)
                self.on_error("Inference configuration error", inference_error)
                self._set_subsystem_status(
                    SubsystemId.LIVE_INFERENCE,
                    SubsystemState.FAILED,
                    error=inference_error,
                )
                inference_preflight_error = inference_error
        else:
            self._inference_queue = None

        reach_synchronization_ready = self._start_reach_camera_domains(
            inference_camera_indices,
        )
        # Each hardware domain starts and fails independently. Readiness is
        # aggregated only when Record is requested.
        can_ready = self._start_can_domain(wait_connected=wait_connected)
        self._start_nidaq_domain()
        self._start_laser_domain()
        self._validate_session_logs_domain()

        watchdog_mon_register = self._analysis.watchdog_monitor.register_watchdog
        if can_ready:
            watchdog_mon_register(
                WatchdogItems.DEVICE_READER,
                lambda: self._hardware.watchdog_reader_perf_c,
            )
            watchdog_mon_register(
                WatchdogItems.DEVICE_WRITER,
                lambda: self._hardware.watchdog_writer_perf_c,
            )

        for cam in self._cameras:
            status = self._subsystem_status_registry.get(
                SubsystemId.camera(cam.name)
            )
            if status is not None and status.is_ready:
                watchdog_mon_register(f"camera.{cam.name}", lambda cam=cam: cam.watchdog_capture_perf_c)

        if __debug__ and os.getenv("_AUTOTRAINER_TEST_WATCHDOG") == "1":
            def fake_watchdog(t_end):
                return t_end
            watchdog_mon_register("test-watchdog", lambda t=time.perf_counter() + 180: fake_watchdog(t))

        if can_ready:
            # We always begin at home when the pellet controller is available.
            log_hardware_initialization(logger, "START | pellet home command")
            try:
                self._behavior.system_machine.pellet.move_home(force=True)
            except Exception as exc:
                error = str(exc) or exc.__class__.__name__
                logger.exception("Pellet home command failed")
                self._set_subsystem_status(
                    SubsystemId.CAN_PELLET,
                    SubsystemState.FAILED,
                    error=error,
                )
                try:
                    self._hardware.safety_shutdown(
                        f"pellet home failure: {error}",
                        wait=True,
                    )
                except Exception:
                    logger.exception("CAN safety shutdown failed after pellet home error")
            else:
                log_hardware_initialization(logger, "QUEUED | pellet home command")

        # once cameras successfully started:
        self._save_project_metadata(project_info, when=datetime.now(), session=None, caller="capture_start")
        #
        # Start inference only after the required reach camera topology is ready.
        if self._start_inference_domain(
            reach_synchronization_ready=reach_synchronization_ready,
            preflight_error=inference_preflight_error,
        ):
            watchdog_mon_register(WatchdogItems.POSE_DATA_MONITOR_PROC,
                                  lambda: self._inference.watchdog_monitor_data_proc_perf_c)

        if not algo.algo_paused:
            analysis.restart()

        animal = self._selected_animal
        plan = (
            None if (animal is None or animal.training.current_protocol is None)
            else self.get_training_plan_by_id(animal.training.current_protocol)
        )
        self.training_plan = plan if animal is not None else None

        if animal is not None:
            self._set_animal_base_positions(animal)

        self._acquisition_started = True
        self._acquisition_starting = False
        self.status = target_status
        self.property_changed(self.Props.ACQUISITION_RUNNING, True, False)
        self._event_manager.post_event_content(
            ApiEventKind.applicationModeChanged,
            dict(mode=app_status_to_api_app_mode(target_status))
        )

        self._log_acquisition_startup_summary()
        log_hardware_initialization(
            logger,
            "READY | acquisition hardware | status=%s elapsed=%.3fs",
            target_status.value,
            time.perf_counter() - hardware_init_started,
        )

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
            recording_status = self._session_recording_status
        if recording_status in {
            SessionRecordingStatus.ARMING,
            SessionRecordingStatus.RECORDING,
        }:
            logger.warning("Acquisition stop requested during recording; aborting the session")
            self.abort_recording()
        # always remove status-file on stop:
        status_file_path = self.status_file_path.expanduser()
        status_file_path.unlink(missing_ok=True)
        try:
            self._capture_stop()
        finally:
            # always:
            self._nidaq_signal_monitor.stop()
            # must be set before try reload training plans, given checked in it
            self._acquisition_started = False
            self._acquisition_stopping = False
            self._acquisition_starting = False
            if self._session_recording_status == SessionRecordingStatus.ABORTING:
                self._finish_abort_recording()
            elif self._session_recording_status != SessionRecordingStatus.READY:
                self._session_data_recorder.abort()
                self._set_session_recording_status(SessionRecordingStatus.READY)
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
        self._detach_training_plan()  # always

        watchdog_mon_unregister = self._analysis.watchdog_monitor.unregister_watchdog
        for item in WatchdogItems:
            watchdog_mon_unregister(item)

        can_status = self._subsystem_status_registry.get(SubsystemId.CAN_PELLET)
        if can_status is not None and can_status.state is SubsystemState.READY:
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.STOPPING,
                reason="stopping CAN/pellet controller",
            )
            try:
                tok = self._hardware.set_color_led(0, 0, 0)
                self._hardware.wait_pending_command_acked(
                    tok,
                    timeout=2,
                    raise_on_timeout=False,
                )
            except Exception:
                logger.exception("Failed to turn off pellet controller LED")

        self._set_subsystem_status(
            SubsystemId.LIVE_INFERENCE,
            SubsystemState.STOPPING,
            reason="stopping live inference",
        )
        try:
            self._inference.stop()
        except Exception:
            logger.exception("Failed to stop live inference")
        self._set_subsystem_status(
            SubsystemId.LIVE_INFERENCE,
            (
                SubsystemState.DISABLED
                if not self._inference.is_enabled
                else SubsystemState.STOPPED
            ),
            reason="live inference stopped",
        )

        self._set_subsystem_status(
            SubsystemId.LASER,
            SubsystemState.STOPPING,
            reason="closing laser controller",
        )
        try:
            self._laser.close()
        except Exception as err:
            logger.exception("Failed to close laser controller during capture stop: %s", err)
        self._set_subsystem_status(
            SubsystemId.LASER,
            (
                SubsystemState.DISABLED
                if self._laser.configuration.backend == "disabled"
                else SubsystemState.STOPPED
            ),
            reason="laser controller stopped",
        )

        self._set_subsystem_status(
            SubsystemId.NIDAQ_STREAM,
            SubsystemState.STOPPING,
            reason="stopping NI-DAQ stream",
        )
        try:
            self._nidaq_signal_monitor.stop()
        except Exception:
            logger.exception("Failed to stop NI-DAQ stream")
        self._set_subsystem_status(
            SubsystemId.NIDAQ_STREAM,
            (
                SubsystemState.DISABLED
                if not (
                    self._nidaq_signal_monitor.hardware_enabled
                    and self._nidaq_signal_monitor.configuration.is_enabled
                )
                else SubsystemState.STOPPED
            ),
            reason="NI-DAQ stream stopped",
        )

        try:
            self._hardware.disconnect()
        except Exception:
            logger.exception("Failed to disconnect CAN/pellet controller")
        self._set_subsystem_status(
            SubsystemId.CAN_PELLET,
            (
                SubsystemState.DISABLED
                if not self._hardware.requires_connection
                else SubsystemState.STOPPED
            ),
            reason="CAN/pellet controller stopped",
        )

        for camera in self._cameras:
            watchdog_mon_unregister(f"camera.{camera.name}")
            if camera.is_enabled and not camera.is_primary:
                logger.verbose("notifying end to %s", camera.name)
                try:
                    camera.on_capture_notify_end()
                except Exception:
                    logger.exception("Failed to notify camera %s to end", camera.name)

        time.sleep(0.01)

        for camera in self._cameras:
            if camera.is_enabled and camera.is_primary:
                logger.verbose("notifying end to %s", camera.name)
                try:
                    camera.on_capture_notify_end()
                except Exception:
                    logger.exception("Failed to notify camera %s to end", camera.name)

        for camera in self._cameras:
            if camera.is_enabled and not camera.is_primary:
                logger.verbose("stopping capture to %s", camera.name)
                try:
                    camera.on_capture_stop()
                except Exception:
                    logger.exception("Failed to stop camera %s", camera.name)

        for camera in self._cameras:
            if camera.is_enabled and camera.is_primary:
                logger.verbose("stopping capture to %s", camera.name)
                try:
                    camera.on_capture_stop()
                except Exception:
                    logger.exception("Failed to stop camera %s", camera.name)

        for camera in self._cameras:
            self._set_subsystem_status(
                SubsystemId.camera(camera.name),
                (
                    SubsystemState.STOPPED
                    if camera.is_enabled
                    else SubsystemState.DISABLED
                ),
                reason="camera stopped" if camera.is_enabled else "camera disabled",
            )
        self._set_subsystem_status(
            SubsystemId.REACH_SYNCHRONIZATION,
            (
                SubsystemState.STOPPED
                if any(camera.is_enabled for camera in self._reach_cameras)
                else SubsystemState.DISABLED
            ),
            reason="reach camera synchronization stopped",
        )

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

    def load_configuration(self, location: Optional[Path] = None, *, random_cameras: bool = False):
        if location is None:
            location = self.get_config_location()

        config_started = time.perf_counter()
        log_hardware_initialization(
            logger,
            "START | hardware configuration | path=%s random_cameras=%s",
            location,
            random_cameras,
        )
        configuration: SystemConfiguration = self.get_config_from_location(location)
        if random_cameras:
            logger.notice("Using random camera override for this run")
            self._apply_random_camera_override(configuration)
        self._ensure_optional_stim_camera(configuration)

        self._sync_reach_cameras_to_configuration(configuration)

        frame_rate = None
        reach_camera_configs = []
        for camera in self._reach_cameras:
            camera_config = configuration.get_camera(camera.camera_id)
            if camera_config is None:
                continue
            reach_camera_configs.append((camera, camera_config))
            if frame_rate is None:
                frame_rate = camera_config.params.get("fps")

        pose_algo = self._pose_algorithm
        pose_algo.frame_rate = frame_rate
        # also reset it to inference:
        self._inference.pose_algorithm = pose_algo
        # which force a sync to pose-result process.
        self._behavior.system_machine.intersession.frame_rate = frame_rate

        for camera, camera_config in reach_camera_configs:
            if camera_config.record_mode != VideoRecordMode.TRIGGER:
                logger.warning(
                    "Ignoring continuous camera mode for manual recording: %s=%s",
                    camera.name,
                    camera_config.record_mode,
                )
            camera_config.record_mode = VideoRecordMode.TRIGGER
            if camera_config.record_prebuffer_duration != 0:
                logger.warning(
                    "Ignoring camera prebuffer for manual recording alignment: %s=%s",
                    camera.name,
                    camera_config.record_prebuffer_duration,
                )
            camera_config.record_prebuffer_duration = 0

        for camera, camera_config in reach_camera_configs:
            camera.load_configuration(camera_config)
            log_hardware_initialization(
                logger,
                "CONFIGURED | camera | name=%s enabled=%s primary=%s source=%s shape=%s fps=%s",
                camera.name,
                camera_config.is_enabled,
                camera_config.params.get("primary", "no"),
                camera.camera_source.url,
                camera.shape,
                camera_config.params.get("fps"),
            )
        self._ensure_reach_primary_camera()

        self._behavior.algorithm.record_prebuffer_duration = 0

        self._hardware.load_config(configuration.hardware)
        log_hardware_initialization(
            logger,
            "CONFIGURED | hardware flags | CAN=%s pellet_controller=%s NI-DAQ=%s",
            configuration.hardware.can_enabled,
            configuration.hardware.pellet_controller_enabled,
            configuration.hardware.nidaq_enabled,
        )
        self.inference.load_configuration(configuration.inference)
        log_hardware_initialization(
            logger,
            "CONFIGURED | live inference | enabled=%s model=%s",
            configuration.inference.is_enabled,
            configuration.inference.pose_model_location,
        )
        # Configuration loading must remain side-effect free. The laser controller
        # is opened as its own acquisition domain in capture_start().
        self.laser.set_configuration_offline(configuration.laser)
        self._nidaq_ports = configuration.nidaq_ports
        self.nidaq_signal_monitor.set_hardware_enabled(
            configuration.hardware.nidaq_enabled,
            auto_start=False,
        )
        hardware_timed_output_devices = tuple(
            dict.fromkeys(
                device_name
                for channel in configuration.laser.channels
                for device_name in (device_name_from_channel(channel.analog_output),)
                if configuration.laser.hardware_timed and device_name is not None
            )
        )
        self.nidaq_signal_monitor.configure_timing(
            configuration.nidaq_ports.timing,
            hardware_timed_output_devices=hardware_timed_output_devices,
            device_identities=configuration.nidaq_ports.device_identities,
        )
        nidaq_acquisition = build_nidaq_acquisition_configuration(
            configuration.nidaq_stream,
            configuration.nidaq_ports,
            configuration.laser,
        )
        configuration.nidaq_stream = nidaq_acquisition
        self.nidaq_signal_monitor.load_configuration(nidaq_acquisition)
        self.behavior.load_configuration(configuration.behavior)

        self._analysis.watchdog_monitor.config = configuration.watchdog

        self._loaded_configuration = configuration
        self._loaded_config_dir_path = location.parent.resolve()
        self._loaded_configuration_has_runtime_override = random_cameras
        self._runtime_live_inference_override = None

        # only at the end:
        self.output_location = configuration.persistence.output_location
        self._configure_subsystem_intent(configuration)

        self.reload_training_plans(reraise_on_error=True)

        # and:
        self._load_animals()

        self.configuration_loaded_event(configuration)

        log_hardware_initialization(
            logger,
            "READY | hardware configuration | elapsed=%.3fs",
            time.perf_counter() - config_started,
        )

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
        if self._loaded_configuration_has_runtime_override:
            logger.info("Skipping configuration save because this run used runtime camera source overrides")
            return
        loc = self._preferences.configuration_location
        logger.info("Saving configuration to %s", loc)
        conf = self._create_configuration()
        conf.save_default(loc)

    def set_runtime_live_inference_override(self, enabled: bool) -> None:
        """Override configured live inference without persisting the CLI choice."""
        if self._loaded_configuration is None:
            raise RuntimeError("Cannot override live inference before configuration is loaded")
        enabled = bool(enabled)
        self._inference.is_enabled = enabled
        self._runtime_live_inference_override = enabled
        logger.notice("Runtime live inference override: enabled=%s", enabled)

    def update_daq_port_configuration(
        self,
        nidaq_ports: NidaqPortConfiguration,
        laser_configuration: LaserSystemConfiguration,
    ) -> None:
        if self._loaded_configuration is None:
            raise RuntimeError("Cannot update DAQ port configuration before a system configuration is loaded")
        self._nidaq_ports = nidaq_ports
        self._laser.set_configuration_offline(laser_configuration)
        self._loaded_configuration.nidaq_ports = nidaq_ports
        self._loaded_configuration.laser = laser_configuration
        acquisition = build_nidaq_acquisition_configuration(
            self._nidaq_signal_monitor.configuration,
            nidaq_ports,
            laser_configuration,
        )
        self._nidaq_signal_monitor.load_configuration(acquisition)
        hardware_timed_output_devices = tuple(
            dict.fromkeys(
                device_name
                for channel in laser_configuration.channels
                for device_name in (device_name_from_channel(channel.analog_output),)
                if laser_configuration.hardware_timed and device_name is not None
            )
        )
        self._nidaq_signal_monitor.configure_timing(
            nidaq_ports.timing,
            hardware_timed_output_devices=hardware_timed_output_devices,
            device_identities=nidaq_ports.device_identities,
        )
        self._loaded_configuration.nidaq_stream = acquisition
        self.configuration_loaded_event(self._loaded_configuration)
        self.save_configuration()

    def update_nidaq_signal_stream_channels(self, channels) -> None:
        """Apply and persist plot selection without changing DAQ acquisition."""
        if self._loaded_configuration is None:
            raise RuntimeError("Cannot update NI-DAQ stream channels before a system configuration is loaded")
        channel_names = tuple(
            channel.name if hasattr(channel, "name") else str(channel)
            for channel in channels
        )
        self._nidaq_signal_monitor.set_display_channels(channel_names)
        self._loaded_configuration.nidaq_stream = self._nidaq_signal_monitor.save_configuration()
        self.save_configuration()

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

    def _stop_periodic_timers(self):
        self._closing_event.set()
        for timer in (
                self._timer_one_minute_repeat,
                self._timer_daily,
        ):
            logger.debug("stopping timer %s", timer)
            timer.cancel()

    def _request_safety_shutdown(self, reason: str, *, wait: bool) -> None:
        self._stop_periodic_timers()
        self._hardware.safety_shutdown(reason, wait=wait)

    def on_close(self):
        logger.debug("AppModel.on_close")
        # Stop command producers first so none can race with CAN teardown or
        # reschedule themselves after their current timer is cancelled.
        self._request_safety_shutdown("application close", wait=True)

        self._session_data_recorder.close()
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
        unregister_fatal_exception_callback(self._on_fatal_exception)

    def _on_fatal_exception(self, source: str, exception: BaseException) -> None:
        self._request_safety_shutdown(
            f"{source}: {exception}",
            wait=False,
        )

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
            parts.append(f"Post-session analysis: {cur_inter_state}")
        text_overlay = None if len(parts) == 0 else "\n".join(parts)
        for camera in self._reach_cameras:
            camera.text_overlay = text_overlay

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

    def _on_preferences_property_changed(self, name: str, value, old_value):
        prefs = UserPreferences
        if name == prefs.SELECTED_ANIMAL:
            for animal in self._animals:
                if animal.name == value:
                    self.selected_animal = animal
                    break

    def _update_led_color(self):
        color = self._behavior.get_led_color()
        cur_led = self._hardware.color_led
        if cur_led is None or color != (cur_led.red, cur_led.green, cur_led.blue):
            self._hardware.set_color_led(*color)

    def _on_watchdog_property_changed(self, name, value, old_value):
        wd_mon = self._analysis.watchdog_monitor
        if name == wd_mon.IS_ENGAGED:
            if value:
                for watchdog_id in wd_mon.engaged_watchdogs:
                    self._handle_watchdog_timeout(watchdog_id)

    def _handle_watchdog_timeout(self, watchdog_id) -> None:
        watchdog_key = (
            watchdog_id.value
            if isinstance(watchdog_id, enum.Enum)
            else str(watchdog_id)
        )
        error = f"watchdog timed out: {watchdog_key}"
        if watchdog_key.startswith("camera."):
            camera_name = watchdog_key.partition(".")[2]
            camera = next(
                (item for item in self._cameras if item.name == camera_name),
                None,
            )
            if camera is None:
                logger.error("Unknown camera watchdog timed out: %s", watchdog_key)
                return
            try:
                camera.on_capture_stop()
            except Exception:
                logger.exception("Failed to stop timed-out camera %s", camera_name)
            self._analysis.watchdog_monitor.unregister_watchdog(watchdog_key)
            root_error = camera.last_error or error
            self._set_subsystem_status(
                SubsystemId.camera(camera_name),
                SubsystemState.FAILED,
                error=root_error,
            )
            if camera in self._reach_cameras:
                if camera.is_primary:
                    for secondary in self._reach_cameras:
                        if not secondary.is_enabled or secondary is camera:
                            continue
                        try:
                            secondary.on_capture_stop()
                        except Exception:
                            logger.exception(
                                "Failed to suspend secondary camera %s",
                                secondary.name,
                            )
                        self._analysis.watchdog_monitor.unregister_watchdog(
                            f"camera.{secondary.name}"
                        )
                        self._set_subsystem_status(
                            SubsystemId.camera(secondary.name),
                            SubsystemState.BLOCKED,
                            reason="primary trigger unavailable",
                        )
                self._set_subsystem_status(
                    SubsystemId.REACH_SYNCHRONIZATION,
                    SubsystemState.FAILED,
                    error=(
                        f"reach camera unavailable: {camera_name}; {root_error}"
                    ),
                )
                try:
                    self._inference.stop()
                except Exception:
                    logger.exception("Failed to stop inference after camera timeout")
                self._set_subsystem_status(
                    SubsystemId.LIVE_INFERENCE,
                    SubsystemState.BLOCKED,
                    reason=f"reach camera unavailable: {camera_name}",
                )
            else:
                logger.error("Unknown camera watchdog failed: %s", camera_name)
            self._abort_recording_for_required_subsystem(
                SubsystemId.camera(camera_name),
                root_error,
            )
        elif watchdog_key in {
            WatchdogItems.DEVICE_READER.value,
            WatchdogItems.DEVICE_WRITER.value,
        }:
            try:
                self._hardware.safety_shutdown(error, wait=False)
            except Exception:
                logger.exception("CAN safety shutdown failed after watchdog timeout")
            self._analysis.watchdog_monitor.unregister_watchdog(watchdog_id)
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.FAILED,
                error=error,
            )
            self._abort_recording_for_required_subsystem(
                SubsystemId.CAN_PELLET,
                error,
            )
        elif watchdog_key in {
            WatchdogItems.POSE_PROCESS.value,
            WatchdogItems.POSE_DATA_MONITOR_PROC.value,
        }:
            try:
                self._inference.stop()
            except Exception:
                logger.exception("Failed to stop inference after watchdog timeout")
            self._analysis.watchdog_monitor.unregister_watchdog(watchdog_id)
            self._set_subsystem_status(
                SubsystemId.LIVE_INFERENCE,
                SubsystemState.FAILED,
                error=error,
            )
            self._abort_recording_for_required_subsystem(
                SubsystemId.LIVE_INFERENCE,
                error,
            )
        else:
            logger.error("Unowned watchdog timed out: %s", watchdog_key)
            return
        self.on_error(
            "Hardware subsystem failure",
            f"{error}. Unrelated hardware remains running; Record is blocked "
            "until the required subsystem is ready.",
        )

    def _on_nidaq_monitor_property_changed(self, name, value, _old_value):
        monitor = self._nidaq_signal_monitor
        if name == monitor.IS_STARTING and value:
            self._begin_subsystem_start(SubsystemId.NIDAQ_STREAM)
        elif name == monitor.IS_RUNNING:
            if value:
                self._set_subsystem_status(
                    SubsystemId.NIDAQ_STREAM,
                    SubsystemState.READY,
                    reason="NI-DAQ tasks running",
                )
            elif (
                monitor.configuration.is_enabled
                and monitor.hardware_enabled
                and not monitor.error_message
            ):
                self._set_subsystem_status(
                    SubsystemId.NIDAQ_STREAM,
                    SubsystemState.STOPPED,
                    reason="NI-DAQ stream stopped",
                )
        elif name == monitor.ERROR_MESSAGE and value:
            self._set_subsystem_status(
                SubsystemId.NIDAQ_STREAM,
                SubsystemState.FAILED,
                error=str(value),
            )
            self._abort_recording_for_required_subsystem(
                SubsystemId.NIDAQ_STREAM,
                str(value),
            )

    def _on_laser_runtime_property_changed(self, name, value, _old_value):
        if name != self._laser.IS_CONNECTED:
            return
        if value:
            self._set_subsystem_status(
                SubsystemId.LASER,
                SubsystemState.READY,
                reason="laser controller connected",
            )
        elif self._laser.configuration.backend != "disabled":
            current = self._subsystem_status_registry.get(SubsystemId.LASER)
            if (
                self._acquisition_started
                and not self._acquisition_stopping
                and current is not None
                and current.state is SubsystemState.READY
            ):
                reason = "laser controller disconnected unexpectedly"
                self._set_subsystem_status(
                    SubsystemId.LASER,
                    SubsystemState.FAILED,
                    error=reason,
                )
                self._abort_recording_for_required_subsystem(
                    SubsystemId.LASER,
                    reason,
                )
            else:
                self._set_subsystem_status(
                    SubsystemId.LASER,
                    SubsystemState.STOPPED,
                    reason="laser controller disconnected",
                )

    def _abort_recording_for_required_subsystem(
        self,
        subsystem_id,
        reason: str,
    ) -> None:
        status = self._subsystem_status_registry.get(subsystem_id)
        if (
            status is not None
            and status.required_for_recording
            and self._session_recording_status in {
                SessionRecordingStatus.ARMING,
                SessionRecordingStatus.RECORDING,
            }
        ):
            logger.error(
                "Required subsystem %s failed during recording: %s; aborting session",
                status.subsystem_id,
                reason,
            )
            self.abort_recording()

    def _on_intersession_property_changed(self, name, value, _):
        if name == IntersessionMachine.Properties.STATE_PROPERTY:
            self._update_status_text_overlay()

    def _on_session_starting_before_record_start(self):
        session_config = self._behavior.algorithm.active_config.session_control
        self._trial_ledger = PelletTrialLedger(
            self._project_info.short_id,
            TrialAccountingConfiguration(
                assignment_policy=AttemptAssignmentPolicy(
                    session_config.attempt_assignment
                ),
                retry_settings_policy=RetrySettingsPolicy(
                    session_config.retry_settings
                ),
                count_basis=TrialCountBasis(
                    session_config.trial_count_basis
                ),
                counted_outcomes=frozenset(
                    TrialOutcome(outcome)
                    for outcome in session_config.counted_trial_outcomes
                ),
            ),
        )
        self._cancel_automatic_stop_timers()
        self._session_stop_policy = SessionStopPolicy(
            SessionStopConfiguration(
                duration_seconds=session_config.duration_limit_seconds,
                trial_limit=session_config.trial_limit,
                stop_on_protocol_complete=(
                    session_config.stop_on_protocol_complete
                ),
                drain_timeout_seconds=(
                    session_config.stop_drain_timeout_seconds
                ),
            ),
        )
        self._session_stop_evaluation = None
        self._recording_ending_reason = RecordingEndingReason.NA
        self._cams_record_start_perf.value = math.nan
        drained = 0
        while self._record_stop_sema.acquire(block=False):
            drained += 1
        if drained:
            logger.verbose("drained record_stop_sema by %s", drained)
        prepare_pose = getattr(
            type(self._inference),
            "prepare_session_recording",
            None,
        )
        if prepare_pose is not None:
            prepare_pose(self._inference, self._project_info)

    def _on_session_capture_ended(self, reason: RecordingEndingReason):
        self._recording_ending_reason = RecordingEndingReason(reason)
        logger.debug("session capture trigger ended: %s", reason)

    def _on_session_ending(self, project: ProjectInfo, result: CaptureAnalysisResult):
        if project.short_id in self._aborted_session_ids:
            logger.info("ignoring session-ending callback for aborted %s", project.short_id)
            return
        self._session_analysis_finished = True
        if self._session_recording_status == SessionRecordingStatus.ANALYZING:
            if self._session_analysis_started_perf is not None:
                self._session_analysis_duration_seconds = (
                    time.perf_counter() - self._session_analysis_started_perf
                )
                logger.info(
                    "Session analysis finished in %.3f seconds: %s",
                    self._session_analysis_duration_seconds,
                    project.short_id,
                )
            try:
                self._save_project_metadata(
                    project,
                    caller="session_analysis_ended",
                )
            except Exception as exc:
                self._set_subsystem_status(
                    SubsystemId.OFFLINE_ANALYSIS,
                    SubsystemState.FAILED,
                    error=f"final metadata save failed: {exc}",
                )
                raise
            else:
                self._set_subsystem_status(
                    SubsystemId.OFFLINE_ANALYSIS,
                    (
                        SubsystemState.READY
                        if self._session_data_complete
                        else SubsystemState.FAILED
                    ),
                    reason=(
                        "offline analysis completed"
                        if self._session_data_complete
                        else "analysis completed; session data is incomplete"
                    ),
                    error=(
                        ""
                        if self._session_data_complete
                        else "; ".join(self._session_data_errors)
                    ),
                )
            finally:
                self._set_session_recording_status(
                    SessionRecordingStatus.READY
                )

    def _complete_stopped_recording(self, project: ProjectInfo) -> None:
        end_perf = self._pending_session_end_perf
        self._pending_session_end_perf = None
        if end_perf is None:
            end_perf = get_perf_now()
            logger.warning("Primary camera did not report a final recorded-frame timestamp")
        boundary = self._session_boundary
        if boundary is None or boundary.session_id != project.short_id:
            raise RuntimeError(
                f"Missing canonical recording boundary for {project.short_id}"
            )
        self._session_boundary = boundary.with_end(end_perf)
        project.start_record_timestamp = self._session_boundary.start_wall_time
        try:
            trial_ledger = self._trial_ledger
            if trial_ledger is not None:
                if trial_ledger.active_attempt is not None:
                    trial_ledger.close_active_for_analysis(
                        end_perf,
                        self._session_boundary.end_wall_time,
                    )
                self._session_data_recorder.set_trial_ledger(
                    trial_ledger.to_records(),
                    trial_ledger.summary(),
                )
            stream_result = self._session_data_recorder.stop(end_perf)
            camera_alignment = (
                None
                if not isinstance(stream_result, dict)
                else stream_result.get("cameraNidaqAlignment")
            )
            matched_sample_index = (
                None
                if camera_alignment is None
                else camera_alignment.get("matchedSampleIndex")
            )
            if matched_sample_index is not None:
                self._session_boundary = (
                    self._session_boundary.with_nidaq_sample_index(
                        matched_sample_index
                    )
                )
            if isinstance(stream_result, dict):
                self._session_data_complete = bool(
                    stream_result.get("sessionComplete", True)
                )
                self._session_data_errors = tuple(
                    stream_result.get("incompleteReasons", ())
                )
                self._session_enabled_sources = tuple(
                    stream_result.get("enabledSources", ())
                )
                if not self._session_data_complete:
                    message = (
                        "Session auxiliary data is incomplete: "
                        + "; ".join(self._session_data_errors)
                    )
                    logger.error(message)
                    self.on_error("Session data incomplete", message)
        except Exception as err:
            logger.exception("Failed to save session auxiliary streams: %s", err)
            self.on_error("Session stream save failed", str(err))
            self._session_data_complete = False
            self._session_data_errors = (
                f"auxiliary stream save failed: {err}",
            )
        if self._session_analysis_finished:
            if self._session_analysis_duration_seconds is None:
                self._session_analysis_duration_seconds = 0.0
            self._set_subsystem_status(
                SubsystemId.OFFLINE_ANALYSIS,
                (
                    SubsystemState.READY
                    if self._session_data_complete
                    else SubsystemState.FAILED
                ),
                reason=(
                    "offline analysis completed"
                    if self._session_data_complete
                    else "analysis completed; session data is incomplete"
                ),
                error=(
                    ""
                    if self._session_data_complete
                    else "; ".join(self._session_data_errors)
                ),
            )
        else:
            self._session_analysis_started_perf = time.perf_counter()
            self._begin_subsystem_start(
                SubsystemId.OFFLINE_ANALYSIS,
                reason="analyzing stopped session",
            )
            self._set_session_recording_status(SessionRecordingStatus.ANALYZING)
        try:
            self._save_project_metadata(
                project,
                caller="raw_writers_closed",
            )
        except Exception as exc:
            self._session_data_complete = False
            self._session_data_errors = (
                *self._session_data_errors,
                f"metadata save failed: {exc}",
            )
            self._set_subsystem_status(
                SubsystemId.OFFLINE_ANALYSIS,
                SubsystemState.FAILED,
                error=f"metadata save failed: {exc}",
            )
            raise
        finally:
            if self._session_analysis_finished:
                self._set_session_recording_status(
                    SessionRecordingStatus.READY
                )

    def _finish_abort_recording(self) -> None:
        self._record_start_timer.cancel()
        self._record_start_timer = no_op_timer
        self._abort_cleanup_timer.cancel()
        self._abort_cleanup_timer = no_op_timer
        project = self._aborting_project
        if project is None:
            logger.error("abort completed without an associated project")
            self._set_session_recording_status(SessionRecordingStatus.READY)
            return
        try:
            day_path = Path(project.get_day_path(skip_ensure=True)[0]).resolve()
            session_path = Path(project.get_session_path(skip_ensure=True).location).resolve()
            try:
                session_path.relative_to(day_path)
            except ValueError as err:
                raise RuntimeError(
                    f"Resolved session path is outside the project day directory: {session_path}"
                ) from err
            if session_path.parent != day_path:
                raise RuntimeError(
                    f"Resolved session path is not a direct session directory: {session_path}"
                )
            if session_path.name != f"session{project.session:03}":
                raise RuntimeError(
                    f"Resolved session directory has an unexpected name: {session_path.name}"
                )
            if session_path.exists():
                shutil.rmtree(session_path)
                logger.notice("aborted session removed: %s", session_path)
            current_project = self._project_info
            if current_project is not None and current_project.session == project.session:
                current_project.session = max(0, project.session - 1)
        except Exception as err:
            logger.exception("Unable to delete aborted session data: %s", err)
            self.on_error("Abort cleanup failed", str(err))
        finally:
            self._behavior.algorithm.reset_session_counts()
            self._session_data_recorder.abort()
            self._pending_session_end_perf = None
            self._session_boundary = None
            self._session_data_complete = True
            self._session_data_errors = ()
            self._session_enabled_sources = ()
            self._trial_ledger = None
            self._session_analysis_finished = True
            self._session_analysis_started_perf = None
            self._session_analysis_duration_seconds = None
            self._set_subsystem_status(
                SubsystemId.OFFLINE_ANALYSIS,
                SubsystemState.DISABLED,
                reason="aborted session has no offline analysis",
            )
            self._abort_had_recording_started = False
            self._aborting_project = None
            self._set_session_recording_status(SessionRecordingStatus.READY)

    def _remove_timestamps_txt_files(
        self,
        project: ProjectInfo,
        cameras: Optional[Tuple[VideoCaptureModel, ...]] = None,
    ):
        removed = []
        for cam in self._cameras if cameras is None else cameras:
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

    def _on_hardware_property_changed(self, name: str, value, _):
        animal = self._selected_animal
        hard = self._hardware
        if animal is not None and name in {hard.SET_X, hard.SET_Y, hard.SET_Z}:
            # Protocol actions temporarily move the motors without redefining
            # the animal's manually configured base position.
            if self._attached_plan is not None:
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
            elif value == InferenceStatus.stopped:
                current = self._subsystem_status_registry.get(
                    SubsystemId.LIVE_INFERENCE
                )
                if (
                    self._acquisition_started
                    and not self._acquisition_stopping
                    and current is not None
                    and current.state is SubsystemState.READY
                ):
                    reason = "live inference stopped unexpectedly"
                    self._set_subsystem_status(
                        SubsystemId.LIVE_INFERENCE,
                        SubsystemState.FAILED,
                        error=reason,
                    )
                    self._abort_recording_for_required_subsystem(
                        SubsystemId.LIVE_INFERENCE,
                        reason,
                    )

            if new_is_live or value == InferenceStatus.intersession:
                self._analysis.watchdog_monitor.register_watchdog(
                    WatchdogItems.POSE_PROCESS, lambda: self._inference.watchdog_pose_process_perf_c)
            else:
                self._analysis.watchdog_monitor.unregister_watchdog(WatchdogItems.POSE_PROCESS)

            for camera in self._reach_cameras:
                camera.display_dots_detection = new_is_live
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
        min_check_delay = 5  # seconds before reporting an invalid/missing measurement
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
        if project.short_id in self._aborted_session_ids:
            logger.info("ignoring analysis result for aborted session %s", project.short_id)
            return
        # SystemMachine applies this result to the session-only behavior counts.
        # Final project metadata is written by _on_session_ending after all
        # analysis callbacks for this session have completed.

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
        dst = Path(self._preferences.animal_location).joinpath(f"{animal.name}.json")
        logger.verbose("Saving %s to %s ; sender=%s", animal, dst, sender)
        if backup_previous and dst.exists():
            now = datetime.now()
            dst.with_suffix(f'.{now.strftime(DATE_TIME_FORMAT)}.json.bak').write_bytes(dst.read_bytes())
        animal.to_file(dst)

    def _create_configuration(self) -> SystemConfiguration:
        loaded_hardware = (
            self._loaded_configuration.hardware
            if self._loaded_configuration is not None
            else HardwareConfiguration(pellet_identifier="CAN")
        )
        hardware_configuration = HardwareConfiguration(
            pellet_identifier=loaded_hardware.pellet_identifier,
            can_enabled=self._hardware.can_enabled,
            pellet_controller_enabled=self._hardware.pellet_controller_enabled,
            nidaq_enabled=self._hardware.nidaq_enabled,
            min_ack_timeout=loaded_hardware.min_ack_timeout,
            board_status_timeout=loaded_hardware.board_status_timeout,
        )

        cameras = []
        for camera in self._cameras:
            cameras.append(camera.save_configuration())

        inference_configuration = self._inference.save_configuration()
        if self._runtime_live_inference_override is not None and self._loaded_configuration is not None:
            inference_configuration.is_enabled = self._loaded_configuration.inference.is_enabled

        configuration = SystemConfiguration(cameras=cameras,
                                            hardware=hardware_configuration,
                                            inference=inference_configuration,
                                            laser=self._laser.save_configuration(),
                                            nidaq_ports=self._nidaq_ports,
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
        if caller in {"raw_writers_closed", "session_analysis_ended"}:
            boundary = self._session_boundary
            if boundary is None or boundary.session_id != project_info.short_id:
                raise RuntimeError(
                    f"Cannot finalize {project_info.short_id} without a canonical "
                    "recording boundary"
                )
            if not (
                math.isfinite(boundary.start_perf_time)
                and math.isfinite(boundary.start_wall_time)
            ):
                raise RuntimeError(
                    f"Cannot finalize {project_info.short_id} with a non-finite "
                    "recording boundary"
                )
            alignment_path = Path(
                project_info.get_session_path().location
            ) / "streams" / "alignment.json"
            if not alignment_path.is_file():
                raise RuntimeError(
                    f"Cannot finalize {project_info.short_id}: "
                    "streams/alignment.json is missing"
                )
            with alignment_path.open(encoding="utf-8") as stream:
                alignment = json.load(stream)
            alignment_boundary = alignment.get("canonicalBoundary", {})
            if (
                alignment_boundary.get("startPerfTime")
                != boundary.start_perf_time
                or alignment_boundary.get("startWallTime")
                != boundary.start_wall_time
            ):
                raise RuntimeError(
                    f"Cannot finalize {project_info.short_id}: canonical "
                    "boundary differs from streams/alignment.json"
                )
        self._save_metadata(project_info, when, file_name, session)

    def _save_metadata(self, project: ProjectInfo, when: datetime, file_name: str, session: Optional[int] = -1):
        when_as_utc = when.astimezone(timezone.utc)
        boundary = self._session_boundary
        if boundary is not None and boundary.session_id == project.short_id:
            start_record_timestamp = boundary.start_wall_time
            session_boundary = boundary.to_metadata()
        else:
            start_record_timestamp = project.start_record_timestamp
            session_boundary = None
        recording_duration = (
            None
            if boundary is None or boundary.end_perf_time is None
            else max(0.0, boundary.end_perf_time - boundary.start_perf_time)
        )
        info: Dict[str, Any] = {
            "date": when.strftime("%Y%m%d_%H%M%S"),
            "created": when.timestamp(),
            "createdUtc": when_as_utc.timestamp(),  # same than created
            "start_record_timestamp": start_record_timestamp,
            "sessionBoundary": session_boundary,
            "serialNumber": self._preferences.serial_number or "",
            "appVersion": self._app_version,
            "animalName": self.animal_name,
            "notes": self.notes or "",
            "cameraNames": list(project.camera_names),
            "session": session,
            "firstPelletDeliveryOffsetSeconds": project.first_pellet_delivery_offset,
            "firstPelletPresentationOffsetSeconds": project.first_pellet_presentation_offset,
            "sessionCounts": {
                "presented": self._behavior.algorithm.pellets_presented,
                "reaches": self._behavior.algorithm.pellet_reaches,
                "successfulReaches": self._behavior.algorithm.successful_reaches,
                "consumed": self._behavior.algorithm.pellets_consumed,
            },
            "recordingStatus": self._session_recording_status.value,
            "recordingStopReason": (
                None
                if self._recording_ending_reason is RecordingEndingReason.NA
                else self._recording_ending_reason.value
            ),
            "recordingDurationSeconds": recording_duration,
            "sessionDataComplete": self._session_data_complete,
            "sessionDataErrors": list(self._session_data_errors),
            "enabledSources": list(self._session_enabled_sources),
            "analysisDurationSeconds": self._session_analysis_duration_seconds,
            "trialSummary": (
                None
                if self._trial_ledger is None
                else self._trial_ledger.summary()
            ),
            "sessionStopPolicy": (
                None
                if self._session_stop_policy is None
                else dataclasses.asdict(
                    self._session_stop_policy.configuration
                )
            ),
            "sessionStopResult": (
                None
                if self._session_stop_evaluation is None
                else {
                    "decision": self._session_stop_evaluation.decision.value,
                    "reason": (
                        None
                        if self._session_stop_evaluation.reason is None
                        else self._session_stop_evaluation.reason.value
                    ),
                    "triggeredReasons": [
                        reason.value
                        for reason in (
                            self._session_stop_evaluation.triggered_reasons
                        )
                    ],
                    "requestedPerfTime": (
                        self._session_stop_evaluation.requested_perf_time
                    ),
                    "timeoutSeconds": (
                        self._session_stop_evaluation.timeout_seconds
                    ),
                }
            ),
            "hardwareConfigured": {
                "canEnabled": self._hardware.can_enabled,
                "pelletControllerEnabled": self._hardware.pellet_controller_enabled,
                "nidaqEnabled": self._hardware.nidaq_enabled,
                "laserBackend": self._laser.configuration.backend,
                "liveInferenceEnabled": self._inference.is_enabled,
                "cameras": {
                    camera.name: {
                        "previewEnabled": camera.is_enabled,
                        "recordEnabled": camera.is_recording_enabled,
                    }
                    for camera in self._cameras
                },
            },
            "hardwareRuntime": self._subsystem_status_registry.snapshot(),
            "hardwareRuntimeAtRecord": self._session_hardware_status_at_record,
            "configuration": None,
        }

        configuration = self._create_configuration()

        out = info.copy()
        out["configuration"] = asdict(configuration)
        out = _metadata_without_nonfinite_numbers(out)
        json_path = Path(file_name + ".json")
        yaml_path = Path(file_name + ".yaml")
        json_temp = json_path.with_name(json_path.name + ".tmp")
        yaml_temp = yaml_path.with_name(yaml_path.name + ".tmp")
        try:
            with json_temp.open("w", encoding="utf-8") as file:
                json.dump(
                    out,
                    file,
                    cls=SystemConfigurationJSONEncoder,
                    allow_nan=False,
                )

            with yaml_temp.open("w", encoding="utf-8") as file:
                yaml.dump(
                    out,
                    file,
                    Dumper=SystemConfigurationDumper,
                    sort_keys=False,
                )
            os.replace(json_temp, json_path)
            os.replace(yaml_temp, yaml_path)
        finally:
            json_temp.unlink(missing_ok=True)
            yaml_temp.unlink(missing_ok=True)

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
        project = self._project_info
        if project is None:
            project = self.make_project_info()
        misplaced_mon = analysis.pellet_misplaced_monitor
        animal = self._selected_animal

        detectors = [
            ApiDetectorStatus(
                detector_id=ApiDetectorKind.pelletMisplaced,
                is_enabled=misplaced_mon.running,
                is_active=misplaced_mon.is_engaged,
            ),
        ]

        # auto-trainer-api 0.9.22 requires alarm/tunnel-shaped fields in its
        # status payload. They are compatibility-only placeholders: ReachAQ
        # has no corresponding alarm, emergency, tunnel, or magnet runtime.
        legacy_api_alarms = []

        dcs_pos_xyz = hard.last_dcs_position
        dcs_send_xyz = hard.last_dcs_set_position
        if dcs_pos_xyz is None or any(map(math.isnan, dcs_pos_xyz)):
            dcs_pos_xyz = Offset3DTuple.get_nan()
        if dcs_send_xyz is None or any(map(math.isnan, dcs_send_xyz)):
            dcs_send_xyz = Offset3DTuple.get_nan()

        reach_status = ApiReachStatus(
            pellets_presented=algo.pellets_presented,
            pellets_consumed=algo.pellets_consumed,
            reaches=algo.pellet_reaches,
            successful_reaches=algo.successful_reaches,
        )

        system_status = ApiSystemStatus(
            application_mode=app_status_to_api_app_mode(self._status),
            training_mode=self._derived_api_training_mode(),
            animal=None if animal is None else animal.to_api_status(),
            project=ApiProjectStatus(
                day_path=project.get_day_path()[0],
                session_index=project.session,
            ),
            detectors=detectors,
            alarms=legacy_api_alarms,
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
                magnet_intensity=math.nan,
                gate_open=False,
            ),
            behavior=ApiBehaviorStatus(
                baseline_magnet_intensity=math.nan,
                reaches=reach_status,
            )
        )
        return system_status

    #

    # pellet machine events

    def _on_trial_protocol_complete(self) -> None:
        logger.success("Selected pellet-trial protocol is complete")
        self.property_changed(
            self.Props.RECORDING_BLOCKERS,
            self.recording_blockers,
            None,
        )
        self._evaluate_automatic_stop_policy(protocol_complete=True)

    def _finish_active_pellet_trial(self, perf_c: float) -> None:
        ledger = self._trial_ledger
        if ledger is None or ledger.active_attempt is None:
            return
        attempt = ledger.close_active_for_analysis(perf_c, time.time())
        if attempt.is_presented:
            self._protocol_runner.finish_trial(
                attempt.attempt_label,
                attempt.outcome,
            )
        self._evaluate_automatic_stop_policy(
            protocol_complete=self._protocol_runner.protocol_complete,
        )

    def _on_pellet_loading_for_trial(self):
        self._finish_active_pellet_trial(get_perf_now())

    def _on_pellet_cycle_completed(self, *, perf_c: float):
        self._finish_active_pellet_trial(perf_c)

    def _on_pellet_sending(self, *, perf_c: float, context: str):
        ledger = self._trial_ledger
        algo = self._behavior.algorithm
        if ledger is None or not algo.is_in_session:
            return
        wall_time = time.time()
        active = ledger.active_attempt
        if active is not None:
            ledger.close_active_for_analysis(perf_c, wall_time)
        attempt = ledger.begin_send(
            perf_c,
            wall_time,
            operation_id=context,
        )
        logger.info(
            "pellet trial attempt started: session=%s attempt=%s context=%s",
            ledger.session_id,
            attempt.attempt_label,
            context,
        )

    def _on_pellet_sent(
        self,
        *,
        perf_c: Optional[float] = None,
        context: Optional[str] = None,
    ):
        algo = self._behavior.algorithm
        logger.debug("on_pellet_sent: recording=%s in_session=%s",
                     self._session_recording_status, algo.is_in_session)
        if algo.is_in_session:
            ledger = self._trial_ledger
            if ledger is not None and ledger.active_attempt is not None:
                if (
                    context is not None
                    and ledger.active_attempt.operation_id != context
                ):
                    logger.error(
                        "Ignoring pellet presentation acknowledgement with mismatched "
                        "context: active=%s received=%s",
                        ledger.active_attempt.operation_id,
                        context,
                    )
                    return
                attempt = ledger.acknowledge_presentation(
                    get_perf_now() if perf_c is None else perf_c,
                    time.time(),
                )
                self._protocol_runner.begin_trial(attempt.attempt_label)
            algo.increase_pellets_presented(1)
            self._evaluate_automatic_stop_policy(
                protocol_complete=self._protocol_runner.protocol_complete,
            )
