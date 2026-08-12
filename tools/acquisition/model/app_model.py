import csv
import ctypes
import dataclasses
import enum
import hashlib
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

import numpy as np
import yaml

from autotrainer.api import ApiDetectorKind, ApiTrainingMode, \
    ApiSystemConfiguration, ApiApplicationMode, ApiCommand, ApiCommandRequestErrorKind

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
    SystemCommandKind,
    SystemStatusMessageKind,
    Offset3DTuple,
    get_perf_now,
)
from autotrainer.core import (
    AnimalSubject,
    ExternalAnimalRecord,
    ExternalIdentity,
    FixedArrayMultiQueue,
)
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
from tools.acquisition.model.api_status import ReachAQSystemStatus
from tools.acquisition.model.session_api_publisher import SessionApiPublisher
from tools.acquisition.model.pellet_cycle_controller import (
    PelletCycleController,
    offset_record,
)
from tools.acquisition.model.recording_session_controller import (
    RecordingSessionController,
    SessionGeneration,
)
from tools.acquisition.model.camera_recording_validation import (
    validate_closed_video,
)
from tools.acquisition.model.camera_timing_alignment import (
    CameraTimestampInput,
    align_camera_timestamp_files,
)
from tools.acquisition.model.acquisition_controller import AcquisitionController
from tools.acquisition.model.coordinate_model import CoordinateModel
from tools.autotrainer_version import __version__ as app_version
from tools.acquisition.model.helpers import get_config_location
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.laser_model import LaserModel
from autotrainer.device import CanFailure, CanTransportConfiguration
from tools.acquisition.model.hardware_scan import HardwareScanEntry, scan_can_adapters, scan_gpus
from tools.acquisition.model.nidaq_discovery import device_name_from_channel, discover_nidaq_devices
from tools.acquisition.model.nidaq_channel_plan import build_nidaq_acquisition_configuration
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.nidaq_timing import (
    remap_nidaq_physical_channel,
    resolve_nidaq_device_aliases,
)
from tools.acquisition.model.rfid_resolution import (
    RfidResolution,
    RfidResolutionKind,
)
from tools.acquisition.model.animal_metadata_sync import AnimalMetadataSyncService
from tools.acquisition.model.animal_reconciliation import (
    AnimalReconciliationChoices,
    reconcile_animals,
)
from tools.acquisition.model.animal_registry import AnimalRegistry
from tools.acquisition.model.rfid_metadata_controller import RfidMetadataController
from tools.acquisition.model.softmouse_spreadsheet_source import (
    SoftMouseMappingProfile,
    SoftMouseSpreadsheetSource,
)
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.acquisition.model.atomic_session_io import (
    atomic_publish_file,
    atomic_write_json,
    file_manifest_entry,
)
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.trial_protocol_schedule import TrialProtocolSchedule
from tools.acquisition.model.session_stop_policy import (
    SessionStopConfiguration,
    SessionStopDecision,
    SessionStopEvaluation,
    SessionStopPolicy,
    SessionStopReason,
)
from tools.acquisition.model.output_storage_monitor import (
    RecordingStorageMonitor,
    StorageSnapshot,
    preflight_storage,
)
from tools.acquisition.model.intertrial_analysis import (
    AnalysisProgressionMode,
    IntertrialAnalysisCoordinator,
    IntertrialAnalysisRequest,
    IntertrialAnalysisResult,
    analyze_tracking_window,
    classify_pellet_state,
)
from tools.acquisition.model.live_tracking_buffer import (
    FrameTimelineAnchor,
    LiveTrackingBuffer,
    LiveTrackingSample,
)
from tools.acquisition.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
    SubsystemStatus,
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


def _compact_subsystem_snapshot(snapshot):
    """Remove fields repeated by the subsystem key and empty defaults."""
    compact = {}
    for subsystem_id, status in (snapshot or {}).items():
        values = {}
        for key, value in status.items():
            if key == "subsystem_id":
                continue
            if key != "state" and value in (None, "", False, 0):
                continue
            values[key] = value
        compact[subsystem_id] = values
    return compact


def _subsystem_snapshot_changes(at_record, at_finalize):
    baseline = _compact_subsystem_snapshot(at_record)
    final = _compact_subsystem_snapshot(at_finalize)
    return {
        subsystem_id: status
        for subsystem_id, status in final.items()
        if baseline.get(subsystem_id) != status
    }


def _metadata_reference(path: Path, *, relative_to: Path):
    if not path.is_file():
        return None
    relative_path = Path(os.path.relpath(path, start=relative_to)).as_posix()
    return {
        "$ref": relative_path,
        "targetSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

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
        ANIMAL_METADATA_STATUS = "animal_metadata_status"
        RFID_READER_STATUS = "rfid_reader_status"
        RFID_SCAN_RESULT = "rfid_scan_result"
        ANIMAL_METADATA_PREVIEW = "animal_metadata_preview"
        ANIMAL_METADATA_REFRESH_BUSY = "animal_metadata_refresh_busy"
        INTERNAL_ERROR_DIAGNOSTIC = "internal_error_diagnostic"
        TRIAL_PROTOCOL_STATE = "trial_protocol_state"

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
        self._acquisition = AcquisitionController()

        self._output_location = PersistenceConfiguration.get_default_output_path().as_posix()
        self._project_info: Optional[ProjectInfo] = None
        self._animal_name = ""
        self._notes = ""
        self._editable_notes_project: Optional[ProjectInfo] = None
        self._trial_protocol_schedule = TrialProtocolSchedule.with_placeholder_rows()
        self._left_camera = self._right_camera = self._stim_camera = None
        self._reach_cameras: Tuple[VideoCaptureModel, ...] = ()

        self._timer_daily: DaemonTimer = _daily_timer(0, self._on_daily_timer)
        self._current_day: Optional[date] = None
        self._log_file_path: Optional[Path] = None

        self.set_log_location()

        self._plan_repo = PlanRepository()
        self._training_plan: Optional[TrainingPlan] = None
        self._training_plan_animal: Optional[AnimalSubject] = None
        self._recording_session = RecordingSessionController()
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
        self._storage_monitor = RecordingStorageMonitor()
        self._storage_preflight = None
        self._internal_error_diagnostic = None
        self._session_invariant_unknown = False
        self._intertrial_resolution_request = None
        self._intertrial_resolution_reason = ""
        self._closing_event = threading.Event()
        self._reload_plans_needed = False
        self._coordinates = CoordinateModel()
        self._p_start_capture = -math.inf
        self._p_inference_live_begin = -math.inf

        self._event_manager = EventManager.default()
        self._session_api = SessionApiPublisher(self._event_manager)

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
        self._record_generation_value = mp_ctx.Value(ctypes.c_uint64, 0)

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
            self._acquisition.subsystems.ensure(
                SubsystemId.camera(camera.name),
                state=SubsystemState.DISABLED,
                reason="camera disabled",
            )
        for subsystem_id in SubsystemId:
            self._acquisition.subsystems.ensure(
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
        system_machine.use_live_intertrial_analysis = True

        self._models: List[ProjectDependentProtocol] = [
            *self._reach_cameras,
            self._inference,
            self._behavior,
            self._nidaq_signal_monitor,
        ]

        self._animals: List[AnimalSubject] = []
        self._animal_by_id: Dict[str, AnimalSubject] = {}
        self._animal_by_external_key: Dict[Tuple[str, str, str], AnimalSubject] = {}
        self._external_link_conflicts = set()
        self._animal_registry = None
        self._animal_metadata_sync = None
        self._rfid_metadata_controller = None
        self._animal_metadata_status = "Not configured"
        self._animal_metadata_preview = None
        self._rfid_reader_status = None
        self._rfid_scan_result = None
        self._animal_metadata_refresh_thread = None
        self._animal_metadata_refresh_lock = threading.Lock()
        self._animal_metadata_refresh_busy = False
        self._animal_metadata_refresh_deferred = False
        self._animal_metadata_refresh_reason = ""

        self._selected_animal: Optional[AnimalSubject] = None
        self._attached_plan: Optional[TrainingPlan] = None
        self._attached_phase: Optional[TrainingPhase] = None
        self._attached_animal: Optional[AnimalSubject] = None
        self._protocol_runner = TrialProtocolRunner(
            on_protocol_complete=self._on_trial_protocol_complete,
        )
        self._pellet_cycles = PelletCycleController(
            self._session_api,
            self._protocol_runner,
            self._session_data_recorder,
        )
        self._live_tracking = LiveTrackingBuffer()
        self._intertrial_analysis = IntertrialAnalysisCoordinator(
            self._on_intertrial_analysis_result,
        )
        self._intertrial_finalize_thread = None
        self._trial_window_start = None
        self._tone2_active = False
        self._intertrial_lock = threading.RLock()

        self._rpc_service: Optional[RpcService] = None

        self._hardware.property_changed += self._on_hardware_property_changed
        self._hardware.command_failed += self._on_hardware_command_failed
        self._nidaq_signal_monitor.property_changed += (
            self._on_nidaq_monitor_property_changed
        )
        self._laser.property_changed += self._on_laser_runtime_property_changed
        self._system_message_handler.decoded_message_received += (
            self._on_intertrial_device_message
        )
        inference.property_changed += self._on_inference_property_changed
        inference.pose_response_ready += self._on_pose_response_ready

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
            record_generation=self._record_generation_value,
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
        if (
            getattr(self._preferences, "softmouse_nightly_refresh", False)
            and self._animal_metadata_sync is not None
        ):
            self._request_animal_metadata_catchup()

    @property
    def app_lock(self) -> threading.RLock:
        return self._behavior.algorithm.thread_lock

    @property
    def _trial_ledger(self) -> Optional[PelletTrialLedger]:
        """Compatibility view; PelletCycleController owns the ledger."""
        return self._pellet_cycles.ledger

    @_trial_ledger.setter
    def _trial_ledger(self, value: Optional[PelletTrialLedger]) -> None:
        self._pellet_cycles.ledger = value

    @property
    def acquisition_started(self):
        return self._acquisition.started

    @property
    def session_recording_status(self) -> SessionRecordingStatus:
        return self._recording_session.status

    def _publish_session_recording_transition(
        self,
        previous: SessionRecordingStatus,
        status: SessionRecordingStatus,
    ) -> None:
        if status == previous:
            return
        logger.info("session recording status: %s -> %s", previous.value, status.value)
        self.property_changed(self.Props.SESSION_RECORDING_STATUS, status, previous)
        self.property_changed(
            self.Props.RECORDING_BLOCKERS,
            self.recording_blockers,
            None,
        )
        if status is SessionRecordingStatus.READY:
            with self._animal_metadata_refresh_lock:
                deferred = self._animal_metadata_refresh_deferred
            if deferred:
                self.request_animal_metadata_refresh("deferred after session")

    def _set_session_recording_status(
        self,
        status: SessionRecordingStatus,
        *,
        expected: Optional[Tuple[SessionRecordingStatus, ...]] = None,
        token: Optional[SessionGeneration] = None,
    ) -> bool:
        previous = self._recording_session.transition(
            status,
            expected=expected,
            token=token,
        )
        if previous is None:
            logger.warning(
                "ignored stale/invalid session transition to %s: expected=%s token=%s",
                status.value,
                expected,
                token,
            )
            return False
        self._publish_session_recording_transition(previous, status)
        return True

    def start_recording(self) -> bool:
        if not self._acquisition.started or self._status == AppModelStatus.IDLE:
            self.on_error("Recording unavailable", "Set System Mode to Running before recording.")
            return False
        if self._selected_animal is None:
            logger.warning("start_recording refused because no subject is selected")
            self.on_error(
                "Recording unavailable",
                "Scan an RFID tag or select a subject before recording.",
            )
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
        if self._recording_session.status != SessionRecordingStatus.READY:
            logger.warning("start_recording refused while %s", self._recording_session.status.value)
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
        project = self._project_info
        if project is None:
            return False
        session_config = self._behavior.algorithm.active_config.session_control
        estimated_rate = self._estimate_session_bytes_per_second()
        session_path = Path(
            project.get_session_path(skip_ensure=True).location
        )
        try:
            storage_preflight = preflight_storage(
                session_path,
                estimated_bytes_per_second=estimated_rate,
                configured_duration_seconds=session_config.duration_limit_seconds,
            )
        except Exception as error:
            logger.exception("Session output write/fsync preflight failed")
            self.on_error("Recording unavailable", f"Storage preflight failed: {error}")
            return False
        self._storage_preflight = storage_preflight
        logger.info(
            "Session storage preflight: target=%s free_bytes=%d "
            "estimated_bytes_per_second=%.1f projected_maximum_minutes=%s",
            storage_preflight.target_directory,
            storage_preflight.free_bytes,
            storage_preflight.estimated_bytes_per_second,
            storage_preflight.projected_maximum_minutes,
        )
        if not storage_preflight.duration_fits_projection:
            message = (
                f"Configured duration {storage_preflight.configured_duration_seconds / 60:.1f} "
                "minutes exceeds the projected storage capacity of "
                f"{storage_preflight.projected_maximum_minutes:.1f} minutes."
            )
            logger.warning(message)
            self.on_error("Storage capacity warning", message)
        self.persist_stopped_session_notes()
        self._editable_notes_project = None
        self.notes = ""
        reservation = self._recording_session.begin_record(
            project.short_id,
            self._acquisition.subsystems.snapshot(),
            animal_snapshot=(
                None
                if self._selected_animal is None
                else self._selected_animal.session_snapshot()
            ),
        )
        if reservation is None:
            logger.warning(
                "start_recording lost session reservation while %s",
                self._recording_session.status.value,
            )
            return False
        previous_status, session_token = reservation
        self._publish_session_recording_transition(
            previous_status,
            SessionRecordingStatus.ARMING,
        )
        self._record_generation_value.value = session_token.generation
        project.session_generation = session_token.generation
        self._recording_session.set_storage_telemetry({
            "preflight": {
                "targetDirectory": storage_preflight.target_directory,
                "freeBytes": storage_preflight.free_bytes,
                "estimatedBytesPerSecond": (
                    storage_preflight.estimated_bytes_per_second
                ),
                "projectedMaximumMinutes": (
                    storage_preflight.projected_maximum_minutes
                ),
                "configuredDurationSeconds": (
                    storage_preflight.configured_duration_seconds
                ),
                "durationFitsProjection": storage_preflight.duration_fits_projection,
            }
        })
        self._set_subsystem_status(
            SubsystemId.INTERTRIAL_ANALYSIS,
            SubsystemState.DISABLED,
            reason="no completed pellet trial pending",
        )
        self._abort_had_recording_started = False
        self._session_data_recorder.arm(
            project,
            source_manifest=self._build_session_source_manifest(),
            metadata_generation_id=(
                self._recording_session.metadata_generation_id
            ),
        )
        try:
            started = self._behavior.algorithm.start_session(reason="manual_record")
        except Exception as err:
            logger.exception("manual recording start failed: %s", err)
            self._session_data_recorder.abort()
            self._set_session_recording_status(
                SessionRecordingStatus.READY,
                expected=(SessionRecordingStatus.ARMING,),
                token=session_token,
            )
            self.on_error("Recording failed", str(err))
            return False
        if not started:
            self._session_data_recorder.abort()
            self._set_session_recording_status(
                SessionRecordingStatus.READY,
                expected=(SessionRecordingStatus.ARMING,),
                token=session_token,
            )
            return False
        self._aborted_session_ids.discard(project.short_id)
        if self._recording_session.status == SessionRecordingStatus.ARMING:
            self._record_start_timer.cancel()
            self._record_start_timer = make_daemon_timer(
                10.0,
                lambda token=session_token: self._record_start_timed_out(token),
            )
            self._record_start_timer.start()
        return True

    def _estimate_session_bytes_per_second(self) -> float:
        """Conservative configuration-only estimate used until writes are observed."""

        total = 16 * 1024.0  # decoded events, timestamps, pose, laser, and logs
        for camera in self._get_recording_cams():
            params = camera.active_config.params
            width = float(params.get("width", 2048))
            height = float(params.get("height", 1536))
            fps = float(params.get("fps", 150))
            explicit_mbps = params.get("estimated_recording_mbps")
            if explicit_mbps is not None:
                total += max(0.0, float(explicit_mbps)) * 1_000_000 / 8.0
            else:
                # MP4V is content dependent. One quarter byte per source pixel is
                # deliberately conservative and observed throughput supersedes it.
                total += max(0.0, width * height * fps * 0.25)
            total += max(0.0, fps * 64.0)  # frame timestamp rows
        monitor = self._nidaq_signal_monitor
        if monitor.hardware_enabled and monitor.configuration.is_enabled:
            configuration = monitor.configuration
            bytes_per_sample = 24 + 4 * len(configuration.channels)
            total += configuration.sample_rate_hz * bytes_per_sample
        return total

    def _start_storage_monitor(self, token: SessionGeneration) -> None:
        project = self._project_info
        if project is None or not self._recording_session.is_current(
            token, statuses=(SessionRecordingStatus.RECORDING,)
        ):
            return
        target = Path(project.get_session_path(skip_ensure=True).location)
        self._storage_monitor.start(
            target,
            estimated_bytes_per_second=self._estimate_session_bytes_per_second(),
            callback=lambda snapshot, threshold, token=token: (
                self._on_storage_sample(token, snapshot, threshold)
            ),
        )

    def _on_storage_sample(
        self,
        token: SessionGeneration,
        snapshot: StorageSnapshot,
        threshold: Optional[int],
    ) -> None:
        if not self._recording_session.is_current(
            token, statuses=(SessionRecordingStatus.RECORDING,)
        ):
            return
        if threshold is None:
            return
        remaining = snapshot.projected_remaining_minutes
        message = (
            f"Projected recording capacity is {remaining:.2f} minutes "
            f"({snapshot.free_bytes} bytes free)."
        )
        if threshold == 1:
            logger.error("Storage critical; requesting normal Stop: %s", message)
            self.on_error("Storage critical — recording will stop", message)
            self._stop_recording(RecordingEndingReason.STORAGE_LIMIT, token=token)
        else:
            logger.warning("Storage capacity crossed %d minutes: %s", threshold, message)
            self.on_error(f"Storage below {threshold} minutes", message)

    def _build_session_source_manifest(self) -> Tuple[dict, ...]:
        def runtime_state(subsystem_id) -> str:
            status = self._acquisition.subsystems.get(subsystem_id)
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

    def _record_start_timed_out(
        self,
        token: Optional[SessionGeneration] = None,
    ) -> None:
        if token is not None and not self._recording_session.is_current(
            token,
            statuses=(SessionRecordingStatus.ARMING,),
        ):
            logger.info("ignoring stale record-start timeout: %s", token)
            return
        if self._recording_session.status != SessionRecordingStatus.ARMING:
            return
        logger.error("Timed out waiting for the primary camera to begin recording")
        self.on_error(
            "Recording failed",
            "The primary camera did not begin recording within 10 seconds. "
            "The partial session will be aborted.",
        )
        if token is None:
            self.abort_recording()
        else:
            self.abort_recording(token=token)

    def _start_automatic_stop_policy(
        self,
        start_perf_time: float,
        token: Optional[SessionGeneration] = None,
    ) -> None:
        policy = self._session_stop_policy
        if policy is None:
            return
        policy.start(start_perf_time)
        duration = policy.configuration.duration_seconds
        if duration is not None:
            self._automatic_stop_timer.cancel()
            self._automatic_stop_timer = make_daemon_timer(
                duration,
                lambda token=token: self._evaluate_automatic_stop_policy(
                    token=token
                ),
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
        token: Optional[SessionGeneration] = None,
    ) -> Optional[SessionStopEvaluation]:
        if token is not None and not self._recording_session.is_current(
            token,
            statuses=(SessionRecordingStatus.RECORDING,),
        ):
            logger.info("ignoring stale automatic-stop callback: %s", token)
            return None
        policy = self._session_stop_policy
        if (
            policy is None
            or not policy.is_started
            or self._recording_session.status is not SessionRecordingStatus.RECORDING
        ):
            return None
        evaluation = policy.evaluate(
            get_perf_now(),
            trial_count=self._pellet_cycles.count(),
            protocol_complete=protocol_complete,
            trial_active=self._pellet_cycles.active_attempt is not None,
        )
        self._session_stop_evaluation = evaluation
        if evaluation.decision is SessionStopDecision.NONE:
            return evaluation
        if evaluation.decision is SessionStopDecision.FINISH_ACTIVE_TRIAL:
            self._behavior.algorithm.pellet_automation_stop_requested = True
            if self._stop_drain_timer is no_op_timer:
                self._stop_drain_timer = make_daemon_timer(
                    policy.configuration.drain_timeout_seconds,
                    lambda token=token: self._evaluate_automatic_stop_policy(
                        token=token
                    ),
                )
                self._stop_drain_timer.start()
            logger.info(
                "automatic session stop requested; finishing active trial: %s",
                evaluation.reason,
            )
            return evaluation
        if evaluation.decision is SessionStopDecision.TIMEOUT_ERROR:
            now_perf = get_perf_now()
            if self._pellet_cycles.active_attempt is not None:
                self._pellet_cycles.finalize_active_incomplete(
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
        if token is None:
            self._stop_recording_with_reason(
                evaluation.reason,
                evaluation.decision,
            )
        else:
            self._stop_recording_with_reason(
                evaluation.reason,
                evaluation.decision,
                token=token,
            )
        return evaluation

    def _stop_recording_with_reason(
        self,
        reason: Optional[SessionStopReason],
        decision: SessionStopDecision = SessionStopDecision.STOP,
        *,
        token: Optional[SessionGeneration] = None,
    ) -> bool:
        ending_reason = {
            SessionStopReason.DURATION_LIMIT: RecordingEndingReason.DURATION_LIMIT,
            SessionStopReason.TRIAL_LIMIT: RecordingEndingReason.TRIAL_LIMIT,
            SessionStopReason.PROTOCOL_COMPLETE: RecordingEndingReason.PROTOCOL_COMPLETE,
        }.get(reason, RecordingEndingReason.STOP_DRAIN_TIMEOUT)
        if decision is SessionStopDecision.TIMEOUT_ERROR:
            ending_reason = RecordingEndingReason.STOP_DRAIN_TIMEOUT
        return self._stop_recording(ending_reason, token=token)

    def _stop_recording(
        self,
        reason: RecordingEndingReason,
        *,
        token: Optional[SessionGeneration] = None,
    ) -> bool:
        token = token or self._recording_session.token()
        if token is not None and not self._recording_session.is_current(
            token, statuses=(SessionRecordingStatus.RECORDING,)
        ):
            logger.warning(
                "stop_recording refused while %s",
                self._recording_session.status.value,
            )
            return False
        self._cancel_automatic_stop_timers()
        self._recording_session.analysis_finished = False
        if not self._set_session_recording_status(
            SessionRecordingStatus.STOPPING,
            expected=(SessionRecordingStatus.RECORDING,),
            token=token,
        ):
            return False
        stopped = self._behavior.algorithm.end_capture_session(reason=reason)
        if not stopped:
            self._set_session_recording_status(
                SessionRecordingStatus.RECORDING,
                expected=(SessionRecordingStatus.STOPPING,),
                token=token,
            )
        return bool(stopped)

    def stop_recording(self) -> bool:
        return self._stop_recording(RecordingEndingReason.MANUAL_STOP)

    def abort_recording(
        self,
        *,
        token: Optional[SessionGeneration] = None,
    ) -> bool:
        token = token or self._recording_session.token()
        previous_status = self._recording_session.status
        if previous_status not in {
            SessionRecordingStatus.ARMING,
            SessionRecordingStatus.RECORDING,
            SessionRecordingStatus.ANALYZING,
        }:
            logger.warning("abort_recording refused while %s", self._recording_session.status.value)
            return False
        if token is not None and not self._recording_session.is_current(
            token,
            statuses=(
                SessionRecordingStatus.ARMING,
                SessionRecordingStatus.RECORDING,
                SessionRecordingStatus.ANALYZING,
            ),
        ):
            logger.warning("abort_recording refused for stale session: %s", token)
            return False
        project = self._project_info
        if project is None:
            return False
        self._aborting_project = project.to_local_value()
        if not self._set_session_recording_status(
            SessionRecordingStatus.ABORTING,
            expected=(previous_status,),
            token=token,
        ):
            self._aborting_project = None
            return False
        self._aborted_session_ids.add(self._aborting_project.short_id)
        self._intertrial_analysis.cancel_session()
        self._live_tracking.clear()
        self._trial_window_start = None
        self._intertrial_resolution_request = None
        self._intertrial_resolution_reason = ""
        self._behavior.algorithm.pellet_send_block_reason = ""
        self._session_data_recorder.abort()
        self._cancel_automatic_stop_timers()
        self._recording_session.pending_end_perf = None
        self._record_start_timer.cancel()
        self._record_start_timer = no_op_timer
        if previous_status is SessionRecordingStatus.ANALYZING:
            logger.notice(
                "Cancelling pending pellet-trial analysis before deleting %s",
                self._aborting_project.short_id,
            )
            self._finish_abort_recording(token=token)
            return True
        stopped = self._behavior.algorithm.end_capture_session(
            reason=RecordingEndingReason.MANUAL_ABORT,
        )
        if not stopped:
            self._set_session_recording_status(
                previous_status,
                expected=(SessionRecordingStatus.ABORTING,),
                token=token,
            )
            self._aborting_project = None
            return False
        if previous_status == SessionRecordingStatus.ARMING:
            self._abort_cleanup_timer.cancel()
            self._abort_cleanup_timer = make_daemon_timer(
                1.0,
                lambda token=token: self._finish_abort_if_never_started(token),
            )
            self._abort_cleanup_timer.start()
        return True

    def _finish_abort_if_never_started(
        self,
        token: Optional[SessionGeneration] = None,
    ) -> None:
        if token is not None and not self._recording_session.is_current(
            token,
            statuses=(SessionRecordingStatus.ABORTING,),
        ):
            logger.info("ignoring stale abort-cleanup callback: %s", token)
            return
        if (
            self._recording_session.status == SessionRecordingStatus.ABORTING
            and not self._abort_had_recording_started
            and all(
                camera.video_status != CaptureProcessStatus.RECORDING
                for camera in self._get_recording_cams()
            )
        ):
            self._finish_abort_recording(token=token)

    def check_target_status_valid(self, target: AppModelStatus):
        current_status = self._status
        if (
            target != current_status
            and self._recording_session.status != SessionRecordingStatus.READY
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
            return None
        timing_path = project.get_frame_timing_path()
        prim_cam = cams[0]  # primary
        main_fps = prim_cam.active_config.params.get('fps', math.nan)
        logger.info("Merging camera timestamp files for session%03d into %s (fps=%s)",
                    project.session, timing_path, main_fps)
        if not isinstance(main_fps, (int, float)) or main_fps == 0 or not math.isfinite(main_fps):
            raise ValueError(
                f"invalid camera config fps={main_fps}; camera={prim_cam.name}"
            )
        boundary = self._recording_session.boundary
        if boundary is None or boundary.session_id != project.short_id:
            raise RuntimeError(
                f"Missing canonical recording boundary for {project.short_id}"
            )
        sources = []
        used_camera_field_names = set()
        for cam in cams:
            base_name = self._camera_timing_field_name(cam)
            name = base_name
            suffix = 1
            while name in used_camera_field_names:
                suffix += 1
                name = f"{base_name}_{suffix}"
            used_camera_field_names.add(name)
            _, timestamp_path, _ = project.get_video_path(
                cam.name,
                allow_overwrite=True,
            )
            sources.append(
                CameraTimestampInput(
                    name=cam.name,
                    field_name=name,
                    path=Path(timestamp_path),
                )
            )
        session_dir = Path(project.get_session_path().location)
        diagnostics = align_camera_timestamp_files(
            tuple(sources),
            output_path=timing_path,
            diagnostics_path=session_dir / "streams" / "camera_alignment.json",
            primary_frame_id=boundary.primary_frame_id,
            primary_fps=float(main_fps),
        )
        logger.info(
            "Written %s entries into %s; synchronization_complete=%s",
            diagnostics["primaryFrameCount"],
            timing_path,
            diagnostics["synchronizationComplete"],
        )
        self._remove_timestamps_txt_files(project, cams)
        return diagnostics

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
                if self._recording_session.status
                is not SessionRecordingStatus.READY
                else self._identify_primary_main_cam_idx()
            )
            if cam_idx == reference_cam_idx:
                if new_status == CaptureProcessStatus.RECORDING:
                    first_frame_perf, first_frame_when, first_frame_time, *r_args = r_args
                    first_frame_id = (
                        int(r_args[0])
                        if r_args
                        else int(self._cams_synced_frame_index.value)
                    )
                    message_generation = int(r_args[1]) if len(r_args) > 1 else None
                    session_token = self._recording_session.token()
                    if (
                        session_token is not None
                        and message_generation not in (None, 0, session_token.generation)
                    ):
                        logger.warning(
                            "ignoring stale camera RECORDING callback: camera=%s "
                            "message_generation=%s active=%s",
                            cam_idx,
                            message_generation,
                            session_token,
                        )
                        return
                    p_now = first_frame_perf
                    project = self._project_info
                    if (
                        session_token is not None
                        and (
                            project is None
                            or project.short_id != session_token.session_id
                        )
                    ):
                        logger.warning(
                            "ignoring camera RECORDING callback for wrong project: "
                            "project=%s active=%s",
                            None if project is None else project.short_id,
                            session_token,
                        )
                        return
                    if project is not None:
                        project.start_record_timestamp = first_frame_time
                        primary = next(
                            camera
                            for camera in self._cameras
                            if camera.camera_index == cam_idx
                        )
                        boundary = SessionBoundary(
                            session_id=project.short_id,
                            primary_camera=primary.name,
                            primary_frame_id=first_frame_id,
                            start_perf_time=float(first_frame_perf),
                            start_wall_time=float(first_frame_time),
                            camera_when=float(first_frame_when),
                        )
                        if session_token is None:
                            self._recording_session.boundary = boundary
                        elif not self._recording_session.set_boundary(
                            boundary,
                            session_token,
                        ):
                            logger.warning(
                                "ignoring stale camera recording boundary: %s",
                                session_token,
                            )
                            return
                    self._session_data_recorder.commit_start(
                        first_frame_perf,
                        first_frame_time,
                        boundary=self._recording_session.boundary,
                    )
                    self._record_start_timer.cancel()
                    self._record_start_timer = no_op_timer
                    self._abort_had_recording_started = True
                    if project is not None:
                        self._session_api.session_started(project.short_id)
                    logger.info(
                        "received RECORDING: frame-0 time=%.3f perf_c=%.3f now=%.3f",
                        first_frame_time,
                        first_frame_perf,
                        get_perf_now(),
                    )
                    if self._recording_session.status == SessionRecordingStatus.ARMING:
                        if self._set_session_recording_status(
                            SessionRecordingStatus.RECORDING,
                            expected=(SessionRecordingStatus.ARMING,),
                            token=session_token,
                        ):
                            if session_token is not None:
                                self._start_storage_monitor(session_token)
                            self._start_automatic_stop_policy(
                                first_frame_perf,
                                session_token,
                            )
                else:
                    p_now = get_perf_now()
                algo.set_capture_status(new_status, perf_now=p_now)
                if new_status == CaptureProcessStatus.RUNNING:
                    message_generation = (
                        int(r_args[2]) if len(r_args) > 2 else None
                    )
                    session_token = self._recording_session.token()
                    if (
                        session_token is not None
                        and message_generation not in (None, 0, session_token.generation)
                    ):
                        logger.warning(
                            "ignoring stale camera RUNNING callback: camera=%s "
                            "message_generation=%s active=%s",
                            cam_idx,
                            message_generation,
                            session_token,
                        )
                        return
                    if self._recording_session.status == SessionRecordingStatus.STOPPING:
                        end_perf = r_args[0] if r_args else get_perf_now()
                        if session_token is None:
                            self._recording_session.pending_end_perf = end_perf
                        else:
                            self._recording_session.set_pending_end(
                                end_perf,
                                session_token,
                            )
                    elif (
                        self._recording_session.status == SessionRecordingStatus.ABORTING
                        and not self._abort_had_recording_started
                    ):
                        self._finish_abort_recording(token=session_token)
            else:
                logger.verbose("not handling non-primary camera status, cam_idx=%s status=%s",
                               cam_idx, new_status)
        elif cmd == SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED:
            cam_idx, frames_written, project, *r_args = args
            if project is None:
                return
            message_generation = int(r_args[0]) if r_args else None
            writer_diagnostics = (
                dict(r_args[1]) if len(r_args) > 1 and r_args[1] else {}
            )
            session_token = self._recording_session.token()
            if session_token is not None and (
                project.short_id != session_token.session_id
                or message_generation not in (None, 0, session_token.generation)
            ):
                logger.warning(
                    "ignoring stale camera-close callback: camera=%s project=%s "
                    "message_generation=%s active=%s",
                    cam_idx,
                    project.short_id,
                    message_generation,
                    session_token,
                )
                return
            recording_cams = self._get_recording_cams()
            recording_cam_indices = tuple(cam.camera_index for cam in recording_cams)
            if cam_idx in recording_cam_indices:
                cams_closed_finished[cam_idx] = (
                    project,
                    frames_written,
                    message_generation,
                    writer_diagnostics,
                )
                if all(cam.camera_index in cams_closed_finished for cam in recording_cams):
                    project = cams_closed_finished[recording_cams[0].camera_index][0]
                    session_dir = Path(project.get_session_path().location)
                    for camera in recording_cams:
                        (
                            camera_project,
                            camera_frames,
                            _camera_generation,
                            camera_writer_diagnostics,
                        ) = (
                            cams_closed_finished[camera.camera_index]
                        )
                        video_path, timestamp_path, _ = camera_project.get_video_path(
                            camera.name,
                            allow_overwrite=True,
                        )
                        validation = validate_closed_video(
                            Path(video_path),
                            Path(timestamp_path),
                            writer_frame_count=camera_frames,
                            writer_diagnostics=camera_writer_diagnostics,
                        )
                        if validation.warnings:
                            logger.warning(
                                "Camera recording validation warning: camera=%s %s",
                                camera.name,
                                "; ".join(validation.warnings),
                            )
                        if validation.failure:
                            logger.error(
                                "Camera recording validation failed: camera=%s %s",
                                camera.name,
                                validation.failure,
                            )
                            self.on_error(
                                "Camera recording incomplete",
                                f"{camera.name}: {validation.failure}",
                            )
                        self._session_data_recorder.set_source_result(
                            f"camera.{camera.name}",
                            sample_count=validation.decoded_frame_count,
                            path=Path(video_path).relative_to(
                                session_dir
                            ).as_posix(),
                            failure=validation.failure,
                            warnings=validation.warnings,
                            diagnostics=validation.diagnostics(),
                        )
                    monitored_cams = tuple(
                        camera for camera in recording_cams
                        if camera in self._reach_cameras
                    )
                    try:
                        alignment_diagnostics = self._merge_camera_timestamp_files(
                            project,
                            monitored_cams,
                        )
                    except Exception as err:
                        logger.exception("Failed to merge camera timestamps: %s", err)
                        self.on_error("Camera timestamp merge failed", str(err))
                        for camera in monitored_cams:
                            self._session_data_recorder.set_source_result(
                                f"camera.{camera.name}",
                                failure=f"camera timestamp alignment failed: {err}",
                            )
                    else:
                        if (
                            alignment_diagnostics is not None
                            and not alignment_diagnostics["synchronizationComplete"]
                        ):
                            message = (
                                "Camera frame synchronization is incomplete; "
                                "see streams/camera_alignment.json"
                            )
                            logger.warning(message)
                            self.on_error("Camera synchronization incomplete", message)
                            for camera in monitored_cams:
                                camera_diagnostics = alignment_diagnostics[
                                    "cameras"
                                ][camera.name]
                                camera_incomplete = bool(
                                    camera_diagnostics["frameIdGapCount"]
                                    or camera_diagnostics["missingPrimaryFrameIds"]
                                    or camera_diagnostics["extraFrameIds"]
                                )
                                self._session_data_recorder.set_source_result(
                                    f"camera.{camera.name}",
                                    failure=message if camera_incomplete else "",
                                    warnings=(message,),
                                    diagnostics={
                                        "frameAlignment": camera_diagnostics,
                                    },
                                )
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
                    if self._recording_session.status == SessionRecordingStatus.ABORTING:
                        if session_token is None:
                            self._finish_abort_recording()
                        else:
                            self._finish_abort_recording(token=session_token)
                    elif self._recording_session.status == SessionRecordingStatus.STOPPING:
                        if session_token is None:
                            self._complete_stopped_recording(project)
                        else:
                            self._complete_stopped_recording(
                                project,
                                token=session_token,
                            )
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
        return dict(self._acquisition.subsystems.statuses)

    @property
    def recording_blockers(self) -> Tuple[str, ...]:
        blockers = list(self._acquisition.subsystems.recording_blockers())
        session_control = self._behavior.algorithm.active_config.session_control
        if session_control.intertrial_analysis_enabled and not self._inference.is_enabled:
            blockers.append(
                "Live intertrial analysis requires live inference; enable live "
                "inference or disable live intertrial analysis"
            )
        if self._protocol_runner.protocol_complete:
            blockers.append("Selected protocol is complete")
        if self._recording_session.status is not SessionRecordingStatus.READY:
            blockers.append(
                f"recording state: {self._recording_session.status.value}"
            )
        if not self._acquisition.started:
            blockers.append("System Mode is not running")
        if self._session_invariant_unknown:
            blockers.append(
                "An internal error left recording lifecycle ownership uncertain; "
                "restart acquisition before recording again"
            )
        return tuple(blockers)

    @property
    def internal_error_diagnostic(self) -> Optional[dict]:
        return (
            None
            if self._internal_error_diagnostic is None
            else dict(self._internal_error_diagnostic)
        )

    def _mark_session_invariant_unknown(self, reason: str) -> None:
        """Explicit lifecycle-owner hook; generic exception reporting never calls it."""

        self._session_invariant_unknown = True
        logger.error("Recording lifecycle invariant is unknown: %s", reason)
        self.property_changed(
            self.Props.RECORDING_BLOCKERS,
            self.recording_blockers,
            None,
        )

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
        with self._acquisition.lock:
            previous_statuses = self._acquisition.subsystems.statuses
            status, _ = self._acquisition.subsystems.transition(
                subsystem_id,
                state,
                reason=reason,
                error=error,
                required_for_recording=required_for_recording,
                generation=generation,
            )
            current_statuses = self._acquisition.subsystems.statuses
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
        with self._acquisition.lock:
            previous_statuses = self._acquisition.subsystems.statuses
            status, _ = self._acquisition.subsystems.begin_retry(
                subsystem_id,
                required_for_recording=required_for_recording,
                reason=reason,
            )
            current_statuses = self._acquisition.subsystems.statuses
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
            SubsystemId.INTERTRIAL_ANALYSIS,
            SubsystemState.DISABLED,
            reason="no completed pellet trial pending",
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
        if self._acquisition.started or self._status != AppModelStatus.IDLE:
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
            physical_camera_count = sum(
                source.url.startswith("spinnaker://")
                for source in camera_sources
            )
            camera_info = (
                f"✓ {len(camera_sources)} selectable source(s) · "
                f"{physical_camera_count} Spinnaker camera(s)"
            )
            if source_names:
                camera_info += "\n" + "\n".join(f"→ {name}" for name in source_names)
            if missing_enabled_cameras:
                missing_text = ", ".join(missing_enabled_cameras)
                warnings_list.append(f"configured camera(s) not currently discovered: {missing_text}")
                camera_info += f"\n! missing: {missing_text}"
            elif physical_camera_count == 0:
                camera_info += "\n! zero Spinnaker cameras found"
            scan_results["cameras"] = HardwareScanEntry(
                camera_info,
                (
                    "warning"
                    if missing_enabled_cameras or physical_camera_count == 0
                    else "ok"
                ),
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
        self._rebuild_animal_indexes()
        self._on_property_changed(self.Props.ANIMALS, value, prev)

    def get_animal_by_id(self, animal_id) -> Optional[AnimalSubject]:
        return self._animal_by_id.get(animal_id)

    def get_animal_by_external_identity(
        self, identity: ExternalIdentity
    ) -> Optional[AnimalSubject]:
        if identity.key in self._external_link_conflicts:
            return None
        return self._animal_by_external_key.get(identity.key)

    @property
    def external_link_conflicts(self):
        return frozenset(self._external_link_conflicts)

    @property
    def animal_metadata_status(self) -> str:
        return self._animal_metadata_status

    @property
    def animal_metadata_preview(self):
        return self._animal_metadata_preview

    @property
    def animal_metadata_refresh_busy(self) -> bool:
        with self._animal_metadata_refresh_lock:
            return self._animal_metadata_refresh_busy

    def _set_animal_metadata_status(self, value: str) -> None:
        previous, self._animal_metadata_status = self._animal_metadata_status, value
        if value != previous:
            self._on_property_changed(
                self.Props.ANIMAL_METADATA_STATUS, value, previous
            )

    def _configure_animal_metadata_services(self) -> None:
        controller = self._rfid_metadata_controller
        if controller is not None:
            controller.stop()
        self._animal_registry = None
        self._animal_metadata_sync = None
        self._rfid_metadata_controller = None
        manifest_value = getattr(self._preferences, "softmouse_manifest_path", "")
        hardware_configuration = (
            None if self._loaded_configuration is None
            else self._loaded_configuration.hardware
        )
        reader_enabled = bool(
            hardware_configuration is not None
            and hardware_configuration.rfid_reader_enabled
        )
        logger.info(
            "Configuring SoftMouse/RFID services: manifest_configured=%s "
            "rfid_enabled=%s rfid_device=%s",
            bool(manifest_value),
            reader_enabled,
            (
                "not configured"
                if hardware_configuration is None or not hardware_configuration.rfid_device
                else hardware_configuration.rfid_device
            ),
        )
        if not manifest_value and not reader_enabled:
            self._set_animal_metadata_status("Not configured")
            self._set_subsystem_status(
                SubsystemId.ANIMAL_REGISTRY,
                SubsystemState.DISABLED,
                reason="SoftMouse/RFID not configured",
                required_for_recording=False,
            )
            self._set_subsystem_status(
                SubsystemId.RFID_READER,
                SubsystemState.DISABLED,
                reason="RFID reader disabled",
                required_for_recording=False,
            )
            return
        cache_directory = Path(self._preferences.configuration_location) / "softmouse"
        try:
            registry = self._animal_registry = AnimalRegistry(
                cache_directory / "animal-registry.sqlite3"
            )
        except Exception as exc:
            logger.exception("SoftMouse animal registry initialization failed")
            self._set_animal_metadata_status(f"Registry unavailable: {exc}")
            self._set_subsystem_status(
                SubsystemId.ANIMAL_REGISTRY,
                SubsystemState.FAILED,
                error=str(exc),
                required_for_recording=False,
            )
            self._set_subsystem_status(
                SubsystemId.RFID_READER,
                SubsystemState.DISABLED,
                reason="local registry unavailable",
                required_for_recording=False,
            )
            return
        registry_reason = "local cache ready"
        if registry.recovered_corrupt_path is not None:
            registry_reason = (
                "rebuilt corrupt cache; backup: "
                f"{registry.recovered_corrupt_path.name}"
            )
            logger.warning(
                "Recovered corrupt SoftMouse registry: backup=%s",
                registry.recovered_corrupt_path,
            )
        self._set_subsystem_status(
            SubsystemId.ANIMAL_REGISTRY,
            SubsystemState.READY,
            reason=registry_reason,
            required_for_recording=False,
        )
        if manifest_value:
            profile = SoftMouseMappingProfile(
                new_animal_name_column=getattr(
                    self._preferences, "softmouse_name_column", "Physical Tag"
                )
            )
            self._animal_metadata_sync = AnimalMetadataSyncService(
                manifest_path=Path(manifest_value),
                local_staging_directory=cache_directory / "source",
                source=SoftMouseSpreadsheetSource(profile),
                registry=registry,
                can_refresh=lambda: (
                    self.session_recording_status is SessionRecordingStatus.READY
                ),
            )
        if reader_enabled:
            self._rfid_metadata_controller = RfidMetadataController(
                app_model=self,
                registry=registry,
                device=hardware_configuration.rfid_device,
            )
            self._set_subsystem_status(
                SubsystemId.RFID_READER,
                SubsystemState.STOPPED,
                reason="configured; not started",
                required_for_recording=False,
            )
            logger.info(
                "RFID metadata controller configured: device=%s",
                hardware_configuration.rfid_device,
            )
        else:
            self._set_subsystem_status(
                SubsystemId.RFID_READER,
                SubsystemState.DISABLED,
                reason="RFID reader disabled",
                required_for_recording=False,
            )
        previous = registry.last_complete_import()
        if previous is None:
            self._set_animal_metadata_status("Ready; no export imported yet")
        else:
            self._set_animal_metadata_status(
                f"Cache has {previous['accepted_rows']} tagged animals; "
                f"last refresh {previous['imported_utc']}; "
                f"source {previous['source_file_sha256'][:12]}"
            )

    def refresh_animal_metadata(self):
        """Compatibility entry point; refresh work is always asynchronous."""
        return self.request_animal_metadata_refresh("manual refresh")

    def request_animal_metadata_refresh(
        self,
        reason: str,
        *,
        only_if_due: bool = False,
    ) -> bool:
        service = self._animal_metadata_sync
        if service is None:
            raise RuntimeError("Configure a SoftMouse publication manifest first")
        if self.session_recording_status is not SessionRecordingStatus.READY:
            with self._animal_metadata_refresh_lock:
                self._animal_metadata_refresh_deferred = True
                self._animal_metadata_refresh_reason = str(reason)
            self._set_animal_metadata_status("Refresh deferred until the session is Ready")
            return False
        with self._animal_metadata_refresh_lock:
            if self._animal_metadata_refresh_busy:
                self._animal_metadata_refresh_deferred = True
                self._animal_metadata_refresh_reason = str(reason)
                logger.info("SoftMouse refresh coalesced while another refresh is active")
                return False
            self._animal_metadata_refresh_busy = True
            self._animal_metadata_refresh_deferred = False
            self._animal_metadata_refresh_reason = str(reason)
        self._on_property_changed(
            self.Props.ANIMAL_METADATA_REFRESH_BUSY, True, False,
        )
        logger.info(
            "SoftMouse local-cache refresh requested: manifest=%s reason=%s",
            service.manifest_path,
            reason,
        )
        self._set_animal_metadata_status("Refreshing…")
        self._set_subsystem_status(
            SubsystemId.ANIMAL_REGISTRY,
            SubsystemState.STARTING,
            reason="refreshing SoftMouse cache",
            required_for_recording=False,
        )
        thread = threading.Thread(
            target=self._run_animal_metadata_refresh,
            args=(service, bool(only_if_due)),
            name="softmouse-cache-refresh",
            daemon=True,
        )
        self._animal_metadata_refresh_thread = thread
        thread.start()
        return True

    def _run_animal_metadata_refresh(self, service, only_if_due: bool) -> None:
        try:
            if only_if_due and not service.refresh_due():
                self._finish_animal_metadata_refresh(None, None, skipped=True)
                return
            result = service.refresh_now()
        except Exception as exc:
            self._finish_animal_metadata_refresh(None, exc)
            return
        self._finish_animal_metadata_refresh(result, None)

    def _finish_animal_metadata_refresh(
        self, result, error: Optional[BaseException], *, skipped: bool = False,
    ) -> None:
        if error is not None:
            logger.error(
                "SoftMouse local-cache refresh failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            self._set_animal_metadata_status(f"Refresh failed: {error}")
            self._set_subsystem_status(
                SubsystemId.ANIMAL_REGISTRY,
                SubsystemState.FAILED,
                error=str(error),
                required_for_recording=False,
            )
        elif skipped:
            self._set_animal_metadata_status("Already current; refresh not due")
            self._set_subsystem_status(
                SubsystemId.ANIMAL_REGISTRY,
                SubsystemState.READY,
                reason="local cache refresh not due",
                required_for_recording=False,
            )
        else:
            self._publish_animal_metadata_refresh_result(result)
        with self._animal_metadata_refresh_lock:
            self._animal_metadata_refresh_busy = False
            rerun = self._animal_metadata_refresh_deferred
            reason = self._animal_metadata_refresh_reason
            self._animal_metadata_refresh_deferred = False
        self._on_property_changed(
            self.Props.ANIMAL_METADATA_REFRESH_BUSY, False, True,
        )
        if (
            rerun
            and not self._closing_event.is_set()
            and self.session_recording_status is SessionRecordingStatus.READY
        ):
            self.request_animal_metadata_refresh(reason or "coalesced refresh")

    def _publish_animal_metadata_refresh_result(self, result) -> None:
        batch = result.preview.batch
        status = (
            f"Imported {batch.accepted_rows} tagged animals; "
            f"ignored {batch.ignored_missing_rfid_rows} without RFID and "
            f"{batch.ignored_ended_rows} ended"
        )
        if result.registry_result.unchanged:
            status = "Already current; " + status
        status += (
            f"; source {batch.source_file_sha256[:12]}; "
            f"refreshed {batch.imported_utc}"
        )
        self._set_animal_metadata_status(status)
        previous_preview, self._animal_metadata_preview = (
            self._animal_metadata_preview,
            result.preview,
        )
        self._on_property_changed(
            self.Props.ANIMAL_METADATA_PREVIEW,
            result.preview,
            previous_preview,
        )
        self._set_subsystem_status(
            SubsystemId.ANIMAL_REGISTRY,
            SubsystemState.READY,
            reason=status,
            required_for_recording=False,
        )
        logger.info(
            "SoftMouse local-cache refresh complete: import_id=%s total_rows=%d "
            "tagged_rows=%d ignored_missing_rfid=%d ignored_ended=%d unchanged=%s "
            "source_sha256=%s",
            batch.import_id,
            batch.total_source_rows,
            batch.accepted_rows,
            batch.ignored_missing_rfid_rows,
            batch.ignored_ended_rows,
            result.registry_result.unchanged,
            batch.source_file_sha256,
        )

    def _run_animal_metadata_catchup(self) -> None:
        try:
            self.request_animal_metadata_refresh(
                "scheduled catch-up", only_if_due=True,
            )
        except Exception as exc:
            logger.warning("Scheduled SoftMouse catch-up refresh failed: %s", exc)

    def _request_animal_metadata_catchup(self) -> None:
        self._run_animal_metadata_catchup()

    def apply_animal_metadata_preferences(self) -> None:
        if self.animal_metadata_refresh_busy:
            raise RuntimeError("SoftMouse settings cannot change during a cache refresh")
        self._configure_animal_metadata_services()
        if self._rfid_metadata_controller is not None:
            self._rfid_metadata_controller.start()

    def current_external_animal_records(self):
        if self._animal_registry is None:
            return ()
        return tuple(self._animal_registry.list_current_records())

    def _current_registry_provenance(self):
        ledger = (
            None
            if self._animal_registry is None
            else self._animal_registry.last_complete_import()
        )
        return {
            "registry_import_id": None if ledger is None else ledger["import_id"],
            "source_file_sha256": (
                None if ledger is None else ledger["source_file_sha256"]
            ),
            "imported_utc": None if ledger is None else ledger["imported_utc"],
        }

    def manually_link_animal(
        self, animal_id: str, record: ExternalAnimalRecord
    ) -> AnimalSubject:
        logger.info(
            "Manual SoftMouse link requested: animal_id=%s subject_id=%s rfid=%s",
            animal_id,
            record.identity.subject_id,
            record.physical_rfid,
        )
        animal = self.get_animal_by_id(animal_id)
        if animal is None:
            raise ValueError(f"Unknown reachAQ animal UUID {animal_id!r}")
        if animal.external_identity is not None:
            raise ValueError("Manual linking requires an unlinked animal JSON")
        self.link_animal_to_external_record(
            animal,
            record,
            **self._current_registry_provenance(),
        )
        self.animals = list(self._animals)
        self.selected_animal = animal
        return animal

    def update_animal_details(
        self,
        animal_id: str,
        *,
        name: str,
        notes: str,
    ) -> AnimalSubject:
        if self.session_recording_status is not SessionRecordingStatus.READY:
            raise RuntimeError("Animal details cannot change while a session is active")
        animal = self.get_animal_by_id(animal_id)
        if animal is None:
            raise ValueError(f"Unknown reachAQ animal UUID {animal_id!r}")
        name = str(name).strip()
        if not name:
            raise ValueError("Animal display name cannot be empty")
        previous_name, previous_notes = animal.name, animal.notes
        animal.name = name
        animal.notes = str(notes or "").strip()
        try:
            self._save_animal_metadata(animal, sender="animal-details")
        except Exception:
            animal.name = previous_name
            animal.notes = previous_notes
            raise
        self.animals = list(self._animals)
        logger.info(
            "Animal details saved: animal_id=%s animal_name=%s notes_present=%s",
            animal.id,
            animal.name,
            bool(animal.notes),
        )
        return animal

    def _rebuild_animal_indexes(self) -> None:
        self._animal_by_id = {animal.id: animal for animal in self._animals}
        self._animal_by_external_key = {}
        self._external_link_conflicts = set()
        for animal in self._animals:
            identity = animal.external_identity
            if identity is None:
                continue
            previous = self._animal_by_external_key.get(identity.key)
            if previous is not None and previous.id != animal.id:
                self._external_link_conflicts.add(identity.key)
                self._animal_by_external_key.pop(identity.key, None)
            elif identity.key not in self._external_link_conflicts:
                self._animal_by_external_key[identity.key] = animal

    def link_animal_to_external_record(
        self,
        animal: AnimalSubject,
        record: ExternalAnimalRecord,
        *,
        registry_import_id: Optional[str] = None,
        source_file_sha256: Optional[str] = None,
        imported_utc: Optional[str] = None,
    ) -> None:
        logger.info(
            "SoftMouse link requested: animal_id=%s animal_name=%s subject_id=%s "
            "rfid=%s import_id=%s",
            animal.id,
            animal.name,
            record.identity.subject_id,
            record.physical_rfid,
            registry_import_id,
        )
        if self.session_recording_status is not SessionRecordingStatus.READY:
            raise RuntimeError("Animal links cannot change while a session is active")
        linked = self.get_animal_by_external_identity(record.identity)
        if linked is not None and linked.id != animal.id:
            raise ValueError(
                f"{record.identity.subject_id!r} is already linked to {linked.name!r}"
            )
        old_identity = animal.external_identity
        old_metadata = animal.external_metadata
        animal.external_identity = record.identity
        animal.external_metadata = record.metadata_snapshot(
            registry_import_id=registry_import_id,
            source_file_sha256=source_file_sha256,
            imported_utc=imported_utc,
        )
        try:
            self._save_animal_metadata(animal, sender="external-link")
        except Exception:
            animal.external_identity = old_identity
            animal.external_metadata = old_metadata
            self._rebuild_animal_indexes()
            logger.exception(
                "SoftMouse link save failed and was rolled back: animal_id=%s "
                "subject_id=%s rfid=%s",
                animal.id,
                record.identity.subject_id,
                record.physical_rfid,
            )
            raise
        self._rebuild_animal_indexes()
        logger.info(
            "SoftMouse link saved: animal_id=%s animal_name=%s subject_id=%s rfid=%s",
            animal.id,
            animal.name,
            record.identity.subject_id,
            record.physical_rfid,
        )

    def resolve_external_record(
        self,
        record: Optional[ExternalAnimalRecord],
        *,
        scanned_rfid: str,
        registry_import_id: Optional[str] = None,
        source_file_sha256: Optional[str] = None,
        imported_utc: Optional[str] = None,
        cache_stale: bool = False,
    ) -> RfidResolution:
        if self.session_recording_status is not SessionRecordingStatus.READY:
            return RfidResolution(
                RfidResolutionKind.BUSY,
                scanned_rfid,
                record=record,
                message="RFID selection is disabled while a session is active",
            )
        if record is None:
            return RfidResolution(
                RfidResolutionKind.UNKNOWN_RFID,
                scanned_rfid,
                message=(
                    "RFID is unknown and the local cache is stale; refresh before linking"
                    if cache_stale
                    else "RFID is not present in the current local cache"
                ),
            )
        if record.identity.key in self._external_link_conflicts:
            return RfidResolution(
                RfidResolutionKind.LINK_CONFLICT,
                scanned_rfid,
                record=record,
                message="Multiple local animals have this external identity",
            )
        animal = self.get_animal_by_external_identity(record.identity)
        if animal is None:
            return RfidResolution(
                RfidResolutionKind.SETUP_REQUIRED,
                scanned_rfid,
                record=record,
                message=(
                    "Create a new animal JSON or link this RFID to an existing "
                    "unlinked animal"
                ),
            )
        self.link_animal_to_external_record(
            animal,
            record,
            registry_import_id=registry_import_id,
            source_file_sha256=source_file_sha256,
            imported_utc=imported_utc,
        )
        self.selected_animal = animal
        return RfidResolution(
            RfidResolutionKind.SELECTED,
            scanned_rfid,
            record=record,
            animal=animal,
            message=("Selected using a stale local cache" if cache_stale else ""),
        )

    def complete_rfid_animal_setup(
        self,
        record: ExternalAnimalRecord,
        *,
        scanned_rfid: str,
        name: str,
        notes: str,
        existing_animal_id: Optional[str] = None,
    ) -> RfidResolution:
        if self.session_recording_status is not SessionRecordingStatus.READY:
            raise RuntimeError("RFID setup is disabled while a session is active")
        if record.physical_rfid != scanned_rfid:
            raise ValueError("Scanned RFID does not match the selected SoftMouse record")
        name = str(name).strip()
        if not name:
            raise ValueError("Animal display name cannot be empty")
        notes = str(notes or "").strip()
        created = existing_animal_id is None
        if created:
            animal = AnimalSubject(name=name, notes=notes)
            self._animals.append(animal)
            self._rebuild_animal_indexes()
            previous_name = previous_notes = None
        else:
            animal = self.get_animal_by_id(existing_animal_id)
            if animal is None:
                raise ValueError(
                    f"Unknown reachAQ animal UUID {existing_animal_id!r}"
                )
            if animal.external_identity is not None:
                raise ValueError("RFID setup can only link an unlinked animal JSON")
            previous_name, previous_notes = animal.name, animal.notes
            animal.name = name
            animal.notes = notes
        try:
            self.link_animal_to_external_record(
                animal,
                record,
                **self._current_registry_provenance(),
            )
        except Exception:
            if created:
                self._animals.remove(animal)
                self._rebuild_animal_indexes()
            else:
                animal.name = previous_name
                animal.notes = previous_notes
            raise
        if created:
            self.animals = list(self._animals)
            self._event_manager.post_event_content(
                ApiEventKind.animalCreated, animal.to_api_status()
            )
        else:
            self.animals = list(self._animals)
        self.selected_animal = animal
        result = RfidResolution(
            (
                RfidResolutionKind.CREATED_AND_SELECTED
                if created
                else RfidResolutionKind.LINKED_AND_SELECTED
            ),
            scanned_rfid,
            record=record,
            animal=animal,
        )
        self._publish_rfid_resolution(result)
        return result

    @property
    def rfid_reader_status(self):
        return self._rfid_reader_status

    @property
    def rfid_scan_result(self):
        return self._rfid_scan_result

    def set_rfid_reader_status(self, status) -> None:
        previous, self._rfid_reader_status = self._rfid_reader_status, status
        self._on_property_changed(self.Props.RFID_READER_STATUS, status, previous)
        state_value = getattr(getattr(status, "state", None), "value", "failed")
        logger.info(
            "RFID reader state applied: state=%s device=%s reason=%s",
            state_value,
            getattr(status, "device", "unknown"),
            getattr(status, "reason", "") or "none",
        )
        state = {
            "ready": SubsystemState.READY,
            "connecting": SubsystemState.STARTING,
            "reconnecting": SubsystemState.STARTING,
            "stopped": SubsystemState.STOPPED,
            "failed": SubsystemState.FAILED,
        }.get(state_value, SubsystemState.FAILED)
        self._set_subsystem_status(
            SubsystemId.RFID_READER,
            state,
            reason=getattr(status, "reason", "") or state_value,
            error=(getattr(status, "reason", "") if state is SubsystemState.FAILED else ""),
            required_for_recording=False,
        )

    def handle_rfid_scan(self, record, **kwargs) -> RfidResolution:
        scanned_rfid = kwargs.get("scanned_rfid", "unknown")
        logger.info(
            "RFID scan resolution started: rfid=%s matched_softmouse=%s",
            scanned_rfid,
            record is not None,
        )
        result = self.resolve_external_record(record, **kwargs)
        self._publish_rfid_resolution(result)
        return result

    def _publish_rfid_resolution(self, result: RfidResolution) -> None:
        previous, self._rfid_scan_result = self._rfid_scan_result, result
        self._on_property_changed(self.Props.RFID_SCAN_RESULT, result, previous)
        resolution_logger = (
            logger.info
            if result.kind in {
                RfidResolutionKind.SELECTED,
                RfidResolutionKind.CREATED_AND_SELECTED,
                RfidResolutionKind.LINKED_AND_SELECTED,
                RfidResolutionKind.SETUP_REQUIRED,
            }
            else logger.warning
        )
        resolution_logger(
            "RFID scan resolution complete: rfid=%s result=%s subject_id=%s "
            "animal_id=%s animal_name=%s message=%s",
            result.rfid,
            result.kind.value,
            None if result.record is None else result.record.identity.subject_id,
            None if result.animal is None else result.animal.id,
            None if result.animal is None else result.animal.name,
            result.message or "none",
        )

    def condense_animal_duplicate(
        self,
        survivor_id: str,
        linked_duplicate_id: str,
    ) -> AnimalSubject:
        survivor = self.get_animal_by_id(survivor_id)
        duplicate = self.get_animal_by_id(linked_duplicate_id)
        if survivor is None or duplicate is None:
            raise ValueError("Both animal JSON records must still exist")
        if duplicate.external_identity is None:
            raise ValueError("The duplicate animal has no SoftMouse/RFID link")
        if (
            survivor.external_identity is not None
            and survivor.external_identity.key != duplicate.external_identity.key
        ):
            raise ValueError(
                "The surviving animal is already linked to a different SoftMouse animal"
            )
        merged = self.condense_animals(
            survivor_id,
            linked_duplicate_id,
            AnimalReconciliationChoices(
                name_from=survivor_id,
                pellet_position_from=survivor_id,
                training_from=survivor_id,
                target_limit_from=survivor_id,
                external_identity_from=linked_duplicate_id,
            ),
        )
        self.selected_animal = merged
        return merged

    def condense_animals(
        self,
        survivor_id: str,
        loser_id: str,
        choices: AnimalReconciliationChoices,
    ) -> AnimalSubject:
        """Condense two local records with explicit field-group decisions.

        Both original JSON files are preserved as backups and the losing UUID
        receives a redirect record. Session directories are never inspected or
        rewritten.
        """
        logger.info(
            "Animal reconciliation requested: survivor_id=%s loser_id=%s choices=%s",
            survivor_id,
            loser_id,
            choices.to_dict(),
        )
        if self.session_recording_status is not SessionRecordingStatus.READY:
            raise RuntimeError("Animals cannot be reconciled while a session is active")
        survivor = self.get_animal_by_id(survivor_id)
        loser = self.get_animal_by_id(loser_id)
        if survivor is None or loser is None:
            raise ValueError("Both survivor and losing UUIDs must be current animals")
        merged = reconcile_animals(survivor, loser, choices)
        animal_directory = Path(self._preferences.animal_location)
        animal_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        survivor_path = animal_directory / f"{survivor.id}.json"
        loser_path = animal_directory / f"{loser.id}.json"
        if not survivor_path.is_file() or not loser_path.is_file():
            raise FileNotFoundError(
                "Both animal JSON files must exist before they can be condensed"
            )

        backups = {}
        for label, path in (("survivor", survivor_path), ("loser", loser_path)):
            backup = animal_directory / (
                f"{path.stem}.{timestamp}.{label}.merge-backup"
            )
            shutil.copy2(path, backup)
            backups[path] = backup

        selected_participant = (
            self._selected_animal
            if self._selected_animal is not None
            and self._selected_animal.id in {
                survivor.id,
                loser.id,
            }
            else None
        )
        if selected_participant is not None:
            # Deselect before moving the losing file: the selection setter saves
            # the outgoing animal and would otherwise recreate that file.
            self.selected_animal = None

        archived_loser = animal_directory / (
            f"{loser.id}.{timestamp}.merged-animal-backup"
        )
        redirect_path = animal_directory / f"{loser.id}.merged-redirect"
        if redirect_path.exists():
            if selected_participant is not None:
                self.selected_animal = selected_participant
            raise FileExistsError(
                f"A merge redirect already exists for losing UUID {loser.id}"
            )
        staged_survivor = survivor_path.with_name(
            f".{survivor_path.name}.{timestamp}.merge-stage"
        )
        temporary_redirect = redirect_path.with_name(
            f".{redirect_path.name}.{timestamp}.tmp"
        )
        redirect = {
            "schemaVersion": 1,
            "mergedUtc": timestamp,
            "losingReachaqId": loser.id,
            "survivingReachaqId": survivor.id,
            "fieldSources": choices.to_dict(),
        }
        merged.to_file(staged_survivor)
        temporary_redirect.write_text(
            json.dumps(redirect, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        try:
            os.replace(staged_survivor, survivor_path)
            os.replace(loser_path, archived_loser)
            os.replace(temporary_redirect, redirect_path)
        except Exception:
            # Restore both authoritative JSON files from the pre-merge copies.
            # Each restore is itself staged so readers never see partial JSON.
            for original, backup in backups.items():
                restore_stage = original.with_name(
                    f".{original.name}.{timestamp}.restore-stage"
                )
                shutil.copy2(backup, restore_stage)
                os.replace(restore_stage, original)
            redirect_path.unlink(missing_ok=True)
            staged_survivor.unlink(missing_ok=True)
            temporary_redirect.unlink(missing_ok=True)
            if selected_participant is not None:
                self.selected_animal = selected_participant
            logger.exception(
                "Animal reconciliation failed and was rolled back: survivor_id=%s "
                "loser_id=%s",
                survivor_id,
                loser_id,
            )
            raise

        self.animals = [
            merged if animal.id == survivor.id else animal
            for animal in self._animals
            if animal.id != loser.id
        ]
        if selected_participant is not None:
            self.selected_animal = merged
        logger.info(
            "Animal reconciliation complete: survivor_id=%s loser_id=%s "
            "archived_loser=%s redirect=%s",
            survivor_id,
            loser_id,
            archived_loser,
            redirect_path,
        )
        return merged

    @property
    def selected_animal(self) -> Optional[AnimalSubject]:
        return self._selected_animal

    @selected_animal.setter
    def selected_animal(self, animal: Optional[AnimalSubject]):
        prev, self._selected_animal = self._selected_animal, animal
        if animal == prev:
            return
        self._detach_training_plan()  # always
        logger.info(
            "Animal selection changed: animal_id=%s animal_name=%s previous_id=%s "
            "previous_name=%s",
            None if animal is None else animal.id,
            None if animal is None else animal.name,
            None if prev is None else prev.id,
            None if prev is None else prev.name,
        )
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
        self._preferences.selected_animal = "" if animal is None else animal.id
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

    def _require_session_ready_for_configuration(self, action: str) -> None:
        status = self._recording_session.status
        if status is not SessionRecordingStatus.READY:
            raise RuntimeError(
                f"{action} is unavailable while session state is {status.value}"
            )

    @output_location.setter
    def output_location(self, value: str):
        self._require_session_ready_for_configuration("Changing the output location")
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

    def persist_stopped_session_notes(self) -> bool:
        """Atomically update notes for the most recently stopped session."""
        project = self._editable_notes_project
        if project is None:
            return False
        try:
            self._save_project_metadata(
                project,
                caller="session_notes_updated",
            )
        except Exception as exc:
            logger.exception("Failed to update stopped-session notes")
            self.on_error("Session notes were not saved", str(exc))
            return False
        return True

    @property
    def trial_protocol_rows(self) -> Tuple[dict, ...]:
        return self._trial_protocol_schedule.to_records()

    @property
    def trial_protocol_state(self) -> dict:
        ledger = self._trial_ledger
        attempts = () if ledger is None else ledger.attempts
        active = None if ledger is None else ledger.active_attempt
        active_trial_id = None if active is None else active.trial_id
        session_control = self._behavior.algorithm.active_config.session_control
        estimate = self._intertrial_analysis.timing_estimate(1.0)
        if active is not None and active_trial_id is None and active.protocol_context:
            active_trial_id = (
                active.protocol_context.get("trial_row") or {}
            ).get("trial_id")
        return {
            "rows": self.trial_protocol_rows,
            "active_trial_id": active_trial_id,
            "completed_trial_ids": tuple(sorted({
                attempt.trial_id
                for attempt in attempts
                if attempt.trial_id is not None and attempt.logical_trial_complete
            })),
            "analysis": {
                "enabled": session_control.intertrial_analysis_enabled,
                "progression_mode": session_control.intertrial_progression_mode,
                "pending_attempts": (
                    0
                    if ledger is None
                    else ledger.summary().get("pending_analysis_attempts", 0)
                ),
                "send_block_reason": self._behavior.algorithm.pellet_send_block_reason,
                "resolution_required": self._intertrial_resolution_request is not None,
                "resolution_reason": self._intertrial_resolution_reason,
                "seconds_per_tracking_second": estimate.mean_realtime_factor,
                "estimate": estimate.display_text,
            },
        }

    def update_trial_protocol_row(self, trial_id: int, field: str, value) -> bool:
        state = self.trial_protocol_state
        trial_id = int(trial_id)
        if trial_id == state["active_trial_id"] or trial_id in state["completed_trial_ids"]:
            return False
        self._trial_protocol_schedule.update(trial_id, field, value)
        self._notify_trial_protocol_state()

    def set_intertrial_analysis_enabled(self, enabled: bool) -> None:
        control = self._behavior.algorithm.active_config.session_control
        control.intertrial_analysis_enabled = bool(enabled)
        if not enabled:
            control.behavioral_retry_outcomes = ()
        self._notify_trial_protocol_state()
        return True

    def retry_pending_intertrial_analysis(self) -> bool:
        request = self._intertrial_resolution_request
        if (
            request is None
            or self._recording_session.status is not SessionRecordingStatus.RECORDING
        ):
            return False
        token = self._recording_session.token()
        if (
            token is None
            or request.generation != token.generation
            or request.session_id != token.session_id
            or not self._intertrial_analysis.submit(request)
        ):
            return False
        self._intertrial_resolution_request = None
        self._intertrial_resolution_reason = ""
        estimate = self._intertrial_analysis.timing_estimate(
            request.window.end_perf - request.window.start_perf
        )
        self._behavior.algorithm.pellet_send_block_reason = (
            "waiting for retried pellet trial analysis; " + estimate.display_text
        )
        self._notify_trial_protocol_state()
        return True

    def continue_without_pending_intertrial_result(self) -> bool:
        request = self._intertrial_resolution_request
        if (
            request is None
            or self._recording_session.status is not SessionRecordingStatus.RECORDING
        ):
            return False
        ledger = self._trial_ledger
        attempt = next(
            (
                item
                for item in (() if ledger is None else ledger.attempts)
                if item.trial_id == request.trial_id
                and item.attempt_id == request.attempt_id
                and item.operation_id == request.operation_id
            ),
            None,
        )
        if attempt is None or attempt.outcome is not TrialOutcome.PENDING_ANALYSIS:
            return False
        tone_references, laser_references = (
            self._session_data_recorder.trial_stream_references(
                request.window.start_perf,
                request.window.end_perf,
            )
        )
        self._pellet_cycles.finalize_intertrial_unavailable(
            self._project_info,
            attempt,
            perf_time=request.window.end_perf,
            wall_time=attempt.capture_end_wall_time or time.time(),
            reason=(
                "operator continued without an analysis result; behavioral retry "
                "decision was skipped"
            ),
            pellet_presence=request.pellet_state.presence.value,
            pellet_misplacement=request.pellet_state.misplacement.value,
            window=request.window,
            tone_references=tone_references,
            laser_references=laser_references,
        )
        self._intertrial_resolution_request = None
        self._intertrial_resolution_reason = ""
        self._behavior.algorithm.pellet_send_block_reason = ""
        self._pellet_cycles.sync_behavior_counts(self._behavior.algorithm)
        self._notify_trial_protocol_state()
        self._evaluate_automatic_stop_policy(
            protocol_complete=self._protocol_runner.protocol_complete,
        )
        return True

    def _hold_intertrial_resolution(
        self,
        request: IntertrialAnalysisRequest,
        reason: str,
    ) -> None:
        self._intertrial_resolution_request = request
        self._intertrial_resolution_reason = str(reason)
        self._behavior.algorithm.pellet_send_block_reason = (
            "pellet trial analysis needs operator action"
        )
        logger.error("%s", reason)
        self._notify_trial_protocol_state()

    def _notify_trial_protocol_state(self) -> None:
        self.property_changed(
            self.Props.TRIAL_PROTOCOL_STATE,
            self.trial_protocol_state,
            None,
        )

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
        return self._coordinates.enabled

    @check_diamond_coord_enabled.setter
    def check_diamond_coord_enabled(self, value):
        self._coordinates.enabled = bool(value)

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
        prepared[primary] = self._acquisition.subsystems.get(
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
                prepared[secondary] = self._acquisition.subsystems.get(
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
                status := self._acquisition.subsystems.get(
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
                timing_plan=(
                    self._nidaq_signal_monitor.timing_plan
                    if configuration.hardware_timed
                    else None
                ),
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
            session_config = self._behavior.algorithm.active_config.session_control
            preflight = preflight_storage(
                output_location,
                estimated_bytes_per_second=self._estimate_session_bytes_per_second(),
                configured_duration_seconds=session_config.duration_limit_seconds,
            )
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
            reason=(
                f"session output durable-write test passed: {output_location}; "
                f"free={preflight.free_bytes} bytes; projected maximum="
                f"{preflight.projected_maximum_minutes:.1f} minutes"
            ),
            generation=generation,
        )
        return True

    def _log_acquisition_startup_summary(self) -> None:
        statuses = self._acquisition.subsystems.statuses
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
        if not self._acquisition.started:
            raise RuntimeError("Subsystem retry requires System Mode to be running")
        if self._recording_session.status is not SessionRecordingStatus.READY:
            raise RuntimeError("Subsystem retry is unavailable during recording or analysis")

        failed = {
            subsystem_id
            for subsystem_id, status in self._acquisition.subsystems.statuses.items()
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

        synchronization = self._acquisition.subsystems.get(
            SubsystemId.REACH_SYNCHRONIZATION
        )
        inference = self._acquisition.subsystems.get(SubsystemId.LIVE_INFERENCE)
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
        remaining = self._acquisition.subsystems.recording_blockers()
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
        primary_status = self._acquisition.subsystems.get(
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
            status = self._acquisition.subsystems.get(
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
            camera_generation = self._acquisition.subsystems.get(
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
                status := self._acquisition.subsystems.get(
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
            if self._acquisition.started:
                if self.is_target_status_valid(target_status):
                    self.status = target_status
                    return True
                self.on_error("AppModelStatus change error",
                              f"Target status {target_status} not valid for source status {before_status}")
                return False
            if not self._acquisition.begin_start():
                logger.warning("Acquisition already starting")
                return False
            previous_internal_error = self._internal_error_diagnostic
            self._internal_error_diagnostic = None
            self._session_invariant_unknown = False
            self._start_count += 1
            is_first_start = self._start_count == 1
        if previous_internal_error is not None:
            self._on_property_changed(
                self.Props.INTERNAL_ERROR_DIAGNOSTIC,
                None,
                previous_internal_error,
            )
            self._set_subsystem_status(
                SubsystemId.RUNTIME_DIAGNOSTICS,
                SubsystemState.READY,
                reason="internal-error diagnostic cleared by acquisition restart",
                required_for_recording=False,
            )

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
            status = self._acquisition.subsystems.get(
                SubsystemId.camera(cam.name)
            )
            if status is not None and status.is_ready:
                watchdog_mon_register(f"camera.{cam.name}", lambda cam=cam: cam.watchdog_capture_perf_c)

        if __debug__ and os.getenv("_AUTOTRAINER_TEST_WATCHDOG") == "1":
            def fake_watchdog(t_end):
                return t_end
            watchdog_mon_register("test-watchdog", lambda t=time.perf_counter() + 180: fake_watchdog(t))

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

        self._acquisition.mark_started()
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
            if not self._acquisition.begin_stop(force=force):
                logger.verbose("acquisition not running or already stopping")
                return
            before_status = self._status
            recording_status = self._recording_session.status
            session_token = self._recording_session.token()
        if recording_status in {
            SessionRecordingStatus.ARMING,
            SessionRecordingStatus.RECORDING,
        }:
            logger.warning("Acquisition stop requested during recording; aborting the session")
            self.abort_recording(token=session_token)
        # always remove status-file on stop:
        status_file_path = self.status_file_path.expanduser()
        status_file_path.unlink(missing_ok=True)
        try:
            self._capture_stop()
        finally:
            # always:
            self._nidaq_signal_monitor.stop()
            # must be set before try reload training plans, given checked in it
            self._acquisition.mark_stopped()
            if self._recording_session.status == SessionRecordingStatus.ABORTING:
                self._finish_abort_recording(token=session_token)
            elif self._recording_session.status != SessionRecordingStatus.READY:
                self._session_data_recorder.abort()
                self._set_session_recording_status(
                    SessionRecordingStatus.READY,
                    token=session_token,
                )
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

        can_status = self._acquisition.subsystems.get(SubsystemId.CAN_PELLET)
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
        self._require_session_ready_for_configuration("Loading configuration")
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
            "CONFIGURED | hardware flags | CAN=%s pellet_controller=%s NI-DAQ=%s RFID=%s",
            configuration.hardware.can_enabled,
            configuration.hardware.pellet_controller_enabled,
            configuration.hardware.nidaq_enabled,
            configuration.hardware.rfid_reader_enabled,
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
        self._configure_animal_metadata_services()

        self.configuration_loaded_event(configuration)

        log_hardware_initialization(
            logger,
            "READY | hardware configuration | elapsed=%.3fs",
            time.perf_counter() - config_started,
        )

        return True

    def reload_training_plans(self, *, refresh: bool = False, reraise_on_error: bool = False):
        if self._acquisition.started or self._status != AppModelStatus.IDLE:
            logger.notice("delaying reload training plans given acquisition started(%s) or status not idle: %s",
                          self._acquisition.started, self._status)
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
        self._require_session_ready_for_configuration(
            "Changing the DAQ port configuration"
        )
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

    def update_hardware_configuration(
        self,
        *,
        can_enabled: bool,
        pellet_controller_enabled: bool,
        nidaq_enabled: bool,
        rfid_reader_enabled: bool,
        rfid_device: str,
    ) -> str:
        """Apply and persist rig-level hardware availability settings."""
        if self._loaded_configuration is None:
            raise RuntimeError("Cannot update hardware settings before configuration is loaded")
        if self._acquisition.started or self._status != AppModelStatus.IDLE:
            raise RuntimeError("Hardware settings can only be changed while acquisition is idle")

        rfid_device = str(rfid_device).strip()
        if rfid_reader_enabled and not rfid_device:
            raise ValueError("RFID serial device is required when the reader is enabled")

        hardware_configuration = self._loaded_configuration.hardware
        previous_settings = {
            "can_enabled": hardware_configuration.can_enabled,
            "pellet_controller_enabled": hardware_configuration.pellet_controller_enabled,
            "nidaq_enabled": hardware_configuration.nidaq_enabled,
            "rfid_reader_enabled": hardware_configuration.rfid_reader_enabled,
            "rfid_device": hardware_configuration.rfid_device,
        }
        requested_settings = {
            "can_enabled": bool(can_enabled),
            "pellet_controller_enabled": bool(pellet_controller_enabled),
            "nidaq_enabled": bool(nidaq_enabled),
            "rfid_reader_enabled": bool(rfid_reader_enabled),
            "rfid_device": rfid_device,
        }
        logger.info(
            "Hardware configuration update requested: previous=%s requested=%s",
            previous_settings,
            requested_settings,
        )
        was_can_required = self._hardware.requires_connection
        will_require_can = bool(can_enabled and pellet_controller_enabled)
        if was_can_required and not will_require_can and self._hardware.connected:
            self._hardware.disconnect()

        hardware_configuration.can_enabled = bool(can_enabled)
        hardware_configuration.pellet_controller_enabled = bool(
            pellet_controller_enabled
        )
        hardware_configuration.nidaq_enabled = bool(nidaq_enabled)
        hardware_configuration.rfid_reader_enabled = bool(rfid_reader_enabled)
        hardware_configuration.rfid_device = rfid_device

        self._hardware.load_config(hardware_configuration)
        self._nidaq_signal_monitor.set_hardware_enabled(
            hardware_configuration.nidaq_enabled,
            auto_start=False,
        )
        self._configure_subsystem_intent(self._loaded_configuration)
        self._configure_animal_metadata_services()
        if self._rfid_metadata_controller is not None:
            self._rfid_metadata_controller.start()
        self.save_configuration()
        self.configuration_loaded_event(self._loaded_configuration)
        logger.info("Hardware configuration update applied: settings=%s", requested_settings)
        return "Hardware settings saved; refresh hardware to connect newly enabled devices"

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
        if self._rfid_metadata_controller is not None:
            self._rfid_metadata_controller.start()
        if getattr(self._preferences, "softmouse_nightly_refresh", False):
            self._request_animal_metadata_catchup()

    def _stop_periodic_timers(self):
        self._closing_event.set()
        for timer in (
                self._timer_one_minute_repeat,
                self._timer_daily,
        ):
            logger.debug("stopping timer %s", timer)
            timer.cancel()

    def _prepare_application_shutdown(self) -> None:
        self._stop_periodic_timers()

    def on_close(self):
        logger.debug("AppModel.on_close")
        # Stop command producers first so none can race with CAN teardown or
        # reschedule themselves after their current timer is cancelled.
        self._prepare_application_shutdown()
        self._storage_monitor.abort()
        self.persist_stopped_session_notes()

        if self._rfid_metadata_controller is not None:
            self._rfid_metadata_controller.stop()
        metadata_thread = self._animal_metadata_refresh_thread
        if metadata_thread is not None and metadata_thread.is_alive():
            metadata_thread.join(5)

        try:
            self._system_message_handler.decoded_message_received -= (
                self._on_intertrial_device_message
            )
        except (KeyError, ValueError):
            pass
        if not self._intertrial_analysis.close():
            logger.warning("Intertrial analysis worker did not stop cleanly")
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
        # Fatal callbacks are process-wide and may originate from cameras,
        # analysis, NI-DAQ, or UI code. They must never reset or disconnect an
        # otherwise healthy shared CAN interface.
        previous = self._internal_error_diagnostic
        if previous is None:
            token = self._recording_session.token()
            diagnostic = {
                "source": str(source),
                "exceptionType": exception.__class__.__name__,
                "message": str(exception) or exception.__class__.__name__,
                "wallTime": time.time(),
                "sessionStatus": self._recording_session.status.value,
                "sessionId": None if token is None else token.session_id,
                "sessionGeneration": None if token is None else token.generation,
                "operationChanged": False,
            }
            self._internal_error_diagnostic = diagnostic
            self._on_property_changed(
                self.Props.INTERNAL_ERROR_DIAGNOSTIC,
                dict(diagnostic),
                None,
            )
            self._set_subsystem_status(
                SubsystemId.RUNTIME_DIAGNOSTICS,
                SubsystemState.FAILED,
                error=(
                    f"{diagnostic['exceptionType']}: {diagnostic['message']} "
                    f"({diagnostic['source']})"
                ),
                required_for_recording=False,
            )
            self.on_error(
                "Internal software error",
                "The error was logged without stopping recording, moving hardware, "
                "resetting CAN, or disconnecting unrelated systems. Stop and Abort "
                "remain available.",
            )
        logger.error(
            "Fatal callback from %s left the active operation and CAN ownership "
            "unchanged: %s",
            source,
            exception,
        )

    def _load_animals(self):
        animals = []
        animals_dir_path = Path(self._preferences.animal_location)

        if animals_dir_path.is_dir():
            files = list(animals_dir_path.glob("*.json"))
            loaded = {}
            skipped = []
            for path in files:
                try:
                    animal = AnimalSubject.from_file(path)
                    if animal is None:
                        raise ValueError("unsupported or invalid animal schema")
                except Exception as error:
                    logger.exception("Skipping invalid animal file %s", path)
                    skipped.append((path.name, str(error) or error.__class__.__name__))
                    continue
                loaded[path] = animal
            animals: Dict[Path, AnimalSubject] = loaded
            if skipped:
                self.on_error(
                    "Some animal files were skipped",
                    "\n".join(f"{name}: {reason}" for name, reason in skipped),
                )

            loaded_by_path = animals
            paths_by_id = {}
            for source_path, animal in loaded_by_path.items():
                previous_path = paths_by_id.get(animal.id)
                if previous_path is not None:
                    raise ValueError(
                        "Duplicate reachAQ animal UUID "
                        f"{animal.id!r} in {previous_path.name!r} and "
                        f"{source_path.name!r}; remove or archive one copy"
                    )
                paths_by_id[animal.id] = source_path
            animals = sorted(loaded_by_path.values(), key=lambda a: a.name)

            # New persistence is UUID-addressed so duplicate display names are
            # safe. Preserve a recoverable copy of every legacy name-based file.
            for source_path, animal in loaded_by_path.items():
                destination = animals_dir_path / f"{animal.id}.json"
                if source_path == destination:
                    continue
                if destination.exists():
                    raise ValueError(
                        f"Cannot migrate {source_path.name!r}: canonical path "
                        f"{destination.name!r} already exists"
                    )
                backup = source_path.with_suffix(source_path.suffix + ".legacy-name-backup")
                if backup.exists():
                    backup = source_path.with_suffix(
                        source_path.suffix
                        + datetime.now(timezone.utc).strftime(
                            ".%Y%m%dT%H%M%S%fZ.legacy-name-backup"
                        )
                    )
                os.replace(source_path, backup)
                try:
                    animal.to_file(destination)
                except Exception:
                    os.replace(backup, source_path)
                    raise

        pref_animal = self._preferences.selected_animal
        self.animals = animals
        for animal in animals:
            if pref_animal in {animal.id, animal.name}:
                self.selected_animal = animal
                break

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
                if value in {animal.id, animal.name}:
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
            self._handle_recording_subsystem_failure(
                SubsystemId.camera(camera_name),
                root_error,
            )
        elif watchdog_key in {
            WatchdogItems.DEVICE_READER.value,
            WatchdogItems.DEVICE_WRITER.value,
        }:
            try:
                self._hardware.report_can_watchdog_failure(error)
            except Exception:
                logger.exception("Failed to start CAN recovery after watchdog timeout")
            self._analysis.watchdog_monitor.unregister_watchdog(watchdog_id)
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.FAILED,
                error=error,
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
            self._handle_recording_subsystem_failure(
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
            self._handle_recording_subsystem_failure(
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
            current = self._acquisition.subsystems.get(SubsystemId.LASER)
            if (
                self._acquisition.started
                and not self._acquisition.stopping
                and current is not None
                and current.state is SubsystemState.READY
            ):
                reason = "laser controller disconnected unexpectedly"
                self._set_subsystem_status(
                    SubsystemId.LASER,
                    SubsystemState.FAILED,
                    error=reason,
                )
                self._handle_recording_subsystem_failure(
                    SubsystemId.LASER,
                    reason,
                )
            else:
                self._set_subsystem_status(
                    SubsystemId.LASER,
                    SubsystemState.STOPPED,
                    reason="laser controller disconnected",
                )

    def _handle_recording_subsystem_failure(
        self,
        subsystem_id,
        reason: str,
    ) -> None:
        status = self._acquisition.subsystems.get(subsystem_id)
        recording_status = self._recording_session.status
        if status is None or recording_status not in {
            SessionRecordingStatus.ARMING,
            SessionRecordingStatus.RECORDING,
        }:
            return
        subsystem_key = status.subsystem_id
        if subsystem_key.startswith("camera."):
            camera_name = subsystem_key.split(".", 1)[1]
            camera = next(
                (item for item in self._reach_cameras if item.name == camera_name),
                None,
            )
            is_session_reference = (
                camera is not None
                and camera.camera_index == self._identify_session_reference_cam_idx()
            )
            if is_session_reference:
                if recording_status is SessionRecordingStatus.RECORDING:
                    logger.error(
                        "Primary camera failed during recording: %s; requesting "
                        "normal Stop and preserving partial data",
                        reason,
                    )
                    self._stop_recording(
                        RecordingEndingReason.REQUIRED_SOURCE_FAILURE,
                        token=self._recording_session.token(),
                    )
                else:
                    logger.error(
                        "Primary camera failed before the recording boundary: %s; "
                        "cleaning the empty armed session",
                        reason,
                    )
                    self.abort_recording(token=self._recording_session.token())
            else:
                logger.error(
                    "Secondary camera failed; healthy primary recording continues: %s",
                    reason,
                )
            return
        logger.error(
            "%s failed during recording; remaining streams and the active "
            "operation continue: %s",
            subsystem_key,
            reason,
        )

    def _on_intersession_property_changed(self, name, value, _):
        if name == IntersessionMachine.Properties.STATE_PROPERTY:
            self._update_status_text_overlay()

    def _on_session_starting_before_record_start(self):
        session_config = self._behavior.algorithm.active_config.session_control
        token = self._recording_session.token()
        if token is None:
            logger.warning(
                "Behavior session started outside the recording controller; "
                "intertrial results will not be applied"
            )
            analysis_generation = 0
            analysis_session_id = self._project_info.short_id
        else:
            analysis_generation = token.generation
            analysis_session_id = token.session_id
        self._live_tracking.clear()
        self._trial_window_start = None
        self._tone2_active = False
        self._intertrial_resolution_request = None
        self._intertrial_resolution_reason = ""
        self._behavior.algorithm.pellet_send_block_reason = ""
        self._intertrial_analysis.begin_session(
            analysis_generation,
            analysis_session_id,
        )
        self._pellet_cycles.start_session(
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
        self._notify_trial_protocol_state()
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
        session_token = self._recording_session.token()
        project_generation = int(getattr(project, "session_generation", 0))
        if session_token is not None and (
            project.short_id != session_token.session_id
            or project_generation not in (0, session_token.generation)
        ):
            logger.warning(
                "ignoring stale session-ending callback: project=%s "
                "generation=%s active=%s",
                project.short_id,
                project_generation,
                session_token,
            )
            return
        if self._recording_session.status not in {
            SessionRecordingStatus.STOPPING,
            SessionRecordingStatus.ANALYZING,
        }:
            logger.warning(
                "ignoring session-ending callback while state is %s: project=%s",
                self._recording_session.status.value,
                project.short_id,
            )
            return
        logger.info(
            "capture lifecycle ended for %s (%s); waiting for closed "
            "pellet-window analyses after writers close",
            project.short_id,
            result.value,
        )

    def _complete_stopped_recording(
        self,
        project: ProjectInfo,
        *,
        token: Optional[SessionGeneration] = None,
    ) -> None:
        token = token or self._recording_session.token()
        if token is not None and (
            not self._recording_session.is_current(
                token,
                statuses=(SessionRecordingStatus.STOPPING,),
            )
            or project.short_id != token.session_id
        ):
            logger.warning(
                "ignoring stale stopped-recording completion: project=%s active=%s",
                project.short_id,
                token,
            )
            return
        end_perf = (
            self._recording_session.pending_end_perf
            if token is None
            else self._recording_session.take_pending_end(token)
        )
        if token is None:
            self._recording_session.pending_end_perf = None
        if end_perf is None:
            end_perf = get_perf_now()
            logger.warning("Primary camera did not report a final recorded-frame timestamp")
        boundary = self._recording_session.boundary
        if boundary is None or boundary.session_id != project.short_id:
            raise RuntimeError(
                f"Missing canonical recording boundary for {project.short_id}"
            )
        completed_boundary = boundary.with_end(end_perf)
        if token is None:
            self._recording_session.boundary = completed_boundary
        elif not self._recording_session.set_boundary(completed_boundary, token):
            logger.warning("stopped-recording boundary became stale: %s", token)
            return
        project.start_record_timestamp = self._recording_session.boundary.start_wall_time
        storage_telemetry = self._storage_monitor.stop()
        current_storage = dict(self._recording_session.storage_telemetry)
        current_storage["recording"] = storage_telemetry
        self._recording_session.set_storage_telemetry(current_storage)
        try:
            self._pellet_cycles.snapshot_for_stop(
                end_perf,
                self._recording_session.boundary.end_wall_time,
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
                self._recording_session.boundary = (
                    self._recording_session.boundary.with_nidaq_sample_index(
                        matched_sample_index
                    )
                )
            if isinstance(stream_result, dict):
                self._recording_session.set_stream_result(stream_result)
                if not self._recording_session.data_complete:
                    message = (
                        "Session auxiliary data is incomplete: "
                        + "; ".join(self._recording_session.data_errors)
                    )
                    logger.error(message)
                    self.on_error("Session data incomplete", message)
        except Exception as err:
            logger.exception("Failed to save session auxiliary streams: %s", err)
            self.on_error("Session stream save failed", str(err))
            self._recording_session.add_data_error(
                f"auxiliary stream save failed: {err}"
            )
        self._request_session_end_home("stop")
        self._recording_session.analysis_started_perf = time.perf_counter()
        self._begin_subsystem_start(
            SubsystemId.INTERTRIAL_ANALYSIS,
            reason="finishing closed pellet-trial analyses",
        )
        transitioned = self._set_session_recording_status(
            SessionRecordingStatus.ANALYZING,
            expected=(SessionRecordingStatus.STOPPING,),
            token=token,
        )
        try:
            self._editable_notes_project = project.to_local_value()
            self._save_project_metadata(
                project,
                caller="raw_writers_closed",
            )
        except Exception as exc:
            self._recording_session.add_data_error(
                f"metadata save failed: {exc}"
            )
            self._set_subsystem_status(
                SubsystemId.INTERTRIAL_ANALYSIS,
                SubsystemState.FAILED,
                error=f"metadata save failed: {exc}",
            )
            raise
        if not transitioned:
            return
        if self._intertrial_analysis.is_idle():
            self._finish_intertrial_session(project, token)
        else:
            thread = threading.Thread(
                target=self._wait_and_finish_intertrial_session,
                args=(project.to_local_value(), token),
                name="FinishIntertrialSession",
                daemon=True,
            )
            self._intertrial_finalize_thread = thread
            thread.start()

    def _wait_and_finish_intertrial_session(
        self,
        project: ProjectInfo,
        token: Optional[SessionGeneration],
    ) -> None:
        while (
            token is None
            and self._recording_session.status is SessionRecordingStatus.ANALYZING
        ) or self._recording_session.is_current(
            token, statuses=(SessionRecordingStatus.ANALYZING,)
        ):
            if self._intertrial_analysis.wait_for_idle(0.25):
                self._finish_intertrial_session(project, token)
                return

    def _finish_intertrial_session(
        self,
        project: ProjectInfo,
        token: Optional[SessionGeneration],
    ) -> None:
        if not (
            (
                token is None
                and self._recording_session.status
                is SessionRecordingStatus.ANALYZING
            )
            or self._recording_session.is_current(
                token, statuses=(SessionRecordingStatus.ANALYZING,)
            )
        ):
            return
        requests, tracking_errors = (
            self._session_data_recorder.load_trial_tracking_requests(project)
        )
        for error in tracking_errors:
            logger.warning("Stored trial tracking could not be validated: %s", error)
        pending_identities = {
            (attempt.trial_id, attempt.attempt_id, attempt.operation_id)
            for attempt in (self._trial_ledger.attempts if self._trial_ledger else ())
            if attempt.outcome is TrialOutcome.PENDING_ANALYSIS
        }
        for request in requests:
            identity = (
                request.trial_id,
                request.attempt_id,
                request.operation_id,
            )
            if identity not in pending_identities:
                continue
            try:
                result = analyze_tracking_window(request)
                self._session_data_recorder.persist_trial_tracking(
                    project,
                    request,
                    result,
                )
                tone_references, laser_references = (
                    self._session_data_recorder.trial_stream_references(
                        request.window.start_perf,
                        request.window.end_perf,
                    )
                )
                self._pellet_cycles.finalize_intertrial_result(
                    project,
                    result,
                    retry=False,
                    tone_references=tone_references,
                    laser_references=laser_references,
                )
                logger.warning(
                    "Repaired pending trial %s from stored live tracking",
                    request.attempt_label,
                )
            except Exception as error:
                logger.exception(
                    "Stored tracking repair failed for %s: %s",
                    request.attempt_label,
                    error,
                )
        self._pellet_cycles.finalize_pending_without_analysis(
            project,
            perf_time=get_perf_now(),
            wall_time=time.time(),
            reason="intertrial analysis ended without a result",
        )
        self._pellet_cycles.sync_behavior_counts(self._behavior.algorithm)
        self._pellet_cycles.end_session()
        self._intertrial_resolution_request = None
        self._intertrial_resolution_reason = ""
        self._behavior.algorithm.pellet_send_block_reason = ""
        self._recording_session.analysis_finished = True
        self._recording_session.analysis_duration_seconds = (
            time.perf_counter()
            - (self._recording_session.analysis_started_perf or time.perf_counter())
        )
        try:
            self._save_project_metadata(
                project,
                caller="intertrial_analysis_finished",
            )
        except Exception as error:
            self._recording_session.add_data_error(
                f"final metadata save failed: {error}"
            )
            self._set_subsystem_status(
                SubsystemId.INTERTRIAL_ANALYSIS,
                SubsystemState.FAILED,
                error=f"final metadata save failed: {error}",
            )
            logger.exception("Final intertrial metadata save failed")
        else:
            self._set_subsystem_status(
                SubsystemId.INTERTRIAL_ANALYSIS,
                (
                    SubsystemState.READY
                    if self._recording_session.data_complete
                    else SubsystemState.FAILED
                ),
                reason=(
                    "all closed pellet-trial analyses completed"
                    if self._recording_session.data_complete
                    else "trial analysis completed; session data is incomplete"
                ),
                error=(
                    ""
                    if self._recording_session.data_complete
                    else "; ".join(self._recording_session.data_errors)
                ),
            )
        finally:
            self._set_session_recording_status(
                SessionRecordingStatus.READY,
                expected=(SessionRecordingStatus.ANALYZING,),
                token=token,
            )

    def _request_session_end_home(self, ending: str) -> None:
        requested_perf = get_perf_now()
        requested_wall = time.time()
        if not self._recording_session.reserve_end_action(
            "pellet_home",
            {
                "ending": str(ending),
                "requestedPerfTime": requested_perf,
                "requestedWallTime": requested_wall,
                "insideRecordedBoundary": False,
            },
        ):
            return
        status = self._acquisition.subsystems.get(SubsystemId.CAN_PELLET)
        if status is None or not status.is_ready or not self._hardware.connected:
            self._recording_session.finish_end_action(
                "pellet_home",
                status="skipped",
                error="pellet board was not Ready at session end",
            )
            logger.warning("Session-end pellet home skipped: pellet board not Ready")
            return
        try:
            token = self._hardware.send_home_and_wait(timeout=15.0)
        except Exception as error:
            message = str(error) or error.__class__.__name__
            logger.exception("Session-end pellet home failed")
            self._recording_session.finish_end_action(
                "pellet_home",
                status="failed",
                completedPerfTime=get_perf_now(),
                error=message,
            )
            self.on_error("Pellet home failed", message)
            return
        self._recording_session.finish_end_action(
            "pellet_home",
            status="completed",
            completedPerfTime=get_perf_now(),
            commandToken=str(token),
            error=None,
        )

    def _finish_abort_recording(
        self,
        *,
        token: Optional[SessionGeneration] = None,
    ) -> None:
        token = token or self._recording_session.token()
        if token is not None and not self._recording_session.is_current(
            token,
            statuses=(SessionRecordingStatus.ABORTING,),
        ):
            logger.warning("ignoring stale abort completion: %s", token)
            return
        self._record_start_timer.cancel()
        self._storage_monitor.abort()
        self._record_start_timer = no_op_timer
        self._abort_cleanup_timer.cancel()
        self._abort_cleanup_timer = no_op_timer
        project = self._aborting_project
        if project is None:
            logger.error("abort completed without an associated project")
            self._set_session_recording_status(
                SessionRecordingStatus.READY,
                expected=(SessionRecordingStatus.ABORTING,),
                token=token,
            )
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
            self._request_session_end_home("abort")
            self._pellet_cycles.abort(
                perf_time=get_perf_now(),
                wall_time=time.time(),
            )
            self._behavior.algorithm.reset_session_counts()
            self._session_data_recorder.abort()
            self._recording_session.reset_after_abort()
            self._set_subsystem_status(
                SubsystemId.INTERTRIAL_ANALYSIS,
                SubsystemState.DISABLED,
                reason="aborted session analysis was cancelled",
            )
            self._abort_had_recording_started = False
            self._aborting_project = None
            self._editable_notes_project = None
            self.notes = ""
            self._set_session_recording_status(
                SessionRecordingStatus.READY,
                expected=(SessionRecordingStatus.ABORTING,),
                token=token,
            )

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
        if name == hard.CAN_CONNECTION_STATE_PROPERTY:
            self._on_can_connection_state_changed(value)
        elif animal is not None and name in {hard.SET_X, hard.SET_Y, hard.SET_Z}:
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

    def _on_can_connection_state_changed(self, value) -> None:
        if not self._acquisition.started:
            return
        state = str((value or {}).get("state", "failed"))
        error = str((value or {}).get("error", ""))
        watchdog = self._analysis.watchdog_monitor
        if state in {"failed", "recovering", "connecting"}:
            watchdog.unregister_watchdog(WatchdogItems.DEVICE_READER)
            watchdog.unregister_watchdog(WatchdogItems.DEVICE_WRITER)
        if state == "ready":
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.READY,
                reason="CAN/pellet controller ready",
            )
            watchdog.register_watchdog(
                WatchdogItems.DEVICE_READER,
                lambda: self._hardware.watchdog_reader_perf_c,
            )
            watchdog.register_watchdog(
                WatchdogItems.DEVICE_WRITER,
                lambda: self._hardware.watchdog_writer_perf_c,
            )
        elif state in {"recovering", "connecting"}:
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.STARTING,
                reason=(
                    "recovering CAN/pellet controller"
                    if state == "recovering"
                    else "connecting CAN/pellet controller"
                ),
                error=error,
            )
        elif state == "failed":
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.FAILED,
                error=error or "CAN/pellet controller failed",
            )
            # Preserve all remaining recording streams. The in-flight command
            # is explicitly failed/unknown by HardwareModel and no new pellet
            # cycle is allowed until this subsystem returns to Ready.
        elif state == "stopped" and not self._acquisition.stopping:
            self._set_subsystem_status(
                SubsystemId.CAN_PELLET,
                SubsystemState.STOPPED,
                reason="CAN/pellet controller stopped",
            )

    def _on_inference_property_changed(self, name: str, value, _):
        if name == InferenceModel.STATUS:
            new_is_live = value == InferenceStatus.live
            if new_is_live:
                self._p_inference_live_begin = time.perf_counter()
            elif value == InferenceStatus.stopped:
                current = self._acquisition.subsystems.get(
                    SubsystemId.LIVE_INFERENCE
                )
                if (
                    self._acquisition.started
                    and not self._acquisition.stopping
                    and current is not None
                    and current.state is SubsystemState.READY
                ):
                    reason = "live inference stopped unexpectedly"
                    self._set_subsystem_status(
                        SubsystemId.LIVE_INFERENCE,
                        SubsystemState.FAILED,
                        error=reason,
                    )
                    self._handle_recording_subsystem_failure(
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
        boundary = self._recording_session.boundary
        if (
            boundary is not None
            and self._recording_session.status
            in {SessionRecordingStatus.RECORDING, SessionRecordingStatus.STOPPING}
        ):
            try:
                primary = self._ordered_reach_cameras(enabled_only=True)[0]
                frame_rate = primary.active_config.params.get("fps")
                if frame_rate is None:
                    frame_rate = self._behavior.system_machine.intersession.frame_rate
                sample = LiveTrackingSample.from_pose_response(
                    response,
                    FrameTimelineAnchor(
                        boundary.primary_frame_id,
                        boundary.start_perf_time,
                        float(frame_rate),
                    ),
                )
                if sample is not None:
                    self._live_tracking.append(sample)
            except (IndexError, TypeError, ValueError) as error:
                logger.warning("Live tracking sample was not buffered: %s", error)
        message = self._coordinates.validate_pose(
            response,
            self._behavior.algorithm.diamond_triangle_config,
            inference_started_perf=self._p_inference_live_begin,
            paused=self._behavior.algorithm.algo_paused,
        )
        if message is not None:
            self.on_error("Diamond not detected or invalid position", message)

    def _on_intertrial_device_message(
        self,
        kind,
        data,
        perf_time: float,
        _wall_time: float,
    ) -> None:
        """Open the physical attempt window on decoded pellet-board tone 2."""
        if kind is not SystemStatusMessageKind.STIMULUS_INPUTS:
            return
        if isinstance(data, dict):
            tone2 = bool(data.get("tone2", False))
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            tone2 = bool(data[1])
        else:
            logger.warning("Cannot decode tone-2 state from stimulus input: %r", data)
            return
        rising = tone2 and not self._tone2_active
        self._tone2_active = tone2
        if not rising:
            return
        attempt = self._pellet_cycles.active_attempt
        if (
            attempt is None
            or self._recording_session.status is not SessionRecordingStatus.RECORDING
        ):
            logger.warning("Tone-2 rising edge received without an active recorded pellet attempt")
            return
        with self._intertrial_lock:
            if self._trial_window_start is None:
                self._trial_window_start = (
                    attempt.operation_id,
                    float(perf_time),
                )
                logger.info(
                    "pellet trial tracking window opened: attempt=%s perf=%.6f",
                    attempt.attempt_label,
                    perf_time,
                )

    def _on_intertrial_analysis_result(
        self,
        result: IntertrialAnalysisResult,
    ) -> None:
        request = result.request
        token = self._recording_session.token()
        if (
            token is None
            or token.generation != request.generation
            or token.session_id != request.session_id
            or self._recording_session.status
            not in {
                SessionRecordingStatus.RECORDING,
                SessionRecordingStatus.STOPPING,
                SessionRecordingStatus.ANALYZING,
            }
        ):
            logger.warning(
                "Ignoring stale intertrial result: %s generation=%s active=%s",
                request.attempt_label,
                request.generation,
                token,
            )
            return
        config = self._behavior.algorithm.active_config.session_control
        retry = result.outcome.value in config.behavioral_retry_outcomes
        self._session_data_recorder.persist_trial_tracking(
            self._project_info,
            request,
            result,
        )
        if result.error and config.behavioral_retry_outcomes:
            self._hold_intertrial_resolution(
                request,
                "Trial analysis failed while a behavioral retry rule requires "
                f"the result: {result.error}",
            )
            return
        tone_references, laser_references = (
            self._session_data_recorder.trial_stream_references(
                request.window.start_perf,
                request.window.end_perf,
            )
        )
        try:
            finalized = self._pellet_cycles.finalize_intertrial_result(
                self._project_info,
                result,
                retry=retry,
                tone_references=tone_references,
                laser_references=laser_references,
            )
        except (KeyError, RuntimeError) as error:
            logger.error(
                "Ignoring unusable intertrial result for %s: %s",
                request.attempt_label,
                error,
            )
            return

        if (
            self._intertrial_resolution_request is not None
            and self._intertrial_resolution_request.operation_id
            == request.operation_id
        ):
            self._intertrial_resolution_request = None
            self._intertrial_resolution_reason = ""

        if result.reaches:
            response = IntersessionResponse(
                rh_max_vp_list=[
                    Offset3DTuple(*trajectory.closest_offset)
                    for trajectory in result.reaches
                    if not trajectory.consumed
                ],
                food_consumed=result.consumption_count,
                successful_reaches=result.success_count,
                pellets_presented=1,
                total_reaches=result.reach_count,
            )
            self._behavior.system_machine.shift_xyz_handler.put_intersession_response(
                self._project_info,
                response,
            )
        self._pellet_cycles.sync_behavior_counts(self._behavior.algorithm)
        self._behavior.algorithm.pellet_send_block_reason = ""
        self._notify_trial_protocol_state()
        self._evaluate_automatic_stop_policy(
            protocol_complete=self._protocol_runner.protocol_complete,
        )
        logger.success(
            "pellet trial analysis finalized: attempt=%s outcome=%s "
            "reaches=%s analysis=%.3fs",
            finalized.attempt_label,
            finalized.outcome.value,
            finalized.reach_count,
            result.analysis_seconds,
        )

    def _sync_session_counts_from_results(
        self,
        ledger: PelletTrialLedger,
    ) -> None:
        """Compatibility wrapper around the authoritative controller summary."""
        if ledger is not self._pellet_cycles.ledger:
            self._pellet_cycles.ledger = ledger
        self._pellet_cycles.sync_behavior_counts(self._behavior.algorithm)

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
        dst = Path(self._preferences.animal_location).joinpath(f"{animal.id}.json")
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
            rfid_reader_enabled=loaded_hardware.rfid_reader_enabled,
            rfid_device=loaded_hardware.rfid_device,
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
        if caller in {
            "raw_writers_closed",
            "session_analysis_ended",
            "session_notes_updated",
        }:
            boundary = self._recording_session.boundary
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
        metadata_generation_id = (
            f"acquisition-{when_as_utc.timestamp()}"
            if session is None
            else self._recording_session.metadata_generation_id
            or f"{project.short_id}-g{getattr(project, 'session_generation', 0)}"
        )
        boundary = self._recording_session.boundary
        if boundary is not None and boundary.session_id == project.short_id:
            session_boundary = boundary.to_metadata()
        else:
            session_boundary = None
        configuration = asdict(self._create_configuration())
        hardware_configured = {
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
        }
        runtime_final = self._acquisition.subsystems.snapshot()

        if session is None:
            out = {
                "metadataSchemaVersion": 2,
                "metadataGenerationId": metadata_generation_id,
                "scope": "acquisition",
                "createdUtc": when_as_utc.timestamp(),
                "serialNumber": self._preferences.serial_number or "",
                "appVersion": self._app_version,
                "hardware": {
                    "configured": hardware_configured,
                    "runtime": _compact_subsystem_snapshot(runtime_final),
                },
                "configuration": configuration,
            }
        else:
            session_dir = Path(file_name).parent
            artifacts = {
                "alignment": _metadata_reference(
                    session_dir / "streams" / "alignment.json",
                    relative_to=session_dir,
                ),
                "trialSummary": _metadata_reference(
                    session_dir / "streams" / "trial_summary.json",
                    relative_to=session_dir,
                ),
            }
            out = {
                "metadataSchemaVersion": 2,
                "metadataGenerationId": metadata_generation_id,
                "scope": "session",
                "sessionId": project.short_id,
                "sessionIndex": session,
                "createdUtc": when_as_utc.timestamp(),
                "serialNumber": self._preferences.serial_number or "",
                "appVersion": self._app_version,
                "boundary": session_boundary,
                "animal": self._recording_session.animal_snapshot,
                "notes": self.notes or "",
                "counts": {
                    "presented": self._behavior.algorithm.pellets_presented,
                    "reaches": self._behavior.algorithm.pellet_reaches,
                    "successfulReaches": self._behavior.algorithm.successful_reaches,
                    "consumed": self._behavior.algorithm.pellets_consumed,
                },
                "recording": {
                    "status": self._recording_session.status.value,
                    "stopReason": (
                        None
                        if self._recording_ending_reason is RecordingEndingReason.NA
                        else self._recording_ending_reason.value
                    ),
                    "dataComplete": self._recording_session.data_complete,
                    "dataErrors": list(self._recording_session.data_errors),
                    "analysisDurationSeconds": (
                        self._recording_session.analysis_duration_seconds
                    ),
                    "firstPelletDeliveryOffsetSeconds": (
                        project.first_pellet_delivery_offset
                    ),
                    "firstPelletPresentationOffsetSeconds": (
                        project.first_pellet_presentation_offset
                    ),
                    "endActions": [
                        dict(action)
                        for action in self._recording_session.end_actions
                    ],
                    "storage": dict(
                        self._recording_session.storage_telemetry
                    ),
                    "internalError": self.internal_error_diagnostic,
                },
                "stopPolicy": (
                    None
                    if self._session_stop_policy is None
                    else dataclasses.asdict(
                        self._session_stop_policy.configuration
                    )
                ),
                "protocolSchedule": list(self._trial_protocol_schedule.to_records()),
                "stopResult": (
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
                "hardware": {
                    "configured": hardware_configured,
                    "atRecord": _compact_subsystem_snapshot(
                        self._recording_session.hardware_status_at_record
                    ),
                    "changesAtFinalize": _subsystem_snapshot_changes(
                        self._recording_session.hardware_status_at_record,
                        runtime_final,
                    ),
                },
                "configuration": configuration,
                "artifacts": artifacts,
            }
        out = _metadata_without_nonfinite_numbers(out)
        json_path = Path(file_name + ".json")
        yaml_path = Path(file_name + ".yaml")
        session_dir = None if session is None else Path(file_name).parent
        generation_id = str(
            out.get("metadataGenerationId") or f"acquisition-{when_as_utc.timestamp()}"
        )
        # Serialize both representations before publishing either one. This
        # preserves the previous pair if either encoder rejects the snapshot;
        # the generation ID detects the narrower power-loss window between the
        # two atomic replacements.
        json_text = json.dumps(
            out,
            cls=SystemConfigurationJSONEncoder,
            allow_nan=False,
            sort_keys=True,
        ) + "\n"
        yaml_text = yaml.dump(
            out,
            Dumper=SystemConfigurationDumper,
            sort_keys=False,
        )
        atomic_publish_file(
            json_path,
            lambda path: path.write_text(json_text, encoding="utf-8"),
            session_dir=session_dir,
            generation_id=generation_id,
            validate=lambda path: json.loads(path.read_text(encoding="utf-8")),
        )

        atomic_publish_file(
            yaml_path,
            lambda path: path.write_text(yaml_text, encoding="utf-8"),
            session_dir=session_dir,
            generation_id=generation_id,
            validate=lambda path: yaml.safe_load(path.read_text(encoding="utf-8")),
        )
        if session_dir is not None:
            stream_manifest_path = session_dir / "streams" / "stream_manifest.json"
            manifest_files = [json_path, yaml_path, stream_manifest_path]
            manifest = {
                "schemaVersion": 1,
                "metadataGenerationId": generation_id,
                "authoritativeMetadata": json_path.relative_to(session_dir).as_posix(),
                "files": [
                    file_manifest_entry(path, relative_to=session_dir)
                    for path in manifest_files
                    if path.is_file()
                ],
            }
            # This is deliberately last: its presence declares that the current
            # auxiliary generation reached its authoritative metadata publish.
            atomic_write_json(
                session_dir / "manifest.json",
                manifest,
                session_dir=session_dir,
                generation_id=generation_id,
            )

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

    def _make_api_system_status_payload(self) -> ReachAQSystemStatus:
        hard = self._hardware
        algo = self._behavior.algorithm
        project = self._project_info
        if project is None:
            project = self.make_project_info()
        animal = self._selected_animal
        plan = self._attached_plan
        phase = None if plan is None else plan.current_phase
        return ReachAQSystemStatus(
            schema_version=1,
            acquisition_state=self._status.value,
            recording_state=self._recording_session.status.value,
            synchronization_ready=not any(
                "camera" in blocker.lower()
                for blocker in self._acquisition.subsystems.recording_blockers()
            ),
            animal=None if animal is None else animal.to_api_status(),
            project={
                "dayPath": project.get_day_path()[0],
                "sessionIndex": project.session,
                "sessionId": project.short_id,
            },
            subsystems=self._acquisition.subsystems.snapshot(),
            pellet_device={
                "connected": hard.connected,
                "position": self._offset_record(hard.last_dcs_position),
                "sendPosition": self._offset_record(hard.last_dcs_set_position),
                "loadArm": hard.load_arm_position,
                "coverArm": hard.cover_arm_position,
            },
            session_counts={
                "reaches": algo.pellet_reaches,
                "presented": algo.pellets_presented,
                "success": algo.successful_reaches,
                "consumed": algo.pellets_consumed,
            },
            protocol={
                "selected": None if plan is None else plan.plan_id,
                "phase": None if phase is None else phase.phase_id,
                "complete": self._protocol_runner.protocol_complete,
                "automaticAdvance": self._protocol_runner.automatic_advance,
            },
        )

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

    def _on_pellet_loading_for_trial(self):
        if self._pellet_cycles.active_attempt is not None:
            logger.warning(
                "Pellet loading closed an attempt without a pellet-cycle event"
            )
            self._complete_pellet_trial_window(
                get_perf_now(),
                close_reason="next pellet load began",
            )

    def _on_pellet_cycle_completed(self, *, perf_c: float):
        self._complete_pellet_trial_window(
            perf_c,
            close_reason="pellet cycle completed",
        )

    def _complete_pellet_trial_window(
        self,
        perf_c: float,
        *,
        close_reason: str,
    ) -> None:
        active = self._pellet_cycles.active_attempt
        if active is None:
            return
        closed = self._pellet_cycles.finish_active(perf_c, time.time())
        with self._intertrial_lock:
            window_start = self._trial_window_start
            self._trial_window_start = None
        config = self._behavior.algorithm.active_config.session_control
        tracking_window = None
        evidence = None
        unavailable_reason = ""
        if window_start is None:
            unavailable_reason = "decoded pellet-board tone-2 rising edge was not observed"
        elif window_start[0] != closed.operation_id:
            unavailable_reason = "tone-2 window belonged to a different pellet operation"
        else:
            tracking_window = self._live_tracking.window(window_start[1], perf_c)
            pellet_config = self._behavior.algorithm.active_config.pellet_delivery
            evidence = classify_pellet_state(
                tracking_window,
                expected_triangle_pellet_distance=(
                    pellet_config.triangle_pellet_expected_distance
                ),
                misplacement_threshold=(
                    pellet_config.triangle_pellet_diff_too_far_threshold
                ),
            )

        if not config.intertrial_analysis_enabled:
            unavailable_reason = "live intertrial analysis is disabled"
        elif tracking_window is not None and not tracking_window.samples:
            unavailable_reason = "no live tracking samples covered the pellet trial"

        token = self._recording_session.token()
        request = None
        if token is not None and tracking_window is not None and evidence is not None:
            request = IntertrialAnalysisRequest(
                generation=token.generation,
                session_id=token.session_id,
                trial_id=closed.trial_id,
                attempt_id=closed.attempt_id,
                operation_id=closed.operation_id,
                window=tracking_window,
                pellet_state=evidence,
            )
            self._session_data_recorder.persist_trial_tracking(
                self._project_info,
                request,
            )
            self._pellet_cycles.annotate_intertrial_capture(
                self._project_info,
                request,
            )

        if unavailable_reason:
            tone_references, laser_references = (
                ((), ())
                if tracking_window is None
                else self._session_data_recorder.trial_stream_references(
                    tracking_window.start_perf,
                    tracking_window.end_perf,
                )
            )
            finalized = self._pellet_cycles.finalize_intertrial_unavailable(
                self._project_info,
                closed,
                perf_time=perf_c,
                wall_time=closed.capture_end_wall_time or time.time(),
                reason=f"{close_reason}: {unavailable_reason}",
                pellet_presence=(
                    "unknown" if evidence is None else evidence.presence.value
                ),
                pellet_misplacement=(
                    "unknown" if evidence is None else evidence.misplacement.value
                ),
                window=tracking_window,
                outcome=(
                    TrialOutcome.UNSCORED
                    if not config.intertrial_analysis_enabled
                    else TrialOutcome.INCOMPLETE
                ),
                tone_references=tone_references,
                laser_references=laser_references,
            )
            self._pellet_cycles.sync_behavior_counts(self._behavior.algorithm)
            self._notify_trial_protocol_state()
            self._evaluate_automatic_stop_policy(
                protocol_complete=self._protocol_runner.protocol_complete,
            )
            logger.warning(
                "pellet trial %s finalized without trajectory analysis: %s",
                finalized.attempt_label,
                unavailable_reason,
            )
            return

        if request is None:
            return
        must_wait = (
            config.intertrial_progression_mode
            == AnalysisProgressionMode.WAIT.value
            or bool(config.behavioral_retry_outcomes)
        )
        if must_wait:
            estimate = self._intertrial_analysis.timing_estimate(
                tracking_window.end_perf - tracking_window.start_perf
            )
            self._behavior.algorithm.pellet_send_block_reason = (
                "waiting for pellet trial analysis; " + estimate.display_text
            )
        if not self._intertrial_analysis.submit(request):
            if must_wait:
                self._hold_intertrial_resolution(
                    request,
                    "Intertrial analysis queue is full. Retry this analysis, "
                    "continue without its result, Stop, or Abort.",
                )
                return
            self._behavior.algorithm.pellet_send_block_reason = ""
            tone_references, laser_references = (
                self._session_data_recorder.trial_stream_references(
                    tracking_window.start_perf,
                    tracking_window.end_perf,
                )
            )
            finalized = self._pellet_cycles.finalize_intertrial_unavailable(
                self._project_info,
                closed,
                perf_time=perf_c,
                wall_time=closed.capture_end_wall_time or time.time(),
                reason="intertrial analysis queue was full",
                pellet_presence=evidence.presence.value,
                pellet_misplacement=evidence.misplacement.value,
                window=tracking_window,
                tone_references=tone_references,
                laser_references=laser_references,
            )
            self._pellet_cycles.sync_behavior_counts(self._behavior.algorithm)
            logger.error(
                "Intertrial analysis queue full; attempt %s was not analyzed",
                finalized.attempt_label,
            )
        self._notify_trial_protocol_state()

    def _on_pellet_sending(self, *, perf_c: float, context: str):
        algo = self._behavior.algorithm
        if self._trial_ledger is None or not algo.is_in_session:
            return
        wall_time = time.time()
        with self._intertrial_lock:
            self._trial_window_start = None
        shift = self._behavior.system_machine.shift_xyz_handler
        attempt = self._pellet_cycles.begin_send(
            perf_c,
            wall_time,
            operation_id=context,
            pellet_position=self._offset_record(self._hardware.last_dcs_set_position),
            planned_shift=self._offset_record(shift.last_shift_xyz),
            applied_shift=self._offset_record(shift.last_processed_shift_xyz),
            protocol_context=self._current_trial_protocol_context(
                self._pellet_cycles.planned_trial_id
            ),
        )
        self._notify_trial_protocol_state()
        logger.info(
            "pellet trial attempt started: session=%s attempt=%s context=%s",
            self._trial_ledger.session_id,
            attempt.attempt_label,
            context,
        )

    def _on_hardware_command_failed(self, failure: CanFailure) -> None:
        """Finalize the matching physical pellet attempt as a hardware error."""
        finalized = self._pellet_cycles.finalize_hardware_failure(
            failure,
            in_session=self._behavior.algorithm.is_in_session,
        )
        if finalized is None:
            return
        self._notify_trial_protocol_state()
        self._evaluate_automatic_stop_policy(
            protocol_complete=self._protocol_runner.protocol_complete,
        )

    def _on_pellet_sent(
        self,
        *,
        perf_c: Optional[float] = None,
        context: Optional[str] = None,
    ):
        algo = self._behavior.algorithm
        logger.debug("on_pellet_sent: recording=%s in_session=%s",
                     self._recording_session.status, algo.is_in_session)
        if algo.is_in_session:
            ledger = self._trial_ledger
            if ledger is not None and self._pellet_cycles.active_attempt is not None:
                attempt = self._pellet_cycles.acknowledge_presentation(
                    get_perf_now() if perf_c is None else perf_c,
                    time.time(),
                    operation_id=context,
                )
                if attempt is None:
                    return
                self._notify_trial_protocol_state()
            if ledger is not None:
                # Presentation is acknowledgement-derived immediately. The
                # other three counts update together when this pellet window's
                # live tracking result is finalized.
                algo.pellets_presented = self._pellet_cycles.count(
                    TrialCountBasis.PRESENTED
                )
            self._evaluate_automatic_stop_policy(
                protocol_complete=self._protocol_runner.protocol_complete,
            )

    @staticmethod
    def _offset_record(value):
        return offset_record(value)

    def _current_trial_protocol_context(self, trial_id: Optional[int] = None):
        plan = self._attached_plan
        phase = None if plan is None else plan.current_phase
        if trial_id is None:
            trial_id = self._pellet_cycles.planned_trial_id
        return {
            "protocol_id": None if plan is None else plan.plan_id,
            "phase_id": None if phase is None else phase.phase_id,
            "automatic_advance": self._protocol_runner.automatic_advance,
            "trial_row": self._trial_protocol_schedule.row(trial_id).to_record(),
        }
