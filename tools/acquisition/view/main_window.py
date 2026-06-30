import os
import pickle
import random
import shutil
import textwrap
import threading
import time
import math
import traceback
from datetime import datetime, date
from functools import partial
from itertools import chain
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable, Union, Literal

from PySide6.QtCore import Qt, QCoreApplication, Signal, QSize, QKeyCombination
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QMainWindow, QStatusBar, QToolBar, QLabel, QMessageBox, QApplication,
                               QSizePolicy, QWidget, QComboBox, QLineEdit, QFileDialog, QPushButton, QHBoxLayout,
                               QSpinBox, QDoubleSpinBox, QFrame, QDialog)
import qtawesome as qta

from autotrainer.core import EventManager, Offset3DTuple, AnimalSubject, SystemConfiguration, CameraConfiguration, \
    calculate_std_dev_manual, ProjectInfo, get_perf_now
from autotrainer.core.analysis.autoclamp_evasion_detector import AutoClampEvasionDetector
from autotrainer.core.animal.animal_subject import AnimalPelletCounts
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.logging import get_console_handler, get_verbose_logger
from autotrainer.core.multiproc import make_daemon_timer, no_op_timer
from autotrainer.core.pose_elements import SceneElement
from autotrainer.core.event.api_event_kind import ApiEventKind
from autotrainer.core.project.project_info import DATE_TIME_FORMAT
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig

from autotrainer.inference import InferenceStatus, PoseResponse
from autotrainer.inference.analysis import IntersessionResponse
from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME, make_cam_offsets_dict

from autotrainer.behavior import TrainingMode, SystemState
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus, BehaviorAlgoProps
from autotrainer.behavior.pellet import PelletState

from autotrainer.pyside.content_widget import InvokeMethod, invoke_method

from autotrainer.training import TrainingPlan, PlanInfo, LoadProgressResult, TrainingPhase

from autotrainer.pyside.xyz_label import XYZQLabel

from tools.autotrainer_version import __version__ as app_version
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus
from tools.acquisition.model.handle_3d_calibration import make_3d_calib
from tools.acquisition.model.training_plan import get_plan_id
from tools.acquisition.model.user_preferences import UserPreferences
from tools.acquisition.view.main_content import MainContent
from tools.acquisition.view.preferences_dialog import PreferencesDialog
from tools.acquisition.view.debug_content import DebugView

logger = get_verbose_logger(__name__)

_this_dir = Path(__file__).parent.resolve()


_calibrate_timer = make_daemon_timer

DEFAULT_DIAMOND_TRIANGLE_CALIB_DURATION = 3  # duration of calibration data acquisition
DEFAULT_DIAMOND_TRIANGLE_CALIB_TIMEOUT = 30  # maximum time before automated stop of calibration
# if not enough data is captured after that time the calib is automatically finished/stopped (and ask for retry)
DEFAULT_DIAMOND_TRIANGLE_NOISY_DISTANCE = 0.2  # distance over which data is considered noisy, and a retry proposed


def _make_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)  # Optional: adds a slight sunken effect
    return sep


def _make_window_title(prefs):
    return f"{prefs.serial_number} - Auto Trainer - Acquisition v{app_version}"


class MainWindow(QMainWindow):

    training_mode_changed = Signal(TrainingMode)
    running_status_changed = Signal(bool)  # True == running/acquiring
    # todo: integrate within on_app_model_status_changed event handling

    def __init__(
        self,
        app: QApplication,
        user_preferences: UserPreferences,
        config_file: Optional[Path],
        *,
        is_dev: bool = False,
    ):
        super().__init__(None)

        self._default_event_manager = EventManager.default()
        self._post_api_event = self._default_event_manager.post_event_content

        self._orig_inference_analysis_feed = None
        self._orig_inference_analysis_process = None

        self._app = app
        self._is_dev = is_dev
        prefs = self._preferences = user_preferences
        self._update_log_level(prefs.log_level)
        self._title = _make_window_title(prefs)
        self._closing = False
        self._close_event = None
        self._start_capture_thread = None
        self._stop_capture_thread = None

        self.setWindowTitle(self._title)

        icon = QIcon(_this_dir.joinpath("autotrainer.png").absolute().as_posix())
        self.setWindowIcon(icon)

        self._open_dialogs = []
        self._debug_view: Optional[DebugView] = None
        self._training_plan_index_by_plan_id: Dict[Optional[str], int] = {}
        self._diamond_triangle_calib_run = None
        self._warned_invalid_dcs_config = False

        self._previous_intersession_analysis_rsp: Optional[Tuple[ProjectInfo, IntersessionResponse]] = None

        app_model = self._app_model = AppModel(prefs)

        try:
            self.setContentsMargins(0, 0, 0, 0)

            self.main_content = MainContent(app_model)

            self._create_actions()
            self._configure_menubar()
            self._configure_statusbar()
            self._configure_toolbar()

            self.setCentralWidget(self.main_content)
            self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
            # self.setMaximumSize(1880, 1080)
        except Exception as err:
            app_model.on_close()
            raise RuntimeError(f"Error setting up UI: {err}") from err

        app_model.on_error += self._show_message
        app_model.configuration_loaded_event += self._on_app_model_configuration_loaded
        app_model.training_plan_deserialized += self._on_training_plan_deserialized

        try:
            app_model.load_configuration(config_file)
        except Exception as err:
            tb = traceback.format_exc()
            app_model.on_error("Failed load configuration",
                               f"\nConfiguration file {config_file} has issue,\n\n"
                               f"please check and fix following error:\n\n{err}\n\n{tb}")
            # app_model.on_close()
            # raise RuntimeError(f"Could not load config: {err}") from err

        app_model.property_changed += self._on_app_model_property_changed
        app_model.current_day_changed += self._on_day_changed
        app_model.hardware.property_changed += self._on_hardware_property_changed
        app_model.inference.property_changed += self._on_inference_property_changed
        app_model.inference.detection_result_ready += self._on_inference_analysis_result_ready
        app_model.behavior.algorithm.property_changed += self._on_behavior_algo_property_changed

        user_preferences.property_changed += self._on_preferences_property_changed

        analysis = app_model.analysis
        analysis.emergency_alarm_monitor.property_changed += self._on_alarm_monitor_property_changed
        analysis.system_maintenance_alarm.property_changed += self._on_system_maintenance_prop_changed
        analysis.autoclamp_evasion_detector.property_changed += self._on_autoclamp_evasion_property_changed

        self.running_status_changed.connect(self._set_start_or_stop)
        #
        self._reload_animals(self._app_model.animals)  # after all property_changed connect above
        #
        # then after everything:
        self._set_reset_vat_text()
        self._set_reset_cage_clean_text()
        self._set_autoclamp_evasion(analysis.autoclamp_evasion_detector)

    @property
    def app_model(self) -> AppModel:
        return self._app_model

    @property
    def has_fully_valid_dcs(self) -> bool:
        cfg = self._app_model.behavior.algorithm.diamond_triangle_config
        return cfg is not None and cfg.fully_valid

    def _add_box_to_open_dialogs(self, item: Union[QMessageBox, QDialog]):
        self._open_dialogs.append(item)
        def close_event(event, orig_close_event=item.closeEvent):
            try:
                self._open_dialogs.remove(item)
            except ValueError:
                pass
            orig_close_event(event)
            event.accept()
        item.closeEvent = close_event

    def _set_start_or_stop(self, started: bool):
        #
        stopped = not started
        self.edit_camera_settings_action.setEnabled(stopped)
        self.make_3d_calib_action.setEnabled(stopped)
        #
        run_action = self.run_action
        run_action.blockSignals(True)  # block signal to ensure we don't re-start/stop
        run_action.setChecked(started)
        run_action.blockSignals(False)
        if started:
            icon = qta.icon('ei.stop')
            run_action.setText("Stop")
            run_action.setIcon(icon)
        else:
            icon = qta.icon('ei.play')
            run_action.setText("Start")
            run_action.setIcon(icon)

    def _on_capture_start_stop(self, is_toggled, *, target_status: AppModelStatus = AppModelStatus.ACQUIRING):
        app_model = self._app_model
        self.run_action.setEnabled(False)
        self.make_3d_calib_action.setEnabled(False)
        self.animal_in_device_action.setEnabled(False)
        self.animal_in_training_action.setEnabled(False)
        self._app_model_status_combo.setEnabled(False)
        self.running_status_changed.emit(is_toggled)
        if is_toggled:
            self._check_diamond_triangle_config()
            self._status_label.setText("Starting acquisition...")
            def exec_start_capture(prev_thread=self._start_capture_thread):
                if prev_thread is not None:
                    logger.verbose("joining previous start thread")
                    prev_thread.join()
                logger.info("starting subprocesses")
                try:
                    started = app_model.capture_start(target_status=target_status)
                except Exception as err:
                    logger.exception("app_model.capture_start failed: %s", err)
                    started = False
                self._start_capture_thread = None
                # following should normally be executed in main UI thread:
                if started:
                    self._status_label.setText("")
                    self._acquisition_started = True
                else:
                    logger.verbose("capture_start failed: %s", app_model.status)
                    self._status_label.setText("Startup failed")
                    self.running_status_changed.emit(False)
                self.run_action.setEnabled(True)
                self.animal_in_device_action.setEnabled(True)
                self.animal_in_training_action.setEnabled(True)
                self._app_model_status_combo.setEnabled(True)
            thread = threading.Thread(target=exec_start_capture, daemon=True, name="StartAcquisition")
            self._start_capture_thread = thread
            thread.start()
        else:
            self._status_label.setText("Stopping acquisition...")
            def exec_stop_capture(prev_start=self._start_capture_thread,
                                  prev_stop=self._stop_capture_thread):
                if prev_start is not None:
                    logger.verbose("joining previous start thread")
                    prev_start.join()
                if prev_stop is not None:
                    prev_stop.join()
                logger.info("stopping subprocesses")
                app_model.capture_stop()
                # following should normally be executed in main UI thread:
                self.running_status_changed.emit(False)
                self.run_action.setEnabled(True)
                self.animal_in_device_action.setEnabled(True)
                self.animal_in_training_action.setEnabled(True)
                self._app_model_status_combo.setEnabled(True)
                self._status_label.setText("")
                self._acquisition_started = False
            thread = threading.Thread(target=exec_stop_capture, daemon=True, name="StopAcquisition")
            self._stop_capture_thread = thread
            thread.start()

    def _on_system_mode_combo_changed(self, idx: int):
        status = self._app_model_status_combo.itemData(idx)
        if status == AppModelStatus.IDLE:
            self._on_capture_start_stop(False)
        elif status == AppModelStatus.ACQUIRING:
            self._on_capture_start_stop(True)
        elif status == AppModelStatus.ANIMAL_IN_DEVICE:
            self._on_animal_in_device_triggered(True)
        elif status == AppModelStatus.ANIMAL_IN_TRAINING:
            self._on_animal_in_training_triggered(True)

    def _on_animal_in_device_triggered(self, is_toggled):
        logger.verbose("_on_animal_in_device_triggered: %s", is_toggled)
        if is_toggled:
            if self._app_model.acquisition_started:
                self._app_model.status = AppModelStatus.ANIMAL_IN_DEVICE
            else:
                self._on_capture_start_stop(True, target_status=AppModelStatus.ANIMAL_IN_DEVICE)
        else:
            self._app_model.status = AppModelStatus.ACQUIRING

    def _on_animal_in_training_triggered(self, is_toggled):
        logger.verbose("_on_animal_in_training_triggered: %s", is_toggled)
        if is_toggled:
            for action in (self.animal_in_device_action,):
                action.blockSignals(True)
                action.setChecked(True)
                action.blockSignals(False)

            if self._app_model.acquisition_started:
                self._app_model.status = AppModelStatus.ANIMAL_IN_TRAINING
            else:
                self._on_capture_start_stop(True, target_status=AppModelStatus.ANIMAL_IN_TRAINING)
        else:
            self._app_model.status = AppModelStatus.ANIMAL_IN_DEVICE

    def on_show_reach_event(self, is_toggled):
        raw = self._previous_intersession_analysis_rsp
        # uncomment for being able show agx001_20260205_11 reach
        # if is_toggled and raw is None:
        #     from tests.real_data_intersession_process_test import (
        #         agx001_20260205_11_project_info,
        #     )
        #     raw = (agx001_20260205_11_project_info, None)
        if raw is None or not is_toggled:
            self.main_content.show_analysis_reach_events(None)
            return
        prj, rsp = raw
        self.main_content.show_analysis_reach_events(prj)

    def on_reset_pellet_loaded_count(self):
        prefs = self._preferences
        prefs.pellet_load_count_total = prefs.pellet_load_count_day = 0
        self._app_model.check_max_pellet_loaded()

    def on_reset_cage_clean(self):
        prefs = self._preferences
        prefs.cage_clean_previous_day = date.today()
        prefs.save()
        self._set_reset_cage_clean_text()

    def on_reset_autoclamp_evasion(self):
        self._app_model.analysis.autoclamp_evasion_detector.pellets_consumed = 0

    def _check_target_next_or_previous_plan_phase(
        self,
        method: Literal["advance", "fallback"],
        target_phase: TrainingPhase,
    ):
        attempts = target_phase.progress.phase_attempts
        if attempts == 0:
            return True
        m_cap = method.capitalize()
        box = QMessageBox()
        box.setWindowTitle("Phase Progress Reset")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f"{m_cap} to Phase {target_phase.name} will reset progress for the phase")
        button = box.addButton(f"{m_cap}", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.show()
        box.exec()
        return box.clickedButton() == button

    def on_previous_plan_phase(self):
        app_model = self._app_model
        plan = app_model.attached_plan
        if plan is None:
            return
        phase_idx = plan.phases.index(plan.current_phase)
        if phase_idx > 0:
            target = plan.phases[phase_idx - 1]
            if not self._check_target_next_or_previous_plan_phase("fallback", target):
                return
        logger.info("fallback on plan %s (%s)", plan.plan_id, hex(id(plan)))
        plan.fallback()
        self._refresh_prev_next_phases()
        self.main_content.training_plan_changed.emit(plan)

    def on_next_plan_phase(self):
        app_model = self._app_model
        plan = app_model.attached_plan
        if plan is None:
            return
        phase_idx = plan.phases.index(plan.current_phase)
        if 0 <= phase_idx <= len(plan.phases) - 1:
            target = plan.phases[phase_idx + 1]
            if not self._check_target_next_or_previous_plan_phase("advance", target):
                return
        logger.info("advance on plan %s (%s)", plan.plan_id, hex(id(plan)))
        plan.advance()
        self._refresh_prev_next_phases()
        self.main_content.training_plan_changed.emit(plan)

    def _handle_diamond_triangle_calib_run(
        self,
        *,
        positions: List[Offset3DTuple],  # motor coordinate system positions
        offsets: List[Offset3DTuple],  # diamond-triangle offsets (inference coordinate system)
        diamond_locs3d: List[Offset3DTuple],  # diamond loc3d (inference coordinate system)
        raw_diamond_3d: List[Offset3DTuple],
    ):
        self._timer_calibrate_diamond_triangle.cancel()
        if len(offsets) < 3 or len(diamond_locs3d) < 10:
            self._post_api_event(
                ApiEventKind.calibrationDcsFailed,
                dict(reason="MissingData")
            )
            self.calib_diamond_triangle_action.setEnabled(False)
            box = QMessageBox()
            box.setWindowTitle("Please")
            box.setText(
                "Could not get enough or data at all,\n"
                "please send pellet to make triangle visible in both cameras,\n"
                "then you can retry after.")
            box.setIcon(QMessageBox.Icon.Critical)

            self._open_dialogs.append(box)

            def remove():
                self.calib_diamond_triangle_action.setEnabled(True)
                # logger.debug("removing dialog from self.open_dialogs")
                try:
                    self._open_dialogs.remove(box)
                except ValueError:  # safer
                    pass

            retry_button = box.addButton("Retry calibration", QMessageBox.ButtonRole.AcceptRole)
            retry_button.clicked.connect(lambda: self.on_calibrate_diamond_triangle(True))
            retry_button.clicked.connect(remove)
            #
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole).clicked.connect(remove)
            box.setWindowModality(Qt.WindowModality.NonModal)
            box.setModal(False)

            def close_event(event):
                remove()
                event.accept()

            box.closeEvent = close_event
            box.show()
            return
        #
        avg_pos, stdev_pos = calculate_std_dev_manual(positions)
        assert isinstance(avg_pos, Offset3DTuple)
        assert isinstance(stdev_pos, Offset3DTuple)
        logger.info("motor-position: avg=%s stdev=%s", avg_pos, stdev_pos)
        #
        avg_offset, stdev_offset = calculate_std_dev_manual(offsets)
        assert isinstance(avg_offset, Offset3DTuple)
        assert isinstance(stdev_offset, Offset3DTuple)
        logger.info("diamond-triangle-inference-offset: avg=%s stdev=%s", avg_offset, stdev_offset)
        #
        avg_dia_loc3, stdev_dia_loc3d = calculate_std_dev_manual(diamond_locs3d)
        assert isinstance(avg_dia_loc3, Offset3DTuple)
        assert isinstance(stdev_dia_loc3d, Offset3DTuple)
        logger.info("diamond-inference-position: avg=%s stdev=%s", avg_dia_loc3, stdev_dia_loc3d)
        #
        avg_rawdia_loc3, stdev_rawdia_loc3d = calculate_std_dev_manual(raw_diamond_3d)
        logger.info("raw-diamond-inference-position: avg=%s stdev=%s", avg_rawdia_loc3, stdev_rawdia_loc3d)
        #
        noisy = False
        for val in chain(stdev_offset, stdev_pos, stdev_dia_loc3d, stdev_rawdia_loc3d):
            if val >= DEFAULT_DIAMOND_TRIANGLE_NOISY_DISTANCE:
                noisy = True
        if noisy:
            self._post_api_event(ApiEventKind.calibrationDcsFailed, dict(reason="NoisyData"))
            rsp = QMessageBox.warning(
                self, "Confirmation", "The data is noisy, do you want retry longer ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if rsp == QMessageBox.StandardButton.Yes:
                self._diamond_triangle_calib_run = self._make_diamond_triangle_calib_run(2 * DEFAULT_DIAMOND_TRIANGLE_CALIB_DURATION)
                self.on_calibrate_diamond_triangle(True)
            return
        # success
        self._post_api_event(ApiEventKind.calibrationDcsCompleted)
        app_model = self._app_model
        algo = app_model.behavior.algorithm
        save_path = algo.diamond_triangle_offset_config_path.expanduser()
        if save_path.exists():
            file_path: Optional[str] = None

            def retain_selected(path):
                nonlocal file_path
                file_path = path

            rsp = QMessageBox.question(
                self, "Confirmation",
                f"The configuration file ({save_path.as_posix()}) already exists, are you sure you want to proceed ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if rsp != QMessageBox.StandardButton.Yes:
                return
            dialog = QFileDialog(self, "Save to configuration file", save_path.parent.as_posix(),
                                 "All yaml files (*.yaml *.yml)")
            dialog.selectFile(save_path.name)
            dialog.fileSelected.connect(retain_selected)
            dialog.exec()
            if file_path is None:
                return
            save_path = Path(file_path)

        # always:
        QMessageBox.information(self, "Information",
                                f"Successfully computed values for diamond-triangle position & offset.\n"
                                f"\nSaving to {save_path.as_posix()}\n\n"
                                f"Calibration is being used immediately by now.",
                                QMessageBox.StandardButton.Ok,
                                )
        new_cfg = DiamondTriangleOffsetConfig(
            used_position=avg_pos,
            measured_offset=avg_offset,
            # once fully calibrated twice,
            # what we get from triangulate+reorient_and_center is fully centered on diamond, which is ~0/0/0.
            diamond_coord=Offset3DTuple(0, 0, 0),  # avg_dia_loc3,
            # so this allows to not have to calibrate twice.
            raw_diamond_coord=avg_rawdia_loc3,
            version=DiamondTriangleOffsetConfig.current_config_version,
        )
        logger.success("Saving new config %s to %s", new_cfg, save_path.as_posix())
        new_cfg.to_file(save_path)
        # also write new camera_offsets.pkl to calib dir:
        cam_offsets = make_cam_offsets_dict()
        cam_off = avg_rawdia_loc3
        cam_offsets.update(
            x_off=cam_off.x,
            y_off=cam_off.y,
            z_off=cam_off.z,
        )
        calib_dir = Path(self._preferences.configuration_location).joinpath(DEFAULT_3D_CALIB_DIR_NAME)
        cam_offsets_path = calib_dir.joinpath(DEFAULT_CAM_OFFSET_FILE_NAME)
        if cam_offsets_path.exists():
            now = datetime.now()
            backup_path = cam_offsets_path.parent.joinpath(f"{cam_offsets_path.name}_{now.strftime(DATE_TIME_FORMAT)}")
            shutil.copy(cam_offsets_path, backup_path)
        logger.info("Writing new camera-offsets %s to file %s", cam_offsets, cam_offsets_path)
        with cam_offsets_path.open("wb") as fh:
            pickle.dump(cam_offsets, fh)
        #
        app_model = self._app_model
        app_model.reload_calib(calib_dir)
        algo = app_model.behavior.algorithm
        algo.diamond_triangle_config = new_cfg
        animal = app_model.selected_animal
        # to ensure animal will get its x/y/z in DCS
        app_model.selected_animal = None
        app_model.selected_animal = animal

    def _make_diamond_triangle_calib_run(self, calib_duration: float = DEFAULT_DIAMOND_TRIANGLE_CALIB_DURATION):
        logger.notice("Starting diamond-triangle calibration .. duration=%.1f second(s)", calib_duration)

        app_model = self._app_model
        prev_status, app_model.status = app_model.status, AppModelStatus.CALIBRATION_DCS

        algo = app_model.behavior.algorithm
        action = self.calib_diamond_triangle_action

        diamond_locs3d = []
        raw_diamond_3d = []

        def record_offsets(pose_response: PoseResponse):
            nonlocal start_perf_c, offsets, positions, recording
            if not recording:
                return
            if len(offsets) > 2:
                self._timer_calibrate_diamond_triangle.cancel()
            new_offset = pose_response.get_parts_3d_offset(SceneElement.Diamond, SceneElement.Triangle)
            if new_offset is not None:
                offsets.append(new_offset)
                positions.append(app_model.hardware.last_position)
            dia_loc3d = pose_response.locations_3d.get(SceneElement.Diamond)
            raw_dia_3d = pose_response.raw_loc_3d.get(SceneElement.Diamond)
            if dia_loc3d is not None:
                diamond_locs3d.append(dia_loc3d)
            if raw_dia_3d is not None:
                raw_diamond_3d.append(raw_dia_3d)
            if time.perf_counter() - start_perf_c > calib_duration:
                # required, to execute the function in the UI/main thread:
                # reminder this record_offsets is executed by some thread handler/worker in some callback
                InvokeMethod(self.on_calibrate_diamond_triangle, False)
                recording = False

        recording = True
        offsets = []
        positions = []
        action.setChecked(True)
        action.setIcon(qta.icon("fa5s.crosshairs", color='red'))
        before_pellet_delivery_enabled = algo.pellet_delivery_enabled
        algo.pellet_delivery_enabled = False
        start_perf_c = time.perf_counter()
        app_model.inference.pose_response_ready += record_offsets
        self._status_label.setText("Capturing data ..")
        timer = self._timer_calibrate_diamond_triangle = _calibrate_timer(
            DEFAULT_DIAMOND_TRIANGLE_CALIB_TIMEOUT,
            lambda: InvokeMethod(self.on_calibrate_diamond_triangle, False)
        )
        timer.start()
        # for i in range(15):
        #     offsets.append(Offset3DTuple(1 + 0.001 * i, 2 + 0.01 * i, 3 + 0.001 * i ** 2))
        #     positions.append(Offset3DTuple(1 + 0.001 * i, 2 + 0.01 * i, 3 + 0.1001 * i ** 2))
        #
        yield
        #
        self._status_label.setText("")
        logger.notice("finished diamond-triangle calibration")
        self._timer_calibrate_diamond_triangle.cancel()
        app_model.inference.pose_response_ready -= record_offsets
        algo.pellet_delivery_enabled = before_pellet_delivery_enabled
        action.setIcon(qta.icon("fa5s.crosshairs"))
        action.setChecked(False)
        #
        self._diamond_triangle_calib_run = None  # MUST be before
        #
        try:
            self._handle_diamond_triangle_calib_run(
                positions=positions,
                offsets=offsets,
                diamond_locs3d=diamond_locs3d,
                raw_diamond_3d=raw_diamond_3d,
            )
        finally:
            app_model.status = prev_status

    def on_activated(self, *, target_status: AppModelStatus = AppModelStatus.ACQUIRING):
        logger.success("main window activated")
        self.main_content.on_activated()
        app_status = self._app_model.status
        loaded_cfg = self._app_model.loaded_configuration
        if loaded_cfg is None:
            logger.verbose("No loaded valid config, skipping auto-start")
        else:
            if app_status == AppModelStatus.IDLE:
                if target_status != AppModelStatus.IDLE:
                    self._on_capture_start_stop(True, target_status=target_status)
            else:
                logger.verbose("AppModelStatus not idle, not starting acquisition", app_status)

    def on_calibrate_diamond_triangle(self, is_toggled):
        if is_toggled and self._diamond_triangle_calib_run is None:
            self._diamond_triangle_calib_run = self._make_diamond_triangle_calib_run()
        calib_run = self._diamond_triangle_calib_run
        if is_toggled:
            # this triggers the execution of the first part of the calib run,
            # which is to record enough data
            next(calib_run)

        if not is_toggled:
            # this then triggers the stop of the data recording, and try to use it and if correct save it to file.
            try:
                next(calib_run)
            except StopIteration:
                pass

    def on_3d_calibrate(self, is_toggled):
        self.run_action.setEnabled(not is_toggled)
        if not is_toggled:
            return
        app_model = self._app_model
        prev_status, app_model.status = app_model.status, AppModelStatus.CALIBRATION_3D
        error = "Processing unfinished"
        result_dir: Path = None
        def handle_3d_calib():
            nonlocal error, result_dir
            try:
                result_dir = make_3d_calib(self._app_model)
            except Exception as err:
                logger.exception("3d-calib failed: %s", err)
                error = err
            else:
                error = None

        @invoke_method
        def show_result():
            self.run_action.setEnabled(True)
            self.make_3d_calib_action.setChecked(False)
            self.make_3d_calib_action.setEnabled(True)
            app_model.status = prev_status
            logger.verbose("3d-calib thread joined, error=%s", error)
            if error is not None:
                self._post_api_event(ApiEventKind.calibration3dFailed, dict(reason=str(error)))
                QMessageBox.warning(self, "3D calibration failed", f"Error received: {error}",
                                    QMessageBox.StandardButton.Ok)
                return
            self._post_api_event(ApiEventKind.calibration3dCompleted)
            backup_path = None
            target_calib_dir = Path(self._preferences.configuration_location).joinpath(DEFAULT_3D_CALIB_DIR_NAME)
            if target_calib_dir.exists():
                rsp = QMessageBox.question(
                    self,
                    "3D calibration success",
                    f"Calibration dir already exists ({target_calib_dir}),\n\n"
                    f"do you want to replace ?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if rsp != QMessageBox.StandardButton.Yes:
                    return
                now = datetime.now()
                backup_path = target_calib_dir.parent.joinpath(f"{target_calib_dir.name}_{now.strftime(DATE_TIME_FORMAT)}")
                target_calib_dir.rename(backup_path)
                logger.debug("Previous 3d-calib moved to %s", backup_path)
            shutil.copytree(
                result_dir, target_calib_dir,
                dirs_exist_ok=False,  # default already, but to be sure we want be clean
            )
            if backup_path is not None:
                prev_cam_offsets = backup_path.joinpath(DEFAULT_CAM_OFFSET_FILE_NAME)
                new_cam_offsets = target_calib_dir.joinpath(DEFAULT_CAM_OFFSET_FILE_NAME)
                if prev_cam_offsets.exists() and not new_cam_offsets.exists():
                    shutil.copyfile(prev_cam_offsets, target_calib_dir)
                    logger.verbose("copied previous %s given no new one", DEFAULT_CAM_OFFSET_FILE_NAME)
            self._app_model.reload_calib(target_calib_dir)
            QMessageBox.information(
                self,
                "3D calibration success", f"Result saved into {target_calib_dir}", QMessageBox.StandardButton.Ok)

        def wait_3d_calib_done(thread):
            thread.join()
            show_result()

        executor_thread = threading.Thread(target=handle_3d_calib, name="3d-calibration", daemon=True)
        executor_thread.start()

        waiter_thread = threading.Thread(target=wait_3d_calib_done, name="wait-3d-calibration",
                                         args=(executor_thread,), daemon=True)
        waiter_thread.start()

    @invoke_method
    def _show_msg_box(self, title, text, icon):
        box = QMessageBox()
        box.setWindowTitle(title)
        # box.setModal(True)
        box.setText(text)
        box.setIcon(icon)
        box.show()
        self._add_box_to_open_dialogs(box)

    def _check_diamond_triangle_config(self):
        if self._warned_invalid_dcs_config:
            return
        if not self.has_fully_valid_dcs:
            self._warned_invalid_dcs_config = True
            title = "Missing, or invalid, Diamond-Triangle config"
            text = textwrap.dedent("""
                The coordinate system calibration data is out of date or missing.\n
                Ensure the System Mode is set to Running and perform the following steps:\n
                1) Use the Set and Send buttons in the Hardware Control Panel to move the pellet spoon to a typical delivery position.\n
                2) Start the calibration using the Tools ->Calibrate Coordinate System menu item. You will be notified when the calibration is complete.\n
                """)
            self._show_msg_box(title, text, QMessageBox.Icon.Warning)

    def _finish_close(self):
        logger.info("finishing close ..")
        self.main_content.close()
        dialogs = self._open_dialogs
        self._open_dialogs = []
        for dialog in dialogs:
            dialog.close()
        super().close()

    def close(self):
        logger.debug("received close")
        with self._app_model.app_lock:
            event = self._close_event
            if self._closing:
                if event is not None:
                    event.accept()
                    self._close_event = None
                else:
                    logger.warning("already closing")
                return
            if event is not None:
                event.ignore()
                self._close_event = None
            self._closing = True

        def execute_close():
            self._on_capture_start_stop(False)
            stop_thread = self._stop_capture_thread
            self._timer_calibrate_diamond_triangle.cancel()
            if stop_thread is not None:
                logger.debug("joining stop capture thread")
                stop_thread.join()
            self._app_model.on_close()
            InvokeMethod(self._finish_close)
        close_thread = threading.Thread(target=execute_close)
        close_thread.start()

    def closeEvent(self, event):
        logger.debug("received closeEvent: %s", event)
        self._close_event = event
        self.close()

    def moveEvent(self, e):
        self._preferences.last_window_x = self.pos().x()
        self._preferences.last_window_y = self.pos().y()
        super(MainWindow, self).moveEvent(e)

    def _edit_camera_settings(self):
        # isChecked() has already swapped to the new value by the time this is called
        if not self.edit_camera_settings_action.isChecked():
            self.main_content.set_is_editable(False)
            self.run_action.setEnabled(True)
        else:
            self.main_content.set_is_editable(True)
            self.run_action.setEnabled(False)

    def _show_preferences(self):
        dialog = PreferencesDialog(self._preferences, self._app_model)
        self._add_box_to_open_dialogs(dialog)
        dialog.exec()

    def _set_reset_vat_text(self):
        analysis = self._app_model.analysis
        maint_cfg = analysis.system_maintenance_alarm.config
        prefs = self._preferences
        txt = (
            f"Reset vat pellet count (vat refilled)\n"
            f"{prefs.pellet_load_count_total} of "
            f"{maint_cfg.max_pellets_loaded_count} pellets presented before refill"
        )
        self._reset_pellet_loaded_count_action.setToolTip(txt)

    def _set_reset_cage_clean_text(self):
        n_days = self._app_model.get_days_before_cage_clean()
        if n_days < 0:
            msg = f"Cage cleaning is overdue by {-n_days} days"
        else:
            msg = f"{n_days} days until next required cleaning"
        txt = f"Mark the cage as cleaned.\n{msg}"
        self._reset_cage_clean_action.setToolTip(txt)

    def _set_autoclamp_evasion(self, det: AutoClampEvasionDetector):
        if not self._app_model.hardware.tunnel_headfix_enabled:
            self._reset_autoclamp_evasion_action.setVisible(False)
            return
        det = self._app_model.analysis.autoclamp_evasion_detector
        action = self._reset_autoclamp_evasion_action
        action.setVisible(det.is_engaged or os.getenv("AUTOTRAINER_SHOW_AUTOCLAMP_EVASION") == "1")
        action.setToolTip(f"Reset AutoClamp Evasion\n{det.pellets_consumed} out of {det.config.pellets_consumed_trigger}")

    def _create_actions(self):
        action = self.edit_camera_settings_action = QAction(QIcon(qta.icon("fa5s.edit")), "Edit Camera Settings", self)
        action.setToolTip("Edit Camera Settings")
        action.setCheckable(True)
        action.setChecked(False)
        action.triggered.connect(self._edit_camera_settings)

        action = self.run_action = QAction(QIcon(qta.icon("ei.play")), "Start", self)
        action.setToolTip("Start or stop acquisition")
        action.setCheckable(True)
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_R))
        action.triggered.connect(self._on_capture_start_stop)

        action = self.animal_in_device_action = QAction(QIcon(qta.icon("fa5s.vector-square")), "Animal in device", self)
        action.setCheckable(True)
        action.triggered.connect(self._on_animal_in_device_triggered)

        action = self.animal_in_training_action = QAction(QIcon(qta.icon("fa5s.chalkboard-teacher")), "Animal in training", self)
        action.setCheckable(True)
        action.triggered.connect(self._on_animal_in_training_triggered)

        action = self.show_reach_event_action = QAction(QIcon(qta.icon("fa5s.bezier-curve")), "Show Previous Reach", self)
        action.setCheckable(True)
        action.setEnabled(False)  # comment me to be able to show 20260205_agx001_trial011 on start
        action.triggered.connect(self.on_show_reach_event)

        action = self._reset_pellet_loaded_count_action = QAction(QIcon(qta.icon("fa5s.fill")), "Reset Pellet VAT Load Count", self)
        action.triggered.connect(self.on_reset_pellet_loaded_count)

        action = self._reset_cage_clean_action = QAction(QIcon(qta.icon("fa5s.broom")), "Reset Cage Clean", self)
        action.triggered.connect(self.on_reset_cage_clean)

        action = self._reset_autoclamp_evasion_action = QAction(QIcon(qta.icon("ei.vimeo")), "Reset AutoClamp Evasion", self)
        action.setVisible(False)
        action.triggered.connect(self.on_reset_autoclamp_evasion)

        action = self.next_training_phase_action = QAction(QIcon(qta.icon("fa5s.arrow-alt-circle-right")), "Next Phase", self)
        action.setVisible(False)
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Right))
        action.triggered.connect(self.on_next_plan_phase)

        action = self.previous_training_phase_action = QAction(QIcon(qta.icon("fa5s.arrow-alt-circle-left")), "Previous Phase", self)
        action.setVisible(False)
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Left))
        action.triggered.connect(self.on_previous_plan_phase)

        self._diamond_triangle_calib_run = None
        self._timer_calibrate_diamond_triangle = no_op_timer
        action = self.calib_diamond_triangle_action = QAction(QIcon(qta.icon("fa5s.crosshairs")), "Calibrate Coordinate System", self)
        action.setToolTip("Calibrate the relative offset between the pellet delivery spoon and the tunnel")
        action.setCheckable(True)
        action.triggered.connect(self.on_calibrate_diamond_triangle)
        action.setEnabled(False)

        action = self.make_3d_calib_action = QAction(QIcon(qta.icon("fa5s.crosshairs")), "Make 3D calibration", self)
        action.setCheckable(True)
        action.triggered.connect(self.on_3d_calibrate)

        action = self.view_diagnostics_action = QAction("Diagnostics", self)
        action.setToolTip("Show or hide diagnostics panel")
        action.setCheckable(True)
        action.setChecked(self.main_content.is_diagnostics_visible)
        action.triggered.connect(self._toggle_diagnostics_view)

        action = self.debug_action = QAction("Debug", self)
        action.setCheckable(True)
        action.triggered.connect(self._toggle_debug_view)

        action = self.load_cell_trigger_action = QAction("Load Cell", self)
        action.setCheckable(True)
        action.triggered.connect(self._internal_simulate_trigger_load_cell)

        action = self.force_headbar_detector_action = QAction("HeadBar", self)
        action.setCheckable(True)
        action.triggered.connect(self._internal_set_force_headbar_detector)

        action = self.pellet_seen_action = QAction("Pellet Seen", self)
        action.triggered.connect(self._internal_set_pellet_seen)

        action = self.mouse_seen_action = QAction("Mouse Seen", self)
        action.triggered.connect(self._internal_set_mouse_seen)

        action = self.mouse_near_pellet_action = QAction("Hands min-Y DCS", self)
        action.triggered.connect(self._internal_mouse_near_pellet)

        action = self.analysis_results_action = QAction("Analysis-Result", self)
        action.setCheckable(True)
        action.triggered.connect(self._internal_detection_result_toggle)

        action = self.preferences_action = QAction(QIcon(qta.icon("fa5s.cog")), "Preferences", self)
        action.triggered.connect(self._show_preferences)

        tooltip = "Reset pellet and reach counts for this animal"
        action = self._reset_animal_pellet_counts_action = QAction(QIcon(qta.icon("fa5s.sync")), tooltip, self)
        action.triggered.connect(self._reset_animal_pellet_counts)

        action = self.emergency_stop_action = QAction("Emergency", self)
        action.setCheckable(True)

        action = self.quit_action = QAction("Quit")
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Q))
        action.triggered.connect(lambda: self._app.quit())

    def _configure_menubar(self):
        menu_bar = self.menuBar()

        menu_bar.setObjectName("MenuBar")
        menu_bar.setStyleSheet("#MenuBar {background-color: #eee}")

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.quit_action)

        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(self.edit_camera_settings_action)

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(self.calib_diamond_triangle_action)
        tools_menu.addAction(self.make_3d_calib_action)

        if self._is_dev:
            view_menu = menu_bar.addMenu("View")
            view_menu.addAction(self.view_diagnostics_action)
            view_menu.addAction(self.debug_action)

    def _configure_toolbar(self):

        app_model = self._app_model
        behavior = app_model.behavior

        toolbar = QToolBar("Run Toolbar")
        toolbar.setFloatable(False)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("System Mode:"))
        combo = self._app_model_status_combo = QComboBox()
        combo.setMinimumWidth(100)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.addItem("Idle", userData=AppModelStatus.IDLE)
        combo.addItem("Running", userData=AppModelStatus.ACQUIRING)
        combo.currentIndexChanged.connect(self._on_system_mode_combo_changed)
        toolbar.addWidget(combo)
        # toolbar.addAction(self.run_action)
        # toolbar.addAction(self.animal_in_device_action)
        # toolbar.addAction(self.animal_in_training_action)
        toolbar.addSeparator()

        toolbar.addAction(self.show_reach_event_action)
        toolbar.addAction(self._reset_pellet_loaded_count_action)
        toolbar.addAction(self._reset_autoclamp_evasion_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        toolbar.addWidget(QLabel("Notes:"))
        self._notes = QLineEdit()
        self._notes.setMinimumWidth(100)
        self._notes.setText(self._app_model.notes)
        self._notes.textChanged.connect(self.notes_changed)
        self._notes.setContentsMargins(4, 0, 8, 0)
        # note: this margin allows the emergency button to be greater
        toolbar.addWidget(self._notes)

        toolbar.addWidget(QLabel("Subject:"))
        combo = self._animal_dropdown_combo = QComboBox()
        combo.setMinimumWidth(100)
        combo.setEditable(True)
        combo.setDuplicatesEnabled(False)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.currentIndexChanged.connect(self._animal_changed)
        combo.lineEdit().editingFinished.connect(self._add_animal)
        toolbar.addWidget(combo)

        toolbar.addAction(self._reset_animal_pellet_counts_action)

        toolbar.addSeparator()

        combo = self._training_mode_combo = QComboBox()
        combo.setDuplicatesEnabled(False)
        for mode in TrainingMode:
            combo.addItem(mode.value,
                          (mode,)  # userData: encapsulated in a tuple,
                          # otherwise pyside drops the enum member and only keep the value string.
                          # this is because it's a typed str-subclass enum.
                          )

        def training_mode_index_changed(_):
            selected_mode = self._training_mode_combo.currentData()[0]  # unpack from tuple, see above.
            app_model.training_mode = selected_mode
        combo.currentIndexChanged.connect(training_mode_index_changed)

        combo = self._training_plan_combo = QComboBox()

        def training_plan_index_changed(_):
            plan_id = self._training_plan_combo.currentData()
            selected_plan: Optional[TrainingPlan] = app_model.get_training_plan_by_id(plan_id)
            logger.debug("plan changed: %s -> %s", plan_id, selected_plan)
            app_model.training_plan = selected_plan

        combo.currentIndexChanged.connect(training_plan_index_changed)
        self._widget_training_plan_action = None

        if self.main_content.protocol_ui_enabled:
            label = QLabel("Training Mode:")
            label.setContentsMargins(8, 0, 0, 0)
            toolbar.addWidget(label)
            toolbar.addWidget(self._training_mode_combo)

            label = QLabel("Protocol:")
            label.setContentsMargins(8, 0, 0, 0)
            widget = QWidget()
            self._widget_training_plan_action = toolbar.addWidget(widget)
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(label)
            layout.addWidget(self._training_plan_combo)

            toolbar.addAction(self.previous_training_phase_action)
            toolbar.addAction(self.next_training_phase_action)

            def update_training_mode(training_mode: TrainingMode):
                logger.debug("Updating training_mode to %s", training_mode)
                is_non_manual = training_mode != TrainingMode.MANUAL
                self._widget_training_plan_action.setVisible(is_non_manual)
                self._status_training_widget.setVisible(is_non_manual)
                animal = app_model.selected_animal
                plan_id = None if animal is None else animal.training.current_protocol
                training_plan_idx = self._training_plan_index_by_plan_id.get(plan_id, -1)
                self._training_plan_combo.blockSignals(True)
                self._training_plan_combo.setCurrentIndex(training_plan_idx)
                self._training_plan_combo.blockSignals(False)
                # self.main_content.training_plan_changed.emit(self._app_model.training_plan)
                self._app_model.training_mode = training_mode
                self._refresh_prev_next_phases()

            update_training_mode(self._app_model.training_mode)
            self.training_mode_changed.connect(update_training_mode)
        else:
            self._training_mode_combo.setCurrentIndex(self._training_mode_combo.findText(TrainingMode.MANUAL.value))
            app_model.training_mode = TrainingMode.MANUAL
            self._status_training_widget.setVisible(False)

        toolbar.addSeparator()

        toolbar.addAction(self.preferences_action)

        toolbar.addSeparator()

        emergency_button = QPushButton("Emergency")
        emergency_button.setCheckable(True)
        emergency_button.setObjectName("EmergencyButton")
        emergency_button.setStyleSheet("#EmergencyButton {background-color: red; color: white; min-width: 100px}")

        @invoke_method
        def update_emergency_ui(is_toggled: bool, source: str):
            emergency_button.setText("Resume" if is_toggled else "Emergency")
            self.setWindowTitle(f"{self._title} - BEHAVIOR ALGORITHM PAUSED - Source: {source}" if is_toggled else self._title)
            if source != "user-button":
                emergency_button.blockSignals(True)  # prevent overwrite of reason with user-button
                emergency_button.setChecked(is_toggled)
                emergency_button.blockSignals(False)

        def emergency_stop_triggered(is_toggled: bool):
            logger.verbose("emergency_stop_triggered: %s", is_toggled)
            (behavior.emergency_stop if is_toggled else behavior.emergency_resume)("user-button")

        emergency_button.toggled.connect(emergency_stop_triggered)
        behavior.emergency_stopped += lambda src: update_emergency_ui(True, source=src)
        behavior.emergency_resumed += lambda src: update_emergency_ui(False, source=src)

        toolbar.addWidget(emergency_button)

        if self._is_dev:
            self.addToolBarBreak()
            toolbar = self._dev_toolbar = QToolBar("Dev Toolbar")
            toolbar.setContentsMargins(0, 0, 0, 0)
            # toolbar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            self.addToolBar(toolbar)
            toolbar.setFloatable(False)
            toolbar.setMovable(False)
            toolbar.addAction(self.load_cell_trigger_action)
            toolbar.addAction(self.force_headbar_detector_action)
            toolbar.addAction(self.pellet_seen_action)
            toolbar.addAction(self.mouse_seen_action)
            toolbar.addAction(self.mouse_near_pellet_action)
            toolbar.addAction(self.analysis_results_action)
            widget = QWidget()
            widget.setContentsMargins(4, 0, 0, 0)
            self._internal_analysis_widget_toolbar = toolbar.addWidget(widget)
            self._internal_analysis_widget_toolbar.setVisible(False)
            hbox = QHBoxLayout(widget)
            label = QLabel("P:")
            label.setToolTip("Pellets Presented")
            hbox.addWidget(label)
            spinbox = self._internal_pellet_presented_spinbox = QSpinBox()
            spinbox.setToolTip(label.toolTip())
            hbox.addWidget(spinbox)
            label = QLabel("R:")
            label.setToolTip("Pellets Reached")
            hbox.addWidget(label)
            spinbox = self._internal_pellet_total_reaches_spinbox = QSpinBox()
            spinbox.setToolTip(label.toolTip())
            hbox.addWidget(spinbox)
            label = QLabel("C:")
            label.setToolTip("Pellets Consumed")
            hbox.addWidget(label)
            spinbox = self._internal_pellet_consumed_spinbox = QSpinBox()
            spinbox.setToolTip(label.toolTip())
            hbox.addWidget(spinbox)
            label = QLabel("S:")
            label.setToolTip("Success Reaches")
            hbox.addWidget(label)
            spinbox = self._internal_success_reaches_spinbox = QSpinBox()
            spinbox.setToolTip(label.toolTip())
            hbox.addWidget(spinbox)
            #
            hbox.addWidget(QLabel("Shift:"))
            spinbox = self._internal_shift_x_spinbox = QDoubleSpinBox()
            spinbox.setToolTip("X shift")
            spinbox.setRange(-9, 9)
            spinbox.setDecimals(1)
            hbox.addWidget(spinbox)
            spinbox = self._internal_shift_y_spinbox = QDoubleSpinBox()
            spinbox.setToolTip("Y shift")
            spinbox.setRange(-9, 9)
            spinbox.setDecimals(1)
            hbox.addWidget(spinbox)
            spinbox = self._internal_shift_z_spinbox = QDoubleSpinBox()
            spinbox.setToolTip("Z shift")
            spinbox.setRange(-9, 9)
            spinbox.setDecimals(1)
            hbox.addWidget(spinbox)
            # for i in range(hbox.count()):
            #     w = hbox.itemAt(i).widget()
            #     # w.setContentsMargins(0, 0, 0, 0)
            #     # w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            # unnecessary with:
            hbox.setContentsMargins(0, 0, 0, 0)
            # plus:
            toolbar.setMaximumHeight(toolbar.minimumSizeHint().height())

    def _update_tunnel_headfix_visibility(self, is_enabled: bool):
        for action in (
            self.load_cell_trigger_action,
            self.force_headbar_detector_action,
        ):
            if not is_enabled and action.isChecked():
                action.setChecked(False)
            action.setVisible(is_enabled)
        self._status_label_magnet_intensity.setVisible(is_enabled)
        self.calib_diamond_triangle_action.setVisible(is_enabled)

    def _configure_statusbar(self):
        self._status_label = QLabel("")
        bar = self._status_bar = QStatusBar(self)
        bar.addWidget(self._status_label)
        bar.setSizeGripEnabled(False)
        widget = self._status_training_widget = QWidget()
        widget.setVisible(False)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(hbox)
        lbl = self._status_label_pos = XYZQLabel(prefix="Position: ", sep=", ", tail=" mm")
        hbox.addWidget(lbl)
        hbox.addWidget(_make_separator())
        lbl = self._status_label_send_pos = XYZQLabel(prefix="Send: ", sep=", ", tail=" mm")
        hbox.addWidget(lbl)
        hbox.addWidget(_make_separator())
        lbl = self._status_label_magnet_intensity = QLabel("Magnet: N/A")
        hbox.addWidget(lbl)
        bar.addPermanentWidget(widget)
        self.setStatusBar(bar)
        self._update_tunnel_headfix_visibility(self._app_model.hardware.tunnel_headfix_enabled)

    def _toggle_diagnostics_view(self):
        self.main_content.set_diagnostics_visible(not self.main_content.is_diagnostics_visible)
        self.view_diagnostics_action.setChecked(self.main_content.is_diagnostics_visible)

    def _toggle_debug_view(self, toggled: True):
        v = self._debug_view
        if v is None:
            v = self._debug_view = DebugView(self)
            def close_debug_view_event(evt, orig_close=v.closeEvent):
                self._debug_view = None
                orig_close(evt)
                self.debug_action.setChecked(False)
            v.closeEvent = close_debug_view_event
            v.show()
            self._add_box_to_open_dialogs(v)
            self.debug_action.setChecked(True)
        else:
            v.close()

    def _show_message(self, title: str, message: str):
        @invoke_method
        def show_in_gui_thread(title=title, message=message):
            dlg = QMessageBox(self)
            dlg.setWindowTitle(title)
            dlg.setText(message)
            dlg.exec()
        show_in_gui_thread()

    @invoke_method
    def _on_preferences_property_changed(self, name, value, _):
        prefs = self._preferences
        if name == prefs.LOG_LEVEL:
            self._update_log_level(value)
        elif name == prefs.REMOVE_RAW_DATA_WHEN_INACTIVE_SESSION:
            self._app_model.behavior.algorithm.clean_raw_data_on_inactive_session = value
        elif name == prefs.PELLET_LOAD_COUNT_TOTAL:
            self._set_reset_vat_text()
        elif name == prefs.SERIAL_NUMBER:
            self._title = _make_window_title(prefs)
            self.setWindowTitle(self._title)

    @invoke_method
    def _on_alarm_monitor_property_changed(self, name, value, _):
        pass

    @invoke_method
    def _on_system_maintenance_prop_changed(self, name, value, _):
        mon = self._app_model.analysis.system_maintenance_alarm
        if name == mon.CONFIG:
            self._set_reset_vat_text()

    def _simulate_intersession_segmentation(self, intersession_block):
        logger.verbose("Simulate feed-intersession-analysis: %s", intersession_block)
        inference = self._app_model.inference
        inference._data_monitor_proc.stop_recorded.wait()
        logger.debug("got stop recorded")
        inference._data_monitor_proc.stop_recorded.clear()
        intersession_block.frame_count = 42
        time.sleep(2.5)

    @staticmethod
    def _simulate_intersession_process(*args, fake_result, **kwargs):
        # NB: must be static method given it's executed in a sub-process, so must no get the whole main-window
        # instance tried to be serialized (would most likely fails/error) !
        logger.verbose("Simulate intersession process: %s", fake_result)
        time.sleep(1.5)
        return fake_result

    def _internal_simulate_trigger_load_cell(self):
        if not self._app_model.hardware.tunnel_headfix_enabled:
            logger.warning("Blocked internal load-cell simulation: tunnel/headfix hardware is disabled")
            self.load_cell_trigger_action.setChecked(False)
            return
        is_checked = self.load_cell_trigger_action.isChecked()
        app_model = self._app_model
        load_cell_monitor = app_model.analysis.load_cell_monitor
        if not is_checked and self.analysis_results_action.isChecked():
            logger.verbose("Patching intersession segmentation and detection with simulate")
            inference = app_model.inference
            inference._feed_intersession_analysis_execute = self._simulate_intersession_segmentation
            x, y, z = (spinbox.value() for spinbox in (
                self._internal_shift_x_spinbox, self._internal_shift_y_spinbox, self._internal_shift_z_spinbox))
            res = IntersessionResponse(
                pellets_presented=self._internal_pellet_presented_spinbox.value(),
                total_reaches=self._internal_pellet_total_reaches_spinbox.value(),
                successful_reaches=self._internal_success_reaches_spinbox.value(),
                food_consumed=self._internal_pellet_consumed_spinbox.value(),
                rh_max_vp_list=[Offset3DTuple(x, y, z)]
            )
            inference._intersession_process_execute = partial(self._simulate_intersession_process, fake_result=res)
        load_cell_monitor.force_engaged(is_checked)

    def _internal_set_force_headbar_detector(self):
        if not self._app_model.hardware.tunnel_headfix_enabled:
            logger.warning("Blocked internal headbar detector simulation: tunnel/headfix hardware is disabled")
            self.force_headbar_detector_action.setChecked(False)
            return
        new_value = self.force_headbar_detector_action.isChecked()
        self._app_model.analysis.headbar_pressure_monitor.force_engaged(new_value)

    def _internal_set_pellet_seen(self):
        self._app_model.behavior.algorithm.update_pellet_seen(True)

    def _internal_set_mouse_seen(self):
        self._app_model.behavior.algorithm.update_mouse_seen(True)

    def _internal_mouse_near_pellet(self):
        # actually nothing needs be done
        algo = self._app_model.behavior.algorithm
        ctx = algo.uncover_context
        cfg = algo.active_config.pellet_uncover
        ctx.start_y_dcs_valid_perf_c = get_perf_now()
        ctx.start_min_y = cfg.min_y_dcs - 5
        ctx.y_dcs_valid = True

    def _internal_detection_result_toggle(self):
        inference = self._app_model.inference
        is_checked = self.analysis_results_action.isChecked()
        self._internal_analysis_widget_toolbar.setVisible(is_checked)
        if is_checked:
            self._orig_inference_analysis_feed = inference._feed_intersession_analysis_execute
            self._orig_inference_analysis_process = inference._intersession_process_execute
        else:
            logger.verbose("Restoring intersession segmentation and detection to real procedures")
            inference._feed_intersession_analysis_execute = self._orig_inference_analysis_feed
            inference._intersession_process_execute = self._orig_inference_analysis_process

    def _simulate_sessions(
        self,
        *,
        n_sessions: int = 1,
        n_trials: int = 1,
        step_sleep: float = 0.2,
        monitor_sleep: float = 0.5,
        load_sleep: float = 0.25,
        sess_sleep: float = 0.5,
        rand_mouse_seen: float = 1,
        rand_hands_near_pellet: float = 1,
        rand_headfix_trigger = 0.5,
        wait_released: bool=True,
        print: Callable[[str], None] = lambda s: logger.info(s),  # for interactive logs output
    ):
        app = self._app_model
        pellet_m = app.behavior.system_machine.pellet
        algo = app.behavior.algorithm
        infe = app.inference
        def do_sleep():
            time.sleep(random.uniform(step_sleep, 3 * step_sleep))
        for sess_idx in range(n_sessions):
            if algo.status != BehaviorAlgoStatus.ANIMAL_IN_TRAINING:
                return
            set_headfix = False
            print("starting new simulate session")
            self.load_cell_trigger_action.trigger()
            do_sleep()
            t_end = time.perf_counter() + 10
            while algo.capture_status != CaptureProcessStatus.RECORDING:
                time.sleep(0.05)
                if time.perf_counter() > t_end:
                    print(f"awaited RECORDING but is still {algo.capture_status}")
                    break
            for idx in range(n_trials):
                set_headfix = False
                if random.random() <= rand_headfix_trigger:
                    # self.force_headbar_detector_action.setChecked(True)
                    set_headfix = True
                    do_sleep()
                    for _ in range(random.randint(1, 3)):
                        print("setting headfix trigger")
                        # self.force_headbar_detector_action.toggle()
                        self.force_headbar_detector_action.trigger()
                        for _ in range(2):
                            do_sleep()
                            algo.update_mouse_seen()
                        do_sleep()
                print("wait monitoring")
                t_end = time.perf_counter() + 15
                while pellet_m.state != PelletState.monitoring:
                    if algo.head_fixation_enabled:
                        # new autoclamp
                        break
                    if algo.status != BehaviorAlgoStatus.ANIMAL_IN_TRAINING:
                        return
                    time.sleep(0.1)
                    if algo.system_state == SystemState.intersession:
                        continue
                    if time.perf_counter() > t_end:
                        print(f"waited monitoring but still {pellet_m.state}")
                        break
                time.sleep(random.uniform(monitor_sleep, 2 * monitor_sleep))
                if random.random() <= rand_mouse_seen:
                    print("setting mouse seen")
                    algo.update_mouse_seen()
                    do_sleep()
                else:
                    set_headfix = False
                if random.random() <= rand_hands_near_pellet:
                    print("setting mouse near pellet")
                    self.mouse_near_pellet_action.toggle()
                    self.mouse_near_pellet_action.trigger()
                    do_sleep()
                if set_headfix:
                    for _ in range(2):
                        # self.force_headbar_detector_action.toggle()
                        self.force_headbar_detector_action.trigger()
                        for _ in range(2):
                            do_sleep()
                            algo.update_mouse_seen()
                        do_sleep()
                if wait_released:
                    print("waiting released")
                    t_end = time.perf_counter() + 3
                    while True:
                        if pellet_m.covered_state is False:
                            print("is released")
                            break
                        if algo.status != BehaviorAlgoStatus.ANIMAL_IN_TRAINING:
                            return
                        if time.perf_counter() > t_end:
                            break
                        time.sleep(0.1)
                time.sleep(random.uniform(load_sleep, 2 * load_sleep))
                print("loading pellet")
                pellet_m.load_pellet(reason="simulate", force=True)  # this basically simulate pellet dropped or eaten
                do_sleep()
                print("waiting pellet monitoring or retract")
                while pellet_m.state not in {PelletState.monitoring, PelletState.retract}:
                    time.sleep(0.05)
                    if algo.status != BehaviorAlgoStatus.ANIMAL_IN_TRAINING:
                        return
                    if infe.status in {
                        InferenceStatus.stopped,
                        InferenceStatus.stopping,
                    }:
                        return
                # if infe.status == InferenceStatus.intersession or pellet_m.state == PelletState.retract:
                #     break
                do_sleep()
            #
            self.load_cell_trigger_action.trigger()
            time.sleep(2)
            while infe.status != InferenceStatus.live:
                if infe.status in {InferenceStatus.stopped, InferenceStatus.stopping}:
                    return
                time.sleep(0.05)
            time.sleep(sess_sleep)
        # end for session_idx in
        print("finished all")

    def notes_changed(self, value: str):
        self._app_model.notes = value

    def _add_animal(self):
        self._app_model.add_animal(self._animal_dropdown_combo.currentText(), select=True)

    def _animal_changed(self, _):
        if self._animal_dropdown_combo.currentIndex() in (0, -1):
            self._app_model.selected_animal = None
        else:
            animal_id = self._animal_dropdown_combo.currentData()
            animal: AnimalSubject = self._app_model.get_animal_by_id(animal_id)
            self._app_model.selected_animal = animal

    @invoke_method
    def _refresh_prev_next_phases(self):
        if not self.main_content.protocol_ui_enabled:
            self.previous_training_phase_action.setVisible(False)
            self.next_training_phase_action.setVisible(False)
            return
        attached = self._app_model.attached_plan
        if attached is None or self._app_model.training_mode != TrainingMode.MANUAL_WITH_PROTOCOL:
            self.previous_training_phase_action.setVisible(False)
            self.next_training_phase_action.setVisible(False)
            return
        for action, can_do, direction in (
            (self.previous_training_phase_action, attached.can_fallback, "left"),
            (self.next_training_phase_action, attached.can_advance, "right"),
        ):
            action.setVisible(True)
            name = f"fa5s.arrow-alt-circle-{direction}"
            action.setIcon(qta.icon(name))
            action.setEnabled(can_do)

    @invoke_method
    def _on_day_changed(self, day: date):
        self._set_reset_cage_clean_text()

    @invoke_method
    def _on_app_model_property_changed(self, name: str, value, prev_value):
        app_model = self._app_model
        props = app_model.Props
        #
        if name == props.ACQUISITION_RUNNING:
            self.running_status_changed.emit(value)

        elif name == props.STATUS:
            logger.debug("got new app model status: %s", value)

            valid_dcs = self.has_fully_valid_dcs

            self.blockSignals(True)

            combo_idx = self._app_model_status_combo.findData(value)
            if combo_idx >= 0:
                self._app_model_status_combo.setCurrentIndex(combo_idx)

            if value is AppModelStatus.IDLE:
                self._warned_invalid_dcs_config = False
                for action in (
                    self.calib_diamond_triangle_action,
                    self.animal_in_device_action,
                    self.animal_in_training_action,
                ):
                    action.setEnabled(False)
                    action.setChecked(False)
                for item in (
                    self.make_3d_calib_action,
                    self.run_action,
                    self.animal_in_device_action,
                    self.animal_in_training_action,
                    self._animal_dropdown_combo,
                    self._training_mode_combo,
                    self._training_plan_combo,
                ):
                    item.setEnabled(True)
                self.animal_in_training_action.setEnabled(valid_dcs)

            elif value is AppModelStatus.ACQUIRING:
                for action in (
                    self.calib_diamond_triangle_action,
                    self.animal_in_device_action,
                    self.animal_in_training_action,
                ):
                    action.setEnabled(True)
                    action.setChecked(False)
                for item in (self._animal_dropdown_combo, self._training_mode_combo, self._training_plan_combo):
                    item.setEnabled(True)
                self.animal_in_training_action.setEnabled(valid_dcs)

            elif value in {AppModelStatus.CALIBRATION_3D, AppModelStatus.CALIBRATION_DCS}:
                for item in (
                    self._training_mode_combo,
                    self._training_plan_combo,
                    self._animal_dropdown_combo,
                    self.calib_diamond_triangle_action,
                    self.animal_in_device_action,
                    self.animal_in_training_action,
                ):
                    item.setEnabled(False)

            elif value is AppModelStatus.ANIMAL_IN_DEVICE:
                self.animal_in_device_action.setChecked(True)
                self.animal_in_training_action.setChecked(False)
                for item in (
                    self._training_mode_combo,
                    self._training_plan_combo,
                ):
                    item.setEnabled(True)
                for item in (
                    self._animal_dropdown_combo,
                    self.calib_diamond_triangle_action,
                    self.make_3d_calib_action,
                    self.calib_diamond_triangle_action,
                ):
                    item.setEnabled(False)
                self.animal_in_training_action.setEnabled(valid_dcs)

            elif value is AppModelStatus.ANIMAL_IN_TRAINING:
                for action in (self.animal_in_device_action, self.animal_in_training_action):
                    action.setChecked(True)
                for item in (
                    self._training_mode_combo,
                    self._training_plan_combo,
                    self._animal_dropdown_combo,
                    self.calib_diamond_triangle_action,
                    self.make_3d_calib_action,
                    self.calib_diamond_triangle_action,
                ):
                    item.setEnabled(False)

            else:
                logger.warning("unhandled app model status: %s", value)

            self.blockSignals(False)

            self.main_content.set_is_capture_active(value != AppModelStatus.IDLE)

        elif name == props.ANIMALS:
            self._reload_animals(value)

        elif name == props.SELECTED_ANIMAL:
            animal_dropdown = self._animal_dropdown_combo
            animal_dropdown.blockSignals(True)
            if value is None:
                animal_dropdown.setCurrentIndex(0)
            else:
                assert isinstance(value, AnimalSubject)
                index = animal_dropdown.findText(value.name)
                if index != -1:
                    animal_dropdown.setCurrentIndex(index)
                else:
                    logger.warning("Cannot select animal %s given not in current list")
                    animal_dropdown.addItem(value.name, value.id)

            animal_dropdown.blockSignals(False)
            self._refresh_prev_next_phases()

        elif name == props.TRAINING_MODE:
            self.training_mode_changed.emit(value)

        elif name == props.TRAINING_PLAN:
            if value is not None:
                assert isinstance(value, TrainingPlan)
                index = self._training_plan_index_by_plan_id.get(value.plan_id, -1)
            else:
                index = -1
            logger.debug("changing plan: index=%s -> %s", index, None if value is None else value.name)
            self._training_plan_combo.blockSignals(True)
            self._training_plan_combo.setCurrentIndex(index)
            self._training_plan_combo.blockSignals(False)
            self._refresh_prev_next_phases()

        elif name == props.TRAINING_PLANS:
            self._set_training_plans(value)

        elif name == props.TRAINING_PHASE:
            self._refresh_prev_next_phases()

    @invoke_method
    def _reload_animals(self, animals: List[AnimalSubject]):
        # get current selected animal before adding them,
        # given when adding that's modifying the currently selected one too,
        # which reset the preference selected to that one...
        find_animal_name = (
            self._preferences.selected_animal if self._app_model.selected_animal is None
            else self._app_model.selected_animal.name
        )

        # prevent on_animal_changed event:
        combo = self._animal_dropdown_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("", None)
        for idx, animal in enumerate(animals):
            combo.addItem(animal.name, animal.id)
        combo.blockSignals(False)

        # we set the good one here:
        if find_animal_name is not None:
            index = combo.findText(find_animal_name)
            if index != -1:
                combo.setCurrentIndex(index)
        else:
            combo.setCurrentIndex(0)

    @staticmethod
    def _update_log_level(value: int):
        # controlled via console and file handlers now
        get_console_handler().setLevel(value)

    @invoke_method
    def _on_inference_property_changed(self, name: str, value, old_value):
        inference = self._app_model.inference
        if name == inference.STATUS:
            self.calib_diamond_triangle_action.setEnabled(value == InferenceStatus.live)

    @invoke_method
    def _on_hardware_property_changed(self, property_name: str, value, _):
        hard = self._app_model.hardware
        if property_name == hard.HEAD_MAGNET_INTENSITY:
            if value is None:
                value = math.nan
            self._status_label_magnet_intensity.setText(f"Magnet: {value:.1f}%")
        elif property_name in {hard.POS_XYZ, hard.SEND_X, hard.SEND_Y, hard.SEND_Z}:
            self._status_label_pos.update_coordinate(hard.last_dcs_position)
            self._status_label_send_pos.update_coordinate(hard.last_dcs_set_position)
        elif property_name == hard.TUNNEL_HEADFIX_ENABLED:
            self._update_tunnel_headfix_visibility(value)

    @invoke_method
    def _on_behavior_algo_property_changed(self, name: str, value, _):
        props = BehaviorAlgoProps
        if name == props.CAGE_CLEAN_CONFIG:
            self._set_reset_cage_clean_text()

    @invoke_method
    def _on_autoclamp_evasion_property_changed(self, name: str, value, _):
        det = self._app_model.analysis.autoclamp_evasion_detector
        # if name in (det.IS_ENGAGED, det.CONFIG, det.PELLETS_CONSUMED):
        self._set_autoclamp_evasion(det)

    @invoke_method
    def _set_training_plans(self, plans: List[PlanInfo]):
        app_model = self._app_model
        combo = self._training_plan_combo
        combo.blockSignals(True)
        combo.clear()
        has_some = len(plans) > 0
        empty_txt = "" if has_some else " " * 64
        tooltip_txt = (
            "Select a training protocol" if has_some
            else "There are no training protocols in the Autotrainer folder"
        )
        combo_indices_map: Dict[Optional[str], int]
        combo_indices_map = self._training_plan_index_by_plan_id = {
            plan.plan_id: idx
            for idx, plan in enumerate(plans)
        }
        for plan_index, plan in enumerate(plans):
            combo.addItem(plan.name, userData=plan.plan_id)
            combo.setItemData(plan_index, plan.description, Qt.ToolTipRole)
        combo.addItem(empty_txt, userData=None)  # put it last
        combo_indices_map[None] = len(plans)
        combo.setItemData(len(plans), tooltip_txt, Qt.ToolTipRole)
        combo.blockSignals(False)
        animal = app_model.selected_animal
        if animal is None:
            combo.blockSignals(True)  # required to not induce loop
            combo.setCurrentIndex(len(plans))
            combo.blockSignals(False)  # required to not induce loop
            return
        plan_id = animal.training.current_protocol
        combo.blockSignals(True)  # required to not induce loop
        if plan_id is None:
            combo.setCurrentIndex(len(plans))
        else:
            plan_combo_index = self._training_plan_index_by_plan_id.get(plan_id, None)
            if plan_combo_index is None:
                logger.warning("Animal has current protocol %r but no such protocol found", plan_id)
                combo.setCurrentIndex(len(plans))
            else:
                combo.setCurrentIndex(plan_combo_index)
        combo.blockSignals(False)  # required to not induce loop

    @invoke_method
    def _on_app_model_configuration_loaded(self, config):
        self._set_training_plans(self._app_model.training_plans)

    @invoke_method
    def _on_inference_analysis_result_ready(self, prj: ProjectInfo, rsp: IntersessionResponse):
        # logger.debug("enabling show_reach_event_action, rsp=%s", rsp)
        self._previous_intersession_analysis_rsp = (prj, rsp)
        self.show_reach_event_action.setEnabled(True)
        if self.show_reach_event_action.isChecked():
            self.on_show_reach_event(True)

    @invoke_method
    def _on_training_plan_deserialized(self, plan: TrainingPlan, result: LoadProgressResult, *,
                                       force_update: bool = False):
        app_model = self._app_model
        #
        if result == LoadProgressResult.OUT_OF_DATE:
            if force_update:  # loop detected
                app_model.training_plan = None
                logger.warning("detected training_plan_deserialize loop, result=%s plan=%s", result, plan)
                return
            msg = (
                "This training protocol has been revised since it was last used with this animal.\n"
                "Progress for this animal for training phases that have changed will be reset for this animal."
            )

        elif result == LoadProgressResult.UNKNOWN_ERROR:
            if force_update:
                app_model.training_plan = None
                app_model.on_error(
                    "Error Loading Training Progress",
                    "There was an unexpected error loading progress for this training protocol with this animal.\n"
                    "It cannot be used with this animal at this time."
                )
                return
            msg = (
                "There was an unexpected error loading progress for this training protocol with this animal.\n"
                "You can reset all progress and attempt to load again."
            )
        else:
            # should be LoadProgressResult.OK
            if result != LoadProgressResult.OK:
                logger.warning("unexpected or unhandled plan deserialize result: %s plan=%s", result, plan)
            return

        app_model.training_plan = None
        msg_box = QMessageBox(self)
        self._add_box_to_open_dialogs(msg_box)
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText(f"\n{msg}\n\nLoad or Cancel ?\n")
        load_btn = msg_box.addButton("Load", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Cancel)
        msg_box.exec()
        if msg_box.clickedButton() == load_btn:
            app_model.set_training_plan(plan, force_update=True)
        msg_box.close()
        try:
            self._open_dialogs.remove(msg_box)
        except ValueError:
            pass

    def _reset_animal_pellet_counts(self):
        app_model = self._app_model
        # apply it to the algo,
        # so that event/change listeners will get reset too
        algo = app_model.behavior.algorithm
        algo.pellets_presented_day = algo.pellets_presented_total = 0
        algo.successful_reaches_day = algo.successful_reaches_total = 0
        algo.pellet_reaches_day = algo.pellet_reaches_total = 0
        algo.pellet_consumed_day = algo.pellet_consumed_total = 0
        # but animal isn't synced with that, so:
        selected = self._app_model.selected_animal
        if selected is not None:
            selected.pellet_counts_day = AnimalPelletCounts()
            selected.pellet_counts_total = AnimalPelletCounts()
            self._app_model._save_animal_metadata(selected, sender="reset_animal_counts",
                                                  backup_previous=True)
