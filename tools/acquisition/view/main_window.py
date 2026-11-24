import os
import logging
import random
import textwrap
import threading
import time
import math
import traceback
import dataclasses
import getpass
from datetime import datetime
from functools import partial
from itertools import chain
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable, Union, Literal, Mapping, Sequence

from PySide6.QtCore import Qt, QCoreApplication, QRect, QTimer, Signal, QSize, QKeyCombination
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (QMainWindow, QStatusBar, QToolBar, QLabel, QMessageBox, QApplication,
                               QSizePolicy, QWidget, QComboBox, QLineEdit, QFileDialog, QHBoxLayout,
                               QSpinBox, QDoubleSpinBox, QFrame, QDialog, QInputDialog, QLayout)
import qtawesome as qta

from autotrainer.core import EventManager, Offset3DTuple, AnimalSubject, SystemConfiguration, CameraConfiguration, \
    calculate_std_dev_manual, ProjectInfo, get_perf_now
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.logging import get_console_handler, get_verbose_logger
from autotrainer.core.multiproc import make_daemon_timer, no_op_timer
from autotrainer.core.pose_elements import SceneElement
from autotrainer.core.event.api_event_kind import ApiEventKind
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig

from autotrainer.inference import InferenceStatus, PoseResponse
from autotrainer.inference.analysis import IntersessionResponse
from autotrainer.inference.analysis.prepare_jetson_data import make_cam_offsets_dict
from autotrainer.inference.multires_calibration import library
from autotrainer.inference.multires_calibration.report import QualityReport
from autotrainer.inference.multires_calibration.store import (CalibrationIndex, CalibrationResolutionError,
                                                             calibration_created, load_index)

from autotrainer.behavior import SystemState
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus, BehaviorAlgoProps
from autotrainer.behavior.pellet import PelletState

from autotrainer.pyside.content_widget import InvokeMethod, invoke_method

from autotrainer.behavior.training import TrainingPlan, PlanInfo, LoadProgressResult, TrainingPhase

from autotrainer.pyside.xyz_label import XYZQLabel

from tools.autotrainer_version import __version__ as app_version
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus, SessionRecordingStatus
from tools.acquisition.model.calibration_install import decide_install, install_calibration, override_record
from tools.acquisition.model.calibration_reminder import (
    MAKE_CALIBRATION_MENU_PATH, calibration_due, due_label_text, reminder_text)
from tools.acquisition.model.dcs_state import DcsState
from tools.acquisition.model.calibration_geometry import camera_serial, match_cameras
from tools.acquisition.model.handle_3d_calibration import capture_reference_entry, make_3d_calib
from tools.acquisition.model.nidaq_discovery import discover_nidaq_devices
from tools.acquisition.model.rfid_resolution import RfidResolutionKind
from tools.acquisition.model.subsystem_status import SubsystemState
from tools.acquisition.model.training_plan import get_plan_id
from tools.acquisition.model.user_preferences import UserPreferences
from tools.acquisition.view.calibration_dialog import (CalibrationDialog, CalibrationRequest, camera_unavailable_reason,
                                                  confirm_reference_capture, confirm_scout_advice, result_page)
from tools.acquisition.view.calibration_library_view import CalibrationLibraryView
from tools.acquisition.view.main_content import MainContent
from tools.acquisition.view.multi_selection_menu import MultiSelectionMenu
from tools.acquisition.view.ui_availability import calculate_ui_availability
from tools.acquisition.view.nidaq_monitor_window import NidaqMonitorWindow
from tools.acquisition.view.nidaq_port_configuration_dialog import NidaqPortConfigurationDialog
from tools.acquisition.view.preferences_dialog import PreferencesDialog
from tools.acquisition.view.animal_metadata_dialog import AnimalMetadataDialog
from tools.acquisition.view.animal_details_dialog import AnimalDetailsDialog
from tools.acquisition.view.debug_content import DebugView
from tools.acquisition.view.status_log_handler import StatusLogHandler

logger = get_verbose_logger(__name__)

_this_dir = Path(__file__).parent.resolve()

_TOOLBAR_ICON_COLOR = "#20242a"
_TOOLBAR_ICON_WARNING_COLOR = "#b00020"
_TRANSITIONAL_APP_MODE = "__transition__"
_MINIMUM_RESIZABLE_WINDOW_SIZE = QSize(320, 240)
def _toolbar_icon(name: str, *, color: str = _TOOLBAR_ICON_COLOR) -> QIcon:
    return QIcon(qta.icon(name, color=color))


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


def _allow_bidirectional_window_resizing(window: QMainWindow) -> None:
    """Prevent child size hints from becoming a fixed top-level resize floor."""
    window.setMinimumSize(_MINIMUM_RESIZABLE_WINDOW_SIZE)
    window.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    window.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)


def _ask_on_gui_thread(ask, *args) -> bool:
    """Run `ask(*args)` on the GUI thread and wait for its answer; from the GUI thread it runs at once.
    An `ask` that raises is logged and answers False (Cancel)."""
    answer = []
    done = threading.Event()

    def run():
        try:
            answer.append(bool(ask(*args)))
        except Exception:
            # A box that fails reads as Cancel, and never escapes the slot.
            logger.exception("the question %s failed; taken as Cancel", getattr(ask, "__name__", ask))
        finally:
            done.set()

    InvokeMethod(run)
    done.wait()
    return bool(answer and answer[0])


def reference_capture_cameras(configured: Sequence, calibrated: Mapping[str, str]):
    """The cameras "Capture reference images now" records: (cameras, names of those that only
    record), or why it cannot run.

    The configured cameras the entry calibrated (`calibrated`: configured name -> its name in
    the entry) that can record. The configured primary triggers the others, so when the
    entry did not calibrate it, it records with them and keeps no reference; a primary that
    cannot record refuses the capture.
    """
    cameras = [c for c in configured if c.name in calibrated and camera_unavailable_reason(c) is None]
    if not cameras:
        return "None of the configured cameras is one this calibration calibrated."
    primary = next((c for c in configured if c.is_primary), None)
    if primary is None or primary in cameras:
        return cameras, []
    reason = camera_unavailable_reason(primary)
    if reason is not None:
        return (f"The primary camera {primary.name} triggers the others but cannot record ({reason}):"
                " enable it, or make a new calibration with the cameras that can.")
    return [primary, *cameras], [primary.name]


class MainWindow(QMainWindow):

    running_status_changed = Signal(bool)  # True == running/acquiring
    capture_start_finished = Signal(bool)
    capture_stop_finished = Signal()
    hardware_refresh_finished = Signal(str, bool)  # message, is_error
    nidaq_discovery_finished = Signal(object, object)  # devices, optional error
    # todo: integrate within on_app_model_status_changed event handling

    def __init__(
        self,
        app: QApplication,
        user_preferences: UserPreferences,
        config_file: Optional[Path],
        *,
        is_dev: bool = False,
        random_cameras: bool = False,
        live_inference: Optional[bool] = None,
        calib_dir: Optional[Path] = None,
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
        self._hardware_refresh_thread = None
        self._nidaq_discovery_thread = None
        self._status_log_handler = None
        self._rfid_setup_dialog = None
        self._normal_geometry_restored = False
        self._hardware_refresh_pending = False

        self.setWindowTitle(self._title)

        icon = QIcon()
        for icon_file in sorted(_this_dir.joinpath("icons").glob("reachaq-*.png")):
            icon.addFile(icon_file.absolute().as_posix())
        self.setWindowIcon(icon)

        self._open_dialogs = []
        # The open calibration library view, refreshed when a reference capture ends.
        self._library_view: Optional[CalibrationLibraryView] = None
        self._debug_view: Optional[DebugView] = None
        self._training_plan_index_by_plan_id: Dict[Optional[str], int] = {}
        self._diamond_triangle_calib_run = None
        self._warned_invalid_dcs_config = False
        self._calibration_reminder_shown = False

        self._previous_intersession_analysis_rsp: Optional[Tuple[ProjectInfo, IntersessionResponse]] = None

        # A simulation's calibration is in its home; a rig run passes none, as before.
        if calib_dir is None:
            app_model = self._app_model = AppModel(prefs)
        else:
            app_model = self._app_model = AppModel(prefs, calib_dir=calib_dir)
        self._notes_save_timer = QTimer(self)
        self._notes_save_timer.setSingleShot(True)
        self._notes_save_timer.setInterval(500)
        self._notes_save_timer.timeout.connect(
            app_model.persist_stopped_session_notes
        )

        try:
            self.setContentsMargins(0, 0, 0, 0)

            self.main_content = MainContent(app_model)

            self._create_actions()
            self.main_content.set_hardware_refresh_action(self.refresh_hardware_action)
            self._configure_menubar()
            self._configure_statusbar()
            self._configure_toolbar()

            self.setCentralWidget(self.main_content)
            self.centralWidget().layout().setContentsMargins(0, 0, 0, 0)
            _allow_bidirectional_window_resizing(self)
            # self.setMaximumSize(1880, 1080)
        except Exception as err:
            app_model.on_close()
            raise RuntimeError(f"Error setting up UI: {err}") from err

        app_model.on_error += self._show_message
        app_model.configuration_loaded_event += self._on_app_model_configuration_loaded
        app_model.training_plan_deserialized += self._on_training_plan_deserialized

        try:
            app_model.load_configuration(config_file, random_cameras=random_cameras)
        except Exception as err:
            tb = traceback.format_exc()
            app_model.on_error("Failed load configuration",
                               f"\nConfiguration file {config_file} has issue,\n\n"
                               f"please check and fix following error:\n\n{err}\n\n{tb}")
            # app_model.on_close()
            # raise RuntimeError(f"Could not load config: {err}") from err
        else:
            if live_inference is not None:
                app_model.set_runtime_live_inference_override(live_inference)

        app_model.property_changed += self._on_app_model_property_changed
        app_model.hardware.property_changed += self._on_hardware_property_changed
        app_model.inference.property_changed += self._on_inference_property_changed
        app_model.inference.detection_result_ready += self._on_inference_analysis_result_ready

        user_preferences.property_changed += self._on_preferences_property_changed

        self.running_status_changed.connect(self._set_start_or_stop)
        self.capture_start_finished.connect(self._on_capture_start_finished)
        self.capture_stop_finished.connect(self._on_capture_stop_finished)
        self.hardware_refresh_finished.connect(self._on_hardware_refresh_finished)
        self.nidaq_discovery_finished.connect(self._on_nidaq_discovery_finished)
        #
        self._reload_animals(self._app_model.animals)  # after all property_changed connect above
        #
        # then after everything:
        QTimer.singleShot(0, self._refresh_hardware_bindings)
        QTimer.singleShot(0, self._refresh_ui_availability)

    def restore_normal_window_geometry(self) -> None:
        """Seed Qt's restore geometry before the initial maximized show."""
        geometry = self._preferences.window_normal_geometry
        screens = QApplication.screens()
        if geometry.isValid() and any(
            geometry.intersects(screen.availableGeometry()) for screen in screens
        ):
            self.setGeometry(geometry)
        else:
            available = (
                self.screen().availableGeometry()
                if self.screen() is not None
                else QRect(0, 0, 1440, 900)
            )
            width = min(max(960, int(available.width() * 0.8)), available.width())
            height = min(max(640, int(available.height() * 0.8)), available.height())
            self.setGeometry(
                available.x() + (available.width() - width) // 2,
                available.y() + (available.height() - height) // 2,
                width,
                height,
            )
        self._normal_geometry_restored = True

    def _remember_normal_window_geometry(self) -> None:
        if (
            self._normal_geometry_restored
            and self.isVisible()
            and not self.isMaximized()
            and not self.isMinimized()
            and not self.isFullScreen()
        ):
            self._preferences.window_normal_geometry = self.geometry()

    @property
    def app_model(self) -> AppModel:
        return self._app_model

    @property
    def has_fully_valid_dcs(self) -> bool:
        """A valid DCS configuration its consumers can use: not while a stale one is held off."""
        cfg = self._app_model.behavior.algorithm.diamond_triangle_config
        return cfg is not None and cfg.fully_valid

    @property
    def has_loaded_valid_dcs(self) -> bool:
        """A valid DCS configuration is loaded, stale or not: what Calibrate Coordinate System
        needs, so that a stale one can always be redone."""
        cfg = self._app_model.behavior.algorithm.loaded_diamond_triangle_config
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
        self.edit_daq_ports_action.setEnabled(stopped and self._nidaq_discovery_thread is None)
        self.refresh_hardware_action.setEnabled(
            (
                stopped
                or self._app_model.session_recording_status
                is SessionRecordingStatus.READY
            )
            and self._hardware_refresh_thread is None
            and self._nidaq_discovery_thread is None
        )
        self.make_3d_calib_action.setEnabled(stopped)
        self._set_hardware_menu_actions_enabled(
            stopped
            and self._app_model.status == AppModelStatus.IDLE
            and self._hardware_refresh_thread is None
            and self._nidaq_discovery_thread is None
        )
        #
        run_action = self.run_action
        run_action.blockSignals(True)  # block signal to ensure we don't re-start/stop
        run_action.setChecked(started)
        run_action.blockSignals(False)
        if started:
            icon = _toolbar_icon('ei.stop')
            run_action.setText("Stop")
            run_action.setIcon(icon)
        else:
            icon = _toolbar_icon('ei.play')
            run_action.setText("Start")
            run_action.setIcon(icon)
        self._refresh_ui_availability()

    def _refresh_ui_availability(self) -> None:
        capture_transition = (
            self._start_capture_thread is not None
            or self._stop_capture_thread is not None
        )
        state = calculate_ui_availability(
            status=self._app_model.status,
            recording_status=self._app_model.session_recording_status,
            acquisition_started=self._app_model.acquisition_started,
            capture_transition=capture_transition,
            hardware_refreshing=self._hardware_refresh_thread is not None,
            nidaq_discovering=self._nidaq_discovery_thread is not None,
            # It gates only Calibrate Coordinate System, which must stay offered while stale.
            has_valid_dcs=self.has_loaded_valid_dcs,
            laser_calibrating=self._app_model.laser_calibration_active,
        )
        self._app_model_status_combo.setEnabled(state.system_mode)
        self.run_action.setEnabled(state.system_mode)
        self.preferences_action.setEnabled(state.preferences)
        self.edit_camera_settings_action.setEnabled(state.idle_configuration)
        self.edit_daq_ports_action.setEnabled(state.idle_configuration)
        self.make_3d_calib_action.setEnabled(state.idle_configuration)
        self.calibration_library_action.setEnabled(state.calibration_library)
        # It pauses the NI-DAQ input stream, which in System Mode is the
        # acquisition's; AppModel refuses the pause then as well.
        self.daq_monitor_action.setEnabled(state.idle_configuration)
        self.refresh_hardware_action.setEnabled(state.hardware_refresh)
        self._set_hardware_menu_actions_enabled(state.idle_configuration)
        self._set_capture_binning_actions_enabled(state.idle_configuration)
        self._animal_dropdown_combo.setEnabled(state.subject)
        self._rfid_entry.setEnabled(state.subject)
        self.animal_metadata_action.setEnabled(state.subject)
        self._training_plan_combo.setEnabled(state.protocol_selection)
        self._notes.setEnabled(state.notes)
        self.calib_diamond_triangle_action.setEnabled(state.calibration)

    def _set_transitional_system_mode(self, text: str) -> None:
        combo = self._app_model_status_combo
        combo.blockSignals(True)
        index = combo.findData(_TRANSITIONAL_APP_MODE)
        if index < 0:
            combo.addItem(text, userData=_TRANSITIONAL_APP_MODE)
            index = combo.count() - 1
        else:
            combo.setItemText(index, text)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _restore_system_mode(self) -> None:
        combo = self._app_model_status_combo
        combo.blockSignals(True)
        transition_index = combo.findData(_TRANSITIONAL_APP_MODE)
        if transition_index >= 0:
            combo.removeItem(transition_index)
        status_index = combo.findData(self._app_model.status)
        if status_index >= 0:
            combo.setCurrentIndex(status_index)
        combo.blockSignals(False)

    def _set_startup_message(self, message: str) -> None:
        self._status_label.setText(message)
        self._status_label.setStyleSheet("font-weight: 600; color: #b45309;")

    def _clear_startup_message(self) -> None:
        self._status_label.setText("")
        self._status_label.setStyleSheet("")

    def _update_runtime_health_label(self) -> None:
        if not self._app_model.acquisition_started:
            return
        degraded = tuple(
            status
            for status in self._app_model.subsystem_statuses.values()
            if status.state in {SubsystemState.FAILED, SubsystemState.BLOCKED}
        )
        if degraded:
            self._status_label.setText("Running (degraded)")
            self._status_label.setStyleSheet(
                "font-weight: 600; color: #b45309;"
            )
            self._status_label.setToolTip("\n".join(
                f"{status.subsystem_id}: {status.error or status.reason}"
                for status in degraded
            ))
        else:
            self._clear_startup_message()

    @invoke_method
    def _on_capture_start_finished(self, started: bool) -> None:
        self._start_capture_thread = None
        self._restore_system_mode()
        if started:
            self._acquisition_started = True
            self._update_runtime_health_label()
        else:
            logger.verbose("capture_start failed: %s", self._app_model.status)
            self._status_label.setText("Startup failed")
            self._status_label.setStyleSheet("font-weight: 600; color: #b00020;")
            self.running_status_changed.emit(False)
        self._refresh_ui_availability()

    @invoke_method
    def _on_capture_stop_finished(self) -> None:
        self._stop_capture_thread = None
        self.running_status_changed.emit(False)
        self._restore_system_mode()
        self._clear_startup_message()
        self._acquisition_started = False
        self._refresh_ui_availability()

    def _on_capture_start_stop(self, is_toggled, *, target_status: AppModelStatus = AppModelStatus.RUNNING):
        app_model = self._app_model
        if is_toggled and self.edit_camera_settings_action.isChecked():
            self.edit_camera_settings_action.blockSignals(True)
            self.edit_camera_settings_action.setChecked(False)
            self.edit_camera_settings_action.blockSignals(False)
            self.main_content.set_is_editable(False)
        self.run_action.setEnabled(False)
        self.edit_camera_settings_action.setEnabled(False)
        self.edit_daq_ports_action.setEnabled(False)
        self.make_3d_calib_action.setEnabled(False)
        self.refresh_hardware_action.setEnabled(False)
        self._app_model_status_combo.setEnabled(False)
        if is_toggled:
            self.running_status_changed.emit(True)
            self._check_diamond_triangle_config()
            self._set_transitional_system_mode("Starting acquisition...")
            self._set_startup_message("Starting acquisition — hardware controls are temporarily unavailable...")
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
                self.capture_start_finished.emit(started)
            thread = threading.Thread(target=exec_start_capture, daemon=True, name="StartAcquisition")
            self._start_capture_thread = thread
            thread.start()
        else:
            self._set_transitional_system_mode("Stopping acquisition...")
            self._set_startup_message("Stopping acquisition — waiting for hardware to close...")
            def exec_stop_capture(prev_start=self._start_capture_thread,
                                  prev_stop=self._stop_capture_thread):
                if prev_start is not None:
                    logger.verbose("joining previous start thread")
                    prev_start.join()
                if prev_stop is not None:
                    prev_stop.join()
                logger.info("stopping subprocesses")
                app_model.capture_stop()
                self.capture_stop_finished.emit()
            thread = threading.Thread(target=exec_stop_capture, daemon=True, name="StopAcquisition")
            self._stop_capture_thread = thread
            thread.start()
        self._refresh_ui_availability()

    def _refresh_hardware_bindings(self):
        if self._hardware_refresh_thread is not None and self._hardware_refresh_thread.is_alive():
            self.statusBar().showMessage("Hardware refresh already in progress", 5000)
            return
        if self._nidaq_discovery_thread is not None and self._nidaq_discovery_thread.is_alive():
            self.statusBar().showMessage("NI-DAQ discovery is already in progress", 5000)
            return
        retry_running = self._app_model.acquisition_started
        if (
            retry_running
            and self._app_model.session_recording_status
            is not SessionRecordingStatus.READY
        ):
            self.statusBar().showMessage(
                "Hardware retry is unavailable during recording or analysis",
                5000,
            )
            return
        if not retry_running and self._app_model.status != AppModelStatus.IDLE:
            self.statusBar().showMessage("Hardware refresh is unavailable while System Mode is changing", 5000)
            return

        self.refresh_hardware_action.setEnabled(False)
        self._set_hardware_menu_actions_enabled(False)
        if not retry_running:
            self.run_action.setEnabled(False)
            self._app_model_status_combo.setEnabled(False)
        self._status_label.setText(
            "Retrying failed hardware..." if retry_running else "Refreshing hardware..."
        )
        self.main_content.set_hardware_refreshing(True)
        self._refresh_ui_availability()
        QCoreApplication.processEvents()

        def refresh_worker():
            try:
                message = (
                    self._app_model.retry_failed_subsystems()
                    if retry_running
                    else self._app_model.refresh_hardware_bindings()
                )
                is_error = False
            except Exception as exc:
                logger.exception("Hardware refresh failed")
                message = f"Hardware refresh failed: {str(exc) or exc.__class__.__name__}"
                is_error = True
            self.hardware_refresh_finished.emit(message, is_error)

        thread = threading.Thread(target=refresh_worker, daemon=True, name="RefreshHardware")
        self._hardware_refresh_thread = thread
        thread.start()

    @invoke_method
    def _on_hardware_refresh_finished(self, message: str, _is_error: bool):
        self._hardware_refresh_thread = None
        self._status_label.setText("")
        self.main_content.set_hardware_refreshing(False)
        self._refresh_ui_availability()
        self.statusBar().showMessage(message, 12000)

    def _on_system_mode_combo_changed(self, idx: int):
        status = self._app_model_status_combo.itemData(idx)
        if status == AppModelStatus.IDLE:
            self._on_capture_start_stop(False)
        elif status == AppModelStatus.RUNNING:
            self._on_capture_start_stop(True)

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
            logger.error(
                "Coordinate-system calibration failed: insufficient data. Send the pellet so the "
                "triangle is visible in both cameras, then retry calibration."
            )
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
            logger.error(
                "Coordinate-system calibration failed: captured data was too noisy; retry the "
                "calibration with a longer, stable capture."
            )
            return
        app_model = self._app_model
        # The offsets are kept with the active calibration; without one (it went while DCS
        # ran) nothing is written: a stand-in folder would be a legacy calibration that is not there.
        refusal = app_model.dcs_refusal()
        if refusal is not None:
            self._post_api_event(ApiEventKind.calibrationDcsFailed, dict(reason="NoActiveCalibration"))
            logger.error("%s", refusal)
            QMessageBox.warning(self, "Calibrate Coordinate System", refusal)
            return
        # success
        self._post_api_event(ApiEventKind.calibrationDcsCompleted)
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
        # The calibration this was measured in, and its folder: the offsets are written
        # beside it and that folder is reloaded, so the DCS fits it at once. A later 3D
        # calibration has another id, which makes this DCS stale; a legacy calibration has
        # no id to record. There is an active folder: dcs_refusal said so above.
        calibration_index = app_model.active_calibration_index
        calib_dir = app_model.active_calibration_dir
        new_cfg = DiamondTriangleOffsetConfig(
            used_position=avg_pos,
            measured_offset=avg_offset,
            # once fully calibrated twice,
            # what we get from triangulate+reorient_and_center is fully centered on diamond, which is ~0/0/0.
            diamond_coord=Offset3DTuple(0, 0, 0),  # avg_dia_loc3,
            # so this allows to not have to calibrate twice.
            raw_diamond_coord=avg_rawdia_loc3,
            version=DiamondTriangleOffsetConfig.current_config_version,
            calibration_id=None if calibration_index is None else calibration_index.id,
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
        # Beside a library entry, never inside it: an entry is not written to once in place.
        # Each run is a new file there; the newest is the one live and offline 3D read.
        cam_offsets_path = app_model.write_camera_offsets(calib_dir, cam_offsets)
        logger.info("Wrote new camera-offsets %s to file %s", cam_offsets, cam_offsets_path)
        #
        app_model = self._app_model
        app_model.reload_calib(calib_dir)
        algo = app_model.behavior.algorithm
        algo.diamond_triangle_config = new_cfg
        # Measured against the active calibration, so a stale DCS is valid again and its
        # consumers get it back, before the animal's position is converted below.
        app_model.refresh_dcs_state()
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
        action.setIcon(_toolbar_icon("fa5s.crosshairs", color=_TOOLBAR_ICON_WARNING_COLOR))
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
        action.setIcon(_toolbar_icon("fa5s.crosshairs"))
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

    def on_activated(self, *, target_status: AppModelStatus = AppModelStatus.IDLE):
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
                logger.verbose("AppModelStatus not idle (%s), not starting acquisition", app_status)

    def on_calibrate_diamond_triangle(self, is_toggled):
        refusal = self._app_model.dcs_refusal() if is_toggled and self._diamond_triangle_calib_run is None else None
        if refusal is not None:
            logger.error("%s", refusal)
            self.calib_diamond_triangle_action.setChecked(False)
            QMessageBox.warning(self, "Calibrate Coordinate System", refusal)
            return
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
        request = self._ask_calibration_cameras()
        if request is None:
            # Cancelled: nothing started, so Run and the action are given back as they were.
            self.run_action.setEnabled(True)
            self.make_3d_calib_action.setChecked(False)
            return
        # The dialog's choice, in its order: the configured primary first, then camera-list order.
        by_name = {camera.name: camera for camera in app_model.cameras}
        cameras = [by_name[name] for name in request.cameras]
        prev_status, app_model.status = app_model.status, AppModelStatus.CALIBRATION_3D
        error = "Processing unfinished"
        result: Optional[Tuple[Path, Path, CalibrationIndex, QualityReport]] = None
        def handle_3d_calib():
            nonlocal error, result
            try:
                result = make_3d_calib(self._app_model, cameras, move_list=request.move_list,
                                       confirm_scout=lambda advice: _ask_on_gui_thread(
                                           confirm_scout_advice, self, advice),
                                       capture_references=request.capture_references,
                                       confirm_references=lambda: _ask_on_gui_thread(
                                           confirm_reference_capture, self),
                                       rescout=request.rescout)
            except Exception as err:
                logger.exception("3d-calib failed: %s", err)
                error = err
            else:
                error = None

        @invoke_method
        def show_result():
            logger.verbose("3d-calib thread joined, error=%s", error)
            self._finish_3d_calib(error, result, prev_status)

        def wait_3d_calib_done(thread):
            thread.join()
            show_result()

        executor_thread = threading.Thread(target=handle_3d_calib, name="3d-calibration", daemon=True)
        executor_thread.start()

        waiter_thread = threading.Thread(target=wait_3d_calib_done, name="wait-3d-calibration",
                                         args=(executor_thread,), daemon=True)
        waiter_thread.start()

    def _ask_calibration_cameras(self) -> Optional[CalibrationRequest]:
        """Which cameras to calibrate, from the calibration dialog; None when cancelled."""
        dialog = CalibrationDialog(self._app_model.cameras, self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            return dialog.request()
        finally:
            dialog.deleteLater()

    def _finish_3d_calib(self, error, result, prev_status: AppModelStatus) -> None:
        """Install the calibration make_3d_calib produced, as its quality report allows, or show why it failed.

        The app is given back (_end_3d_calib) only once the install decision, the install
        and the reload are done, on every path. Until then it stays in CALIBRATION_3D, so
        nothing, a remote client included, starts a session under the failed-report dialog
        or while the new calibration loads.

        One API event per calibration: calibration3dFailed with the reason when the
        calibration or its install raised, else calibration3dCompleted, also when the
        operator kept a failed calibration out. Once installed, the calibration has
        completed: a reload that fails after the install says so and asks for a restart,
        and a status restore or a dialog that fails is logged without losing the event.
        """
        installed_report = None
        kept_id = None
        load_error = None
        try:
            if error is None:
                try:
                    _capture_dir, new_dir, index, report = result
                    installed = self._install_3d_calib(new_dir, index, report)
                except Exception as err:
                    logger.exception("3d-calib could not be completed: %s", err)
                    error = err
                else:
                    if installed is not None and not installed[2]:
                        kept_id = index.id
                    elif installed is not None:
                        installed_dir, installed_report, _activated = installed
                        try:
                            self._app_model.reload_calib(installed_dir)
                        except Exception as err:
                            # reload_calib has turned live 3D off for this folder (a DCS measured
                            # against the previous calibration is STALE), and the next
                            # configuration load or a restart retries it.
                            logger.exception("3D calibration %s: installed, but could not be loaded", index.id)
                            load_error = err
        finally:
            try:
                self._end_3d_calib(prev_status)
            except Exception:
                logger.exception("could not give the app back after the 3D calibration")
        if error is not None:
            reason = str(error)
            self._post_api_event(ApiEventKind.calibration3dFailed, dict(reason=reason))
            QMessageBox.critical(self, "3D calibration failed", reason or type(error).__name__)
            return
        if installed_report is not None:
            try:
                self._show_3d_calib_installed(installed_report, load_error)
            except Exception:
                logger.exception("the installed 3D calibration's outcome could not be shown")
        elif kept_id is not None:
            try:
                self._show_msg_box(
                    "3D calibration kept",
                    f"3D calibration {kept_id[:8]} is in the calibration library, not active. Set it active"
                    " from Tools → Calibration library when you want to use it.",
                    QMessageBox.Icon.Information)
            except Exception:
                logger.exception("the kept 3D calibration's outcome could not be shown")
        self._post_api_event(ApiEventKind.calibration3dCompleted)

    def _show_3d_calib_installed(self, report: QualityReport, load_error: Optional[BaseException]) -> None:
        """Tell the operator a new calibration is installed: its report, and that it is not
        loaded yet when its reload failed, or that DCS must be redone when it is stale."""
        # A new calibration is active, so the monthly reminder's status-bar label is out of date. It
        # stays when the previous calibration is kept or the install failed: that one is still active.
        self._set_calibration_due_label(None)
        if load_error is not None:
            reason = str(load_error) or type(load_error).__name__
            self._show_msg_box(
                "3D calibration not loaded",
                f"3D calibration installed, but it could not be loaded: {reason}. Restart the app.",
                QMessageBox.Icon.Warning)
        self._show_msg_box("3D calibration installed", result_page(report), QMessageBox.Icon.Information)
        # A DCS measured against the previous calibration is held off and blocks Record until
        # it is redone. With no DCS at all, the existing missing-DCS warning applies instead.
        if self._app_model.dcs_state is DcsState.STALE:
            self._show_msg_box(
                "Coordinate system out of date",
                "Redo Tools → Calibrate Coordinate System before running sessions.",
                QMessageBox.Icon.Warning)

    def _end_3d_calib(self, prev_status: AppModelStatus) -> None:
        """Give Run, the 3D calibration action and the app status back to the operator."""
        self.run_action.setEnabled(True)
        self.make_3d_calib_action.setChecked(False)
        self.make_3d_calib_action.setEnabled(True)
        self._app_model.status = prev_status

    def _install_3d_calib(self, new_dir: Path, index: CalibrationIndex,
                          report: QualityReport) -> Optional[Tuple[Path, QualityReport, bool]]:
        """Install the calibration in `new_dir` on a pass or warn report, or on a fail the operator overrides.

        A pass or warn asks Set as active or Keep, don't activate (_ask_activate_3d_calib);
        kept, the calibration is added to the library and the previous one stays active.
        Returns (the library entry's folder, the report as installed, with any override,
        whether it was made active), or None when a failed calibration is kept out. Loading it
        is left to the caller.
        """
        install, override, activate = decide_install(
            report.verdict, lambda: self._ask_install_failed_3d_calib(report),
            lambda: self._ask_activate_3d_calib(report))
        if not install:
            logger.notice("kept the previous 3D calibration: %s (%s) failed its quality checks", new_dir, index.id)
            return None
        if override:
            record = override_record(getpass.getuser(), datetime.now())
            # The library keeps it as a decision about the calibration; the calibration is not edited.
            report = dataclasses.replace(report, override=record)
            logger.warning("3D calibration %s: override recorded (install pending): %s", index.id, record)
        try:
            installed = install_calibration(
                new_dir, Path(self._preferences.configuration_location), now=datetime.now(),
                by=getpass.getuser(), override=record if override else None, activate=activate)
        except Exception:
            logger.error("3D calibration %s: install failed", index.id)
            raise
        if not activate:
            logger.success("3D calibration %s: kept in the library, not active", index.id)
            return installed, report, False
        logger.success("3D calibration %s: installed", index.id)
        return installed, report, True

    def _ask_activate_3d_calib(self, report: QualityReport) -> bool:
        """Whether to make a calibration whose report passed or warned the active one.

        The result page, with Set as active (the default, on Enter) and Keep, don't activate.
        Closing the box (Escape) keeps the calibration without activating it.
        """
        box = QMessageBox(self)
        box.setWindowTitle("3D calibration done")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(result_page(report))
        activate = box.addButton("Set as active", QMessageBox.ButtonRole.AcceptRole)
        keep = box.addButton("Keep, don't activate", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(activate)
        box.setEscapeButton(keep)
        box.exec()
        return box.clickedButton() != keep

    def _ask_install_failed_3d_calib(self, report: QualityReport) -> bool:
        """Whether to install a calibration whose quality report failed.

        Keep previous is the default; Install anyway… asks a second time.
        """
        box = QMessageBox(self)
        box.setWindowTitle("3D calibration failed its quality checks")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(result_page(report))
        keep = box.addButton("Keep previous", QMessageBox.ButtonRole.RejectRole)
        install = box.addButton("Install anyway…", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(keep)
        box.setEscapeButton(keep)
        box.exec()
        if box.clickedButton() != install:
            return False
        rsp = QMessageBox.question(
            self,
            "3D calibration override",
            "Install a calibration that failed its quality checks?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return rsp == QMessageBox.StandardButton.Yes

    def _on_calibration_library(self) -> None:
        view = self._library_view = CalibrationLibraryView(
            self._app_model.calibration_library_root, set_active=self._set_active_calibration,
            capture_references=self._capture_reference_images, parent=self)
        try:
            view.exec()
        finally:
            self._library_view = None
            view.deleteLater()

    def _capture_reference_images(self, calibration_id: str) -> Optional[str]:
        """Start capturing reference images for library entry `calibration_id`, into a new entry.

        The configured cameras that entry calibrated (by serial, by name without one) record
        a short still clip (capture_reference_entry) on a worker thread, with the app held in
        CALIBRATION_3D as for a calibration. Returns None when it started or the operator
        cancelled, or why it cannot run. Once done, the operator is offered to set the new
        entry active.
        """
        app_model = self._app_model
        if (app_model.status == AppModelStatus.CALIBRATION_3D or not self.make_3d_calib_action.isEnabled()
                or app_model.session_recording_status != SessionRecordingStatus.READY):
            return ("Reference images are captured as a calibration is: stop the acquisition, the recording or"
                    " the calibration first.")
        root = app_model.calibration_library_root
        try:
            index = load_index(library.entry_path(root, calibration_id))
        except (ValueError, CalibrationResolutionError) as err:
            return f"This calibration cannot take reference images: {err}. Make a new calibration instead."
        match = match_cameras(index.cameras, {c.name: camera_serial(c.active_config) for c in app_model.cameras})
        chosen = reference_capture_cameras(app_model.cameras, match.calibrated)
        if isinstance(chosen, str):
            return chosen
        cameras, record_only = chosen
        if not confirm_reference_capture(self, skip_label="Cancel"):
            return None
        names = {camera.name: match.calibrated[camera.name] for camera in cameras if camera.name in match.calibrated}
        prev_status, app_model.status = app_model.status, AppModelStatus.CALIBRATION_3D
        self.run_action.setEnabled(False)
        self.make_3d_calib_action.setEnabled(False)
        outcome = {}

        @invoke_method
        def finish():
            try:
                self._end_3d_calib(prev_status)
            finally:
                self._finish_reference_capture(calibration_id, outcome)

        def run():
            try:
                outcome["id"] = capture_reference_entry(app_model, root, calibration_id, cameras, names=names,
                                                        record_only=record_only)
            except Exception as err:
                logger.exception("reference images for %s not captured: %s", calibration_id, err)
                outcome["error"] = err
            finally:
                finish()

        threading.Thread(target=run, name="reference-images", daemon=True).start()
        return None

    def _finish_reference_capture(self, source_id: str, outcome: dict) -> None:
        """Say how the reference capture ended and offer to set the new entry active."""
        if self._library_view is not None:
            self._library_view.refresh()
        error = outcome.get("error")
        if error is not None or "id" not in outcome:
            QMessageBox.critical(self, "Reference images not captured", str(error or "the capture did not finish"))
            return
        new_id = outcome["id"]
        rsp = QMessageBox.question(
            self,
            "Reference images captured",
            f"The reference images are in a new calibration, {new_id[:8]}, with the fit of {source_id[:8]}. "
            "Set it active now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if rsp == QMessageBox.StandardButton.Yes:
            refusal = self._set_active_calibration(new_id)
            if refusal:
                QMessageBox.warning(self, "Calibration not set active", refusal)
        if self._library_view is not None:
            self._library_view.refresh()

    def _set_active_calibration(self, calibration_id: str) -> Optional[str]:
        """Make library entry `calibration_id` the active calibration and load it
        (AppModel.set_active_calibration, under the session lock).

        Returns None, or why it was not set or not loaded. When it was set active, the
        monthly reminder's label is reset, as after an install, even if it then could not be
        loaded; a refused call leaves the label as it was.
        """
        app_model = self._app_model
        before = library.pointer_id(app_model.calibration_library_root)
        refusal = app_model.set_active_calibration(calibration_id)
        moved = before != calibration_id and library.pointer_id(app_model.calibration_library_root) == calibration_id
        if refusal is None or moved:
            self._set_calibration_due_label(None)
        if refusal is not None:
            return refusal
        if app_model.dcs_state is DcsState.STALE:
            self._show_msg_box(
                "Coordinate system out of date",
                "Redo Tools → Calibrate Coordinate System before running sessions.",
                QMessageBox.Icon.Warning)
        return None

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
        status_log_handler = self._status_log_handler
        self._status_log_handler = None
        if status_log_handler is not None:
            logging.getLogger().removeHandler(status_log_handler)
            status_log_handler.close()
        self.main_content.close()
        dialogs = self._open_dialogs
        self._open_dialogs = []
        for dialog in dialogs:
            dialog.close()
        self._close_finishing = True
        super().close()

    def _confirm_close_with_session(self) -> bool:
        """Ask before closing over a session; True to go on closing.

        Closing stops acquisition, which once aborted a recording and deleted
        the session with no prompt. It now stops and saves; deleting is the
        operator's explicit choice here, as with Abort.
        """
        app_model = self._app_model
        status = app_model.session_recording_status
        if status is SessionRecordingStatus.READY:
            return True
        project = app_model.project
        session = "The session" if project is None else project.short_id
        box = QMessageBox(self)
        box.setWindowTitle("Session in progress")
        box.setIcon(QMessageBox.Icon.Warning)
        abort = None
        if status in {SessionRecordingStatus.ARMING, SessionRecordingStatus.RECORDING}:
            box.setText(
                f"{session} is recording. Closing stops the recording and "
                "saves it first."
            )
            proceed = box.addButton("Stop, save and close", QMessageBox.ButtonRole.AcceptRole)
            abort = box.addButton(
                "Abort (delete session) and close", QMessageBox.ButtonRole.DestructiveRole)
        elif status is SessionRecordingStatus.ABORTING:
            box.setText(
                f"{session} is being aborted. Close when that finishes?"
            )
            proceed = box.addButton("Close when done", QMessageBox.ButtonRole.AcceptRole)
        else:
            box.setText(
                f"{session} is still being saved. Close when saving finishes?"
            )
            proceed = box.addButton("Close when saved", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if abort is not None and clicked == abort:
            if not app_model.abort_recording():
                self._show_msg_box(
                    "Abort refused",
                    f"{session} could not be aborted; it was not closed.",
                    QMessageBox.Icon.Warning,
                )
                return False
            return True
        return clicked == proceed

    def close(self):
        logger.debug("received close")
        with self._app_model.app_lock:
            closing = self._closing
        if not closing:
            # The prompt runs a nested event loop, in which a second close
            # (another window close, Ctrl-C) can arrive: it waits for this one.
            event = self._close_event
            self._close_event = None
            if getattr(self, "_confirming_close", False):
                if event is not None:
                    event.ignore()
                return
            self._confirming_close = True
            try:
                confirmed = self._confirm_close_with_session()
            finally:
                self._confirming_close = False
            if not confirmed:
                if event is not None:
                    event.ignore()
                return
            self._close_event = event
        with self._app_model.app_lock:
            event = self._close_event
            if self._closing:
                if event is not None:
                    # Only the close _finish_close makes is accepted: one
                    # made while the close thread still saves a session would
                    # take the window, and its error dialogs, away early.
                    if getattr(self, "_close_finishing", False):
                        event.accept()
                    else:
                        event.ignore()
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
        self._remember_normal_window_geometry()
        # An expanded panel is its own top-level window, so it has to be told
        # where the main window went.
        self.main_content.update_right_panel_tracking()
        super(MainWindow, self).moveEvent(e)

    def resizeEvent(self, event):
        self._remember_normal_window_geometry()
        self.main_content.update_right_panel_tracking()
        super().resizeEvent(event)

    def _edit_camera_settings(self):
        # isChecked() has already swapped to the new value by the time this is called
        if not self.edit_camera_settings_action.isChecked():
            self.main_content.set_is_editable(False)
            self._app_model.save_configuration()
            # Changing a camera's source rebuilds its properties without the
            # capture_binning preset, so the menu checkmark is stale until synced.
            self._sync_capture_binning_actions()
            self.run_action.setEnabled(True)
        else:
            self.main_content.set_is_editable(True)
            self.run_action.setEnabled(False)

    def _edit_daq_ports(self):
        configuration = self._app_model.loaded_configuration
        if configuration is None:
            logger.error("DAQ port configuration unavailable: no system configuration is loaded")
            return

        if self._nidaq_discovery_thread is not None and self._nidaq_discovery_thread.is_alive():
            self.statusBar().showMessage("NI-DAQ discovery is already in progress", 5000)
            return

        self.edit_daq_ports_action.setEnabled(False)
        self.refresh_hardware_action.setEnabled(False)
        self._set_hardware_menu_actions_enabled(False)
        self._set_startup_message("Discovering NI-DAQ devices...")

        def discover_worker():
            try:
                devices, discovery_error = discover_nidaq_devices()
            except Exception as exc:
                logger.exception("NI-DAQ discovery failed")
                devices = tuple()
                discovery_error = str(exc) or exc.__class__.__name__
            self.nidaq_discovery_finished.emit(devices, discovery_error)

        thread = threading.Thread(target=discover_worker, daemon=True, name="DiscoverNidaq")
        self._nidaq_discovery_thread = thread
        thread.start()
        self._refresh_ui_availability()

    @invoke_method
    def _on_nidaq_discovery_finished(self, devices, discovery_error) -> None:
        self._nidaq_discovery_thread = None
        if self._start_capture_thread is None:
            self._clear_startup_message()
        is_idle = (
            not self._app_model.acquisition_started
            and self._app_model.status == AppModelStatus.IDLE
            and self._start_capture_thread is None
            # A ramp holds the NI-DAQ lines, and its save would be refused.
            and not self._app_model.laser_calibration_active
        )
        self.edit_daq_ports_action.setEnabled(is_idle)
        # The monitor streams every line on every card, so it needs the
        # hardware the acquisition would otherwise hold.
        self.daq_monitor_action.setEnabled(is_idle)
        self.refresh_hardware_action.setEnabled(is_idle and self._hardware_refresh_thread is None)
        self._set_hardware_menu_actions_enabled(
            is_idle and self._hardware_refresh_thread is None
        )
        self._refresh_ui_availability()
        if not is_idle or self._closing:
            self.statusBar().showMessage("NI-DAQ discovery finished; acquisition is no longer idle", 5000)
            return

        configuration = self._app_model.loaded_configuration
        if configuration is None:
            logger.error("DAQ port configuration unavailable: no system configuration is loaded")
            return
        if discovery_error:
            logger.error("NI-DAQ discovery failed: %s", discovery_error)
        dialog = NidaqPortConfigurationDialog(
            configuration,
            self,
            devices=devices,
            discovery_error=discovery_error,
        )
        self._add_box_to_open_dialogs(dialog)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self._app_model.update_daq_port_configuration(
                dialog.nidaq_ports,
                dialog.laser_configuration,
            )
        except Exception as exc:
            logger.exception("Failed to save DAQ port configuration: %s", exc)
            return
        self.statusBar().showMessage("DAQ port configuration saved", 5000)

    def _open_daq_monitor(self):
        """Open the monitor, or raise the one already open.

        One window: it holds a second stream process and a set of tasks on
        the same lines, and two of those would fight over the hardware rather
        than show twice as much.
        """
        existing = getattr(self, "_daq_monitor_window", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        try:
            window = NidaqMonitorWindow(self._app_model, self)
        except Exception as exc:
            logger.exception("DAQ monitor could not be opened: %s", exc)
            self.statusBar().showMessage(
                f"DAQ monitor could not be opened: {exc}", 8000)
            return
        self._daq_monitor_window = window
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        window.show()

    def _open_pose_training(self):
        """Open the pose training window, or raise the one already open.

        Closing it only hides the view: a training run belongs to the app
        model and keeps going.
        """
        existing = getattr(self, "_pose_training_window", None)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        # Imported here: the window pulls in the keypoint dataset code, which
        # the rest of the acquisition UI does not need at start-up.
        from tools.acquisition.view.pose_training_window import PoseTrainingWindow
        try:
            window = PoseTrainingWindow(self._app_model, self)
        except Exception as exc:
            logger.exception("Pose training could not be opened: %s", exc)
            self.statusBar().showMessage(
                f"Pose training could not be opened: {exc}", 8000)
            return
        self._pose_training_window = window
        window.show()

    def _show_preferences(self):
        if (
            self._app_model.session_recording_status
            is not SessionRecordingStatus.READY
        ):
            self.statusBar().showMessage(
                "Preferences are unavailable during recording or finalization",
                5000,
            )
            return
        dialog = PreferencesDialog(self._preferences, self._app_model)
        self._add_box_to_open_dialogs(dialog)
        dialog.exec()

    def _show_animal_metadata(self):
        dialog = AnimalMetadataDialog(self._app_model, self)
        self._add_box_to_open_dialogs(dialog)
        dialog.exec()

    def _create_actions(self):
        action = self.animal_metadata_action = QAction(
            _toolbar_icon("fa5s.id-card"), "RFID links…", self
        )
        action.setToolTip("Manually link or reconcile reachAQ animals and source records")
        action.triggered.connect(self._show_animal_metadata)

        action = self.edit_camera_settings_action = QAction(_toolbar_icon("fa5s.edit"), "Edit Camera Settings", self)
        action.setToolTip("Edit Camera Settings")
        action.setCheckable(True)
        action.setChecked(False)
        action.triggered.connect(self._edit_camera_settings)

        action = self.edit_daq_ports_action = QAction(_toolbar_icon("fa5s.plug"), "Edit DAQ Ports", self)
        action.setToolTip("Edit NI-DAQ port assignments")
        action.triggered.connect(self._edit_daq_ports)

        action = self.refresh_hardware_action = QAction(_toolbar_icon("fa5s.sync"), "Refresh Hardware", self)
        action.setToolTip(
            "Scan hardware while idle, or retry failed subsystems while System Mode is running"
        )
        action.triggered.connect(self._refresh_hardware_bindings)

        self.hardware_enable_actions = {}
        for field_name, label in (
            ("can_enabled", "Enable CAN Adapter"),
            ("pellet_controller_enabled", "Enable Pellet Controller"),
            ("nidaq_enabled", "Enable NI-DAQ"),
            ("rfid_reader_enabled", "Enable USB RFID Reader"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setEnabled(False)
            action.toggled.connect(
                partial(self._set_hardware_enabled_from_menu, field_name)
            )
            self.hardware_enable_actions[field_name] = action

        # Left/right capture binning, for testing higher-resolution recording.
        # Live inference, the display and calibration keep the base frame
        # size: the capture process averages each frame back down for them.
        # The text is set from the loaded configuration (_sync_capture_binning_actions).
        self.capture_binning_actions = {}
        for binning in (4, 2, 1):
            action = QAction(f"bin {binning}", self)
            action.setCheckable(True)
            action.setEnabled(False)
            action.triggered.connect(partial(self._set_capture_binning_from_menu, binning))
            self.capture_binning_actions[binning] = action

        self.rfid_device_action = QAction("RFID Serial Device…", self)
        self.rfid_device_action.setEnabled(False)
        self.rfid_device_action.triggered.connect(self._edit_rfid_device_from_menu)

        action = self.run_action = QAction(_toolbar_icon("ei.play"), "Start", self)
        action.setToolTip("Start or stop acquisition")
        action.setCheckable(True)
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_R))
        action.triggered.connect(self._on_capture_start_stop)

        action = self.show_reach_event_action = QAction(_toolbar_icon("fa5s.bezier-curve"), "Show Previous Reach", self)
        action.setCheckable(True)
        action.setEnabled(False)
        action.triggered.connect(self.on_show_reach_event)

        action = self.next_training_phase_action = QAction(_toolbar_icon("fa5s.arrow-alt-circle-right"), "Next Phase", self)
        action.setVisible(False)
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Right))
        action.triggered.connect(self.on_next_plan_phase)

        action = self.previous_training_phase_action = QAction(_toolbar_icon("fa5s.arrow-alt-circle-left"), "Previous Phase", self)
        action.setVisible(False)
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Left))
        action.triggered.connect(self.on_previous_plan_phase)

        self._diamond_triangle_calib_run = None
        self._timer_calibrate_diamond_triangle = no_op_timer
        action = self.calib_diamond_triangle_action = QAction(_toolbar_icon("fa5s.crosshairs"), "Calibrate Coordinate System", self)
        action.setToolTip("Calibrate the pellet delivery coordinate system from the diamond and triangle references")
        action.setCheckable(True)
        action.triggered.connect(self.on_calibrate_diamond_triangle)
        action.setEnabled(False)

        action = self.overlay_parts_action = QAction("Overlay Parts and Legend…", self)
        action.setToolTip(
            "Which colour means which part, and which parts the live "
            "overlay draws"
        )
        action.triggered.connect(self._on_overlay_parts)

        action = self.make_3d_calib_action = QAction(_toolbar_icon("fa5s.crosshairs"), "Make 3D calibration", self)
        action.setCheckable(True)
        action.triggered.connect(self.on_3d_calibrate)

        action = self.calibration_library_action = QAction("Calibration library…", self)
        action.setToolTip("Every 3D calibration, which one is active, and which cameras each left 2D only")
        action.triggered.connect(self._on_calibration_library)

        action = self.daq_monitor_action = QAction(
            _toolbar_icon("fa5s.wave-square"), "DAQ Monitor", self)
        action.setToolTip(
            "Watch every line on every DAQ card and drive the stimulus, to "
            "work out which line is which. Available while acquisition is "
            "idle, because it opens the same hardware; the application's "
            "NI-DAQ input stream pauses while it is open.")
        action.triggered.connect(self._open_daq_monitor)

        action = self.pose_training_action = QAction("Pose Training…", self)
        action.setToolTip(
            "Label frames from recorded sessions, train a live pose model, "
            "compare it with the current one and make it live. Training "
            "pauses whenever a recording starts.")
        action.triggered.connect(self._open_pose_training)

        action = self.view_diagnostics_action = QAction("Logging", self)
        action.setToolTip("Show or hide the application logging panel")
        action.setCheckable(True)
        action.setChecked(self.main_content.is_diagnostics_visible)
        action.triggered.connect(self._toggle_diagnostics_view)

        action = self.debug_action = QAction("Debug", self)
        action.setCheckable(True)
        action.triggered.connect(self._toggle_debug_view)

        action = self.pellet_seen_action = QAction("Pellet Seen", self)
        action.triggered.connect(self._internal_set_pellet_seen)

        action = self.mouse_seen_action = QAction("Mouse Seen", self)
        action.triggered.connect(self._internal_set_mouse_seen)

        action = self.mouse_near_pellet_action = QAction("Hands min-Y DCS", self)
        action.triggered.connect(self._internal_mouse_near_pellet)

        action = self.analysis_results_action = QAction("Analysis-Result", self)
        action.setCheckable(True)
        action.triggered.connect(self._internal_detection_result_toggle)

        action = self.preferences_action = QAction(_toolbar_icon("fa5s.cog"), "Preferences", self)
        action.triggered.connect(self._show_preferences)

        action = self.quit_action = QAction("Quit")
        action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Q))
        action.triggered.connect(lambda: self._app.quit())

    def _configure_menubar(self):
        menu_bar = self.menuBar()

        menu_bar.setObjectName("MenuBar")
        menu_bar.setStyleSheet(
            "#MenuBar {background-color: #f3f4f6; color: #20242a;}"
            "#MenuBar::item {background: transparent; color: #20242a; padding: 2px 8px;}"
            "#MenuBar::item:selected {background: #e1e5eb;}"
            "#MenuBar::item:disabled {color: #68717d;}"
        )

        file_menu = menu_bar.addMenu("File")
        hardware_menu = self.hardware_menu = MultiSelectionMenu("Hardware", file_menu)
        file_menu.addMenu(hardware_menu)
        for action in self.hardware_enable_actions.values():
            hardware_menu.addAction(action)
        hardware_menu.addSeparator()
        hardware_menu.addAction(self.rfid_device_action)
        hardware_menu.aboutToHide.connect(self._flush_pending_hardware_refresh)
        file_menu.addAction(self.overlay_parts_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction(self.edit_camera_settings_action)
        edit_menu.addAction(self.edit_daq_ports_action)

        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction(self.daq_monitor_action)
        tools_menu.addAction(self.pose_training_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.calib_diamond_triangle_action)
        tools_menu.addAction(self.make_3d_calib_action)
        tools_menu.addAction(self.calibration_library_action)
        tools_menu.addSeparator()
        resolution_menu = tools_menu.addMenu("Camera resolution")
        for action in self.capture_binning_actions.values():
            resolution_menu.addAction(action)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(self.view_diagnostics_action)
        if self._is_dev:
            view_menu.addAction(self.debug_action)

    def _sync_hardware_menu_actions(self) -> None:
        configuration = self._app_model.loaded_configuration
        if configuration is None:
            return
        hardware = configuration.hardware
        for field_name, action in self.hardware_enable_actions.items():
            action.blockSignals(True)
            action.setChecked(bool(getattr(hardware, field_name)))
            action.blockSignals(False)
        self.rfid_device_action.setText("RFID Serial Device…")
        self.rfid_device_action.setToolTip(
            hardware.rfid_device or "No RFID serial device is configured"
        )

    def _set_hardware_menu_actions_enabled(self, enabled: bool) -> None:
        for action in self.hardware_enable_actions.values():
            action.setEnabled(enabled)
        self.rfid_device_action.setEnabled(enabled)

    def _sync_capture_binning_actions(self) -> None:
        app_model = self._app_model
        try:
            current = app_model.reach_capture_binning
            left = app_model.left_camera
            rows, cols = left.shape if left is not None else (0, 0)
            base = None if left is None else left.base_binning
        except ValueError as exc:
            # A malformed capture_binning, hbin or vbin in the configuration.
            # The model still refuses Run for it, but this runs in the
            # configuration-load and availability handlers, so raising would
            # skip every enablement after it. Show the entries as unavailable.
            logger.warning("Camera resolution menu shows every entry unavailable: %s", exc)
            current, rows, cols, base = None, 0, 0, None
        for binning, action in self.capture_binning_actions.items():
            k = base // binning if base and base % binning == 0 else None
            if k is None:
                text = f"bin {binning} (unavailable)"
            else:
                # The larger presets record H.264 (the base keeps mp4v): say so
                # where the operator chooses, since the files change format.
                note = ", H.264" if k > 1 else ""
                text = f"{cols * k} x {rows * k} (bin {binning}{note})"
            action.blockSignals(True)
            action.setText(text)
            action.setChecked(binning == current)
            action.blockSignals(False)

    def _set_capture_binning_actions_enabled(self, enabled: bool) -> None:
        try:
            for binning, action in self.capture_binning_actions.items():
                action.setEnabled(enabled and self._app_model.reach_capture_binning_available(binning))
        except ValueError as exc:
            # See _sync_capture_binning_actions: leave the entries disabled
            # rather than skip the enablement that follows this call.
            logger.warning("Camera resolution menu left disabled: %s", exc)
            for action in self.capture_binning_actions.values():
                action.setEnabled(False)

    def _set_capture_binning_from_menu(self, binning: int, _checked: bool = False) -> None:
        logger.info("Camera resolution menu selection: capture binning %s", binning)
        try:
            self._app_model.set_reach_capture_binning(binning)
        except Exception as exc:
            logger.exception("Camera resolution was not changed")
            self.statusBar().showMessage(f"Camera resolution not changed: {exc}", 8000)
        else:
            self.statusBar().showMessage(
                f"Left/right cameras capture at binning {binning} from the next Start", 8000)
        # Checkmarks always follow the model, including after a refusal or a
        # click on the entry that was already checked.
        self._sync_capture_binning_actions()

    def _hardware_menu_values(self, **overrides):
        configuration = self._app_model.loaded_configuration
        if configuration is None:
            raise RuntimeError("No system configuration is loaded")
        hardware = configuration.hardware
        values = {
            "can_enabled": hardware.can_enabled,
            "pellet_controller_enabled": hardware.pellet_controller_enabled,
            "nidaq_enabled": hardware.nidaq_enabled,
            "rfid_reader_enabled": hardware.rfid_reader_enabled,
            "rfid_device": hardware.rfid_device,
        }
        values.update(overrides)
        return values

    def _apply_hardware_menu_values(self, values, *, refresh_if_enabled: bool) -> None:
        try:
            message = self._app_model.update_hardware_configuration(**values)
        except Exception as exc:
            logger.exception("Hardware menu setting was not applied")
            self._sync_hardware_menu_actions()
            self.statusBar().showMessage(f"Hardware setting not saved: {exc}", 8000)
            return
        self._sync_hardware_menu_actions()
        self.statusBar().showMessage(message, 8000)
        if refresh_if_enabled:
            self._hardware_refresh_pending = True
            if not self.hardware_menu.isVisible():
                self._flush_pending_hardware_refresh()

    def _flush_pending_hardware_refresh(self) -> None:
        if not self._hardware_refresh_pending:
            return
        self._hardware_refresh_pending = False
        QTimer.singleShot(0, self._refresh_hardware_bindings)

    def _set_hardware_enabled_from_menu(self, field_name: str, enabled: bool) -> None:
        logger.info(
            "Hardware menu selection: %s=%s",
            field_name,
            bool(enabled),
        )
        try:
            values = self._hardware_menu_values(**{field_name: bool(enabled)})
        except Exception as exc:
            self._sync_hardware_menu_actions()
            self.statusBar().showMessage(f"Hardware setting unavailable: {exc}", 8000)
            return
        self._apply_hardware_menu_values(
            values,
            refresh_if_enabled=bool(enabled) and field_name != "rfid_reader_enabled",
        )

    def _edit_rfid_device_from_menu(self) -> None:
        try:
            values = self._hardware_menu_values()
        except Exception as exc:
            self.statusBar().showMessage(f"RFID setting unavailable: {exc}", 8000)
            return
        device, accepted = QInputDialog.getText(
            self,
            "RFID Serial Device",
            "Stable /dev/serial/by-id path:",
            QLineEdit.EchoMode.Normal,
            values["rfid_device"],
        )
        if not accepted:
            return
        device = device.strip()
        logger.info("RFID serial-device selection changed: device=%s", device or "<empty>")
        values["rfid_device"] = device
        self._apply_hardware_menu_values(values, refresh_if_enabled=False)

    def _configure_toolbar(self):

        app_model = self._app_model

        toolbar = QToolBar("Run Toolbar")
        toolbar.setObjectName("RunToolbar")
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setStyleSheet(
            "#RunToolbar {background-color: #f3f4f6; border-bottom: 1px solid #c9cdd3; spacing: 6px; padding: 2px;}"
            "#RunToolbar QLabel {color: #20242a;}"
            "#RunToolbar QToolButton {color: #20242a;}"
            "#RunToolbar QToolButton:disabled {color: #68717d;}"
        )
        toolbar.setFloatable(False)
        toolbar.setMovable(False)
        toolbar.setMinimumWidth(0)
        toolbar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("System Mode:"))
        combo = self._app_model_status_combo = QComboBox()
        combo.setMinimumWidth(100)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.addItem("Idle", userData=AppModelStatus.IDLE)
        combo.addItem("Running", userData=AppModelStatus.RUNNING)
        combo.currentIndexChanged.connect(self._on_system_mode_combo_changed)
        toolbar.addWidget(combo)
        # toolbar.addAction(self.run_action)
        toolbar.addSeparator()

        toolbar.addAction(self.show_reach_event_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        toolbar.addWidget(QLabel("Notes:"))
        self._notes = QLineEdit()
        self._notes.setMinimumWidth(100)
        self._notes.setText(self._app_model.notes)
        self._notes.textChanged.connect(self.notes_changed)
        self._notes.setContentsMargins(4, 0, 8, 0)
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

        self._rfid_status_label = QLabel("RFID: —")
        self._rfid_status_label.setContentsMargins(8, 0, 8, 0)
        toolbar.addWidget(self._rfid_status_label)
        # A USB barcode scanner types its code and Enter here, like a keyboard.
        entry = self._rfid_entry = QLineEdit()
        entry.setPlaceholderText("Scan or type RFID, Enter")
        entry.setToolTip(
            "An RFID, barcode or other unique animal ID. Enter selects the "
            "animal as an RFID reader scan does."
        )
        entry.setClearButtonEnabled(True)
        entry.setMinimumWidth(150)
        entry.returnPressed.connect(self._rfid_entered)
        toolbar.addWidget(entry)
        toolbar.addAction(self.animal_metadata_action)

        toolbar.addSeparator()

        combo = self._training_plan_combo = QComboBox()

        def training_plan_index_changed(_):
            plan_id = self._training_plan_combo.currentData()
            selected_plan: Optional[TrainingPlan] = app_model.get_training_plan_by_id(plan_id)
            logger.debug("plan changed: %s -> %s", plan_id, selected_plan)
            app_model.training_plan = selected_plan

        combo.currentIndexChanged.connect(training_plan_index_changed)
        self._widget_training_plan_action = None

        if self.main_content.protocol_ui_enabled:
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
        else:
            self._status_training_widget.setVisible(False)

        toolbar.addSeparator()

        toolbar.addAction(self.preferences_action)

        if self._is_dev:
            self.addToolBarBreak()
            toolbar = self._dev_toolbar = QToolBar("Dev Toolbar")
            toolbar.setObjectName("DevToolbar")
            toolbar.setIconSize(QSize(18, 18))
            toolbar.setStyleSheet(
                "#DevToolbar {background-color: #f3f4f6; border-bottom: 1px solid #d6d9de; spacing: 6px; padding: 1px;}"
                "#DevToolbar QLabel {color: #20242a;}"
                "#DevToolbar QToolButton {color: #20242a;}"
                "#DevToolbar QToolButton:disabled {color: #68717d;}"
            )
            toolbar.setContentsMargins(0, 0, 0, 0)
            # toolbar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            self.addToolBar(toolbar)
            toolbar.setFloatable(False)
            toolbar.setMovable(False)
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

    def _configure_statusbar(self):
        self._status_label = QLabel("")
        bar = self._status_bar = QStatusBar(self)
        bar.addWidget(self._status_label)
        bar.setSizeGripEnabled(True)
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
        # Shown by the monthly reminder (_remind_if_calibration_due) once the app finds the calibration
        # due, and hidden again when a new one is installed. Left of the position widget, which stays rightmost.
        lbl = self._calibration_due_label = QLabel("")
        lbl.setStyleSheet("font-weight: 600; color: #b45309;")
        lbl.setToolTip(f"Choose {MAKE_CALIBRATION_MENU_PATH} when the rig is ready.")
        lbl.setVisible(False)
        bar.addPermanentWidget(self._calibration_due_label)
        bar.addPermanentWidget(widget)
        self.setStatusBar(bar)
        status_log_handler = self._status_log_handler = StatusLogHandler(self._show_logged_error)
        logging.getLogger().addHandler(status_log_handler)

    def _show_logged_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 15000)

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

    def _on_overlay_parts(self) -> None:
        """Show the overlay legend and let the operator narrow it."""
        from tools.acquisition.view.overlay_parts_dialog import (
            OverlayPartsDialog)

        inference = self._app_model.inference
        parts = list(inference.pose_parts)
        if not parts:
            self._show_message(
                "Overlay parts",
                "The pose model has not reported its parts yet. Start the "
                "system and try again once inference is running.")
            return
        # The model emits the hand composites in addition to its own
        # keypoints, so they belong in the list even though the model does
        # not name them.
        for composite in (SceneElement.L_Hand, SceneElement.R_Hand):
            if composite not in parts:
                parts.append(str(composite))

        dialog = OverlayPartsDialog(parts, inference.overlay_parts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        inference.overlay_parts = dialog.selected_parts()

    def _show_message(self, title: str, message: str):
        @invoke_method
        def show_in_gui_thread(title=title, message=message):
            summary = next((line.strip() for line in message.splitlines() if line.strip()), title)
            self.statusBar().showMessage(f"Error: {title}: {summary}", 15000)
        show_in_gui_thread()

    @invoke_method
    def _on_preferences_property_changed(self, name, value, _):
        prefs = self._preferences
        if name == prefs.LOG_LEVEL:
            self._update_log_level(value)
        elif name == prefs.SERIAL_NUMBER:
            self._title = _make_window_title(prefs)
            self.setWindowTitle(self._title)

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

    @staticmethod
    def _simulate_intersession_offline_3d(*args, **kwargs):
        # A simulated session has no pose files to triangulate.
        logger.verbose("Simulate intersession offline 3D: skipped")

    def _prepare_simulated_analysis_result(self):
        app_model = self._app_model
        if self.analysis_results_action.isChecked():
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
            inference._intersession_offline_3d_execute = self._simulate_intersession_offline_3d

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
            self._orig_inference_offline_3d = inference._intersession_offline_3d_execute
        else:
            logger.verbose("Restoring intersession segmentation and detection to real procedures")
            inference._feed_intersession_analysis_execute = self._orig_inference_analysis_feed
            inference._intersession_process_execute = self._orig_inference_analysis_process
            inference._intersession_offline_3d_execute = self._orig_inference_offline_3d

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
            if algo.status != BehaviorAlgoStatus.RUNNING:
                return
            print("starting new simulate session")
            app.start_recording()
            do_sleep()
            t_end = time.perf_counter() + 10
            while algo.capture_status != CaptureProcessStatus.RECORDING:
                time.sleep(0.05)
                if time.perf_counter() > t_end:
                    print(f"awaited RECORDING but is still {algo.capture_status}")
                    break
            for idx in range(n_trials):
                print("wait monitoring")
                t_end = time.perf_counter() + 15
                while pellet_m.state != PelletState.monitoring:
                    if algo.status != BehaviorAlgoStatus.RUNNING:
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
                if random.random() <= rand_hands_near_pellet:
                    print("setting mouse near pellet")
                    self.mouse_near_pellet_action.toggle()
                    self.mouse_near_pellet_action.trigger()
                    do_sleep()
                if wait_released:
                    print("waiting released")
                    t_end = time.perf_counter() + 3
                    while True:
                        if pellet_m.covered_state is False:
                            print("is released")
                            break
                        if algo.status != BehaviorAlgoStatus.RUNNING:
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
                    if algo.status != BehaviorAlgoStatus.RUNNING:
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
            self._prepare_simulated_analysis_result()
            app.stop_recording()
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
        self._notes_save_timer.start()

    def _add_animal(self):
        self._app_model.add_animal(self._animal_dropdown_combo.currentText(), select=True)

    def _rfid_entered(self) -> None:
        text = self._rfid_entry.text()
        self._rfid_entry.clear()
        if not text.strip():
            return
        try:
            self._app_model.resolve_entered_rfid(text)
        except ValueError as exc:
            self.statusBar().showMessage(f"RFID not accepted: {exc}", 8000)
        except Exception:
            logger.exception("Typed animal identifier resolution failed")
            self.statusBar().showMessage("RFID lookup failed; see the log", 8000)

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
        if attached is None or attached.is_automatic:
            self.previous_training_phase_action.setVisible(False)
            self.next_training_phase_action.setVisible(False)
            return
        for action, can_do, direction in (
            (self.previous_training_phase_action, attached.can_fallback, "left"),
            (self.next_training_phase_action, attached.can_advance, "right"),
        ):
            action.setVisible(True)
            name = f"fa5s.arrow-alt-circle-{direction}"
            action.setIcon(_toolbar_icon(name))
            action.setEnabled(can_do)

    @invoke_method
    def _on_app_model_property_changed(self, name: str, value, prev_value):
        app_model = self._app_model
        props = app_model.Props
        #
        if name == props.ACQUISITION_RUNNING:
            self.running_status_changed.emit(value)

        elif name == props.STATUS:
            logger.debug("got new app model status: %s", value)

            # For Calibrate Coordinate System only, which redoes a stale DCS.
            valid_dcs = self.has_loaded_valid_dcs

            self.blockSignals(True)

            combo_idx = self._app_model_status_combo.findData(value)
            if combo_idx >= 0:
                self._app_model_status_combo.setCurrentIndex(combo_idx)

            if value is AppModelStatus.IDLE:
                self._warned_invalid_dcs_config = False
                self.calib_diamond_triangle_action.setEnabled(False)
                self.calib_diamond_triangle_action.setChecked(False)
                for item in (
                    self.make_3d_calib_action,
                    self.run_action,
                    self._animal_dropdown_combo,
                    self._training_plan_combo,
                ):
                    item.setEnabled(True)

            elif value is AppModelStatus.RUNNING:
                self.calib_diamond_triangle_action.setEnabled(valid_dcs)
                self.calib_diamond_triangle_action.setChecked(False)
                for item in (self._animal_dropdown_combo, self._training_plan_combo):
                    item.setEnabled(True)

            elif value in {AppModelStatus.CALIBRATION_3D, AppModelStatus.CALIBRATION_DCS}:
                for item in (
                    self._training_plan_combo,
                    self._animal_dropdown_combo,
                    self.calib_diamond_triangle_action,
                ):
                    item.setEnabled(False)

            else:
                logger.warning("unhandled app model status: %s", value)

            self.refresh_hardware_action.setEnabled(
                (
                    (
                        value is AppModelStatus.IDLE
                        and not app_model.acquisition_started
                    )
                    or (
                        app_model.acquisition_started
                        and app_model.session_recording_status
                        is SessionRecordingStatus.READY
                    )
                )
                and self._hardware_refresh_thread is None
            )
            self._set_hardware_menu_actions_enabled(
                value is AppModelStatus.IDLE
                and not app_model.acquisition_started
                and self._hardware_refresh_thread is None
                and self._nidaq_discovery_thread is None
            )
            self.blockSignals(False)

            self.main_content.set_is_capture_active(value != AppModelStatus.IDLE)
            self._refresh_ui_availability()

        elif name == props.SESSION_RECORDING_STATUS:
            if value == SessionRecordingStatus.READY:
                self._on_app_model_property_changed(
                    props.STATUS,
                    app_model.status,
                    app_model.status,
                )
            else:
                for item in (
                    self._app_model_status_combo,
                    self.run_action,
                    self.refresh_hardware_action,
                    self.calib_diamond_triangle_action,
                    self.make_3d_calib_action,
                ):
                    item.setEnabled(False)
            self._refresh_ui_availability()

        elif name == props.NOTES:
            signals_were_blocked = self._notes.blockSignals(True)
            self._notes.setText(value or "")
            self._notes.blockSignals(signals_were_blocked)

        elif name == props.SUBSYSTEM_STATUSES:
            if self._start_capture_thread is None:
                self._update_runtime_health_label()

        elif name == props.LASER_CALIBRATION_ACTIVE:
            self._refresh_ui_availability()

        elif name == props.RFID_READER_STATUS:
            state = getattr(getattr(value, "state", None), "value", "disabled")
            reason = getattr(value, "reason", "")
            if (
                self._app_model.rfid_scan_result is None
                and self._app_model.selected_animal is None
            ):
                # Reader health belongs in Hardware Status. The title-bar value
                # represents only the current session's tag and begins empty.
                self._rfid_status_label.setText("RFID: —")
                self._rfid_status_label.setToolTip(reason or f"RFID reader: {state}")
                self._rfid_status_label.setStyleSheet(
                    ""
                    if state in {"ready", "stopped", "disabled"}
                    else "color: #b36b00;"
                )

        elif name == props.RFID_SCAN_RESULT:
            kind = getattr(getattr(value, "kind", None), "value", "unknown")
            animal = getattr(value, "animal", None)
            rfid = getattr(value, "rfid", "")
            self._rfid_status_label.setText(f"RFID: {rfid or 'unknown'}")
            if animal is not None:
                self._rfid_status_label.setStyleSheet("color: #217a3c;")
                self._rfid_status_label.setToolTip(
                    f"Subject: {animal.name}\nResult: {kind.replace('_', ' ')}"
                    + (
                        f"\n{getattr(value, 'message', '')}"
                        if getattr(value, "message", "")
                        else ""
                    )
                )
            elif kind == RfidResolutionKind.SETUP_REQUIRED.value:
                self._rfid_status_label.setStyleSheet("color: #b36b00;")
                self._rfid_status_label.setToolTip(
                    f"First scan: set up or link this animal\nTag: {rfid}"
                )
                QTimer.singleShot(0, partial(self._open_rfid_setup_dialog, value))
            else:
                self._rfid_status_label.setStyleSheet("color: #a33;")
                self._rfid_status_label.setToolTip(
                    (getattr(value, "message", "") + "\n" if getattr(value, "message", "") else "")
                    + f"Tag: {rfid}\nUse Animal metadata > Refresh local cache or manual linking."
                )

        elif name == props.ANIMALS:
            self._reload_animals(value)

        elif name == props.SELECTED_ANIMAL:
            self._sync_selected_animal_widget(value)

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
            self._status_training_widget.setVisible(
                self.main_content.protocol_ui_enabled and value is not None
            )
            self._refresh_prev_next_phases()

        elif name == props.TRAINING_PLANS:
            self._set_training_plans(value)

        elif name == props.TRAINING_PHASE:
            self._refresh_prev_next_phases()

    def _open_rfid_setup_dialog(self, resolution) -> None:
        if self._rfid_setup_dialog is not None:
            logger.warning(
                "RFID first-scan dialog already open; setup remains pending: rfid=%s",
                resolution.rfid,
            )
            return
        if (
            resolution.kind is not RfidResolutionKind.SETUP_REQUIRED
            or resolution.record is None
        ):
            return
        dialog = AnimalDetailsDialog(
            self._app_model,
            self,
            record=resolution.record,
            scanned_rfid=resolution.rfid,
        )
        self._rfid_setup_dialog = dialog
        logger.info("Opening RFID first-scan setup dialog: rfid=%s", resolution.rfid)
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            self._rfid_setup_dialog = None
        if accepted:
            if dialog.saved_animal is not None:
                # Creation and linking select in the model. Reassert the dialog
                # result here so a concurrent list refresh cannot leave the UI
                # displaying a previous subject.
                self._app_model.selected_animal = dialog.saved_animal
                self._sync_selected_animal_widget(dialog.saved_animal)
            logger.info("RFID first-scan setup accepted: rfid=%s", resolution.rfid)
        else:
            logger.info(
                "RFID first-scan setup canceled; no animal was created: rfid=%s",
                resolution.rfid,
            )
            self.statusBar().showMessage(
                f"RFID {resolution.rfid} setup canceled; nothing was created",
                8000,
            )

    def _sync_selected_animal_widget(
        self, animal: Optional[AnimalSubject]
    ) -> None:
        combo = self._animal_dropdown_combo
        signals_were_blocked = combo.blockSignals(True)
        try:
            if animal is None:
                combo.setCurrentIndex(0)
                combo.setEditText("")
            else:
                index = combo.findData(animal.id)
                if index == -1:
                    logger.warning(
                        "Selected animal was not in the subject widget; adding it: "
                        "animal_id=%s animal_name=%s",
                        animal.id,
                        animal.name,
                    )
                    combo.addItem(animal.name, animal.id)
                    index = combo.count() - 1
                combo.setCurrentIndex(index)
                # QComboBox can retain its pre-dialog edit text even after the
                # model index changes. Set it explicitly for first-time animals.
                combo.setEditText(animal.name)
        finally:
            combo.blockSignals(signals_were_blocked)
        self._show_selected_animal_identity(animal)
        self._refresh_prev_next_phases()

    def _show_selected_animal_identity(self, animal: Optional[AnimalSubject]) -> None:
        if animal is None:
            self._rfid_status_label.setText("RFID: —")
            self._rfid_status_label.setStyleSheet("")
            self._rfid_status_label.setToolTip("No RFID scanned or selected")
            return
        metadata = animal.external_metadata
        rfid = None if metadata is None else metadata.rfid
        if rfid:
            self._rfid_status_label.setText(f"RFID: {rfid}")
            self._rfid_status_label.setStyleSheet("color: #217a3c;")
            self._rfid_status_label.setToolTip(f"Subject: {animal.name}\nLinked RFID")
        else:
            self._rfid_status_label.setText("RFID: unlinked")
            self._rfid_status_label.setStyleSheet("color: #b36b00;")
            self._rfid_status_label.setToolTip(
                f"Subject: {animal.name}\nNo RFID linked"
            )

    @invoke_method
    def _reload_animals(self, animals: List[AnimalSubject]):
        # get current selected animal before adding them,
        # given when adding that's modifying the currently selected one too,
        # which reset the preference selected to that one...
        find_animal_id = (
            self._preferences.selected_animal if self._app_model.selected_animal is None
            else self._app_model.selected_animal.id
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
        if find_animal_id is not None:
            index = combo.findData(find_animal_id)
            if index == -1:
                # One-release compatibility with name-based preferences.
                index = combo.findText(find_animal_id)
            if index != -1:
                combo.setCurrentIndex(index)
        else:
            combo.setCurrentIndex(0)
        self._show_selected_animal_identity(self._app_model.selected_animal)

    @staticmethod
    def _update_log_level(value: int):
        # controlled via console and file handlers now
        get_console_handler().setLevel(value)

    @invoke_method
    def _on_inference_property_changed(self, name: str, value, old_value):
        inference = self._app_model.inference
        if name == inference.STATUS:
            self._refresh_ui_availability()

    @invoke_method
    def _on_hardware_property_changed(self, property_name: str, value, _):
        hard = self._app_model.hardware
        if property_name in {hard.POS_XYZ, hard.SEND_X, hard.SEND_Y, hard.SEND_Z}:
            self._status_label_pos.update_coordinate(hard.last_dcs_position)
            self._status_label_send_pos.update_coordinate(hard.last_dcs_set_position)

    @invoke_method
    def _set_training_plans(self, plans: List[PlanInfo]):
        app_model = self._app_model
        combo = self._training_plan_combo
        combo.blockSignals(True)
        combo.clear()
        has_some = len(plans) > 0
        empty_txt = "Manual pellet control"
        tooltip_txt = (
            "Use the pellet controls without a protocol" if has_some
            else "Manual pellet control (no protocols are configured)"
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
            self._status_training_widget.setVisible(False)
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
        self._status_training_widget.setVisible(
            self.main_content.protocol_ui_enabled
            and app_model.training_plan is not None
        )

    @invoke_method
    def _on_app_model_configuration_loaded(self, config):
        self._set_training_plans(self._app_model.training_plans)
        self._sync_hardware_menu_actions()
        self._sync_capture_binning_actions()
        self._set_hardware_menu_actions_enabled(
            not self._app_model.acquisition_started
            and self._app_model.status == AppModelStatus.IDLE
            and self._hardware_refresh_thread is None
            and self._nidaq_discovery_thread is None
        )
        self._refresh_ui_availability()
        self._remind_if_calibration_due()

    def _remind_if_calibration_due(self) -> None:
        """Suggest a new 3D calibration, once per run, when the active one is old or there is none.

        Checked at the run's first configuration load, due or not: by then the app has loaded its
        calibration, and a month-old threshold crossed later in a run is not worth a second look.
        It only shows the message, non-modal, and a status-bar label that stays until a new
        calibration is installed (_finish_3d_calib); the operator starts a calibration from the
        menu they name. An unreadable calibration counts as due, and nothing here may fail a
        configuration load.
        """
        if self._calibration_reminder_shown:
            return
        self._calibration_reminder_shown = True
        created = None
        unreadable = False
        calib_dir = self._app_model.active_calibration_dir
        if calib_dir is not None:
            try:
                created = calibration_created(calib_dir)
            except Exception as err:
                unreadable = True
                # An unreadable or corrupt calibration is expected; anything else is a bug worth its traceback.
                logger.warning("3D calibration reminder: the date of %s is unreadable, taken as due: %s: %s",
                               calib_dir, type(err).__name__, err,
                               exc_info=not isinstance(err, (OSError, ValueError)))
        # A format-2 index may carry a UTC offset; comparing with a naive now would raise.
        now = datetime.now(None if created is None else created.tzinfo)
        if calibration_due(created, now, self._preferences.calibration_reminder_days):
            self._set_calibration_due_label(due_label_text(created, unreadable=unreadable))
            self._show_msg_box(
                "Calibration due", reminder_text(created, unreadable=unreadable), QMessageBox.Icon.Information)

    @invoke_method
    def _set_calibration_due_label(self, text: Optional[str]) -> None:
        """Show `text` in the status bar's calibration-due label, or hide the label for None."""
        label = self._calibration_due_label
        if text is None:
            label.setVisible(False)
            return
        label.setText(text)
        label.setVisible(True)

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
