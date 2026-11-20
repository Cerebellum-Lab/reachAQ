from types import SimpleNamespace

import pytest

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from tools.acquisition.model.app_model_status import (
    AppModelStatus,
    SessionRecordingStatus,
)
from tools.acquisition.view.main_window import MainWindow
from tools.acquisition.view.ui_availability import calculate_ui_availability


def _state(**overrides):
    values = {
        "status": AppModelStatus.RUNNING,
        "recording_status": SessionRecordingStatus.READY,
        "acquisition_started": True,
        "capture_transition": False,
        "hardware_refreshing": False,
        "nidaq_discovering": False,
        "has_valid_dcs": True,
    }
    values.update(overrides)
    return calculate_ui_availability(**values)


def test_running_ready_keeps_mode_selector_available():
    state = _state()

    assert state.system_mode
    assert state.preferences
    assert state.hardware_refresh
    assert state.subject
    assert state.protocol_selection
    assert not state.idle_configuration


def test_recording_locks_identity_and_system_controls_but_not_notes():
    state = _state(recording_status=SessionRecordingStatus.RECORDING)

    assert not state.system_mode
    assert not state.preferences
    assert not state.subject
    assert not state.protocol_selection
    assert not state.hardware_refresh
    assert state.notes


def test_idle_enables_configuration_and_identity_controls():
    state = _state(
        status=AppModelStatus.IDLE,
        acquisition_started=False,
        has_valid_dcs=False,
    )

    assert state.system_mode
    assert state.preferences
    assert state.idle_configuration
    assert state.hardware_refresh
    assert state.subject
    assert not state.calibration


def test_a_laser_calibration_ramp_disables_what_it_would_refuse():
    # Run, Edit DAQ Ports, the DAQ Monitor and hardware refresh stayed
    # enabled during a ramp and were refused only once used.
    state = _state(
        status=AppModelStatus.IDLE,
        acquisition_started=False,
        laser_calibrating=True,
    )

    assert not state.system_mode
    assert not state.idle_configuration
    assert not state.hardware_refresh
    assert state.notes


def test_background_operation_temporarily_disables_conflicting_controls():
    state = _state(hardware_refreshing=True)

    assert not state.system_mode
    assert not state.preferences
    assert not state.subject
    assert state.notes


class _Widget:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _Window(SimpleNamespace):
    """What MainWindow's availability refresh touches, with its real DCS properties."""

    has_fully_valid_dcs = MainWindow.has_fully_valid_dcs
    has_loaded_valid_dcs = MainWindow.has_loaded_valid_dcs

    def _set_hardware_menu_actions_enabled(self, enabled):
        pass

    def _set_capture_binning_actions_enabled(self, enabled):
        pass


_WIDGETS = (
    "_app_model_status_combo", "run_action", "preferences_action", "edit_camera_settings_action",
    "edit_daq_ports_action", "make_3d_calib_action", "daq_monitor_action", "refresh_hardware_action",
    "_animal_dropdown_combo", "_rfid_entry", "animal_metadata_action", "_training_plan_combo", "_notes",
    "calib_diamond_triangle_action",
)


@pytest.fixture
def algo(monkeypatch, tmp_path) -> BehaviorAlgorithm:
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    return BehaviorAlgorithm(diamond_triangle_offset_config_path=tmp_path / "diamond_triangle_offset.yaml")


def _running_window(algo) -> _Window:
    app_model = SimpleNamespace(
        status=AppModelStatus.RUNNING,
        session_recording_status=SessionRecordingStatus.READY,
        acquisition_started=True,
        laser_calibration_active=False,
        behavior=SimpleNamespace(algorithm=algo),
    )
    return _Window(
        _app_model=app_model,
        _start_capture_thread=None,
        _stop_capture_thread=None,
        _hardware_refresh_thread=None,
        _nidaq_discovery_thread=None,
        **{name: _Widget() for name in _WIDGETS},
    )


def test_dcs_action_enabled_while_stale(algo):
    # Calibrate Coordinate System is how a stale DCS is redone, so staleness must not
    # disable it: it follows the loaded configuration, which AppModel holds off as STALE.
    algo.diamond_triangle_config = DiamondTriangleOffsetConfig(
        used_position=Offset3DTuple(7.0, 4.5, 6.5),
        measured_offset=Offset3DTuple(4.7, -34.0, -8.4),
        diamond_coord=Offset3DTuple(0, 0, 0),
        raw_diamond_coord=Offset3DTuple(3.6, -19.3, 0.7),
        version=DiamondTriangleOffsetConfig.current_config_version,
    )
    algo.set_dcs_suppressed(True)
    window = _running_window(algo)

    MainWindow._refresh_ui_availability(window)

    assert window.calib_diamond_triangle_action.enabled is True
    # Every other use sees what the consumers see, so starting acquisition still warns.
    assert window.has_fully_valid_dcs is False


def test_dcs_action_needs_a_loaded_dcs(algo):
    window = _running_window(algo)

    MainWindow._refresh_ui_availability(window)

    assert window.calib_diamond_triangle_action.enabled is False
