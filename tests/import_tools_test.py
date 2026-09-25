

import inspect
import re
from types import SimpleNamespace

import tools


def test_we_can_import_acquisition_main_window():
    from tools.acquisition.view import main_window


def test_main_window_subscribes_only_existing_callbacks():
    from tools.acquisition.view.main_window import MainWindow

    source = inspect.getsource(MainWindow.__init__)
    callback_names = set(re.findall(r"\+= self\.(\w+)", source))

    assert callback_names
    assert not [name for name in callback_names if not hasattr(MainWindow, name)]


def test_hardware_switches_use_persistent_file_submenu_and_deferred_refresh():
    from tools.acquisition.view.hardware_status_content import HardwareStatusContent
    from tools.acquisition.view.main_window import MainWindow

    menu_source = inspect.getsource(MainWindow._configure_menubar)
    apply_source = inspect.getsource(MainWindow._apply_hardware_menu_values)
    status_source = inspect.getsource(HardwareStatusContent)

    assert 'MultiSelectionMenu("Hardware", file_menu)' in menu_source
    assert "aboutToHide.connect(self._flush_pending_hardware_refresh)" in menu_source
    assert "update_hardware_configuration" in apply_source
    assert "_hardware_refresh_pending" in apply_source
    assert "Hardware Configuration" not in status_source


def test_hardware_menu_checkmarks_follow_loaded_configuration():
    from tools.acquisition.view.main_window import MainWindow

    class ActionStub:
        def __init__(self):
            self.checked = None
            self.tooltip = None

        def blockSignals(self, _blocked):
            pass

        def setChecked(self, checked):
            self.checked = checked

        def setText(self, _text):
            pass

        def setToolTip(self, tooltip):
            self.tooltip = tooltip

    hardware = SimpleNamespace(
        can_enabled=True,
        pellet_controller_enabled=False,
        nidaq_enabled=True,
        rfid_reader_enabled=True,
        rfid_device="/dev/serial/by-id/test-rfid",
    )
    actions = {
        name: ActionStub()
        for name in (
            "can_enabled",
            "pellet_controller_enabled",
            "nidaq_enabled",
            "rfid_reader_enabled",
        )
    }
    window = SimpleNamespace(
        _app_model=SimpleNamespace(
            loaded_configuration=SimpleNamespace(hardware=hardware)
        ),
        hardware_enable_actions=actions,
        rfid_device_action=ActionStub(),
    )

    MainWindow._sync_hardware_menu_actions(window)

    assert actions["can_enabled"].checked is True
    assert actions["pellet_controller_enabled"].checked is False
    assert actions["nidaq_enabled"].checked is True
    assert actions["rfid_reader_enabled"].checked is True
    assert window.rfid_device_action.tooltip == hardware.rfid_device


def test_we_can_import_headless():
    from tools.acquisition import headless


def test_we_can_import_pellet_delivery_window():
    from tools.pellet_delivery.view import main_window


class _EnableRecorder:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


def test_the_daq_monitor_is_offered_only_while_idle():
    # It pauses the application's NI-DAQ stream, which in System Mode is the
    # acquisition's. The action was enabled only once, after a DAQ ports
    # discovery, and otherwise stayed enabled in every state.
    from tools.acquisition.model.app_model_status import (
        AppModelStatus,
        SessionRecordingStatus,
    )
    from tools.acquisition.view.main_window import MainWindow

    controls = (
        "_app_model_status_combo", "run_action", "preferences_action",
        "edit_camera_settings_action", "edit_daq_ports_action",
        "make_3d_calib_action", "refresh_hardware_action",
        "_animal_dropdown_combo", "animal_metadata_action",
        "_training_plan_combo", "_notes", "calib_diamond_triangle_action",
        "daq_monitor_action",
    )
    offered = {}
    for label, status, recording, started in (
        ("idle", AppModelStatus.IDLE, SessionRecordingStatus.READY, False),
        ("running", AppModelStatus.RUNNING, SessionRecordingStatus.READY, True),
        ("recording", AppModelStatus.RUNNING, SessionRecordingStatus.RECORDING, True),
    ):
        window = SimpleNamespace(
            _start_capture_thread=None,
            _stop_capture_thread=None,
            _hardware_refresh_thread=None,
            _nidaq_discovery_thread=None,
            has_fully_valid_dcs=False,
            _app_model=SimpleNamespace(
                status=status,
                session_recording_status=recording,
                acquisition_started=started,
            ),
            _set_hardware_menu_actions_enabled=lambda _enabled: None,
            **{name: _EnableRecorder() for name in controls},
        )
        MainWindow._refresh_ui_availability(window)
        offered[label] = window.daq_monitor_action.enabled

    assert offered == {"idle": True, "running": False, "recording": False}
