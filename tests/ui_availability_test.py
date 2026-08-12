from tools.acquisition.model.app_model_status import (
    AppModelStatus,
    SessionRecordingStatus,
)
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


def test_background_operation_temporarily_disables_conflicting_controls():
    state = _state(hardware_refreshing=True)

    assert not state.system_mode
    assert not state.preferences
    assert not state.subject
    assert state.notes
