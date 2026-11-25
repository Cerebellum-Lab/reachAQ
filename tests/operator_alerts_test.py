from types import SimpleNamespace

import pytest

from reachaq.core.interfaces import RecordingEndingReason
from reachaq.app.model.app_model_status import SessionRecordingStatus
from reachaq.app.model.operator_alerts import (
    AlertSeverity,
    OperatorAlertBridge,
    OperatorAlertCenter,
    alert_for_error,
    alert_for_subsystem,
)
from reachaq.app.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
    SubsystemStatus,
)


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def __call__(self, *args):
        for handler in tuple(self.handlers):
            handler(*args)


class FakeModel:
    Props = SimpleNamespace(
        SUBSYSTEM_STATUSES="subsystem_statuses",
        SESSION_RECORDING_STATUS="session_recording_status",
    )

    def __init__(self):
        self.on_error = _Event()
        self.property_changed = _Event()
        self.hardware = SimpleNamespace(
            property_changed=_Event(),
            PELLET_NEEDS_HOME_PROPERTY="pellet_needs_home",
            PELLET_DRIVER_FAULT_PROPERTY="pellet_driver_fault",
            PELLET_ESTOP_PROPERTY="pellet_estop",
        )
        self.subsystem_statuses = {}
        self.session_recording_status = SessionRecordingStatus.READY
        self.recording_ending_reason = RecordingEndingReason.NA

    def set_status(self, subsystem_id, state, **kwargs):
        key = getattr(subsystem_id, "value", subsystem_id)
        self.subsystem_statuses = {
            **self.subsystem_statuses,
            key: SubsystemStatus(subsystem_id=key, state=state, **kwargs),
        }
        self.property_changed(self.Props.SUBSYSTEM_STATUSES, self.subsystem_statuses, None)

    def set_recording(self, status):
        previous, self.session_recording_status = self.session_recording_status, status
        self.property_changed(self.Props.SESSION_RECORDING_STATUS, status, previous)


@pytest.fixture
def wired():
    model = FakeModel()
    center = OperatorAlertCenter(clock=iter(range(10_000)).__next__)
    dialogs = []
    center.dialog_requested.append(dialogs.append)
    bridge = OperatorAlertBridge(model, center)
    bridge.connect()
    return model, center, dialogs


def test_an_unlisted_on_error_title_stays_a_status_bar_line_only(wired):
    model, center, dialogs = wired
    model.on_error("Recording unavailable", "Set System Mode to Running before recording.")
    assert center.active_alerts() == ()
    assert dialogs == []


def test_data_at_risk_errors_open_one_dialog_and_wait_for_dismiss(wired):
    model, center, dialogs = wired
    model.on_error("Camera writer close timeout", "left did not close in 30 s")
    model.on_error("Camera writer close timeout", "left did not close in 30 s")
    (alert,) = center.active_alerts()
    assert alert.severity is AlertSeverity.CRITICAL
    assert alert.count == 2
    assert len(dialogs) == 1, "a repeat within a minute gets no second dialog"
    center.dismiss(alert.key)
    assert center.active_alerts() == ()
    assert [item.active for item in center.history()] == [True, True, False]


def test_storage_warnings_are_warnings_not_errors(wired):
    model, center, _ = wired
    model.on_error("Storage below 30 minutes", "29 min left")
    (alert,) = center.active_alerts()
    assert alert.severity is AlertSeverity.WARNING
    assert not alert.needs_dialog
    assert "Data location" in alert.remedy


def test_a_failed_subsystem_alerts_and_clears_itself_when_it_recovers(wired):
    model, center, _ = wired
    model.set_status(SubsystemId.NIDAQ_STREAM, SubsystemState.FAILED, error="worker exited (code -9)")
    (alert,) = center.active_alerts()
    assert alert.title == "NI-DAQ input stream failed"
    assert "worker exited" in alert.message
    assert "Refresh Hardware" in alert.remedy
    assert alert.is_condition and not alert.needs_dialog
    model.set_status(SubsystemId.NIDAQ_STREAM, SubsystemState.READY)
    assert center.active_alerts() == ()


def test_a_failure_while_recording_says_the_recording_carries_on(wired):
    model, center, _ = wired
    model.set_recording(SessionRecordingStatus.ARMING)
    model.set_recording(SessionRecordingStatus.RECORDING)
    model.set_status(SubsystemId.LASER, SubsystemState.FAILED, error="controller fault")
    (alert,) = center.active_alerts()
    assert "recording continues without it" in alert.note
    assert alert.message == "controller fault"


def test_a_dismissed_failure_stays_dismissed_when_recording_starts_and_stops(wired):
    model, center, _ = wired
    model.set_status(SubsystemId.RFID_READER, SubsystemState.FAILED, error="no device")
    center.dismiss("subsystem:rfid_reader")
    for status in (SessionRecordingStatus.ARMING, SessionRecordingStatus.RECORDING,
                   SessionRecordingStatus.STOPPING, SessionRecordingStatus.READY):
        model.set_recording(status)
    (alert,) = center.active_alerts()
    assert alert.dismissed and alert.count == 1
    assert center.visible_alerts() == ()


def test_a_dismissed_condition_stays_hidden_until_it_changes(wired):
    model, center, _ = wired
    model.set_status("camera.left", SubsystemState.FAILED, error="no frame in 5 s")
    (alert,) = center.active_alerts()
    assert alert.title == "Camera left failed"
    center.dismiss(alert.key)
    assert center.visible_alerts() == ()
    model.set_status("camera.left", SubsystemState.FAILED, error="no frame in 5 s")
    assert center.visible_alerts() == ()
    model.set_status("camera.left", SubsystemState.FAILED, error="device removed")
    assert center.visible_alerts()[0].message.endswith("device removed")


def test_blocked_alerts_only_for_a_refused_nidaq_plan():
    paused = SubsystemStatus(SubsystemId.NIDAQ_STREAM.value, SubsystemState.BLOCKED,
                             reason="NI-DAQ stream is paused: the configuration is loading")
    refused = SubsystemStatus(SubsystemId.NIDAQ_STREAM.value, SubsystemState.BLOCKED,
                              reason="NI-DAQ plan refused; open Edit DAQ Ports: duplicate pin ai3")
    secondary = SubsystemStatus("camera.right", SubsystemState.BLOCKED, reason="primary trigger unavailable")
    assert alert_for_subsystem(paused, recording=False) is None
    assert alert_for_subsystem(secondary, recording=False) is None
    alert = alert_for_subsystem(refused, recording=False)
    assert alert is not None and "Edit DAQ Ports" in alert.remedy


@pytest.mark.parametrize("reason", [
    RecordingEndingReason.REQUIRED_SOURCE_FAILURE,
    RecordingEndingReason.STORAGE_LIMIT,
])
def test_an_early_stop_opens_a_dialog_once_the_reason_is_known(wired, reason):
    model, center, dialogs = wired
    model.set_recording(SessionRecordingStatus.ARMING)
    model.set_recording(SessionRecordingStatus.RECORDING)
    # The status turns STOPPING before the ending reason is recorded.
    model.set_recording(SessionRecordingStatus.STOPPING)
    assert dialogs == []
    model.recording_ending_reason = reason
    model.set_recording(SessionRecordingStatus.ANALYZING)
    model.set_recording(SessionRecordingStatus.READY)
    assert [alert.title for alert in dialogs] == ["Recording stopped early"]


def test_a_failed_start_after_an_early_stop_reports_nothing_new(wired):
    """The ending reason keeps the last session's value until the next one starts
    recording; a start that fails while ARMING must not reuse it."""
    model, center, dialogs = wired
    model.set_recording(SessionRecordingStatus.ARMING)
    model.set_recording(SessionRecordingStatus.RECORDING)
    model.recording_ending_reason = RecordingEndingReason.STORAGE_LIMIT
    model.set_recording(SessionRecordingStatus.STOPPING)
    model.set_recording(SessionRecordingStatus.READY)
    assert len(dialogs) == 1
    model.set_recording(SessionRecordingStatus.ARMING)
    model.set_recording(SessionRecordingStatus.ABORTING)
    model.set_recording(SessionRecordingStatus.READY)
    assert len(dialogs) == 1


def test_start_failures_and_storage_critical_get_no_dialog_of_their_own(wired):
    model, center, dialogs = wired
    model.on_error("Recording failed", "The primary camera did not begin recording within 10 seconds.")
    model.on_error("Storage critical — recording will stop", "1 min left")
    assert dialogs == []
    assert {alert.title for alert in center.active_alerts()} == {
        "Recording failed", "Storage critical — recording will stop"}


def test_an_internal_error_alerts_once(wired):
    model, center, _ = wired
    model.on_error("Internal software error", "ZeroDivisionError")
    model.set_status(SubsystemId.RUNTIME_DIAGNOSTICS, SubsystemState.FAILED, error="ZeroDivisionError")
    assert [alert.title for alert in center.active_alerts()] == ["Internal software error"]


def test_a_failing_listener_never_reaches_the_reporting_thread(wired):
    model, center, _ = wired

    def broken():
        raise RuntimeError("deleted relay")

    center.changed.append(broken)
    model.on_error("Pellet home failed", "limit switch")  # must not raise
    model.set_status(SubsystemId.LASER, SubsystemState.FAILED, error="x")


def test_reconcile_reads_the_latest_statuses_not_the_published_value(wired):
    model, center, _ = wired
    model.set_status(SubsystemId.LASER, SubsystemState.FAILED, error="fault")
    model.subsystem_statuses = {}
    # A late, older snapshot: the bridge must use what the model holds now.
    stale = {"laser": SubsystemStatus("laser", SubsystemState.FAILED, error="fault")}
    model.property_changed(model.Props.SUBSYSTEM_STATUSES, stale, None)
    assert center.active_alerts() == ()


def test_an_operator_stop_opens_no_dialog(wired):
    model, center, dialogs = wired
    model.set_recording(SessionRecordingStatus.ARMING)
    model.set_recording(SessionRecordingStatus.RECORDING)
    model.recording_ending_reason = RecordingEndingReason.MANUAL_STOP
    model.set_recording(SessionRecordingStatus.STOPPING)
    model.set_recording(SessionRecordingStatus.READY)
    assert dialogs == [] and center.active_alerts() == ()


def test_pellet_axes_needing_home_alert_and_clear(wired):
    model, center, _ = wired
    model.hardware.property_changed("pellet_needs_home", ("x", "y"), ())
    (alert,) = center.active_alerts()
    assert alert.title == "Pellet axes need homing" and "x, y" in alert.message
    model.hardware.property_changed("pellet_needs_home", (), ("x", "y"))
    assert center.active_alerts() == ()


def test_alerts_sort_by_severity_then_newest():
    center = OperatorAlertCenter(clock=iter(range(100)).__next__)
    for title in ("Storage capacity warning", "Pellet home failed", "Recording failed", "Storage below 10 minutes"):
        center.raise_alert(alert_for_error(title, "detail"))
    assert [alert.title for alert in center.active_alerts()] == [
        "Recording failed", "Pellet home failed", "Storage below 10 minutes", "Storage capacity warning"]


def test_the_bridge_reads_a_real_app_model(app_model):
    """The names the bridge subscribes to exist on AppModel and carry what it expects."""
    center = OperatorAlertCenter()
    bridge = OperatorAlertBridge(app_model, center)
    bridge.connect()
    try:
        app_model.on_error("Session data incomplete", "trial_summary.json was not written")
        app_model._set_subsystem_status(
            SubsystemId.CAN_PELLET, SubsystemState.FAILED, error="ack timeout")
        titles = {alert.title for alert in center.active_alerts()}
        assert {"Session data incomplete", "Pellet board (CAN) failed"} <= titles
        assert app_model.recording_ending_reason is RecordingEndingReason.NA
        app_model._set_subsystem_status(SubsystemId.CAN_PELLET, SubsystemState.READY)
        assert "Pellet board (CAN) failed" not in {alert.title for alert in center.active_alerts()}
    finally:
        bridge.disconnect()


def _estop(reason="MOVE_Z was not acknowledged within 3 s", *, operator=False, sent=True):
    return {"wallTime": 1.0, "perfTime": 2.0, "reason": reason, "axes": ["X", "Y", "Z"],
            "operator": operator, "sent": sent}


def test_a_stopped_stage_raises_a_blocking_error_until_it_is_homed(wired):
    model, center, dialogs = wired
    record = _estop()
    model.hardware.property_changed("pellet_estop", record, None)

    (alert,) = center.active_alerts()
    assert alert.severity is AlertSeverity.ERROR and alert.needs_dialog and alert.is_condition
    assert alert.title == "Pellet stage stopped"
    assert alert.message == (
        "Pellet stage stopped: MOVE_Z was not acknowledged within 3 s. Home before continuing "
        "(Hardware Control ▸ Home); the session-end home also clears it when the pellet board "
        "is Ready at session end.")
    assert "Hardware Control ▸ Home" in alert.remedy
    # Review M-6: Home is enabled only with the board connected and acquisition started.
    assert "acquisition is started" in alert.remedy and "Refresh Hardware" in alert.remedy
    assert dialogs == [alert]

    model.hardware.property_changed("pellet_estop", None, record)
    assert center.active_alerts() == ()


def test_the_operators_stop_says_stop_was_pressed_and_is_no_failure(wired):
    model, center, dialogs = wired
    model.hardware.property_changed("pellet_estop", _estop("operator STOP", operator=True), None)

    (alert,) = center.active_alerts()
    assert alert.title == "Pellet stage stopped by the operator"
    assert alert.message == (
        "STOP was pressed in Hardware Control. Home before continuing (Hardware Control ▸ Home) "
        "or reconnect (Refresh Hardware, or a calibration capture's connect); the session-end "
        "home does not clear it.")
    assert "fail" not in (alert.title + alert.message).lower()
    assert alert.severity is AlertSeverity.WARNING and not alert.needs_dialog
    assert dialogs == []


def test_a_stop_the_bus_did_not_carry_says_so(wired):
    model, center, _ = wired
    model.hardware.property_changed("pellet_estop", _estop(sent=False), None)

    (alert,) = center.active_alerts()
    assert "could not be written to the CAN bus" in alert.message


def test_the_bridge_raises_the_stop_from_a_real_hardware_model(app_model):
    center = OperatorAlertCenter()
    bridge = OperatorAlertBridge(app_model, center)
    bridge.connect()
    try:
        app_model.hardware._on_device_estop("a test stop", (), True)
        assert "Pellet stage stopped" in {alert.title for alert in center.active_alerts()}
        app_model.hardware._clear_estop("homed")
        assert "Pellet stage stopped" not in {alert.title for alert in center.active_alerts()}
    finally:
        bridge.disconnect()
