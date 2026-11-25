"""Operator-facing alerts for the failures an operator has to act on.

Nothing here detects a failure. The app already reports them through
``AppModel.on_error``, the subsystem status registry, the recording status and
the pellet board's properties; this module turns those into alerts with a
severity, a plain remedy and, for the few that put data at risk, a request for
a dialog. The status bar keeps showing what it showed before.

Two kinds of alert:

- a *condition* (a subsystem FAILED, the pellet axes need homing) stays active
  until the condition clears, and clears itself;
- an *event* (an ``on_error`` such as "Camera writer close timeout") stays
  active until the operator dismisses it.

Callbacks run on whatever thread reported the failure; listeners marshal.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import itertools
import threading
import time

from reachaq.core.logging import get_verbose_logger
from typing import Callable, Deque, Dict, List, Mapping, Optional, Tuple

from reachaq.app.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
    SubsystemStatus,
)


logger = get_verbose_logger(__name__)


def _guarded(handler):
    """Model events do not catch handler exceptions, so one raised here would
    land in the thread that reported the failure and skip later handlers."""

    def run(self, *args):
        try:
            handler(self, *args)
        except Exception:
            logger.exception("Operator alert handler %s failed", handler.__name__)

    run.__name__ = handler.__name__
    return run


class AlertSeverity(enum.IntEnum):
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclasses.dataclass(frozen=True)
class OperatorAlert:
    key: str
    severity: AlertSeverity
    title: str
    message: str
    remedy: str = ""
    #: Context that changes without the condition changing (whether a
    #: recording is running); shown, but not news on its own.
    note: str = ""
    #: Data were lost or a recording ended early: the operator gets a dialog.
    needs_dialog: bool = False
    #: A condition clears itself; an event waits for Dismiss.
    is_condition: bool = False
    raised_wall_time: float = 0.0
    count: int = 1
    active: bool = True
    dismissed: bool = False
    sequence: int = 0


REFRESH = "choose Refresh Hardware on the toolbar"

#: The remedy shown for a subsystem that FAILED, by subsystem id.
SUBSYSTEM_REMEDIES: Mapping[str, str] = {
    "camera": (
        f"Check the camera's USB3 cable and that it is not open in another program, then {REFRESH}."
    ),
    SubsystemId.REACH_SYNCHRONIZATION.value: (
        "Check that every reach camera is connected and that exactly one is primary "
        f"(Edit ▸ Edit Camera Settings), then {REFRESH}."
    ),
    SubsystemId.TOP_CAPTURE.value: f"Check the top camera's cable, then {REFRESH}.",
    SubsystemId.LIVE_INFERENCE.value: (
        "Check the pose model in Preferences and that the GPU is available (nvidia-smi), "
        f"then {REFRESH}."
    ),
    SubsystemId.CAN_PELLET.value: (
        f"Check the pellet board's power and the CAN adapter cable, then {REFRESH}."
    ),
    SubsystemId.NIDAQ_STREAM.value: (
        f"Check that the NI chassis is on and its cable is connected, then {REFRESH}."
    ),
    SubsystemId.LASER.value: (
        f"Check the laser controller's power and its NI lines, then {REFRESH}. "
        "No stimulation is delivered until it recovers."
    ),
    SubsystemId.RFID_READER.value: (
        "Check the RFID reader's USB cable and File ▸ Hardware ▸ RFID Serial Device."
    ),
}

SUBSYSTEM_NAMES: Mapping[str, str] = {
    SubsystemId.REACH_SYNCHRONIZATION.value: "Reach camera synchronization",
    SubsystemId.TOP_CAPTURE.value: "Top camera",
    SubsystemId.LIVE_INFERENCE.value: "Live pose",
    SubsystemId.CAN_PELLET.value: "Pellet board (CAN)",
    SubsystemId.NIDAQ_STREAM.value: "NI-DAQ input stream",
    SubsystemId.LASER.value: "Laser",
    SubsystemId.SESSION_LOGS.value: "Session logs",
    SubsystemId.INTERTRIAL_ANALYSIS.value: "Intertrial analysis",
    SubsystemId.ANIMAL_REGISTRY.value: "Animal registry",
    SubsystemId.RFID_READER.value: "RFID reader",
    SubsystemId.RUNTIME_DIAGNOSTICS.value: "Runtime diagnostics",
}

DIAGNOSTIC_REMEDY = (
    "Note the time, then choose Help ▸ Export Diagnostic Bundle and send the file to support."
)
NIDAQ_PLAN_REMEDY = "Fix the refused lines in Edit ▸ Edit DAQ Ports; the rest of the rig keeps running."
STORAGE_REMEDY = "Free space on the data drive, or choose a larger drive in Preferences ▸ Data location."


@dataclasses.dataclass(frozen=True)
class _ErrorRule:
    severity: AlertSeverity
    remedy: str = ""
    needs_dialog: bool = False


_DATA_AT_RISK = (
    "Check the session folder before the next recording, and send support a diagnostic "
    "bundle (Help ▸ Export Diagnostic Bundle).")

#: ``on_error`` titles that become alerts. Any other title stays a status-bar
#: line only: most are refusals of something the operator just tried.
ERROR_RULES: Mapping[str, _ErrorRule] = {
    # The recording ended, or data were not written.
    # A start that failed: nothing was recorded, and the operator is at the rig.
    "Recording failed": _ErrorRule(
        AlertSeverity.ERROR,
        f"No data were recorded. Check the primary camera, then {REFRESH} and record again."),
    "Camera writer close timeout": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Pose writer close timeout": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Camera recording incomplete": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Camera timestamp merge failed": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Session data incomplete": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Session stream save failed": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Abort cleanup failed": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    "Automatic recording stop timeout": _ErrorRule(AlertSeverity.CRITICAL, _DATA_AT_RISK, True),
    # The "Recording stopped early" alert that follows carries the dialog.
    "Storage critical — recording will stop": _ErrorRule(AlertSeverity.CRITICAL, STORAGE_REMEDY),
    # Data are still being written, but something needs attention.
    "Abort cleanup waiting for writer": _ErrorRule(
        AlertSeverity.ERROR, "Wait for the writer to finish before closing the app."),
    "Camera synchronization incomplete": _ErrorRule(
        AlertSeverity.WARNING, "Review this session's camera alignment before analysis."),
    "Storage capacity warning": _ErrorRule(AlertSeverity.WARNING, STORAGE_REMEDY),
    "Session notes were not saved": _ErrorRule(
        AlertSeverity.ERROR, "Copy the notes elsewhere and check the session folder is writable."),
    "Pellet command refused": _ErrorRule(
        AlertSeverity.ERROR,
        "Check the pellet board's emergency stop and motor driver, then Hardware Control ▸ Home."),
    "Pellet home failed": _ErrorRule(
        AlertSeverity.ERROR, "Check nothing blocks the pellet axes, then home them again."),
    "Inference configuration error": _ErrorRule(
        AlertSeverity.ERROR, "Check the live pose cameras in Preferences; live pose is off."),
    "Pose model pre-validate failed": _ErrorRule(
        AlertSeverity.ERROR, "Choose a different pose model in Preferences."),
    "Failed load configuration": _ErrorRule(
        AlertSeverity.CRITICAL,
        "Fix the file named above, or restore its backup, then restart reachAQ.",
        True),
    "Internal software error": _ErrorRule(AlertSeverity.ERROR, DIAGNOSTIC_REMEDY),
}

#: Titles that carry a number, matched by prefix.
ERROR_PREFIX_RULES: Tuple[Tuple[str, _ErrorRule], ...] = (
    ("Storage below ", _ErrorRule(AlertSeverity.WARNING, STORAGE_REMEDY)),
)

#: Why a recording ended that the operator did not ask for, by ending reason value.
EARLY_STOP_MESSAGES: Mapping[str, Tuple[str, str]] = {
    "RequiredSourceFailure": (
        "A source the recording needs (the primary camera) failed, so the recording was "
        "stopped. The data up to the failure were kept.",
        f"Check the primary camera, then {REFRESH} before the next recording.",
    ),
    "StorageLimit": (
        "The data drive was almost full, so the recording was stopped. The data up to the "
        "stop were kept.",
        STORAGE_REMEDY,
    ),
    "StopDrainTimeout": (
        "The recording did not finish writing in time. Some of the last frames may be missing.",
        _DATA_AT_RISK,
    ),
}

_RECORDING_STATES = {"arming", "recording"}


def _first_line(text: str, fallback: str = "") -> str:
    return next((line.strip() for line in str(text or "").splitlines() if line.strip()), fallback)


def subsystem_display_name(subsystem_id: str) -> str:
    if subsystem_id.startswith("camera."):
        return f"Camera {subsystem_id.split('.', 1)[1]}"
    return SUBSYSTEM_NAMES.get(subsystem_id, subsystem_id.replace("_", " ").capitalize())


def subsystem_remedy(subsystem_id: str) -> str:
    if subsystem_id.startswith("camera."):
        return SUBSYSTEM_REMEDIES["camera"]
    return SUBSYSTEM_REMEDIES.get(subsystem_id, DIAGNOSTIC_REMEDY)


def alert_for_subsystem(
    status: SubsystemStatus, *, recording: bool,
) -> Optional[OperatorAlert]:
    """The alert a subsystem's status calls for, or None when it calls for none.

    FAILED always alerts. BLOCKED alerts only for a refused NI-DAQ plan: every
    other BLOCKED is either a consequence of a FAILED subsystem, which has its
    own alert, or a pause the app makes and lifts itself.
    """
    subsystem_id = status.subsystem_id
    detail = status.error or status.reason or status.state.value
    name = subsystem_display_name(subsystem_id)
    if status.state is SubsystemState.BLOCKED:
        if subsystem_id != SubsystemId.NIDAQ_STREAM.value or "plan refused" not in detail:
            return None
        return OperatorAlert(
            key=f"subsystem:{subsystem_id}",
            severity=AlertSeverity.ERROR,
            title="NI-DAQ inputs are not acquired",
            message=detail,
            remedy=NIDAQ_PLAN_REMEDY,
            is_condition=True,
        )
    if status.state is not SubsystemState.FAILED:
        return None
    if subsystem_id == SubsystemId.RUNTIME_DIAGNOSTICS.value:
        # The "Internal software error" event alert reports it already.
        return None
    return OperatorAlert(
        key=f"subsystem:{subsystem_id}",
        severity=AlertSeverity.ERROR,
        title=f"{name} failed",
        message=str(detail),
        remedy=subsystem_remedy(subsystem_id),
        note=(
            "The recording continues without it; review this session before analysis."
            if recording else ""
        ),
        is_condition=True,
    )


def alert_for_error(title: str, message: str) -> Optional[OperatorAlert]:
    rule = ERROR_RULES.get(title)
    if rule is None:
        rule = next(
            (item for prefix, item in ERROR_PREFIX_RULES if str(title).startswith(prefix)),
            None,
        )
    if rule is None:
        return None
    return OperatorAlert(
        key=f"error:{title}",
        severity=rule.severity,
        title=str(title),
        message=str(message or "").strip() or str(title),
        remedy=rule.remedy,
        needs_dialog=rule.needs_dialog,
    )


def alert_for_early_stop(reason_value: str, session_number: int) -> Optional[OperatorAlert]:
    entry = EARLY_STOP_MESSAGES.get(str(reason_value))
    if entry is None:
        return None
    message, remedy = entry
    return OperatorAlert(
        key=f"recording-stopped:{session_number}",
        severity=AlertSeverity.CRITICAL,
        title="Recording stopped early",
        message=message,
        remedy=remedy,
        needs_dialog=True,
    )


def alert_for_pellet_axes(kind: str, axes: Tuple[str, ...]) -> Optional[OperatorAlert]:
    if not axes:
        return None
    joined = ", ".join(str(axis) for axis in axes)
    if kind == "needs_home":
        return OperatorAlert(
            key="pellet:needs_home",
            severity=AlertSeverity.WARNING,
            title="Pellet axes need homing",
            message=f"Axis {joined} must be homed before pellets can be sent.",
            remedy="Home the pellet axes with Hardware Control ▸ Home.",
            is_condition=True,
        )
    return OperatorAlert(
        key="pellet:driver_fault",
        severity=AlertSeverity.ERROR,
        title="Pellet motor driver fault",
        message=f"The stepper driver reports a fault on axis {joined}.",
        remedy="Check nothing blocks the axis, then Hardware Control ▸ Home to clear it.",
        is_condition=True,
    )


_HOME_TO_CLEAR = (
    "Home before continuing (Hardware Control ▸ Home); the session-end home also clears it."
)
_HOME_TO_CLEAR_OPERATOR = (
    "Home before continuing (Hardware Control ▸ Home) or reconnect (Refresh Hardware); "
    "the session-end home does not clear it."
)
# Hardware Control's Home is enabled only with the board connected and acquisition started
# (HardwareControlContent._refresh_enabled_state); a reconnect homes as well.
_ESTOP_REMEDY = (
    "Check nothing blocks the pellet axes, then Hardware Control ▸ Home (enabled when the pellet "
    "board is connected and acquisition is started), or Refresh Hardware."
)


def alert_for_pellet_estop(record: Optional[Mapping]) -> Optional[OperatorAlert]:
    """The alert for a stopped pellet stage (HardwareModel.pellet_estop), or None once homed.

    A condition: it clears itself when a home succeeds. A stop the host decided
    on asks for a dialog; the operator's own STOP does not, and is not called a
    failure.
    """
    if not record:
        return None
    remedy = _ESTOP_REMEDY
    if record.get("kind", "operator" if record.get("operator") else "failure") == "operator":
        return OperatorAlert(
            key="pellet:estop",
            severity=AlertSeverity.WARNING,
            title="Pellet stage stopped by the operator",
            message=f"STOP was pressed in Hardware Control. {_HOME_TO_CLEAR_OPERATOR}",
            remedy=remedy,
            is_condition=True,
        )
    sent = "" if record.get("sent", True) else " The ESTOP could not be written to the CAN bus."
    return OperatorAlert(
        key="pellet:estop",
        severity=AlertSeverity.ERROR,
        title="Pellet stage stopped",
        message=f"Pellet stage stopped: {record.get('reason', 'unknown reason')}. {_HOME_TO_CLEAR}{sent}",
        remedy=remedy,
        needs_dialog=True,
        is_condition=True,
    )


Listener = Callable[[], None]
DialogListener = Callable[[OperatorAlert], None]


class OperatorAlertCenter:
    """Active alerts and a bounded history, safe to call from any thread."""

    HISTORY_LIMIT = 500
    #: An event alert raised again within this many seconds gets no second dialog.
    DIALOG_REPEAT_SECONDS = 60.0

    def __init__(self, *, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._active: Dict[str, OperatorAlert] = {}
        self._history: Deque[OperatorAlert] = collections.deque(maxlen=self.HISTORY_LIMIT)
        self._sequence = itertools.count(1)
        self._last_dialog: Dict[str, float] = {}
        self.changed: List[Listener] = []
        self.dialog_requested: List[DialogListener] = []

    # -- reading ---------------------------------------------------------

    def active_alerts(self) -> Tuple[OperatorAlert, ...]:
        """Active alerts, most severe first, then newest first."""
        with self._lock:
            alerts = tuple(self._active.values())
        return tuple(sorted(alerts, key=lambda item: (-int(item.severity), -item.sequence)))

    def visible_alerts(self) -> Tuple[OperatorAlert, ...]:
        return tuple(alert for alert in self.active_alerts() if not alert.dismissed)

    def history(self) -> Tuple[OperatorAlert, ...]:
        """Every raise and clear, oldest first."""
        with self._lock:
            return tuple(self._history)

    # -- writing ---------------------------------------------------------

    def raise_alert(self, alert: OperatorAlert) -> OperatorAlert:
        now = self._clock()
        request_dialog = False
        with self._lock:
            previous = self._active.get(alert.key)
            if (
                previous is not None
                and previous.is_condition
                and alert.is_condition
                and (previous.title, previous.message) == (alert.title, alert.message)
            ):
                if previous.note == alert.note:
                    return previous  # the same condition published again
                # Only its context changed: no news, so it stays dismissed if
                # it was, and its count does not grow.
                stored = self._active[alert.key] = dataclasses.replace(previous, note=alert.note)
                news = False
            else:
                news = True
                stored = dataclasses.replace(
                    alert,
                    raised_wall_time=now,
                    count=1 if previous is None else previous.count + 1,
                    active=True,
                    # A changed condition is news; a repeated event stays
                    # dismissed only if the operator already dismissed it.
                    dismissed=bool(previous and previous.dismissed and not alert.is_condition),
                    sequence=next(self._sequence),
                )
                self._active[alert.key] = stored
                self._history.append(stored)
            if news and stored.needs_dialog:
                last = self._last_dialog.get(alert.key)
                if last is None or now - last >= self.DIALOG_REPEAT_SECONDS:
                    self._last_dialog[alert.key] = now
                    request_dialog = True
        self._notify()
        if request_dialog:
            for listener in tuple(self.dialog_requested):
                listener(stored)
        return stored

    def clear(self, key: str) -> None:
        with self._lock:
            previous = self._active.pop(key, None)
            if previous is None:
                return
            self._history.append(dataclasses.replace(
                previous,
                active=False,
                raised_wall_time=self._clock(),
                sequence=next(self._sequence),
            ))
        self._notify()

    def clear_matching(self, prefix: str, keep: Tuple[str, ...] = ()) -> None:
        with self._lock:
            keys = [key for key in self._active if key.startswith(prefix) and key not in keep]
        for key in keys:
            self.clear(key)

    def dismiss(self, key: str) -> None:
        """Hide an alert from the banner. An event is gone; a condition stays
        listed as active, hidden until it clears or changes."""
        with self._lock:
            alert = self._active.get(key)
            if alert is None:
                return
            if alert.is_condition:
                self._active[key] = dataclasses.replace(alert, dismissed=True)
            else:
                del self._active[key]
                self._history.append(dataclasses.replace(
                    alert, active=False, dismissed=True,
                    raised_wall_time=self._clock(), sequence=next(self._sequence)))
        self._notify()

    def _notify(self) -> None:
        for listener in tuple(self.changed):
            listener()


class OperatorAlertBridge:
    """Feeds an OperatorAlertCenter from an AppModel's existing events.

    It only subscribes; it changes nothing on the model. ``app_model`` needs
    ``on_error``, ``property_changed``, ``Props``, ``subsystem_statuses``,
    ``session_recording_status`` and ``hardware``.
    """

    def __init__(self, app_model, center: OperatorAlertCenter, *, dispatch=None):
        self._app_model = app_model
        self.center = center
        #: Runs a reconcile. The window passes one that queues it on the GUI
        #: thread: one thread then applies them in order and each reads the
        #: latest statuses, with no lock held while it reads the model's.
        self._dispatch = dispatch or (lambda function: function())
        self._lock = threading.RLock()
        self._recording = False
        self._sessions = 0
        #: The current session reached RECORDING. A session that fails while
        #: ARMING never recorded, and the ending reason still holds the last
        #: session's until the next one starts recording.
        self._reached_recording = False
        #: Set when a recorded session stops and cleared once its ending
        #: reason is read. The reason is recorded a step after the status
        #: turns STOPPING, so it is read on every later transition too.
        self._stop_check_pending = False
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        model = self._app_model
        model.on_error += self._on_error
        model.property_changed += self._on_app_property_changed
        hardware = getattr(model, "hardware", None)
        if hardware is not None:
            hardware.property_changed += self._on_hardware_property_changed
        self._connected = True
        self._recording = self._recording_now()
        self._reconcile_subsystems()

    def disconnect(self) -> None:
        if not self._connected:
            return
        model = self._app_model
        model.on_error -= self._on_error
        model.property_changed -= self._on_app_property_changed
        hardware = getattr(model, "hardware", None)
        if hardware is not None:
            hardware.property_changed -= self._on_hardware_property_changed
        self._connected = False

    # -- event handlers --------------------------------------------------

    @_guarded
    def _on_error(self, title, message) -> None:
        alert = alert_for_error(str(title), str(message or ""))
        if alert is not None:
            self.center.raise_alert(alert)

    @_guarded
    def _on_app_property_changed(self, name, value, _previous) -> None:
        props = self._app_model.Props
        if name == props.SUBSYSTEM_STATUSES:
            # Not from ``value``: snapshots are published after the model's
            # lock is released, so two threads can deliver them out of order.
            self._dispatch(self._reconcile_subsystems)
        elif name == props.SESSION_RECORDING_STATUS:
            self._on_recording_status(value)

    @_guarded
    def _on_hardware_property_changed(self, name, value, _previous) -> None:
        hardware = self._app_model.hardware
        if name == getattr(hardware, "PELLET_ESTOP_PROPERTY", None):
            alert = alert_for_pellet_estop(value)
            if alert is None:
                self.center.clear("pellet:estop")
            else:
                self.center.raise_alert(alert)
            return
        kinds = {
            getattr(hardware, "PELLET_NEEDS_HOME_PROPERTY", None): "needs_home",
            getattr(hardware, "PELLET_DRIVER_FAULT_PROPERTY", None): "driver_fault",
        }
        kind = kinds.get(name)
        if kind is None:
            return
        alert = alert_for_pellet_axes(kind, tuple(value or ()))
        if alert is None:
            self.center.clear(f"pellet:{kind}")
        else:
            self.center.raise_alert(alert)

    # -- helpers ---------------------------------------------------------

    def _recording_now(self) -> bool:
        status = getattr(self._app_model, "session_recording_status", None)
        return str(getattr(status, "value", status)) in _RECORDING_STATES

    def _on_recording_status(self, status) -> None:
        value = str(getattr(status, "value", status))
        with self._lock:
            was_recording = self._recording
            self._recording = value in _RECORDING_STATES
            if value == "arming":
                self._sessions += 1
                self._reached_recording = False
                self._stop_check_pending = False
            elif value == "recording":
                self._reached_recording = True
            if was_recording and not self._recording and self._reached_recording:
                self._stop_check_pending = True
            check_stop = self._stop_check_pending and not self._recording
            session_number = self._sessions
        if check_stop:
            reason = getattr(self._app_model, "recording_ending_reason", None)
            reason_value = str(getattr(reason, "value", reason))
            alert = alert_for_early_stop(reason_value, session_number)
            with self._lock:
                if alert is not None or value == "ready" or reason_value not in ("NA", "None"):
                    self._stop_check_pending = False
            if alert is not None:
                self.center.raise_alert(alert)
        if was_recording != self._recording:
            self._dispatch(self._reconcile_subsystems)

    def _reconcile_subsystems(self) -> None:
        try:
            statuses = self._app_model.subsystem_statuses
        except Exception:
            logger.exception("Operator alerts: subsystem statuses unavailable")
            return
        with self._lock:
            recording = self._recording
        raised = []
        for status in dict(statuses or {}).values():
            alert = alert_for_subsystem(status, recording=recording)
            if alert is None:
                continue
            raised.append(alert.key)
            self.center.raise_alert(alert)
        self.center.clear_matching("subsystem:", keep=tuple(raised))


__all__ = [
    "AlertSeverity",
    "OperatorAlert",
    "OperatorAlertBridge",
    "OperatorAlertCenter",
    "alert_for_early_stop",
    "alert_for_error",
    "alert_for_pellet_axes",
    "alert_for_pellet_estop",
    "alert_for_subsystem",
    "subsystem_display_name",
]
