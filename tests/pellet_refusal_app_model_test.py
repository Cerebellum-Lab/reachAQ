"""The operator learns of a command the pellet board refused.

A refusal that matches the active pellet attempt finalizes it as a hardware
error. Every refusal, in a trial or not (a refused home, a manual move),
reaches on_error, which the main window shows in its footer: non-blocking, no
window. While a refusal says an axis needs a home, Record is blocked.
"""

from autotrainer.behavior.pellet_trial import HardwareErrorKind, TrialOutcome
from autotrainer.core import Motor, SystemCommandKind
from autotrainer.device import (
    CanFailure,
    CanFailureKind,
    CommandRefusal,
    RefusedCommandClass,
)
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState

_XYZ = (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)


def _refusal(token, error, command, command_class, motors):
    return CommandRefusal(
        token=token, error=error, command=command, step="step",
        command_class=command_class, motors=motors,
    )


def _errors(app_model):
    shown = []
    app_model.on_error += lambda title, message: shown.append((title, message))
    return shown


def _needs_home_blockers(app_model):
    return [b for b in app_model.recording_blockers if "needs homing" in b]


def test_a_refused_send_pellet_finalizes_the_attempt_as_a_hardware_error(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="refused-send-test")
    app_model._on_pellet_sending(perf_c=10.0, context="send-1")
    shown = _errors(app_model)

    app_model.hardware._command_refused("send-1", refusal=_refusal(
        "send-1", -5, SystemCommandKind.SEND_PELLET, RefusedCommandClass.FIXED_XYZ, _XYZ))

    attempt = app_model._trial_ledger.attempts[0]
    assert attempt.outcome is TrialOutcome.HARDWARE_ERROR
    assert attempt.hardware_error_kind is HardwareErrorKind.MOTOR_FAILURE
    assert app_model._trial_ledger.count() == 0
    ((title, message),) = shown
    assert "refused" in title.lower()
    assert "EIO" in message


def test_a_refused_home_with_no_active_attempt_reaches_on_error(app_model):
    shown = _errors(app_model)

    app_model.hardware._command_refused("home-1", refusal=_refusal(
        "home-1", -1, SystemCommandKind.SEND_HOME, RefusedCommandClass.STEPPER_HOME,
        (Motor.PELLET_Y_MOTOR,)))

    ((title, message),) = shown
    assert "refused" in title.lower()
    assert "E-stop" in message


def test_a_failure_that_is_not_a_refusal_still_reaches_no_dialog_without_an_attempt(app_model):
    shown = _errors(app_model)

    app_model._on_hardware_command_failed(CanFailure(
        CanFailureKind.COMMAND, "command could not be queued",
        command=SystemCommandKind.SEND_HOME, context="home-1",
    ))

    assert shown == []


def test_recording_is_blocked_while_a_pellet_axis_needs_homing(app_model):
    app_model._acquisition.started = True
    app_model._set_subsystem_status(
        SubsystemId.CAN_PELLET, SubsystemState.READY, required_for_recording=True,
    )
    published = []
    app_model.property_changed += (
        lambda name, value, old:
        published.append(value) if name == app_model.Props.RECORDING_BLOCKERS else None)
    assert _needs_home_blockers(app_model) == []

    app_model.hardware._command_refused("move-1", refusal=_refusal(
        "move-1", -16, SystemCommandKind.MOVE_X, RefusedCommandClass.STEPPER_MOVE,
        (Motor.PELLET_X_MOTOR,)))

    (blocker,) = _needs_home_blockers(app_model)
    assert "X" in blocker
    assert any(any("needs homing" in b for b in blockers) for blockers in published)

    app_model.hardware._set_axes_needing_home(())
    assert _needs_home_blockers(app_model) == []


def test_on_error_shows_in_the_footer_and_opens_no_window(monkeypatch):
    """The operator asked for refusal notices that block nothing: the status bar."""
    import os
    from types import SimpleNamespace

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from tools.acquisition.view import main_window

    app = QApplication.instance() or QApplication([])
    footer = []
    window = SimpleNamespace(statusBar=lambda: SimpleNamespace(
        showMessage=lambda text, timeout: footer.append((text, timeout))))

    def no_window(*args, **kwargs):
        raise AssertionError("a refusal notice must not open a window")

    monkeypatch.setattr(main_window, "QMessageBox", no_window)

    main_window.MainWindow._show_message(
        window, "Pellet command refused",
        "Pellet motors are in E-stop. Release the E-stop, then home the pellet axes.")
    app.processEvents()

    assert footer == [(
        "Error: Pellet command refused: Pellet motors are in E-stop. Release the "
        "E-stop, then home the pellet axes.",
        15000,
    )]
