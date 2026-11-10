"""The operator learns of a command the pellet board refused.

A refusal that matches the active pellet attempt finalizes it as a hardware
error. Every refusal, in a trial or not (a refused home, a manual move),
reaches on_error, which the main window shows in its footer: non-blocking, no
window. While a refusal says an axis needs a home, Record is blocked.
"""

from types import SimpleNamespace

from autotrainer.behavior.pellet_trial import HardwareErrorKind, TrialOutcome
from autotrainer.core import Motor, SystemCommandKind
from autotrainer.device import (
    CanFailure,
    CanFailureKind,
    CommandRefusal,
    RefusedCommandClass,
)
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState
from tools.acquisition.model.trial_action import PreparedState, PreparedTrialOperation
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute

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


class _StimTrialExecutor:
    """Stands in for TrialActionExecutor holding the attempt's hardware_stim3 trial.

    The real one needs a compiled protocol recipe; only what the failure path
    reads is here.
    """

    def __init__(self, send_context, route=LaserTriggerRoute.HARDWARE_STIM3):
        self._send_context = send_context
        self.failed = []
        self.operation = SimpleNamespace(
            state=PreparedState.ACTIVE,
            TERMINAL=PreparedTrialOperation.TERMINAL,
            recipe=SimpleNamespace(
                logical_trial_id=-1,
                laser_firing=SimpleNamespace(trigger_route=route),
            ),
        )

    def matches_send_context(self, context):
        return context is not None and str(context) == self._send_context

    def fail(self, error):
        self.failed.append(error)
        self.operation.state = PreparedState.FAILED

    def operation_record(self):
        return {}


def _pulse_refusal(token, error=-16):
    return _refusal(token, error, SystemCommandKind.PULSE_DIGITAL_OUTPUT,
                    RefusedCommandClass.PULSE, ())


def test_a_refused_trial_stimulus_pulse_ends_the_attempt_as_a_hardware_error(app_model):
    """The first-reach STIM pulse goes out after the pellet: it has its own token."""
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="refused-stim-test")
    app_model._on_pellet_sending(perf_c=10.0, context="send-1")
    executor = app_model._trial_action_executor = _StimTrialExecutor("send-1")
    shown = _errors(app_model)

    app_model.hardware._command_refused("pulse-1", refusal=_pulse_refusal("pulse-1"))

    attempt = app_model._trial_ledger.attempts[0]
    assert attempt.outcome is TrialOutcome.HARDWARE_ERROR
    assert app_model._trial_ledger.count() == 0
    assert app_model._trial_ledger.active_attempt is None
    assert len(executor.failed) == 1, "the laser operation is failed as well"
    assert len(shown) == 1


def test_a_refused_pulse_that_is_not_the_trials_stimulus_ends_no_attempt(app_model):
    """A software-triggered trial's laser does not use the board pulse."""
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="unrelated-pulse-test")
    app_model._on_pellet_sending(perf_c=10.0, context="send-1")
    app_model._trial_action_executor = _StimTrialExecutor(
        "send-1", route=LaserTriggerRoute.DIRECT_NI_SOFTWARE)

    app_model.hardware._command_refused("pulse-1", refusal=_pulse_refusal("pulse-1"))

    assert app_model._trial_ledger.active_attempt is not None


def test_automatic_sends_are_held_while_a_pellet_axis_needs_homing(app_model):
    pellet = app_model.behavior.system_machine.pellet
    readiness = pellet._send_readiness_callback
    assert readiness() is True

    app_model.hardware._command_refused("move-1", refusal=_refusal(
        "move-1", -1, SystemCommandKind.MOVE_X, RefusedCommandClass.STEPPER_MOVE,
        (Motor.PELLET_X_MOTOR,)))
    assert readiness() is False

    app_model.hardware._set_axes_needing_home(())  # an acknowledged home
    assert readiness() is True


def test_a_manual_send_is_not_held_by_a_needs_home_axis(app_model):
    pellet = app_model.behavior.system_machine.pellet
    app_model.hardware._set_axes_needing_home(("X", "Y", "Z"))

    assert pellet.can_send_pellet(force=True) is True
