"""The pellet-delivery tool clears "command pending" on a refusal, as on an ack."""

from autotrainer.core import Motor, ObservableObject, SystemCommandKind
from autotrainer.device import CommandRefusal, RefusedCommandClass
from tools.pellet_delivery.model.app_model import AppModel


def _model_waiting_on(token):
    model = object.__new__(AppModel)
    ObservableObject.__init__(model)
    model._command_pending = True
    model._last_command = token
    return model


def _refusal(token):
    return CommandRefusal(
        token=token, error=-1, command=SystemCommandKind.SEND_HOME, step="{'home': Y}",
        command_class=RefusedCommandClass.STEPPER_HOME, motors=(Motor.PELLET_Y_MOTOR,),
    )


def test_a_refusal_of_the_pending_command_clears_command_pending():
    model = _model_waiting_on("home-1")

    model.reader_command_refused("home-1", refusal=_refusal("home-1"))

    assert model._last_command is None
    assert model.command_pending is False


def test_a_refusal_of_another_command_leaves_the_pending_one():
    model = _model_waiting_on("home-1")

    model.reader_command_refused("other", refusal=_refusal("other"))

    assert model._last_command == "home-1"
    assert model.command_pending is True
