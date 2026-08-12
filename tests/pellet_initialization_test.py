from unittest import mock

from autotrainer.core import SystemCommandKind
from autotrainer.device import Motor
from tools.acquisition.model.hardware_model import HardwareModel


def test_pellet_initialization_homes_and_detaches_without_release():
    send = mock.Mock()

    HardwareModel._home_and_detach_pellet(send)

    assert send.call_args_list == [
        mock.call(SystemCommandKind.SEND_HOME),
        mock.call(SystemCommandKind.SERVO_DETACH, Motor.PELLET_LOAD_SERVO),
        mock.call(SystemCommandKind.SERVO_DETACH, Motor.PELLET_COVER_SERVO),
    ]
    assert all(
        call.args[0] not in {
            SystemCommandKind.RELEASE_PELLET,
            SystemCommandKind.SERVO_ATTACH,
        }
        for call in send.call_args_list
    )
