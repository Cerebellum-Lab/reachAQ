import dataclasses
from typing import Any, Tuple, Dict


@dataclasses.dataclass
class SystemDataArgsKwargs:
    """A dedicated dataclass to pass data any desired args and/or kwargs as a single value for system commands"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    args: Tuple[Any] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


from .audio_spectrum_message import AudioSpectrumMessage
from .measurement_message import MeasurementMessageProtocol
from .message_handler import MessageHandler
from .motor_configuration_message import MotorConfigurations, ServoConfigMessage, \
    StepperConfigMessage, Motor
from .system_command_kind import SystemCommandKind
from .system_message_handler import SystemMessageHandler
from .system_status_message import SystemStatusMessageKind

