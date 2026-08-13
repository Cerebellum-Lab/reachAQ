from enum import IntEnum
from typing import Protocol


class SystemStatusMessageKind(IntEnum):
    """
    This class defines the messages types that are sent by the device.  If there is a request/command that generates
    a response, the command will be defined in the SystemCommandKind class.

    A message type is sent to the caller with an optional context/data value.  This value is dependent on the message
    type.

    This is implemented and represented in the send_message() method in the DeviceApi class.
    """

    ACKNOWLEDGE = 1
    """
    This message should be send when a command completes along with the context that is provided with the command.  It
    should not be sent with status messages that are generated independently by the hardware.  The is the mechanism by
    which callers determine if and when specific commands are completed.
    """

    FIRMWARE_VERSION = 101
    """Message will contain the firmware version of the device as a string."""
    # TODO: This needs to be more complex than a string now that different modules are part of the same device and
    #  connection.

    PELLET_MOTOR_X = 201
    """Message will contain the pellet X motor status as a StepperStatusMessage."""

    PELLET_MOTOR_Y = 202
    """Message will contain the pellet Y motor status as a StepperStatusMessage."""

    PELLET_MOTOR_Z = 203
    """Message will contain the pellet Z motor status as a StepperStatusMessage."""

    PELLET_LOAD = 204
    """Pellet load arm servo position"""

    PELLET_COVER = 205
    """Pellet cover arm servo position"""

    PELLET_X = 208
    """Pellet X motor position as a float."""

    PELLET_Y = 209
    """Pellet Y motor position as a float."""

    PELLET_Z = 210
    """Pellet Z motor position as a float."""

    STIMULUS_INPUTS = 305
    """Message will contain a list of 4 states as True/False"""

    TONE_STATUS = 306
    """Immediate pellet-board tone-generator state and requested duration."""

    COLOR_LED = 403
    """Message is the original ColorLed message"""

    MOTOR_CONFIGURATION = 501

    CAMERA_STATUS_CHANGE = 1001

    CAMERA_RECORDING_CLOSED_FINISHED = 1002  # current associated args is (cam_idx, tot_frames_written, project_info)
    """Sent when camera recording files are closed by record thread"""



class StepperStatusMessage(Protocol):
    @property
    def position(self) -> float:
        """Current motor position in mm."""

    @property
    def send_position(self) -> float:
        """Current motor delivery position in mm."""

    @property
    def is_at_limit(self) -> bool:
        """True if the limit switch has been hit, otherwise False."""


class ServoStatusMessage(Protocol):

    @property
    def location(self) -> float:
        """Current servo position in degrees."""

    @property
    def status(self) -> int:
        """Current servo status value."""
