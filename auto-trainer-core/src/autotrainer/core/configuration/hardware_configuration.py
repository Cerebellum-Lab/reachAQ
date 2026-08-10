from dataclasses import dataclass

from typing import Optional
from typing_extensions import Self


from autotrainer.core import make_camelize_representer, make_decamelize_constructor


@dataclass(init=False)
class HardwareConfiguration:
    pellet_identifier: str = ""
    can_enabled: bool = True
    """Enable CAN bus hardware connection attempts."""

    pellet_controller_enabled: bool = True
    """Enable the physical pellet delivery controller on the CAN bus."""

    nidaq_enabled: bool = False
    """Enable NI-DAQ hardware for this rig."""

    min_ack_timeout: Optional[float] = None  # min device-ack-timeout
    """CAN uuid ACK timeout, if not set here then default code value of 3s is used."""

    board_status_timeout: Optional[float] = None
    """If any of the boards sub-device misses its status message for more than this delay -> system fault.
    If not set here then default code value of 15s is used.
    """

    def __init__(
        self,
        pellet_identifier: str = "",
        can_enabled: bool = True,
        pellet_controller_enabled: bool = True,
        nidaq_enabled: bool = False,
        min_ack_timeout: Optional[float] = None,
        board_status_timeout: Optional[float] = None,
        *,
        tunnel_identifier=None,
        tunnel_headfix_enabled=None,
    ):
        # Old rig files may still contain tunnel keys. They are accepted once,
        # discarded, and omitted from all newly written configurations.
        self.pellet_identifier = pellet_identifier
        self.can_enabled = can_enabled
        self.pellet_controller_enabled = pellet_controller_enabled
        self.nidaq_enabled = nidaq_enabled
        self.min_ack_timeout = min_ack_timeout
        self.board_status_timeout = board_status_timeout

    @classmethod
    def from_version_zero(cls, content: dict) -> Self:
        configuration = cls()
        if "pellet_delivery" in content:
            configuration.pellet_identifier = content["pellet_delivery"].get("port", "")
            configuration.pellet_controller_enabled = bool(configuration.pellet_identifier)
        configuration.can_enabled = bool(configuration.pellet_identifier)

        return configuration
