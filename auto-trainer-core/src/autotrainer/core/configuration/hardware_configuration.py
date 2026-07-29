from dataclasses import dataclass

from typing import Optional
from typing_extensions import Self


from autotrainer.core import make_camelize_representer, make_decamelize_constructor


@dataclass
class HardwareConfiguration:
    tunnel_identifier: str = ""
    pellet_identifier: str = ""
    can_enabled: bool = True
    """Enable CAN bus hardware connection attempts."""

    pellet_controller_enabled: bool = True
    """Enable the physical pellet delivery controller on the CAN bus."""

    nidaq_enabled: bool = False
    """Enable NI-DAQ hardware for this rig."""

    tunnel_headfix_enabled: bool = False
    """Enable tunnel gate, head magnet, and tunnel fan commands."""

    min_ack_timeout: Optional[float] = None  # min device-ack-timeout
    """CAN uuid ACK timeout, if not set here then default code value of 3s is used."""

    board_status_timeout: Optional[float] = None
    """If any of the boards sub-device misses its status message for more than this delay -> system fault.
    If not set here then default code value of 15s is used.
    """

    @classmethod
    def from_version_zero(cls, content: dict) -> Self:
        configuration = cls()

        if "head_fix" in content:
            configuration.tunnel_identifier = content["head_fix"].get("port", "")
            configuration.tunnel_headfix_enabled = bool(configuration.tunnel_identifier)
        if "pellet_delivery" in content:
            configuration.pellet_identifier = content["pellet_delivery"].get("port", "")
            configuration.pellet_controller_enabled = bool(configuration.pellet_identifier)
        configuration.can_enabled = bool(configuration.tunnel_identifier or configuration.pellet_identifier)

        return configuration
