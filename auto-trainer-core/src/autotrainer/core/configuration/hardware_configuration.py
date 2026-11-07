from dataclasses import dataclass

from typing import Optional
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

    rfid_reader_enabled: bool = False
    """Enable the USB RFID reader for this rig."""

    rfid_device: str = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG01OZ68-if00-port0"
    """Stable serial-device path for the USB RFID reader."""

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
        rfid_reader_enabled: bool = False,
        rfid_device: str = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG01OZ68-if00-port0",
        min_ack_timeout: Optional[float] = None,
        board_status_timeout: Optional[float] = None,
    ):
        self.pellet_identifier = pellet_identifier
        self.can_enabled = can_enabled
        self.pellet_controller_enabled = pellet_controller_enabled
        self.nidaq_enabled = nidaq_enabled
        self.rfid_reader_enabled = rfid_reader_enabled
        self.rfid_device = str(rfid_device).strip()
        self.min_ack_timeout = min_ack_timeout
        self.board_status_timeout = board_status_timeout
