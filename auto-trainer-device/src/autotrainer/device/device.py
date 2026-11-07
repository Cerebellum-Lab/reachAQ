import math
from typing import Optional, Any

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core import EventManager, SystemStatusMessageKind, ApiEventKind, ObservableObject

from .device_interface import DeviceInterface
from .device_api import DeviceApi


logger = get_verbose_logger(__name__)


class Device(ObservableObject):
    """Defines the required methods to represent a device."""

    UUID_ACK_TIMEOUT_ENGAGED = "uuid_ack_timeout_engaged"
    PELLET_UUID_ACK_TIMEOUT_ENGAGED = "pellet_uuid_ack_timeout_engaged"  # actually unused
    PELLET_STATUS_TIMEOUT_ENGAGED = "pellet_status_timeout_engaged"

    def __init__(
        self,
        dev_interface: DeviceInterface = None,
        api: Optional[DeviceApi] = None,
        *,
        event_names=(),
    ):
        super().__init__(event_names=event_names)
        self._api = api
        self._interface = dev_interface
        self._pellet_status_timeout_engaged = False

    @property
    def connected(self) -> bool:
        """Say if it's connected to device"""

    @property
    def writer_watchdog_perf_c(self) -> float:
        """The writer command thread watchdog perf counter"""

    def connect(self):
        """Request connect to the device"""

    def disconnect(self):
        """Request disconnect from the device"""

    def notify_data(self, data: Any) -> None:
        """Notification for data received from the device

        :param data: one or more bytes received from the device to be handled/interpreted by this Device instance
        """

    def notify_message(self, kind: int, data: object, context: object = None) -> None:
        """Notification for messages received from the client script or application

        A set of message kind values and any expected associated data are typically defined by specific Device subclass
        implementation.  Context is an optional argument whose value is returned to the caller when the action taken by
        the message kind is complete.  For example, if the action is to send data to the physical device and received a
        response, the context will be returned as part of the DeviceApi message_callback.

        :param kind: A value whose interpretation will be Device dependent
        :param data: Any additional data required for the message beyond the kind
        :param context: A value to be returned to caller upon completion of the message
        """

    def _acknowledge_command(self, token: object, *, perf_c: float=math.nan):
        logger.verbose("sending command ack: %s perf_c=%.3f", token, perf_c)
        EventManager.default().post_event_content(
            ApiEventKind.deviceCommandAcknowledge, data=dict(context=token))
        if self._api is not None:
            self._api.send_message(SystemStatusMessageKind.ACKNOWLEDGE, (token, perf_c))

    @property
    def api(self):
        return self._api

    @api.setter
    def api(self, value: DeviceApi):
        self._api = value

    @property
    def device_interface(self) -> DeviceInterface:
        return self._interface

    @property
    def pellet_status_timeout_engaged(self) -> bool:
        return self._pellet_status_timeout_engaged

    @pellet_status_timeout_engaged.setter
    def pellet_status_timeout_engaged(self, value):
        prev, self._pellet_status_timeout_engaged = self._pellet_status_timeout_engaged, value
        if prev != value:
            logger.verbose("pellet_status_timeout=%s", value)
        self._on_property_changed(self.PELLET_STATUS_TIMEOUT_ENGAGED, value, prev)
