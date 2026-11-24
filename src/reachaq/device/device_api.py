import math

from queue import Queue
from typing import Callable, Optional

from reachaq.core import get_perf_now
from reachaq.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)


MessageCallbackT = Optional[Callable[[int, object], None]]


class DeviceApi:
    """
    DeviceApi is a simple interface for sending messages from the device to higher-level users of the hardware.

    DeviceApi and derivatives implement support for queues, callbacks, or other means of notifying those users,
    allowing actual hardware implementations to have a single means of sending messages ( `DeviceApi`'s
    `send_message()`).
    """

    def __init__(self, message_callback: MessageCallbackT = None, message_queue: Queue = None):
        self._message_callback = message_callback
        self._message_queue = message_queue
        self._prev_perf_c = -math.inf
        self._tot_msgs = 0

    @property
    def message_callback(self) -> MessageCallbackT:
        return self._message_callback

    @message_callback.setter
    def message_callback(self, value: MessageCallbackT):
        self._message_callback = value

    def send_message(self, kind: int, context: object):
        """Sends a message identifier and optional data to client script or application"""
        if self._message_queue is not None:
            # logger.debug("putting %s", kind)
            self._message_queue.put((kind, context))

        if self._message_callback is not None:
            self._message_callback(kind, context)

        self._tot_msgs += 1
        p_now = get_perf_now()
        if p_now > self._prev_perf_c + 5:
            logger.debug("%.1f tot msgs / seconds", self._tot_msgs / (p_now - self._prev_perf_c))
            self._prev_perf_c = p_now
            self._tot_msgs = 0
