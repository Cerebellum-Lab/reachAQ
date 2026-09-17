import queue
import time
import logging
from threading import Event, Thread
from typing import Union

from autotrainer.core import FixedArrayQueue

logger = logging.getLogger(__name__)


class VideoReader(Thread):
    """
    Block on an image Queue off of the GUI thread
    """

    def __init__(self, name: str, image_queue: Union[queue.Queue, FixedArrayQueue], update_fcn, *,
                 stop_event: Event, reset_event: Event = None, decimation: int = 10):
        super().__init__(name=name, daemon=True)
        self._image_queue = image_queue
        self._update_fcn = update_fcn
        self._name = name
        self._stop_event = stop_event if stop_event is not None else Event()
        self._reset_event = reset_event if reset_event is not None else Event()
        self._decimation = decimation

    @property
    def decimation(self) -> int:
        return self._decimation

    @decimation.setter
    def decimation(self, value: int) -> None:
        self._decimation = max(1, value)
        if self._decimation != 1:
            logger.debug(f"<{self._name}> decimation set to {self._decimation}")

    def run(self):
        count = 0

        want_stop = self._stop_event.is_set
        want_reset = self._reset_event.is_set
        q_get = self._image_queue.get

        while not want_stop():

            if want_reset():
                count = 0
                self._reset_event.clear()

            try:
                data, frame_id = q_get(timeout=0.5, with_frame_id=True)
            except queue.Empty:
                continue
            except TypeError:
                # A plain queue.Queue has no with_frame_id; the display then
                # has no id to pair against and falls back to showing the
                # newest pose, which is what it did before any of this.
                data, frame_id = q_get(timeout=0.5), -1
            # if count % self._decimation == 0:
            # applied on camera capture side
            self._update_fcn(data, frame_id)
            count += 1
