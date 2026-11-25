import sys
from queue import Empty
from multiprocessing import Queue
from typing import Union

from reachaq.core import FixedArrayQueue
from reachaq.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)


def clear_queue(queue: Union[Queue, FixedArrayQueue],
                *, log_dumped: bool=False, name: str=""):
    """Empty a queue."""
    if queue is None or sys.platform == "darwin":
        return

    if not name:
        name = str(queue)

    flushed = 0
    task_done = getattr(queue, "task_done", lambda: None)
    try:
        while not queue.empty() or queue.qsize() > 0:
            try:
                obj = queue.get_nowait()
                flushed += 1
                task_done()
                if log_dumped:
                    logger.debug("queue %s: flushed %s: %s", name, type(obj), obj)
            except Empty:
                empty = queue.empty()
                qsize = queue.qsize()
                if not empty or qsize > 0:
                    logger.warning("queue %s: raised Empty but empty()=%s and qsize()=%s",
                                   name, empty, qsize)
                    # continue
                break
    except Exception as err:
        logger.error("Could not clear queue %s: %s", name, err)

    logger.debug("queue %s: flushed %s items", name, flushed)


def trim_queue(queue: Queue, limit: int) -> bool:
    """Trim queue length to the specified limit."""
    # unused atm
    if queue is None or sys.platform == "darwin":
        return False

    trimmed = False

    while queue.qsize() > limit:
        try:
            queue.get_nowait()
            trimmed = True
        except Empty:
            pass

    return trimmed
