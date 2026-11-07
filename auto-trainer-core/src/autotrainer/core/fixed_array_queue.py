import ctypes
import logging
import queue
import time
from enum import IntEnum
from multiprocessing import RawArray, Value, sharedctypes
from multiprocessing.context import BaseContext
from typing import List, Optional, Tuple

import numpy

from autotrainer.core.multiproc import get_mp_ctx

logger = logging.getLogger(__name__)


class BufferResult(IntEnum):
    Ok = 0
    Overflow = 1


class FixedArrayQueue:
    # NB:
    # this is a single producer & singler consumer implementation

    def __init__(self, depth: int, shape: Tuple[int, int], name: str="noname", *,
                 mp_ctx: Optional[BaseContext] = None,
    ):
        if mp_ctx is None:
            mp_ctx = get_mp_ctx()

        self._name = name
        # indexing: [buffer]
        self._buffers: List[sharedctypes.SynchronizedArray[ctypes.c_ubyte]] = []
        self._is_dirty: List[sharedctypes.Synchronized[bool]] = []
        # The camera frame id each buffer holds, so a consumer can say which
        # frame it is looking at. Parallel array rather than a field on the
        # payload: the payload is raw shared bytes with no room for one.
        # -1 means the producer did not say.
        self._frame_ids: List[sharedctypes.Synchronized] = []

        self._depth = depth
        self._shape = shape
        self._byte_count = shape[0] * shape[1]

        self._buffer_index = 0
        self._read_index = 0

        self._buff_views = []
        for idx in range(depth):
            self._buffers.append(mp_ctx.RawArray(ctypes.c_ubyte, self._byte_count))
            self._is_dirty.append(mp_ctx.Value(ctypes.c_bool, False))
            self._frame_ids.append(mp_ctx.Value(ctypes.c_int64, -1))

        self._next_counts_log_time = time.time()
        self._overflow_count = 0
        self._put_count = 0

    def __str__(self):
        return f"{self.__class__.__name__}({self._name!r})"

    @property
    def shape(self):
        return self._shape

    @property
    def buffer_index(self) -> int:
        return self._buffer_index

    def put(self, content: numpy.ndarray, frame_id: int = -1):
        return self._put(content, frame_id)

    def _put(self, content: numpy.ndarray, frame_id: int = -1):
        self._buff_views.clear()
        for idx in range(self._depth):
            self._buff_views.append(memoryview(self._buffers[idx]).cast("B"))
        self.put = self._put = self._put_view
        return self._put_view(content, frame_id)

    def _put_view(self, content: numpy.ndarray,
                  frame_id: int = -1) -> BufferResult:
        buffer_index = self._buffer_index
        is_overflow = self._is_dirty[buffer_index].value
        if is_overflow:
            self._overflow_count += 1
            return BufferResult.Overflow

        self._put_count += 1
        self._buff_views[buffer_index][:] = content.reshape(-1)
        self._frame_ids[buffer_index].value = int(frame_id)
        # Dirty last: it is what publishes the slot, so the id has to be
        # in place before a consumer can see the frame.
        self._is_dirty[buffer_index].value = True

        buffer_index += 1
        buffer_index %= self._depth
        self._buffer_index = buffer_index

        return BufferResult.Ok

    def get(self, block: bool = True, timeout: float = 0.01,
            *, with_frame_id: bool = False):
        """The next frame, or (frame, frame_id) when asked for the id.

        The id is opt-in so existing callers are untouched; -1 means the
        producer did not supply one.
        """
        perf_timeout = time.perf_counter() + timeout
        read_index = self._read_index
        dirty = self._is_dirty[read_index]
        while True:
            if dirty.value:
                break
            if not block or time.perf_counter() > perf_timeout:
                raise queue.Empty
            time.sleep(0.001)
        buffer = self._buffers[read_index]
        v = numpy.frombuffer(buffer, ctypes.c_uint8, self._byte_count).reshape(self.shape)  # noqa
        output = v.copy()  # numpy.frombuffer() returns a view
        frame_id = self._frame_ids[read_index].value
        dirty.value = False  # after copy of content
        self._read_index = (read_index + 1) % self._depth
        return (output, frame_id) if with_frame_id else output

    def empty(self) -> bool:
        return all(not dirty.value for dirty in self._is_dirty)

    def qsize(self) -> int:
        return sum(1 if dirty.value else 0 for dirty in self._is_dirty)
