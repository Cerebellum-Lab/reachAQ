import ctypes
import math
import logging
import multiprocessing
import time

from multiprocessing import RawArray, Value
from multiprocessing import synchronize
from multiprocessing.context import BaseContext
from typing import Tuple, List, Optional

import numpy

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.frame_index import FrameIndexCategory
from autotrainer.core.fixed_array_queue import BufferResult
from autotrainer.core.multiproc import get_mp_ctx

logger = get_verbose_logger(__name__)


class FixedArrayMultiQueue:
    """
    Alternative to multiprocessing.Queue.

    FixedArrayMultiQueue is an alternative to multiprocessing.Queue that
    * Explicitly queues batches of 2D data from one or more sources
    * Pre-allocates RawArray instances to hold the data for performance

    The class does not share the exact put/get_xyz interface of the standard Queue classes as it is not a drop-in
    replacement for Queue in terms of behavior.
    """
    def __init__(self,
                 depth: int,
                 cam_count: int,
                 frames_per_camera: int,
                 shape: Tuple[int, int],
                 *,
                 primary: int = 0,
                 name: str="noname",
                 mp_ctx: Optional[BaseContext] = None,
                 ):
        """
        :param depth: queue length before discarding old data if consumers can not keep up
        :param cam_count: number of sources (e.g., cameras) providing 2D data (frames)
        :param frames_per_camera: number of frames per source
        :param shape: size of each frame
        :param primary: indicates which source (index) defines when frames_per_camera is met and a buffer rotates
        """
        if mp_ctx is None:
            mp_ctx = get_mp_ctx()

        self._name = name
        self._barrier = mp_ctx.Barrier(cam_count)
        # indexing: [buffer][camera][batch_frame]
        self._buffers: List[List[List[RawArray]]] = []

        self._depth = depth
        self._cam_count = cam_count
        self._frames_per_camera = frames_per_camera
        self._primary = primary

        # writers:
        self._buffer_index = [0] * self._cam_count
        self._batch_index = [0] * self._cam_count

        # reader:
        self._read_index = 0
        self._read_sem_acquired = [False] * self._cam_count

        self._shape = shape
        self._byte_count = shape[0] * shape[1]  # 1 frame byte count, no RGB, only 8-bit gray.

        self._buff_views = []
        for idx in range(self._depth):
            buffer = []
            view = []
            for cdx in range(self._cam_count):
                cam_buffer = []
                cam_view = []
                for bdx in range(self._frames_per_camera):
                    # TODO: could use a single rawarray per camera
                    cam_buffer.append(mp_ctx.RawArray(ctypes.c_ubyte, self._byte_count))
                buffer.append(cam_buffer)
                view.append(cam_view)

            self._buffers.append(buffer)
            self._buff_views.append(view)

        # "free" buckets:
        self._sem_free = [mp_ctx.Semaphore(self._depth) for _ in range(self._cam_count)]
        # "busy" buckets:
        self._sem_busy = [mp_ctx.Semaphore(0) for _ in range(self._cam_count)]

        self._cams_tot_frames = [
            mp_ctx.Value(ctypes.c_int64)
            for _ in range(self._cam_count)
        ]

        # a single shared array for all frames indices of all cameras:
        self._frame_indices = mp_ctx.RawArray(ctypes.c_int64, self._depth * self._frames_per_camera * self._cam_count)
        # The host clock time each frame was exposed, carried beside its index.
        #
        # Without this the consumer knows WHICH frames it has but not WHEN they
        # were taken, so it can time its own work and nothing else. Sensor to
        # result - the figure a closed loop actually waits on - needs the
        # exposure time to travel with the frame. perf_counter is the same
        # monotonic clock in every process on both platforms, so the two ends
        # are directly comparable. NaN means the writer supplied nothing.
        self._frame_perf_c = mp_ctx.RawArray(
            ctypes.c_double, self._depth * self._frames_per_camera * self._cam_count)

        self._frame_indexing: List[int] = list(numpy.repeat(range(self._frames_per_camera), self._cam_count))
        self._camera_indexing: List[int] = list(numpy.tile(range(self._cam_count), self._frames_per_camera))

        self._next_counts_log_time = time.perf_counter()
        self._overflow_count = 0
        self._put_count = 0

    def __str__(self):
        return f"{self.__class__.__name__}({self._name!r})"

    @property
    def depth(self):
        return self._depth

    @property
    def camera_count(self):
        return self._cam_count

    @property
    def frames_per_camera(self):
        return self._frames_per_camera

    @property
    def batch_size(self):
        return self.camera_count * self.frames_per_camera

    @property
    def shape(self):
        return self._shape

    @property
    def buffer_index(self) -> int:
        return self._buffer_index[0]

    def set_cam_tot_frames(self, cam_idx: int, tot_frames: int):
        """Set the total nbr of frames for cam_idx for eventual sync between camera writers"""
        self._cams_tot_frames[cam_idx].value = tot_frames

    def pad_to_batch_size(self, cam_idx: int, empty_frame, cnt_net_q_put, *, timeout: float=10):
        """Pad the queue so that all the cams are on same bucket, and the start of it."""
        barrier_wait = self._barrier.wait
        #
        p_before = time.perf_counter()
        barrier_wait(timeout=timeout)
        p_now = time.perf_counter()
        timeout -= p_now - p_before
        # set the tot_frames in different sync_barrier session than the next get_cam_missing_frames
        self.set_cam_tot_frames(cam_idx, cnt_net_q_put)
        p_before = p_now
        barrier_wait(timeout=timeout)
        p_now = time.perf_counter()
        timeout -= p_now - p_before
        missing = self.get_cam_missing_frames(cam_idx)
        p_before = p_now
        barrier_wait(timeout=timeout)
        logger.verbose("padding %s frames (put=%s) to sync with others writers",
                     missing, cnt_net_q_put)
        for _ in range(missing):
            p_now = time.perf_counter()
            timeout -= p_now - p_before
            p_before = p_now
            self.put_block(empty_frame, cam_idx, FrameIndexCategory.PADDING,
                           timeout=timeout)

    def get_cam_missing_frames(self, cam_idx: int) -> int:
        """Returns the number of missing frame for camera idx,
        to fill in the batch up to the max camera writer"""
        tot_puts = [self._cams_tot_frames[cdx].value for cdx in range(self._cam_count)]
        max_tot_put = max(tot_puts)
        reminder_max = max_tot_put % self._frames_per_camera
        missing_max = (self._frames_per_camera - reminder_max) % self._frames_per_camera
        max_tot_put += missing_max
        res = max_tot_put - tot_puts[cam_idx]
        logger.debug("get_cam_missing_frames(%s): res=%s tot_puts=%s", cam_idx, res, tot_puts)
        return res

    def put_block(self, content: numpy.ndarray, camera: int, frame_idx: int, *, timeout: float=10):
        if self.put(content, camera, frame_idx, timeout=timeout) != BufferResult.Ok:
            raise RuntimeError(f"Timeout waiting space in queue for cam-{camera}")

    def put(self, content: numpy.ndarray, camera: int, frame_idx: Optional[int],
            *, block=True, timeout=0.01,
            frame_perf_c: Optional[float] = None) -> BufferResult:
        batch_index = self._batch_index[camera]  # 0 ... up to frames per camera - 1
        if batch_index == 0:
            if not self._sem_free[camera].acquire(block, timeout):
                return BufferResult.Overflow
        buffer_index = self._buffer_index[camera]  # 0 ... up to depth - 1
        cur_view = memoryview(self._buffers[buffer_index][camera][batch_index]).cast("B")
        # reshape does not copy if not necessary
        cur_view[:] = content.reshape(-1)
        if frame_idx is not None:
            b = numpy.frombuffer(
                memoryview(self._frame_indices).cast("B"), "int64", len(self._frame_indices)
            ).reshape((self._cam_count, self._depth, self._frames_per_camera))[camera]
            b[buffer_index][batch_index] = frame_idx
        stamps = numpy.frombuffer(
            memoryview(self._frame_perf_c).cast("B"), "float64",
            len(self._frame_perf_c)
        ).reshape((self._cam_count, self._depth, self._frames_per_camera))[camera]
        stamps[buffer_index][batch_index] = (
            math.nan if frame_perf_c is None else frame_perf_c)
        self._put_count += 1
        #
        batch_index = self._batch_index[camera] = (batch_index + 1) % self._frames_per_camera
        if batch_index == 0:
            self._sem_busy[camera].release()
            self._buffer_index[camera] = (1 + buffer_index) % self._depth

        return BufferResult.Ok  # if not is_overflow else BufferResult.Overflow

    def get_output(self, output: numpy.ndarray, frames_indices: Optional[numpy.ndarray] = None, *, timeout: float=0.01,
                   frames_perf_c: Optional[numpy.ndarray] = None) -> bool:
        """Get the next available "output" : i.e: 1 batch of frames_per_camera * nbr_cameras"""
        for cdx in range(self._cam_count):
            if not self._read_sem_acquired[cdx]:
                p0 = time.perf_counter()
                if not self._sem_busy[cdx].acquire(timeout=timeout):
                    return False
                p1 = time.perf_counter()
                timeout -= p1 - p0
                self._read_sem_acquired[cdx] = True
        read_idx_value = self._read_index
        buffer = self._buffers[read_idx_value]
        for idx, cdx in enumerate(self._camera_indexing):  # cdx: 0 1 0 1 0 1 0 1
            v = numpy.frombuffer(
                buffer[cdx][self._frame_indexing[idx]], "uint8", self._byte_count
            ).reshape(self.shape)
            # NB: current predict model expects an RGB frame,
            # we have so to copy 3 times the current gray image/frame into the 3 planes:
            # for fn in range(3):
            #     output[idx, :, :, fn] = v
            # we do that after releasing the sema free below
            output[idx, :, :, 0] = v

        if frames_indices is not None:
            # copy frames indices
            frames_indices[:, :] = numpy.frombuffer(
                memoryview(self._frame_indices).cast("B"), "int64", len(self._frame_indices)
            ).reshape(
                (self._cam_count, self._depth, self._frames_per_camera)
            )[:, read_idx_value, :]

        if frames_perf_c is not None:
            frames_perf_c[:, :] = numpy.frombuffer(
                memoryview(self._frame_perf_c).cast("B"), "float64",
                len(self._frame_perf_c)
            ).reshape(
                (self._cam_count, self._depth, self._frames_per_camera)
            )[:, read_idx_value, :]

        # put back the used/copied bucket as free:
        for cdx in range(self._cam_count):
            self._sem_free[cdx].release()
            self._read_sem_acquired[cdx] = False

        # make the copy for rgb after having released sema_free
        for idx, cdx in enumerate(self._camera_indexing):
            for fn in (1, 2):
                output[idx, :, :, fn] = output[idx, :, :, 0]

        # don't forget to:
        self._read_index = (read_idx_value + 1) % self._depth
        return True

    def put_frame_index_category(self, frame, frame_idx: FrameIndexCategory, *,
                                 cam_idx: Optional[int] = None, timeout: float = 10):
        logger.verbose("putting frame index category %s", frame_idx)
        for _ in range(self._frames_per_camera):
            for cdx in range(self._cam_count):
                if cam_idx is None or cam_idx == cdx:
                    t0 = time.perf_counter()
                    # put_block() will take care to raise if timeout occurs
                    self.put_block(frame, cdx, frame_idx, timeout=timeout)
                    timeout -= time.perf_counter() - t0

    def get_frame_indices(self):
        return list(self._frame_indices)
