# process pool usage for live inference 3d-triangulation
import dataclasses
import sys
import functools
import math
import multiprocessing.managers
import queue
import signal
import logging.config
import time
from typing import Optional, Union, Dict, List, Tuple

from autotrainer.core import get_perf_now
from autotrainer.core.logging import (
    get_verbose_logger,
    setup_logging,
    install_log_exception_hook,
    make_log_dict_config,
)
from autotrainer.core.multiproc import pool_init, get_mp_ctx
from autotrainer.inference import PoseAlgorithm, InferenceMonitorDataMsg


logger = get_verbose_logger(__name__)


class LivePoseResultProcessWorker(multiprocessing.Process):
    """Dedicated process worker, with "is_ready" event handling,
    This is necessary since the start of a new worker takes up to ~3-4-5 seconds sometimes, depending on system load.
    And during that start/import time you don't want to use the new worker yet.
    Or else it would delay the live data stream of that amount of time.
    """

    def __init__(
        self,
        *,
        pose_algo: PoseAlgorithm,
        monitored_parts_offsets: List[Tuple[str, str]],
        input_q: multiprocessing.Queue,
        output_q: multiprocessing.Queue,
        camera_count: int = 2,
        generation: int,
        log_config: Optional[Dict] = None,
    ):
        super().__init__(name="LivePoseProcess", daemon=True)
        self._log_dict_config = make_log_dict_config() if log_config is None else log_config
        mp_ctx = get_mp_ctx()
        self._is_ready_event = mp_ctx.Event()
        self._stop_requested = mp_ctx.Event()
        self._stop_request_perf_c = math.nan
        self._input_q = input_q
        self._pose_algo = pose_algo
        self._monitored_parts_offsets = monitored_parts_offsets
        self._output_q = output_q
        self._camera_count = camera_count
        self._generation = generation

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def pose_algo(self) -> PoseAlgorithm:
        return self._pose_algo

    def is_ready(self):
        return self._is_ready_event.is_set()

    def request_stop(self):
        if self.is_alive():
            logger.debug("requesting stop to %s", self)
            self._stop_requested.set()
            self._stop_request_perf_c = get_perf_now()
        else:
            self.join(0)  # just to ensure collect the exit code
            logger.verbose("worker already stopped: %s", self)

    def get_stop_request_age(self) -> float:
        return get_perf_now() - self._stop_request_perf_c

    def _interrupted(self, sig, frame):
        self._stop_requested.set()

    def run(self):
        signal.signal(signal.SIGINT, self._interrupted)
        log_dict_config = self._log_dict_config
        if log_dict_config is None:
            setup_logging(logger_level=logging.DEBUG)
        else:
            logging.config.dictConfig(log_dict_config)
            install_log_exception_hook()
        # logging.getLogger("autotrainer").setLevel(logging.DEBUG)
        warn_full = False
        count_processed = 0
        count_out_full = 0
        log_every_delay = 60
        p_next_log_info = time.perf_counter() + log_every_delay
        logger.verbose("setting ready to work")
        self._is_ready_event.set()
        while True:
            if self._stop_requested.is_set():
                break
            p_now = time.perf_counter()
            if p_now > p_next_log_info:
                logger.debug("%.2f / sec tot=%s out_full=%s",
                             count_processed / log_every_delay, count_processed, count_out_full)
                count_processed = count_out_full = 0
                p_next_log_info = p_now + log_every_delay
            try:
                payload = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if len(payload) == 2:
                # Compatibility with callers/tests that predate source-frame
                # propagation. New monitor processes always send three items.
                pose_data, seq_nr = payload
                source_frame_ids = ()
            else:
                pose_data, seq_nr, source_frame_ids = payload
            count_processed += 1
            rsp = self._pose_algo.process(
                pose_data,
                pairs_3d_offsets=self._monitored_parts_offsets,
                sequence=seq_nr,
                camera_count=self._camera_count,
            )
            rsp = dataclasses.replace(
                rsp,
                source_frame_ids=tuple(
                    tuple(int(frame_id) for frame_id in camera_ids)
                    for camera_ids in source_frame_ids
                ),
            )
            data = (
                InferenceMonitorDataMsg.POSE_RESULT_READY,  # cmd
                ((rsp,), None)  # args, kwargs
            )
            try:
                self._output_q.put(data, block=False)
            except queue.Full:
                count_out_full += 1
                if not warn_full:
                    warn_full = True
                    logger.warning("output queue full")
            else:
                if warn_full:
                    logger.info("recovered from output queue full. count=%s", count_out_full)
                    count_out_full = 0
                    warn_full = False
        logger.debug("exiting")
