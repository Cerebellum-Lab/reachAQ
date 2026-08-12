import collections
import logging.config
import multiprocessing.managers
import os
import queue
import statistics
import signal
import threading
import time
from itertools import chain
from multiprocessing import synchronize
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Optional, List, TextIO, Tuple, Callable, Dict

import h5py
import numpy

from autotrainer.core import ProjectInfo, get_perf_now
from autotrainer.core.frame_index import FrameIndexCategory
from autotrainer.core.multiproc import get_mp_ctx
from autotrainer.core.logging import get_verbose_logger, make_log_dict_config, setup_logging, install_log_exception_hook, \
    get_multiprocess_log_queue

from autotrainer.inference import InferenceMode, PoseAlgorithm, InferenceMonitorDataMsg
from .analysis.intersession_inference import intersession_inference
from .h5_tools import (
    get_h5_pose_data,
    get_h5_frame_index,
    open_h5_file,
    write_h5_batch,
    close_h5_fhs,
)
from .pose_result_live_process import LivePoseResultProcessWorker

logger = get_verbose_logger(__name__)


# default 1 hour renew delay:
LIVE_WORKERS_RENEW_TIMER_DELAY = float(os.getenv("AUTOTRAINER_LIVE_WORKERS_RENEW_TIMER_DELAY",
                                                 60 * 60))


def _get_project_camera_names(project: ProjectInfo) -> Tuple[str, ...]:
    return tuple(project.camera_names)


# even better is to use __debug__ and use "python -O ..."
# see https://docs.python.org/3/using/cmdline.html#cmdoption-O
_local_do_debug = False


def _shorten_text_file(lines: List[str], path: Path, limit: int):
    with path.open("w") as fh:
        fh.write("\n".join(chain(lines[:limit], [''])))


def _close_fhs(cams_frame_idx_fhs: Optional[List[Optional[TextIO]]]):
    if cams_frame_idx_fhs is None:
        return
    closed = []
    for idx, fh in enumerate(cams_frame_idx_fhs):
        if fh is not None:
            closed.append(fh.name)
            logger.debug("closing %s", fh.name)
            fh.flush()
            fh.close()
            cams_frame_idx_fhs[idx] = None
    if len(closed) > 0:
        logger.verbose("closed fhs: %s", closed)

#

class InferenceMonitorDataProc(multiprocessing.Process):

    Msg = InferenceMonitorDataMsg

    def __init__(
        self,
        *,
        project: ProjectInfo,
        pose_data_queue: multiprocessing.Queue,
        cmd_queue: multiprocessing.Queue,
        cmd_ack_event: synchronize.Event,
        msg_queue: multiprocessing.Queue,
        frames_per_cam: int,
        monitored_parts_offsets: List[Tuple[str, str]],
        watchdog_perf_c: Synchronized,
    ):
        mp_ctx = get_mp_ctx()
        log_dict_config = make_log_dict_config()
        logger.verbose("using log_config=%s", log_dict_config)
        super().__init__(
            name=self.__class__.__name__,
            target=self._do_run,
            kwargs=dict(
                project=project,
                log_dict_config=log_dict_config,
            ),
            # daemon=True,
            # cannot use anymore daemon=True given using multiprocess.Process,
            # which refuse to work with daemon=True. This should be ok though.
        )
        self._project = project
        self._data_queue = pose_data_queue
        self._cmd_queue = cmd_queue
        self._cmd_ack_event = cmd_ack_event
        self._msg_queue = msg_queue
        self._frames_per_camera = frames_per_cam
        self._camera_count = len(_get_project_camera_names(project))
        self._recording_live_batch = int(os.getenv("INFERENCE_LIVE_BATCH", 150 * 5))  # 5s at 150 FPS
        self._monitored_parts_offsets = monitored_parts_offsets
        self._parts_offsets = 0
        self._stop_recorded = mp_ctx.Event()
        self._watchdog_perf_c = watchdog_perf_c
        self._pose_algo: Optional[PoseAlgorithm] = None
        self._is_running = True
        self._feed_intersession_project: Optional[ProjectInfo] = None
        self._feed_intersession_error: Optional[str] = None
        self._tot_live_workers = 4  # nbr live workers to use
        self._live_input_q = mp_ctx.Queue(maxsize=32)  # for live 3d processing
        self._live_new_workers: List[LivePoseResultProcessWorker] = []
        self._live_pose_workers: List[LivePoseResultProcessWorker] = []
        self._live_old_workers: List[LivePoseResultProcessWorker] = []
        self._live_generation_renew_age = LIVE_WORKERS_RENEW_TIMER_DELAY

    @property
    def stop_recorded(self) -> synchronize.Event:
        return self._stop_recorded

    def _do_run(self, *, project, log_dict_config):
        self._log_dict_config = log_dict_config
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if log_dict_config is None:
            setup_logging()
        else:
            logging.config.dictConfig(log_dict_config)
            install_log_exception_hook()
        #
        cmd_thread = threading.Thread(target=self._monitor_cmd_queue, daemon=True, name="monitor_cmd_queue")
        cmd_thread.start()
        logger.info("Running monitor_data_queue")
        try:
            self._monitor_data_queue(project)
        except BaseException as err:
            logger.exception("Fatal error: %s", err)
        self._is_running = False  # before below put
        self._cmd_queue.put(None)  # ensure monitor cmd thread will exit too
        cmd_thread.join(3)
        flushed = 0
        while True:
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            flushed += 1
        self._terminate_workers()
        logger.debug("Exiting ; cmd_thread alive: %s ; cmd_queue_flushed=%s", cmd_thread.is_alive(), flushed)

    def _make_new_worker(self, pose_algo, generation):
        worker = LivePoseResultProcessWorker(
            pose_algo=pose_algo,
            monitored_parts_offsets=self._monitored_parts_offsets,
            output_q=self._msg_queue,
            input_q=self._live_input_q,
            camera_count=self._camera_count,
            generation=generation,
            log_config=self._log_dict_config,
        )
        logger.debug("starting %s", worker)
        worker.start()
        logger.verbose("started ok %s", worker)
        return worker

    def _monitor_cmd_queue(self):
        while self._is_running:
            raw = self._cmd_queue.get()
            if raw is None:
                self._is_running = False
                break
            cmd, args, kwargs = raw
            logger.debug("Processing cmd %s with %s // %s", cmd, args, kwargs)
            if cmd is self.Msg.SET_POSE_ALGO:
                pose_algo = args[0]
                self._pose_algo = pose_algo
            elif cmd is self.Msg.SET_PROJECT_INFO:
                self._project = args[0]
            elif cmd is self.Msg.SET_FEED_INTERSESSION_RESULT:
                project, error = args
                self._feed_intersession_project = project
                self._feed_intersession_error = error
            self._cmd_ack_event.set()

    def _send_msg(self, msg, *args, block_msg_queue_put: bool=True, **kwargs):
        self._msg_queue.put((msg, (args, kwargs)), block=block_msg_queue_put)

    def _intersession_offline_process(
        self,
        project_info: ProjectInfo,
        perf_c_start_offline: float,
        pose_algo: PoseAlgorithm,
        range_cams, ib_pose_data_list, ib_pose_data_dict, cams_read_h5_idx,
        cams_read_h5_dss, cams_read_h5_fhs,
    ):
        feed_prj = self._feed_intersession_project
        try:
            if feed_prj != project_info:
                raise RuntimeError(f"Projects mismatch: feed={feed_prj} pose_data_process={project_info}")
            feed_error = self._feed_intersession_error
            if feed_error is not None:
                raise RuntimeError(f"feed analysis failed with {feed_error}")
            logger.notice(
                "Processing intersession offline post-process on %s", project_info
            )
            shape = self._intersession_offline_process2(
                project_info, perf_c_start_offline, pose_algo, range_cams,
                ib_pose_data_list, ib_pose_data_dict, cams_read_h5_idx, cams_read_h5_dss
            )
        except Exception as err:
            logger.exception("Error during intersession_inference: %s", err)
            success = False
            error = str(err)
        else:
            success = True
            error = None
        close_h5_fhs(cams_read_h5_fhs)
        self._send_msg(self.Msg.INTERSESSION_SEGMENTATION_FINISHED, project_info, success, error=error)

    def _intersession_offline_process2(
        self,
        project_info: ProjectInfo,
        perf_c_start_offline: float,
        pose_algo: PoseAlgorithm,
        range_cams,
        ib_pose_data_list: List[List],
        ib_pose_data_dict,
        cams_read_h5_idx,
        cams_read_h5_dss,
    ):
        for cdx, pdl, cur_h5_idx, cur_h5_dss in zip(
            range_cams, ib_pose_data_list, cams_read_h5_idx, cams_read_h5_dss
        ):
            skipped = 0
            while cur_h5_idx < len(cur_h5_dss):
                ds_row = cur_h5_dss[cur_h5_idx]
                pose_data = get_h5_pose_data(ds_row)
                pdl.append(pose_data)
                if __debug__ and _local_do_debug:
                    pdd = ib_pose_data_dict[cdx]
                    pdd[get_h5_frame_index(ds_row)] = pose_data
                cur_h5_idx += 1
                skipped += 1
            logger.debug("cam-%s: read %s final entries from h5 live file",
                         cdx, skipped)

        # if _local_do_debug:
        #     diffs = [
        #         set(range(len(ib_pose_data_dict[cdx]))) - set(ib_pose_data_dict[cdx])
        #         for cdx in range(self._camera_count)
        #     ]
        #     if any(diffs):
        #         logger.warning("seen missing frame indices: %s", diffs)
        #     for cdx, p in enumerate(pose_paths):
        #         with open(str(p) + ".idx_monitor_data_q.txt", "w") as fh:
        #             fh.write("\n".join(chain(map(str, sorted(ib_pose_data_dict[cdx])), [''])))

        # we must ensure same number of data for each sub-list:
        min_nbr_pd = min(map(len, ib_pose_data_list))
        for idx, lst in enumerate(ib_pose_data_list):
            cut = len(lst) - min_nbr_pd
            if cut > 0:
                logger.verbose("cut ib_pose_data_list[%s] to %s", idx, cut)
                del lst[-cut:]

        final_pose_data = numpy.stack(
            (ib_pose_data_list[0], ib_pose_data_list[1]),
            axis=1,
        ).reshape(-1, pose_algo.part_count * 3)

        if False and __debug__:  # flip False to True to ensure same vs previous:
            ib_pose_data = numpy.empty(
                (0, pose_algo.part_count * 3), dtype=numpy.float32
            )
            final_pose_data_prev = numpy.vstack(
                list(chain(
                    [ib_pose_data],  # supposed the empty init array
                    (
                        pdl[ix]
                        for ix in range(min_nbr_pd)
                        for pdl in ib_pose_data_list
                    )
                ))
            )
            is_same = (final_pose_data == final_pose_data_prev)
            if isinstance(is_same, numpy.ndarray):
                is_same = is_same.all()
            logger.verbose("check same previous: %s, shape=%s prev=%s",
                           is_same, final_pose_data.shape, final_pose_data_prev.shape)

        logger.info("assembled %s pose responses, speed=%.3f/s (vstack=%s)"
                  " now calling intersession_inference() ; shape=%s",
                  min_nbr_pd,
                   2 * min_nbr_pd / (time.perf_counter() - perf_c_start_offline),
                   final_pose_data.shape[0], final_pose_data.shape)

        intersession_inference(final_pose_data, pose_algo.part_names, project_info)
        logger.success("fully processed session-%s inference with %s total pose responses",
                       project_info.session, final_pose_data.shape[0])
        return final_pose_data.shape

    def _terminate_workers(self):
        all_workers = self._live_pose_workers + self._live_new_workers + self._live_old_workers
        logger.debug("terminating live workers: %s", all_workers)
        for wrk in all_workers:
            wrk.request_stop()
        p_end = time.perf_counter() + 1
        for wrk in tuple(all_workers):
            p_now = time.perf_counter()
            wrk.join(0 if p_now > p_end else p_end - p_now)
            if not wrk.is_alive():
                all_workers.remove(wrk)
        for wrk in all_workers:
            wrk.terminate()
        p_end = time.perf_counter() + 1
        for wrk in tuple(all_workers):
            p_now = time.perf_counter()
            wrk.join(0 if p_now > p_end else p_end - p_now)
            if not wrk.is_alive():
                all_workers.remove(wrk)
        for wrk in all_workers:
            try:
                os.kill(wrk.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for lst in (self._live_pose_workers, self._live_old_workers, self._live_new_workers):
            lst.clear()

    def _monitor_data_queue(self, project: ProjectInfo):
        pose_data: Optional[List[numpy.ndarray]]
        frames_indices: Optional[numpy.ndarray]

        frames_per_batch = self._frames_per_camera
        cams = list(_get_project_camera_names(project))
        n_cams = len(cams)
        range_cams: List[int] = list(range(n_cams))
        cams_frame_idx_fhs = None
        pose_paths: List[Path] = []
        cams_read_h5_dss: List[h5py.Dataset] = []
        cams_read_h5_idx: List[int] = []
        cams_read_h5_fhs = []
        recording_in_progress = False
        next_prev_mode = None
        tot_written_to_live = None
        cnt_data_received = 0
        pose_data = []
        prev_pose_algo = None
        live_new_workers = self._live_new_workers
        live_pose_workers = self._live_pose_workers
        live_old_workers = self._live_old_workers
        live_check_worker_tasks: Dict[LivePoseResultProcessWorker, Callable] = {}
        live_workers_generation = 0
        live_pose_sequence = 0

        thread_post_process: Optional[threading.Thread] = None
        ib_pose_data_list = []
        ib_pose_data_dict = []

        writes_h5_live_durations = []
        cur_h5_live_batch = [[] for _ in range_cams]
        cur_cams_indices = [[] for _ in range_cams]
        tot_skipped = 0
        perf_c_start_offline = 0
        skip_next_pose_data = 0

        perf_c_log_counters = time.perf_counter()
        t_perf_live_check_data_queue_size = time.perf_counter() + 5

        def get_next_pose_data(timeout: Optional[float] = 0.25):
            nonlocal pose_data
            prev_pose_data = pose_data
            tot_flushed = 0
            if prev_pose_data is None:
                cur_qsize = self._data_queue.qsize()
                if prev_mode != InferenceMode.Offline or recording_in_progress:
                    logger.warning("unexpected state: prev_pose_data is None"
                                   " but prev_mode=%s or recording_in_progress=%s",
                                   prev_mode, recording_in_progress)
                # ensure we won't try flush again if the queue is actually empty on first try:
                pose_data = []
            else:
                cur_qsize = 0

            next_pose_data = next_mode = next_frames_indices = None

            while self._is_running:
                if prev_pose_data is None:
                    try:
                        next_pose_data, next_mode, next_frames_indices = self._data_queue.get_nowait()
                    except queue.Empty:
                        if tot_flushed > 0:
                            # defensive:
                            # will so return the previous next_pose_data, next_mode, next_frames_indices
                            break
                        raise  # nothing we can do, so re-raise to caller
                else:
                    next_pose_data, next_mode, next_frames_indices = self._data_queue.get(timeout=timeout)
                    break
                if (
                    next_mode != InferenceMode.Live
                    or next_frames_indices is None
                    or not (next_frames_indices == FrameIndexCategory.ONLINE_NO_RECORDING).all()
                    or tot_flushed >= cur_qsize # - 1  # so return the last one
                    # removed -1 :
                    # all the previous data in the queue might be, or is, from BEFORE the start of the intersession,
                    # otherwise we would be possibly executing some state events using too old data.
                ):
                    break
                tot_flushed += 1
            # end while True
            if prev_pose_data is None:
                logger.verbose("flushed queue after end of offline processing ; size=%s flushed=%s",
                               cur_qsize, tot_flushed)
                nonlocal skip_next_pose_data
                skip_next_pose_data = 0
            return next_pose_data, next_mode, next_frames_indices

        def live_old_worker_give_up(w: LivePoseResultProcessWorker):
            if not w.is_alive():
                w.join(0)
                logger.debug("joined old worker %s", w)
                del live_check_worker_tasks[w]
            elif w.get_stop_request_age() >= 240:
                logger.warning("giving up waiting on old worker not exited yet: %s", wrk)
                # ensure it does not stay forever in list.
                del live_check_worker_tasks[w]
                # even if totally blocked. this could happen if you SIGSTOP a process eventually
                # (although not sure that this would survive to the SIGKILL),
                # or if there is really something hang/blocked on kernel side for the process eventually.
                # but I doubt this code path will be ever taken.

        def live_old_worker_check_kill(w: LivePoseResultProcessWorker):
            if not w.is_alive():
                w.join(0)
                logger.debug("joined old worker %s", w)
                del live_check_worker_tasks[w]
            elif w.get_stop_request_age() >= 180:
                logger.warning("killing old worker not yet exited: %s", wrk)
                try:
                    os.kill(wrk.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                live_check_worker_tasks[w] = live_old_worker_give_up

        cur_local_prj = self._project
        tot_count_data_received = 0
        last_generation_renew_perf_c = get_perf_now()

        # main loop
        while self._is_running:

            perf_now = get_perf_now()
            self._watchdog_perf_c.value = perf_now

            if perf_now > perf_c_log_counters:
                perf_c_log_counters = perf_now + 15
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("qlen=%s data=%s avg_writes_h5_live=%.6f skipped_h5_live=%s",
                                self._data_queue.qsize(), cnt_data_received,
                                0 if len(writes_h5_live_durations) == 0 else statistics.mean(writes_h5_live_durations),
                                 tot_skipped)
                cnt_data_received = 0
                tot_skipped = 0
                writes_h5_live_durations.clear()

            prev_mode = next_prev_mode  # don't forget

            try:
                (pose_data, mode, frames_indices) = get_next_pose_data()
            except queue.Empty:
                continue

            tot_count_data_received += 1
            perf_now = get_perf_now()

            if prev_mode != mode:
                logger.verbose("Detected inference mode change -> %s frames=%s",
                               mode, frames_indices.tolist())
                if mode == InferenceMode.Live:
                    skip_next_pose_data = 3
                    t_perf_live_check_data_queue_size = perf_now + 0.5
                    # skip next 3 pose data to flush anything remaining
                    # NB: this looks necessary/required to ensure the inference gives back "reliable" result,
                    # with skip=2, for instance, we ~always get a first result without all visible elements detected.

            if mode == InferenceMode.Live:
                if perf_now >= t_perf_live_check_data_queue_size:
                    data_queue_size = self._data_queue.qsize()
                    live_q_size = self._live_input_q.qsize()
                    skip_update = data_queue_size > 2 or live_q_size > 16 or data_queue_size + live_q_size > 24
                    if skip_update:
                        skip_next_pose_data = 1 + data_queue_size // 3
                        logger.warning("data queue size=%s live_q_size=%s skip_next=%s",
                                       data_queue_size, live_q_size, skip_next_pose_data)
                    else:
                        skip_next_pose_data = 0
                    if live_q_size >= 24:
                        pass
                        # keep current t_perf_live_check_data_queue_size
                        # so that next turn will also get skip_update=True,
                        # if still too high number of output async tasks in progress.
                    else:
                        t_perf_live_check_data_queue_size = perf_now + (0.25 if skip_update else 1.5)
                else:
                    skip_update = False
            else:
                skip_update = False
                skip_next_pose_data = 0

            next_prev_mode = mode

            if __debug__ and frames_indices is not None:
                if (not (frames_indices >= 0).all()
                    and not (frames_indices == FrameIndexCategory.ONLINE_NO_RECORDING).all()
                ):
                    logger.debug("mode=%s frames_indices=%s", mode, frames_indices.tolist())

            pose_algo = self._pose_algo
            if pose_algo is None:
                continue

            # check for new worker(s) ready:
            if len(live_new_workers) > 0:
                for wrk in tuple(live_new_workers):
                    if wrk.is_ready():
                        logger.verbose("using new ready worker %s", wrk)
                        live_new_workers.remove(wrk)
                        live_pose_workers.append(wrk)

            has_current_gen = any(
                wrk.generation == live_workers_generation and wrk.is_alive()
                for wrk in live_pose_workers
            )
            if has_current_gen:
                # consider exit/terminate previous workers, but only when there is at least one new ready.
                # this is only when pose_algo renewed or new worker replacing older
                for wrk in tuple(live_pose_workers):
                    if pose_algo is not wrk.pose_algo or wrk.generation != live_workers_generation:
                        wrk.request_stop()
                        live_pose_workers.remove(wrk)
                        live_old_workers.append(wrk)

            to_remove = set()
            for wrk in live_pose_workers:
                if not wrk.is_alive():
                    wrk.join(0)
                    logger.warning("detected unexpected dead live worker: %s", wrk)
                    to_remove.add(wrk)
                    live_new_workers.append(self._make_new_worker(pose_algo, live_workers_generation))
            for wrk in to_remove:
                live_pose_workers.remove(wrk)
            # renew workers every X delay:
            if (perf_now - last_generation_renew_perf_c >= self._live_generation_renew_age
                or pose_algo is not prev_pose_algo
            ):
                # renew every that nbr of pose_data, which at current rate of ~15-20 / second,
                # that makes about ~4-5h of runtime for the workers.
                live_workers_generation += 1
                last_generation_renew_perf_c = perf_now
                logger.notice("making new workers generation-%s..", live_workers_generation)
                for _ in range(self._tot_live_workers):
                    live_new_workers.append(self._make_new_worker(pose_algo, live_workers_generation))
            # check for finished old workers:
            for wrk in tuple(live_old_workers):
                if wrk.is_alive():
                    age = wrk.get_stop_request_age()
                    if age >= 150:
                        logger.warning("terminating old worker not yet exited: %s", wrk)
                        wrk.terminate()
                        live_old_workers.remove(wrk)
                        live_check_worker_tasks[wrk] = live_old_worker_check_kill
                else:
                    wrk.join(0)
                    logger.debug("joined old worker %s", wrk)
                    live_old_workers.remove(wrk)

            # eventual checks for old workers exiting very slowly:
            for wrk, tsk in tuple(live_check_worker_tasks.items()):
                tsk(wrk)

            prev_pose_algo = pose_algo  # NB: after previous blocks

            if (
                not recording_in_progress
                and pose_data is not None
                and mode == InferenceMode.Live
                and frames_indices is not None
                and (frames_indices[:, 0] >= 0).any()
            ):
                tot_written_to_live = 0
                recording_in_progress = True
                cur_local_prj = self._project.to_local_value()
                logger.notice("Detected new record in progress ; session=%s ; mode=%s frames indices: %s",
                              cur_local_prj.session, mode, frames_indices.tolist())
                self._stop_recorded.clear()
                logger.debug("cleared stop_recorded")
                cams_frame_idx_fhs = []
                pose_paths = []
                cur_h5_live_batch = [[] for _ in range_cams]  # safer
                cur_cams_indices = [[] for _ in range_cams]  # safer
                ib_pose_data_list = [[] for _ in range_cams]
                ib_pose_data_dict = []
                if __debug__:
                    ib_pose_data_dict = [{} for _ in range_cams]
                for cam in cams:
                    _, _, p_indices = cur_local_prj.get_video_path(cam, allow_overwrite=True)
                    cams_frame_idx_fhs.append(Path(p_indices).open("w"))
                    pose_path = Path(
                        cur_local_prj.get_intersession_pose_path(cam, suffix="_live"))
                    # ensure live data files are not reused from eventual previous trial,
                    # although that would be an issue of project/session reuse then.
                    write_h5_batch(pose_path, [], [],
                                   columns=pose_algo.pose_result_columns, mode="w")
                    pose_paths.append(pose_path)

            elif recording_in_progress and frames_indices is not None:
                # thx to camera capture which send a full EOF_RECORDING batch frames indices,
                # this condition allows to know when to close/stopping writing to live files,
                # and reopen for offline mode
                if (frames_indices[:, 0] == FrameIndexCategory.EOF_RECORDING).any():
                    recording_in_progress = False
                    logger.notice("Detected stop of recording in progress ; mode=%s prev=%s frames_indices=%s",
                                  mode, prev_mode, frames_indices.tolist())
                    _close_fhs(cams_frame_idx_fhs)
                    cams_frame_idx_fhs = None
                    for cam_pose_path, cam_indices, cam_h5_live in zip(pose_paths, cur_cams_indices, cur_h5_live_batch):
                        if len(cam_h5_live) == 0:
                            continue
                        write_duration = write_h5_batch(cam_pose_path, cam_h5_live, cam_indices,
                                                        columns=pose_algo.pose_result_columns)
                        writes_h5_live_durations.append(write_duration)
                    #
                    logger.debug("setting stop_recorded event")
                    self._stop_recorded.set()  # this is for the feeder thread to know when it can open the data files
                    self._send_msg(
                        InferenceMonitorDataMsg.LIVE_RECORDING_CLOSED,
                        cur_local_prj,
                    )

            cnt_data_received += 1

            try:

                if mode == InferenceMode.Live:

                    if recording_in_progress:
                        tot_written_to_live += 1
                        for fh, cam_fr_indices in zip(cams_frame_idx_fhs, frames_indices):
                            cam_fr_indices = list(filter(lambda i: i >= 0, cam_fr_indices))
                            if fh is not None and len(cam_fr_indices) > 0:
                                fh.write("\n".join(map(str, chain(cam_fr_indices, [""]))))
                                fh.flush()

                        for cdx, cam_fr_indices, cam_pose_path, cam_h5_live, cam_indices in zip(
                            range_cams, frames_indices, pose_paths, cur_h5_live_batch, cur_cams_indices
                        ):
                            # reminder: pose_data has 1 frame cam1, 1 frame cam2, 1 frame cam1, etc..
                            cur = pose_data[cdx::n_cams]
                            cur = {
                                fx: f.flatten()
                                for fx, f in zip(cam_fr_indices, cur)
                                if fx >= 0
                            }
                            cur = [cur[ix] for ix in sorted(cur)]  # should not be needed
                            if len(cur) == 0:
                                continue
                            cam_h5_live.append(cur)
                            cam_indices.extend(filter(lambda ix: ix >= 0, cam_fr_indices))
                            if len(cam_h5_live) * frames_per_batch >= self._recording_live_batch:
                                write_duration = write_h5_batch(cam_pose_path, cam_h5_live, cam_indices,
                                                                columns=pose_algo.pose_result_columns)
                                writes_h5_live_durations.append(write_duration)

                    if skip_update:
                        continue

                    if skip_next_pose_data > 0 and cnt_data_received % 2 == 0:
                        skip_next_pose_data -= 1
                        continue

                    if (frames_indices < FrameIndexCategory.ONLINE_NO_RECORDING).any():
                        # this is normal for stop recording
                        if not (frames_indices < FrameIndexCategory.ONLINE_NO_RECORDING).all():
                            logger.verbose("Skipping incomplete frame index pose_data: %s",frames_indices.tolist())
                        continue

                    if len(live_pose_workers) > 0:
                        try:
                            source_frame_ids = tuple(
                                tuple(int(frame_id) for frame_id in cam_frame_ids if frame_id >= 0)
                                for cam_frame_ids in frames_indices
                            )
                            self._live_input_q.put(
                                (pose_data, live_pose_sequence, source_frame_ids),
                                block=False,
                            )
                        except queue.Full:
                            pass
                        live_pose_sequence += 1
                        # always increase, even on queue full, so that consumers can infer there was drop, in case of.

                elif mode == InferenceMode.Offline:

                    if (
                        pose_data is not None
                        and len(cams_read_h5_dss) == 0
                        and frames_indices is not None and (frames_indices >= 0).any()
                        # with random cam there might be no frame to replay, so we get immediately all < 0
                    ):
                        _close_fhs(cams_frame_idx_fhs)  # just to be sure
                        # logger.debug("setting stop recorded")
                        # self._stop_recorded.set()  # this is for the feeder thread to know when it can open the data files
                        cams_frame_idx_fhs = None
                        perf_c_start_offline = time.perf_counter()
                        logger.notice("Opening live files for offline processing ; prev_mode=%s frames=%s",
                                      prev_mode, frames_indices.tolist())
                        cur_local_prj = self._project.to_local_value()
                        # re-obtain the paths, project info might be from a batch session
                        pose_paths = [
                            Path(cur_local_prj.get_intersession_pose_path(cam, suffix="_live"))
                            for cam in cams
                        ]
                        cams_read_h5_dss = []
                        cams_read_h5_fhs = []
                        for cam_pose_path in pose_paths:
                            f5fh, f5dss = open_h5_file(cam_pose_path)
                            cams_read_h5_fhs.append(f5fh)
                            cams_read_h5_dss.append(f5dss)
                        cams_read_h5_idx = [0] * n_cams
                        ib_pose_data_list = [[] for _ in range_cams]
                        ib_pose_data_dict = []
                        if __debug__:
                            ib_pose_data_dict = [{} for _ in range_cams]
                        tot_skipped = 0

                    if pose_data is None:
                        # end of intersession/offline replay
                        # cur_local_prj = self._project.to_local_value()
                        logger.info("detected end of inference offline processing ; project=%s",
                                       cur_local_prj)
                        # we can reset the offline queue here, it's safe :
                        # the pose process has switched to its online queue at this point
                        # self._stop_recorded.clear()
                        # logger.debug("cleared stop_recorded")
                        _close_fhs(cams_frame_idx_fhs)  # defensive as supposed to be close already
                        cams_frame_idx_fhs = None
                        if thread_post_process is not None:
                            logger.debug("joining previous thread_post_process")
                            thread_post_process.join(1)
                            if thread_post_process.is_alive():
                                # should not happen
                                logger.error("previous post_process thread still alive: %s",
                                             thread_post_process)
                            thread_post_process = None
                        thread_post_process = threading.Thread(
                            name="OfflineProcess",
                            target=self._intersession_offline_process,
                            args=(
                                cur_local_prj,
                                perf_c_start_offline,
                                pose_algo,
                                range_cams,
                                ib_pose_data_list,
                                ib_pose_data_dict,
                                cams_read_h5_idx,
                                cams_read_h5_dss,
                                cams_read_h5_fhs,
                            ),
                            daemon=True,
                        )
                        thread_post_process.start()
                        cams_read_h5_dss = []
                        cams_read_h5_fhs = []
                        ib_pose_data_list = [[] for _ in range_cams]
                        ib_pose_data_dict = []
                    else:
                        if (frames_indices < FrameIndexCategory.ONLINE_NO_RECORDING).all():
                            # happens for EOF_OFFLINE_PROCESSING
                            continue
                        # we can now append the received/processed frame data:
                        skipped = 0
                        # append any of the live processed frame data that are before current
                        # received/processed frames indices:
                        for cdx, pdl, cur_h5_dss, cam_fr_indices in zip(
                            range_cams, ib_pose_data_list, cams_read_h5_dss, frames_indices
                        ):
                            cur_h5_ix = cams_read_h5_idx[cdx]
                            for fx, frame in enumerate(pose_data[cdx::len(cams)]):
                                frame_idx = cam_fr_indices[fx]
                                if frame_idx < 0:  # == FrameIndexCategory.PADDING:
                                    __debug__ and \
                                    logger.spam("cam-%s : fx=%s got negative frame idx: %s",
                                                 cdx, fx, cam_fr_indices)
                                    continue
                                while cur_h5_ix < len(cur_h5_dss):
                                    h5row = cur_h5_dss[cur_h5_ix]
                                    ix = get_h5_frame_index(h5row)
                                    if frame_idx <= ix:
                                        break
                                    f = get_h5_pose_data(h5row)
                                    if __debug__ and _local_do_debug:
                                        pdd = ib_pose_data_dict[cdx]
                                        if ix != len(pdl) or ix in pdd:
                                            logger.warning(
                                                "cam-%s: detected invalid live frame ix: %s vs %s - double=%s",
                                                cdx, ix, len(pdl), ix in pdd)
                                        pdd[ix] = f
                                    pdl.append(f)
                                    cur_h5_ix += 1
                                    skipped += 1
                                cams_read_h5_idx[cdx] = cur_h5_ix
                                f = frame.flatten()
                                if __debug__ and _local_do_debug:
                                    pdd = ib_pose_data_dict[cdx]
                                    if (frame_idx != len(pdl) and frame_idx < len(cur_h5_dss)) or frame_idx in pdd:
                                        logger.warning("cam-%s: detected invalid frame idx: %s vs %s - double=%s",
                                                       cdx, frame_idx, len(pdl), frame_idx in pdd)
                                    pdd[frame_idx] = f
                                pdl.append(f)

                        tot_skipped += skipped

            except Exception as err:
                logger.exception("_monitor_data_queue: loop error processing mode=%s %s: %s",
                                 mode, type(pose_data), err)

        # end while self._is_running
