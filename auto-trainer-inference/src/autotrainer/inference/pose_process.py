import logging.config
import queue
import re
import signal
import threading
import time
from enum import IntEnum
from multiprocessing import Process, Queue, synchronize
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Semaphore as SemaphoreType
from pathlib import Path
from queue import Empty
from typing import Optional, Dict

import numpy
import numpy as np

from autotrainer.core import FixedArrayMultiQueue, PerfMonitor, get_perf_now
from autotrainer.core.logging import (
    get_verbose_logger,
    make_log_dict_config,
    setup_logging,
    install_log_exception_hook,
)
from autotrainer.core.frame_index import FrameIndexCategory
from . import DlcPoseModel, MemoryPoseModel
from .pose_model import PoseModel
from .pose_offline_input import OfflineInputProcess

logger = get_verbose_logger(__name__)

# Frames per camera the pose model graph is built for. DeepLabCut fixes its
# batch size at construction (`setup_pose_prediction` builds a placeholder with
# a literal batch dimension), so this value sizes the model and the offline
# input, while the live queue may deliver fewer real frames and be padded.
DEFAULT_MODEL_FRAMES_PER_CAMERA = 3

_local_do_debug = True


class InferenceCommandMessageKind(IntEnum):
    """Set of messages used to control inference processing"""
    Start = 0
    Terminate = 1
    ProcessLive = 2  # nb: not anymore used as command message.
    ProcessOffline = 3  # used for immediate 1 session/trial offline trigger
    SetOfflineToLive = 4  # used either after end-of-recording, or at end of analysis, to switch back to live
    SetLoggerLevel = 20


class InferenceStatusMessageKind(IntEnum):
    Created = 0
    Loading = 1
    Initialized = 2
    Running = 3
    Terminated = 4
    Performance = 5


class InferenceMode(IntEnum):
    Live = 0
    Offline = 1


class PoseProcess(Process):
    """
    Defines an independent Process for loading and processing a pose interference model.

    The primary behavior is to process frames from a "live" queue.  It is possible to toggle to between the primary and
    an optional secondary "offline" queue.  The purpose of the dual queues is to avoid contention between two different
    sources (having to turn one or both on or off from submitting to the queue, etc.).

    The PoseModel can be passed unloaded as it will be loaded in this separate Process.  Loaded values (such as part
    names) will not be available to the calling Process.  These values are provided as part of the Initialized status
    message.

    Live and Offline queues are FixedArrayMultiQueue instances that will drop batches based on the specified buffer
    depth.  The data, command, and message queues are standard multiprocessing.Queue instances that are not depth
    limited or defined with pre-allocated Array slots due to their expected small payload sizes.  Readers of the data
    and messages queues are expected to keep up with or drain the queues as needed.
    """

    def __init__(
        self,
        live_queue: FixedArrayMultiQueue,
        *,
        model_location: str,
        data_queue: Queue,
        cmd_queue: Queue,
        cmd_queue_ack: synchronize.Event,
        msg_queue: Queue,
        stop_recorded_event: synchronize.Event,
        offline_input_event_cb_ack: synchronize.Event,
        watchdog_perf_c: Synchronized,
        record_stop_sema: Optional[SemaphoreType] = None,
        model_frames_per_camera: int = DEFAULT_MODEL_FRAMES_PER_CAMERA,
    ):
        """
        :param live_queue: a FixedArrayMultiQueue as the default source of input frames
        :param model_location: The model directory to use
        :param data_queue: an output Queue for pose data passed as tuple of pose data and the mode (live or offline)
        :param cmd_queue: an input Queue for starting, terminating, and changing queues
        :param msg_queue: an output Queue for status and performance messages
        :param stop_recorded_event: DataMonitorProc stop recorded
        :param model_frames_per_camera: frames per camera the model graph is built for.
            The offline input uses the same value. The live queue may provide fewer, in
            which case the remaining rows of the predict buffer stay zero-padded.
        """
        log_dict_config = make_log_dict_config()
        super().__init__(
            name=self.__class__.__name__,
            target=self._do_run,
            kwargs=dict(log_dict_config=log_dict_config),
            daemon=True,
        )
        self._pose_model: PoseModel
        self._model_location = model_location
        self._live_input_queue = live_queue
        self._cmd_queue = cmd_queue
        self._cmd_queue_ack = cmd_queue_ack
        self._msg_queue = msg_queue
        self._data_queue = data_queue
        self._stop_recorded_event = stop_recorded_event
        self._offline_input_event_cb_ack = offline_input_event_cb_ack
        self._watchdog_perf_c = watchdog_perf_c
        self._mode = InferenceMode.Live
        self._input_queue = live_queue
        self._record_stop_sema = record_stop_sema
        self._model_frames_per_camera = int(model_frames_per_camera)
        self._process_live_when_ready = False
        self._is_running = True
        self._perf_monitor = PerfMonitor(name="<pose-predict>", units="predict calls/s", report_window=30,
                                         enable_log=False)
        #

    def _do_run(self, *, log_dict_config: Optional[Dict]):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if log_dict_config is None:
            setup_logging()
        else:
            logging.config.dictConfig(log_dict_config)
            install_log_exception_hook()

        logger.debug("started with %s", log_dict_config)
        try:
            self.__do_run()
        except BaseException as err:
            logger.exception("Fatal error: %s", err)

    def __do_run(self):
        self._send_message(InferenceStatusMessageKind.Created)

        model_path = self._model_location
        # Sized from the model frame count, not the live queue: the live queue now
        # delivers a single frame per camera and its batch is padded before predict.
        model_batch_size = (
            self._live_input_queue.camera_count * self._model_frames_per_camera
        )
        if model_path is None or len(model_path) == 0:
            logger.warning("pellet model not specified; using in-memory random data")
            model = MemoryPoseModel(model_batch_size)
        else:
            logger.notice("Loading DLC model %r", model_path)
            model = DlcPoseModel(model_path, 1, 0, model_batch_size)

        if not model.is_valid():
            self._send_message(InferenceStatusMessageKind.Terminated)
            logger.critical("pellet not started because the model does not exist or is not valid"
                           " at the specified location: %s", model_path)
            raise RuntimeError(f"Model at {model_path} not valid")

        self._pose_model = model

        self._send_message(InferenceStatusMessageKind.Loading)

        prev_lvl = logging.root.level
        logging.root.setLevel(logging.WARN)
        self._pose_model.load()
        logging.root.setLevel(prev_lvl)

        self._send_message(InferenceStatusMessageKind.Initialized, self._pose_model.body_parts)

        #
        input_q = self._live_input_queue
        offline_input = OfflineInputProcess(
            stop_recorded=self._stop_recorded_event,
            frame_shape=input_q.shape,
            frames_per_cam=self._model_frames_per_camera,
            nr_cams=input_q.camera_count,
            msg_queue=self._msg_queue,
            event_cb_ack=self._offline_input_event_cb_ack,
            record_stop_sema=self._record_stop_sema,
        )

        try:
            should_process = self._wait_for_start()
            if should_process:
                thread = threading.Thread(target=self._handle_cmd_queue, args=(offline_input,),
                                          daemon=True, name="CmdQueueHandler")
                thread.start()
                logger.info("entering pose_predict")
                self._process(offline_input)
        except Exception as err:
            logger.exception("Error during processing: %s", err)
        finally:
            offline_input.set_live(True)  # ensure it's interrupted if was running
            self._send_message(InferenceStatusMessageKind.Terminated)
        logger.notice("exiting pose_predict")

    def _send_message(self, kind: InferenceStatusMessageKind, context=None):
        self._msg_queue.put((kind, context))

    def _set_process_live(self, *, reason: str="na"):
        logger.notice("setting processing live: %s", reason)
        self._input_queue = self._live_input_queue
        self._mode = InferenceMode.Live

    def _set_process_offline(self):
        logger.debug("got processing offline")
        # self._input_queue = self._offline_input_queue
        # do not change immediately the used input queue to offline,
        # we'll wait the camera capture sends the EOF_RECORDING frame index batch,
        # so that we process entirely the live queue up to eof_recording,
        # handling all possible live recorded frames.
        self._mode = InferenceMode.Offline

    def _wait_for_start(self) -> bool:
        """
        Wait for Start or Terminate command.
        """
        while True:
            try:
                cmd, context = self._cmd_queue.get_nowait()
                self._cmd_queue_ack.set()
                if cmd == InferenceCommandMessageKind.Terminate:
                    return False
                elif cmd == InferenceCommandMessageKind.Start:
                    break
                elif cmd == InferenceCommandMessageKind.ProcessLive:
                    self._set_process_live(reason=str(cmd))
                elif cmd == InferenceCommandMessageKind.ProcessOffline:
                    logger.warning("unexpected cmd %s while wait for start", cmd)
                    # self._set_process_offline()
                else:
                    logger.warning("Unhandled command: %s", cmd)
            except Empty:
                time.sleep(0.01)

        return True

    def _handle_cmd_queue(self, offline_input: OfflineInputProcess):
        while True:
            try:
                cmd, context = self._cmd_queue.get(timeout=1)
            except Empty:
                continue
            logger.info("Handling command %s ...", cmd)
            try:
                if cmd == InferenceCommandMessageKind.Terminate:
                    self._is_running = False
                    return
                elif cmd == InferenceCommandMessageKind.ProcessLive:
                    self._set_process_live(reason=str(cmd))
                elif cmd == InferenceCommandMessageKind.SetOfflineToLive:
                    offline_input.set_live(True)
                elif cmd == InferenceCommandMessageKind.ProcessOffline:  # received from perform_segmentation
                    prj, wait_stop_recorded = context
                    offline_input.set_project_info(prj, wait_stop_recorded=wait_stop_recorded)
                else:
                    logger.warning("Unhandled command: %s", cmd)
            except Exception as err:
                logger.warning("Error processing %s: %s", cmd, err)
            finally:
                self._cmd_queue_ack.set()

    def _process(self, offline_input: OfflineInputProcess):
        # import tensorflow as tf
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        sent_live = False  # on first processed capture
        #
        input_q = self._live_input_queue
        # The predict buffer is sized for the model graph. The live queue fills only
        # its leading rows; the rest stay zero. CNN inference is per-image, so the
        # padding rows cannot influence the real ones, they only cost compute.
        model_batch_size = input_q.camera_count * self._model_frames_per_camera
        live_batch_size = input_q.batch_size
        predict_buffer = numpy.zeros(
            (model_batch_size,  # nbr cams * frames per cam the model expects
             *input_q.shape,  # W, H
             3,  # current model takes RGB
             ))
        # A view, so the queue writes straight into the padded buffer.
        frame_buffer1 = predict_buffer[:live_batch_size]
        frames_indices1 = numpy.ndarray(
            (input_q.camera_count, input_q.frames_per_camera), dtype="int64")
        #
        frame_buffer = frame_buffer1
        frames_indices = frames_indices1

        # Live and offline no longer share a frame count, so the zero result has to
        # match whichever mode produced the batch.
        _zero_pose_row = np.asarray([0] * 3 * len(self._pose_model.body_parts))
        empty_zero_pose_live = [_zero_pose_row] * frames_indices1.size
        empty_zero_pose_offline = [_zero_pose_row] * model_batch_size

        # use a pre-allocated copy for outputting the frames indices:
        prev_mode = None
        logger.info("%s: starting processing ..", self)
        d_q_put = self._data_queue.put
        predict = self._pose_model.predict
        perf_add_c = self._perf_monitor.add_cycle

        live_input = self._live_input_queue

        # always begin with live input:
        i_q: Optional[FixedArrayMultiQueue] = live_input

        def get_live_input():
            res = live_input.get_output(frame_buffer1, frames_indices1, timeout=0.1)
            return res

        def get_offline_input():
            nonlocal frame_buffer, frames_indices
            try:
                frame_buffer, frames_indices = offline_input.get_output(timeout=0.1)
            except queue.Empty:
                if offline_input.live_requested:
                    # ensure we won't immediately return on live,
                    # if we switch back to offline later with an EndOfRecording mark,
                    # while we don't have yet received the related project info in the offline side/thread:
                    offline_input.set_live(False)
                    self._set_process_live(reason="live-requested")
                return False
            if (frames_indices <= 0).any():
                logger.verbose("get_output: out indices: %s ; frames==0: %s",
                               frames_indices.tolist(),
                               [(frame_buffer[idx] == 0).all()
                                for idx in range(live_input.camera_count * live_input.frames_per_camera)])
            return True

        cur_get_output = get_live_input
        def live_release_output():
            pass
        cur_release_output = live_release_output

        def reset_locals():
            nonlocal i_q, cur_get_output, cur_release_output
            nonlocal frame_buffer, frames_indices
            prev_iq = i_q
            i_q = self._input_queue
            if prev_iq is not i_q:
                if i_q is live_input:
                    cur_get_output = get_live_input
                    cur_release_output = live_release_output
                    logger.notice("Switched to online/live queue: %s", frames_indices.tolist())
                    frame_buffer = frame_buffer1
                    frames_indices = frames_indices1
                    self._send_message(InferenceStatusMessageKind.Running, InferenceMode.Live)
                else:
                    cur_get_output = get_offline_input
                    cur_release_output = offline_input.release_output
                    logger.notice("Switched to offline queue: %s", frames_indices.tolist())
                    self._send_message(InferenceStatusMessageKind.Running, InferenceMode.Offline)

        p_last_data = get_perf_now()
        recording_in_progress = False

        while self._is_running:
            p_now = get_perf_now()

            self._watchdog_perf_c.value = time.perf_counter()  # p_now

            if prev_mode != self._mode:
                logger.verbose("Detected change of mode: %s", self._mode)
                prev_mode = self._mode

            # should be removed once more confident
            if i_q is offline_input and p_now > p_last_data + 15:
                logger.warning("timeout waiting offline data ; auto-switching to online")
                self._set_process_live(reason="timeout-offline")

            if i_q is not self._input_queue:
                reset_locals()  # always, so we get eventual change from command handler

            if not cur_get_output():
                continue
            mode_used = InferenceMode.Live if i_q is live_input else InferenceMode.Offline
            actual_release_output = cur_release_output

            p_last_data = p_now

            if (frames_indices[:, -1] < 0).any():
                # live or "signaling" (frameIndexCategory)
                if __debug__:
                    if not (frames_indices == FrameIndexCategory.ONLINE_NO_RECORDING).all():
                        logger.debug("mode=%s prev=%s indices=%s", self._mode, prev_mode, frames_indices.tolist())

            if (
                i_q is live_input
                and (frames_indices[:, -1] == FrameIndexCategory.EOF_RECORDING).any()
            ):
                logger.notice("Got EOF_RECORDING, switching immediately to offline input")
                self._set_process_offline()
                self._input_queue = offline_input
                # # always get new ref:
                # reset_locals()  # no need given done before next get_output
                recording_in_progress = False
            elif i_q is offline_input:
                recording_in_progress = False
            elif i_q is live_input and (frames_indices[:, -1] >= 0).any():
                recording_in_progress = True

            # only predict for not fully incomplete frames buffer:
            if (frames_indices >= FrameIndexCategory.ONLINE_NO_RECORDING).any():
                if i_q is live_input:
                    # Only the leading rows hold real frames; the rest is padding.
                    pose = predict(predict_buffer)[:live_batch_size]
                else:
                    pose = predict(frame_buffer)
            else:
                logger.debug("indices=%s skipped inference", frames_indices.tolist())
                # otherwise gives a full "0" result:
                pose = (
                    empty_zero_pose_live if i_q is live_input
                    else empty_zero_pose_offline
                )
                # that will anyway be skipped in the consumer when needed

            # ensure we make a copy of the frames_indices:
            frames_indices_out = frames_indices.copy()
            # getting frame indices corruption in reader side without this.
            # It could be eventually explained if the serialization
            # of the frames_indices numpy array happens after the return of the queue put()..
            # which is not totally impossible.

            # reminder: on the other side: we don't copy the pose_data output, given we assume
            # it's already a new array from the inner call to predict.

            # better after copy frames_indices, but before put to output data queue
            actual_release_output()

            # NB:
            # the data queue reader/consumer takes care of deciding what to do with the result data:
            d_q_put((pose,
                     mode_used,
                     frames_indices_out,
                     ))

            if not sent_live:
                self._send_message(InferenceStatusMessageKind.Running, InferenceMode.Live)
                sent_live = True

            if perf_add_c():
                self._send_message(InferenceStatusMessageKind.Performance, self._perf_monitor.cps)

            # could only check the frame index, given is only emitted from offline mode:
            if mode_used == InferenceMode.Offline and (
                frames_indices_out[:, -1] == FrameIndexCategory.EOF_OFFLINE_PROCESSING
            ).any():
                logger.notice("Detected end of offline queue processing: %s",
                              frames_indices_out.tolist())
                d_q_put((None, InferenceMode.Offline, None))  # tells data monitor this is EOF current offline data
                # the swap to live queue will be requested explicitly by main app,
                # there can be many/multiple offline sessions analyzed one after the other,
                # without going back to live at all in-between them.
                # Ensure we don't repeat:
                frames_indices[:] = frames_indices1[:] = FrameIndexCategory.PADDING

            if i_q is live_input and not recording_in_progress and offline_input.has_project_waiting():
                # we have to wait that there is no more recording in progress before switching to offline
                logger.notice("Switching to offline input given project waiting")
                self._set_process_offline()
                self._input_queue = offline_input

        # end while True
