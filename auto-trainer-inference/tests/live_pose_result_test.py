
import copy
import ctypes
import multiprocessing
import os
import queue
import time
import logging

import pytest

import autotrainer.core.logging

import numpy as np

from autotrainer.core import ProjectInfo
from autotrainer.core.pose_elements import AllHandsParts, SceneElement
from autotrainer.inference import (
    InferenceMode,
    InferenceMonitorDataMsg,
    PoseResponse,
    pose_result_process,
)
from autotrainer.inference.pose_result_process import InferenceMonitorDataProc
from top_fixtures import collect_log_queue_to_caplog

frames_idc_online_no_recording = np.asarray([(-1, -1, -1), (-1, -1, -1)])

zero_pose_data = np.zeros((2, 42, 3), dtype=float)


@pytest.fixture()
def inference_data_proc(pose_algo, capture_multiprocess_logs, monkeypatch, caplog) -> InferenceMonitorDataProc:  # noqa
    pairs_offset = [
        (SceneElement.Diamond, SceneElement.Triangle),
        (SceneElement.Star, SceneElement.Triangle),
        (SceneElement.Triangle, SceneElement.Pellet),
        *(
            (SceneElement.Pellet, hand_part)
            for hand_part in AllHandsParts
        ),
    ]

    # pose_result_process.make_log_dict_config
    monkeypatch.setattr(pose_result_process, "make_log_dict_config", lambda: {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'queue': {
                'class': 'autotrainer.core.logging.WithThreadIdQueueHandler',
                'queue': capture_multiprocess_logs,
                'level': logging.NOTSET,  # pass everything to the listener
            }
        },
        # root logger is here:
        'root': {
            'handlers': ['queue'],
            # with its own level here:
            'level': logging.NOTSET,  # root_log_level,
            # FORCE NOTSET to relay everything so that file handler can properly get DEBUG as well
        },
        # but eventual level of other loggers have to be defined here:
        'loggers': copy.deepcopy(autotrainer.core.logging._limit_loggers_level),  # noqa
    })

    proc = InferenceMonitorDataProc(
        project=ProjectInfo(camera_names=("left", "right")),
        pose_data_queue=multiprocessing.Queue(),
        cmd_queue=multiprocessing.Queue(),
        msg_queue=multiprocessing.Queue(),
        cmd_ack_event=multiprocessing.Event(),
        frames_per_cam=3,
        watchdog_perf_c=multiprocessing.Value(ctypes.c_double),
        monitored_parts_offsets=pairs_offset,
    )
    proc._cmd_queue.put(
        (proc.Msg.SET_POSE_ALGO, (pose_algo,), None)
    )
    yield proc  # noqa
    # Let the monitor process stop its live workers before falling back to a
    # hard termination. Terminating the parent first leaves its workers
    # orphaned and can make subsequent inference tests appear to hang.
    proc._cmd_queue.put(None)
    proc.join(3)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)


def test_live_no_recording(inference_data_proc):
    proc = inference_data_proc
    proc.start()
    for _ in range(5):
        # NB: the 3 first are discarded by the pose result process
        proc._data_queue.put((
            zero_pose_data,
            InferenceMode.Live,
            frames_idc_online_no_recording,
        ))
        time.sleep(0.1)
    msg, (args, kwargs) = proc._msg_queue.get(timeout=3)
    print(msg, args, kwargs)
    assert msg is InferenceMonitorDataMsg.POSE_RESULT_READY
    pose_rsp = args[0]
    assert isinstance(pose_rsp, PoseResponse)
    assert pose_rsp.sequence == 0
    #
    msg, (args, kwargs) = proc._msg_queue.get(timeout=3)
    pose_rsp = args[0]
    # print(msg, args, kwargs)
    assert pose_rsp.sequence == 1


def test_renew_workers(request, monkeypatch, caplog, capture_multiprocess_logs):
    # os.environ["AUTOTRAINER_LIVE_WORKERS_RENEW_TIMER_DELAY"] = "3"
    proc = request.getfixturevalue("inference_data_proc")
    proc._live_generation_renew_age = 3
    proc.start()
    # proc = inference_data_proc
    # monkeypatch.setattr(pose_result_process, "LIVE_WORKERS_RENEW_TIMER_DELAY", 3)
    for _ in range(3):
        # NB: the 3 first are discarded by the pose result process
        proc._data_queue.put(
            (
                zero_pose_data,
                InferenceMode.Live,
                frames_idc_online_no_recording,
            )
        )
    tot_put = 8
    for _ in range(tot_put):
        proc._data_queue.put(
            (
                zero_pose_data,
                InferenceMode.Live,
                frames_idc_online_no_recording,
            )
        )
        time.sleep(1)
        # increase_simulate_perf_now(1)

    collect_log_queue_to_caplog(capture_multiprocess_logs)
    assert "making new workers generation-1.." in caplog.text
    assert "making new workers generation-2.." in caplog.text

    seq_nr = None

    for _ in range(5):
        msg, (args, kwargs) = proc._msg_queue.get(timeout=3)
        pose_rsp = args[0]
        if seq_nr is None:
            seq_nr = pose_rsp.sequence
        assert pose_rsp.sequence == seq_nr
        seq_nr += 1
