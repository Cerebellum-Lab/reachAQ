"""Live and replayed poses follow the live queue's cameras, in the queue's order.

The live queue holds the enabled reach cameras, primary first, without the
stimulation-mode stimCam. The project's camera_names are the recording cameras, in
configuration order. When the two differ, poses were attributed to the wrong camera:
1. the primary is not the first reach camera in configuration order;
2. a queued camera does not record;
3. the stimCam records in stimulation mode but is not queued.

Each camera's frames carry their own recognisable x, so a frame that lands in another
camera's slice or file shows up as the wrong number.
"""

import copy
import ctypes
import logging
import multiprocessing
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas
import pytest

import autotrainer.core.logging
from autotrainer.core import FrameIndexCategory, ProjectInfo
from autotrainer.core.pose_elements import AllHandsParts, SceneElement
from autotrainer.inference import InferenceMode, InferenceMonitorDataMsg, pose_offline_input, pose_result_process
from autotrainer.inference.h5_tools import get_h5_frame_index, get_h5_pose_data, open_h5_file
from autotrainer.inference.pose_offline_input import OfflineInputProcess
from autotrainer.inference.pose_result_process import InferenceMonitorDataProc
from top_fixtures import collect_log_queue_to_caplog

X_OF = {"left": 11.0, "right": 22.0, "camera4": 33.0, "stimCam": 44.0}

CONDITIONS = {
    # condition: (queue cameras in queue order, project.camera_names in configuration order)
    "primary_not_first": (("left", "right"), ("right", "left")),
    "posed_camera_not_recording": (("left", "right", "camera4"), ("left", "right")),
    "recording_stim_camera": (("left", "right"), ("left", "right", "stimCam")),
}

RECORDED = 5  # frames recorded and posed live
REPLAYED = 3  # frames left to the offline replay


def _project(tmp_path, recording):
    return ProjectInfo(root=str(tmp_path), when=datetime(2026, 1, 2, 3, 4, 5), session=1,
                       ensure_exists=True, camera_names=recording)


def _frame(name, part_count):
    frame = np.zeros((part_count, 3))
    frame[:, 0] = X_OF[name]
    frame[:, 1] = X_OF[name] + 0.5
    frame[:, 2] = 0.99
    return frame


def _batch(queue_names, part_count):
    return np.stack([_frame(name, part_count) for name in queue_names])


def _indices(queue_names, recording, value, missing=FrameIndexCategory.ONLINE_NO_RECORDING):
    return np.asarray([[value if name in recording else missing] for name in queue_names])


@pytest.fixture()
def make_proc(initialized_pose_algo, capture_multiprocess_logs, monkeypatch):
    procs = []
    monkeypatch.setattr(pose_result_process, "make_log_dict_config", lambda: {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {'queue': {
            'class': 'autotrainer.core.logging.WithThreadIdQueueHandler',
            'queue': capture_multiprocess_logs,
            'level': logging.NOTSET,
        }},
        'root': {'handlers': ['queue'], 'level': logging.NOTSET},
        'loggers': copy.deepcopy(autotrainer.core.logging._limit_loggers_level),  # noqa
    })

    def make(project, queue_names):
        proc = InferenceMonitorDataProc(
            project=project,
            camera_names=queue_names,
            pose_data_queue=multiprocessing.Queue(),
            cmd_queue=multiprocessing.Queue(),
            msg_queue=multiprocessing.Queue(),
            cmd_ack_event=multiprocessing.Event(),
            frames_per_cam=1,
            watchdog_perf_c=multiprocessing.Value(ctypes.c_double),
            monitored_parts_offsets=[
                (SceneElement.Diamond, SceneElement.Triangle),
                *((SceneElement.Pellet, hand_part) for hand_part in AllHandsParts),
            ],
        )
        proc._cmd_queue.put((proc.Msg.SET_POSE_ALGO, (initialized_pose_algo,), None))
        procs.append(proc)
        proc.start()
        assert proc._cmd_ack_event.wait(5)
        proc._cmd_ack_event.clear()
        return proc

    yield make
    for proc in procs:
        proc._cmd_queue.put(None)
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(3)


def _wait_for(proc, wanted, *, timeout=20.0, feed=None):
    """Messages until one of kind `wanted`, putting `feed()` meanwhile when given."""
    seen = []
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if feed is not None:
            feed()
        try:
            msg, (args, kwargs) = proc._msg_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        seen.append((msg, args))
        if msg is wanted:
            return args, seen
    raise AssertionError(f"no {wanted} within {timeout}s; saw {[m for m, _ in seen][:20]}")


def _live_responses(proc, queue_names, recording, part_count, count=3):
    """Live pose responses for batches outside a recording."""
    responses = []

    def feed():
        proc._data_queue.put((_batch(queue_names, part_count), InferenceMode.Live,
                              _indices(queue_names, (), 0)))
    while len(responses) < count:
        args, _ = _wait_for(proc, InferenceMonitorDataMsg.POSE_RESULT_READY, feed=feed)
        responses.append(args[0])
    return responses


def _record(proc, queue_names, recording, part_count):
    for frame_idx in range(RECORDED):
        proc._data_queue.put((_batch(queue_names, part_count), InferenceMode.Live,
                              _indices(queue_names, recording, frame_idx)))
    eof = np.full((len(queue_names), 1), FrameIndexCategory.EOF_RECORDING)
    proc._data_queue.put((_batch(queue_names, part_count), InferenceMode.Live, eof))
    _wait_for(proc, InferenceMonitorDataMsg.LIVE_RECORDING_CLOSED)


def _h5_rows(path):
    fh, rows = open_h5_file(Path(path))
    try:
        return [(get_h5_frame_index(row), float(get_h5_pose_data(row)[0])) for row in rows]
    finally:
        fh.close()


def _live_path(project, name):
    return Path(project.get_intersession_pose_path(name, suffix="_live"))


def _check_live(proc, project, queue_names, recording, part_count):
    for rsp in _live_responses(proc, queue_names, recording, part_count):
        assert len(rsp.locations) == len(queue_names)
        for slot, name in enumerate(queue_names):
            assert rsp.locations[slot]["Nose"].x == X_OF[name], (slot, name)

    _record(proc, queue_names, recording, part_count)
    for name in X_OF:
        path = _live_path(project, name)
        if name in queue_names and name in recording:
            assert _h5_rows(path) == [(i, X_OF[name]) for i in range(RECORDED)], name
            idx_path = Path(project.get_video_path(name, allow_overwrite=True)[2])
            assert idx_path.read_text().split() == [str(i) for i in range(RECORDED)], name
        else:
            assert not path.exists(), f"{name} is not posed and recording, yet has {path.name}"


def test_live_pose_order_follows_the_queue_when_the_primary_is_not_first(
        make_proc, initialized_pose_algo, tmp_path):
    queue_names, recording = CONDITIONS["primary_not_first"]
    project = _project(tmp_path, recording)
    _check_live(make_proc(project, queue_names), project, queue_names, recording,
                initialized_pose_algo.part_count)


def test_a_posed_camera_that_does_not_record_is_skipped_without_shifting_the_others(
        make_proc, initialized_pose_algo, tmp_path):
    queue_names, recording = CONDITIONS["posed_camera_not_recording"]
    project = _project(tmp_path, recording)
    _check_live(make_proc(project, queue_names), project, queue_names, recording,
                initialized_pose_algo.part_count)


def test_a_recording_stim_camera_does_not_shift_the_posed_cameras(
        make_proc, initialized_pose_algo, tmp_path):
    queue_names, recording = CONDITIONS["recording_stim_camera"]
    project = _project(tmp_path, recording)
    _check_live(make_proc(project, queue_names), project, queue_names, recording,
                initialized_pose_algo.part_count)


@pytest.mark.parametrize("condition", [
    "primary_not_first", "posed_camera_not_recording", "recording_stim_camera"])
def test_offline_replay_merges_with_the_live_poses_by_queue_slot(
        condition, make_proc, initialized_pose_algo, tmp_path):
    """The replay of the frames not posed live: read back from the queued recording cameras'
    raw2D files only (a recording stimCam has none), and written per camera by name."""
    queue_names, recording = CONDITIONS[condition]
    part_count = initialized_pose_algo.part_count
    project = _project(tmp_path, recording)
    proc = make_proc(project, queue_names)
    _record(proc, queue_names, recording, part_count)

    local = project.to_local_value()
    proc._cmd_queue.put((proc.Msg.SET_FEED_INTERSESSION_RESULT, (local, None), None))
    assert proc._cmd_ack_event.wait(5)
    for frame_idx in range(RECORDED, RECORDED + REPLAYED):
        proc._data_queue.put((_batch(queue_names, part_count), InferenceMode.Offline,
                              _indices(queue_names, recording, frame_idx,
                                       missing=FrameIndexCategory.PADDING)))
    proc._data_queue.put((None, InferenceMode.Offline, None))
    args, _ = _wait_for(proc, InferenceMonitorDataMsg.INTERSESSION_SEGMENTATION_FINISHED)
    _, success = args[:2]
    assert success, args

    pair = [name for name in recording if name in queue_names][:2]
    for name in X_OF:
        path = Path(project.get_intersession_pose_path(name))
        if name in pair:
            df = pandas.read_hdf(path, "df_with_missing")
            assert len(df) == RECORDED + REPLAYED, name
            assert (df.iloc[:, 0] == X_OF[name]).all(), name
        else:
            assert not path.exists(), name


def test_a_batch_that_does_not_split_into_the_queue_cameras_is_skipped_with_one_error(
        make_proc, initialized_pose_algo, tmp_path, capture_multiprocess_logs, caplog):
    project = _project(tmp_path, ("left", "right"))
    proc = make_proc(project, ("left", "right"))
    part_count = initialized_pose_algo.part_count
    odd = _batch(("left", "right", "camera4"), part_count)
    for _ in range(3):
        proc._data_queue.put((odd, InferenceMode.Live, _indices(("left", "right"), (), 0)))
    # A good batch afterwards still goes through.
    _live_responses(proc, ("left", "right"), (), part_count, count=1)
    proc._cmd_queue.put(None)
    proc.join(5)
    collect_log_queue_to_caplog(capture_multiprocess_logs)
    errors = [r for r in caplog.records
              if r.levelno >= logging.ERROR and "queue camera" in r.getMessage()]
    assert len(errors) == 1, [r.getMessage() for r in errors]


# -- the offline feeder ----------------------------------------------------


class _Capture:
    def __init__(self, value, count):
        self._value = value
        self._left = count

    def read(self):
        if self._left == 0:
            return False, None
        self._left -= 1
        return True, np.full((8, 8), self._value, np.uint8)

    def release(self):
        pass


@pytest.mark.parametrize("condition", [
    "primary_not_first", "posed_camera_not_recording", "recording_stim_camera"])
def test_the_offline_feeder_replays_each_recording_queue_camera_into_its_own_slot(
        condition, tmp_path, monkeypatch):
    queue_names, recording = CONDITIONS[condition]
    project = _project(tmp_path, recording)
    frame_count = 4
    opened = []

    def fake_check(path):
        name = next(n for n in X_OF if f"_{n}." in Path(path).name)
        opened.append(name)
        return _Capture(X_OF[name], frame_count), frame_count

    monkeypatch.setattr(pose_offline_input, "_check_frame_count", fake_check)
    monkeypatch.setattr(pose_offline_input, "_local_do_debug", False)
    for name in recording:
        # frame 0 was posed live
        Path(project.get_video_path(name, allow_overwrite=True)[2]).write_text("0\n")

    msg_queue = queue.Queue()
    ack = threading.Event()
    offline = OfflineInputProcess(
        stop_recorded=threading.Event(), frame_shape=(8, 8), frames_per_cam=1,
        camera_names=queue_names, msg_queue=msg_queue, event_cb_ack=ack)
    offline.set_project_info(project, wait_stop_recorded=False)

    batches = []
    end = time.perf_counter() + 20
    while time.perf_counter() < end:
        try:
            kind, _ = msg_queue.get_nowait()
            if kind is InferenceMonitorDataMsg.SET_FEED_INTERSESSION_RESULT:
                ack.set()
        except queue.Empty:
            pass
        try:
            buffer, idc = offline.get_output(timeout=0.1)
        except queue.Empty:
            continue
        batches.append((buffer[:, 0, 0, 0].copy(), idc.copy()))
        offline.release_output()
        if (idc[:, -1] == FrameIndexCategory.EOF_OFFLINE_PROCESSING).all():
            break
    else:
        raise AssertionError("the feeder never sent EOF_OFFLINE_PROCESSING")

    assert sorted(opened) == sorted(n for n in queue_names if n in recording)
    replayed = {name: [] for name in queue_names}
    for values, idc in batches:
        for slot, name in enumerate(queue_names):
            frame_idx = int(idc[slot, 0])
            if frame_idx >= 0:
                assert values[slot] == X_OF[name], (slot, name)
                replayed[name].append(frame_idx)
    for name in queue_names:
        if name in recording:
            assert replayed[name] == list(range(1, frame_count)), name
        else:
            assert replayed[name] == [], name
            assert all(idc[queue_names.index(name), 0] < 0 for _, idc in batches)
