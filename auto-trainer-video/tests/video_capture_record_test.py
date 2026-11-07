import queue
import re
import time

import pytest

from autotrainer.core import SystemStatusMessageKind
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.video import video_capture
from top_fixtures import collect_log_queue_to_caplog


@pytest.mark.parametrize("fps", (30, 100))
@pytest.mark.parametrize("prebuffer_duration", (0.5, 2.5))
@pytest.mark.parametrize("record_duration", (0.5, 2.5))
def test_with_prebuffer(
    fps,
    prebuffer_duration,
    record_duration,
    video_capture_model,
    capture_multiprocess_logs,
    make_log_dict_multiproc,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(video_capture, "make_log_dict_config", make_log_dict_multiproc)
    conf = video_capture_model.save_configuration()
    conf.params["primary"] = "true"
    conf.params["fps"] = fps
    conf.record_prebuffer_duration = prebuffer_duration
    video_capture_model.load_configuration(conf)
    assert video_capture_model.on_prepare_capture() is True
    video_capture_model.on_capture_start()
    assert video_capture_model.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=5) is True
    assert video_capture_model.wait_for_first_frame(timeout=5) is True
    assert video_capture_model.last_captured_frame_index >= 0
    time.sleep(prebuffer_duration + 0.5)  # ensure prebuffer is filled enough
    video_capture_model.on_trigger_recording(True)
    time.sleep(record_duration)
    assert video_capture_model.wait_for_capture_status(CaptureProcessStatus.RECORDING, timeout=5) is True
    video_capture_model.on_trigger_recording(False)
    assert video_capture_model.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=5) is True
    # ensure _record_stop_sema has been released()/increased:
    assert video_capture_model._record_stop_sema.acquire(timeout=3) is True
    collect_log_queue_to_caplog(capture_multiprocess_logs)
    # print(caplog.text)
    m = re.search(r"Closed video file: tot frames written: (\d+)", caplog.text)
    assert m is not None
    captured = int(m.group(1))
    tot_duration = prebuffer_duration + record_duration
    assert fps * (tot_duration - 0.75) <= captured < fps * (tot_duration + 0.75), \
        "Assuming no overloaded or too slow test executor, this should be respected."
    # using 0.75s extra on left+right to be sure. test runner might be sometimes slow


def get_all_msgs_into(q, msgs):
    while True:
        try:
            msgs.append(q.get(block=False))
        except queue.Empty:
            break


def check_cam_status_change(received_msgs, cam_id, status):
    for m in received_msgs:
        if m[0] == SystemStatusMessageKind.CAMERA_STATUS_CHANGE and m[1][:2] == (
            cam_id,
            status,
        ):  #  ( cam-id, status ))
            return m
    return None


def test_with_primary_secondary(
    video_capture_model,
    video_capture_model2,
    project_info,
    capture_multiprocess_logs,
    make_log_dict_multiproc,
    monkeypatch,
    caplog,
):
    msg_q = video_capture_model._msg_queue  # noqa
    monkeypatch.setattr(video_capture, "make_log_dict_config", make_log_dict_multiproc)
    #
    conf = video_capture_model.save_configuration()
    conf.params["primary"] = "true"
    conf.record_prebuffer_duration = 1  # ensure not 0
    video_capture_model.load_configuration(conf)
    #
    conf2 = video_capture_model2.save_configuration()
    conf2.params["primary"] = "false"  # just to double ensure it
    conf2.record_prebuffer_duration = conf.record_prebuffer_duration  # ensure same than cam1
    video_capture_model2.load_configuration(conf2)
    #
    assert video_capture_model.on_prepare_capture() is True
    assert video_capture_model2.on_prepare_capture() is True
    video_capture_model.on_capture_start()
    video_capture_model2.on_capture_start()
    assert video_capture_model.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=5) is True
    assert video_capture_model2.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=5) is True
    assert video_capture_model.wait_for_first_frame(timeout=5) is True
    assert video_capture_model2.wait_for_first_frame(timeout=5) is True
    time.sleep(conf.record_prebuffer_duration + 0.5)  # ensure prebuffer is filled enough
    received_msgs = []
    get_all_msgs_into(msg_q, received_msgs)
    assert check_cam_status_change(received_msgs, 0, CaptureProcessStatus.RUNNING) is not None
    assert check_cam_status_change(received_msgs, 1, CaptureProcessStatus.RUNNING) is not None
    received_msgs.clear()
    #
    for mod in (video_capture_model, video_capture_model2):
        mod.on_trigger_recording(True)
    for mod in (video_capture_model, video_capture_model2):
        assert mod.wait_for_capture_status(CaptureProcessStatus.RECORDING, timeout=5) is True
    #
    time.sleep(1.5)
    get_all_msgs_into(msg_q, received_msgs)
    assert check_cam_status_change(received_msgs, 0, CaptureProcessStatus.RECORDING) is not None
    assert check_cam_status_change(received_msgs, 1, CaptureProcessStatus.RECORDING) is not None
    received_msgs.clear()
    #
    for mod in (video_capture_model, video_capture_model2):
        mod.on_trigger_recording(False)
    for mod in (video_capture_model, video_capture_model2):
        assert mod.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=5) is True
    time.sleep(0.5)  # give little extra time
    collect_log_queue_to_caplog(capture_multiprocess_logs)
    # print(caplog.text)
    #
    m = re.findall(r"Starting record with frame_idx=(\d+)", caplog.text)
    assert m is not None
    assert len(m) == 2, "both cams should have started recording"
    assert m[0] == m[1], "and both cams should have same start frame-id"
    #
    m = re.findall(r"Closed video file: tot frames written: (\d+)", caplog.text)
    assert m is not None
    assert len(m) == 2, "both cams should have closed their video file"
    assert m[0] == m[1], "assuming not overloaded test executor, both cams should have recorded same nbr of frames"
    get_all_msgs_into(msg_q, received_msgs)
    assert check_cam_status_change(received_msgs, 0, CaptureProcessStatus.RUNNING) is not None
    assert check_cam_status_change(received_msgs, 1, CaptureProcessStatus.RUNNING) is not None
