import math

import numpy as np
import pytest

from autotrainer.core.latency.schema import (
    CAN_EVENT_DTYPE,
    LIVE_POSE_DTYPE,
    POSE_FORWARD_DTYPE,
    RECORD_BATCH_DTYPE,
    RECORD_WRITE_DTYPE,
    pose_batch_dtype,
)
from tools.latency import analysis


def _stages(result):
    return {row["stage"].decode(): row for row in result["stages"]}


def test_summarize_reports_percentiles_and_misses():
    row = analysis.summarize("x", [0.001, 0.002, 0.003, np.nan], analysis.SOFTWARE, deadline=0.0025)
    assert row[2] == 3
    assert row[3] == pytest.approx(0.002)
    assert row[8] == pytest.approx(1 / 3)


def test_take_by_key_fills_what_is_missing():
    out = analysis.take_by_key([3, 1, 2], [30.0, 10.0, 20.0], [1, 4, 3])
    assert out[0] == 10.0 and math.isnan(out[1]) and out[2] == 30.0


def test_first_after_respects_the_window():
    out = analysis.first_after([1.0, 2.0], [1.004, 2.5], within=0.01)
    assert out[0] == pytest.approx(1.004) and math.isnan(out[1])


def test_pose_loop_without_ni_falls_back_to_arrival_and_says_so():
    batches = np.zeros(2, dtype=pose_batch_dtype(2))
    batches["pose_seq"] = [0, 1]
    batches["frame_ids"] = [[10, 10], [12, 12]]
    batches["put_perf"] = [[1.003, 1.003], [1.017, 1.017]]
    batches["dequeue_perf"] = [1.004, 1.018]
    batches["predict_start_perf"] = [1.004, 1.018]
    batches["predict_done_perf"] = [1.008, 1.022]
    batches["data_put_perf"] = [1.0081, 1.0221]
    batches["monitor_recv_perf"] = [1.009, 1.023]
    frames = np.zeros(3, dtype=[("frame_id", "<i8"), ("arrival_perf", "<f8")])
    frames["frame_id"] = [10, 11, 12]
    frames["arrival_perf"] = [1.002, 1.009, 1.016]

    result = analysis.pose_loop(
        batches=batches, forwards=np.zeros(0, POSE_FORWARD_DTYPE), live_poses=None, gate=None,
        primary_frames=frames, camera_frames=[frames, frames], exposure_of=None,
        frame_period=1 / 150,
    )

    stages = _stages(result)
    assert result["status"] == "partial"
    assert "start at host arrival" in result["reason"]
    assert stages["sensor_to_pose"]["confidence"] == b"software"
    assert stages["sensor_to_pose"]["p50"] == pytest.approx(0.006)
    assert result["summary"]["coverage"] == pytest.approx(2 / 3)
    # Software fallback does not emit exposure_to_arrival (would measure inter-camera skew, not exposure time)
    assert "exposure_to_arrival" not in stages
    # Real ids reached the table, so nothing is missing for that reason.
    assert "frame ids" not in result["reason"]


@pytest.mark.parametrize("exposure_of", [None, lambda ids: np.full(len(ids), np.nan)],
                         ids=["no_ni_mapping", "ni_mapping"])
def test_pose_loop_without_frame_ids_says_why_the_frame_anchored_stages_are_empty(exposure_of):
    # Emulated and playback cameras are not trigger-synchronized, so the queue
    # passes no frame id and every batch carries -1.
    batches = np.zeros(2, dtype=pose_batch_dtype(2))
    batches["pose_seq"] = [0, 1]
    batches["frame_ids"] = [[-1, -1], [-1, -1]]
    batches["put_perf"] = [[1.003, 1.003], [1.017, 1.017]]
    batches["dequeue_perf"] = [1.004, 1.018]
    batches["predict_start_perf"] = [1.004, 1.018]
    batches["predict_done_perf"] = [1.008, 1.022]
    batches["data_put_perf"] = [1.0081, 1.0221]
    batches["monitor_recv_perf"] = [1.009, 1.023]
    frames = np.zeros(3, dtype=[("frame_id", "<i8"), ("arrival_perf", "<f8")])
    frames["frame_id"] = [10, 11, 12]
    frames["arrival_perf"] = [1.002, 1.009, 1.016]
    # Live 3D and the GUI answered, so the missing ids are the only gap an
    # emulated session would otherwise hide behind a "complete" status.
    forwards = np.zeros(2, POSE_FORWARD_DTYPE)
    forwards["pose_seq"] = [0, 1]
    forwards["live_sequence"] = [0, 1]
    forwards["live_put_perf"] = [1.0095, 1.0235]
    live_poses = np.zeros(2, LIVE_POSE_DTYPE)
    live_poses["live_sequence"] = [0, 1]
    live_poses["live_recv_perf"] = [1.010, 1.024]
    live_poses["triangulated_perf"] = [1.011, 1.025]
    live_poses["gui_recv_perf"] = [1.012, 1.026]

    result = analysis.pose_loop(
        batches=batches, forwards=forwards, live_poses=live_poses, gate=None,
        primary_frames=frames, camera_frames=[frames, frames], exposure_of=exposure_of,
        frame_period=1 / 150,
    )

    stages = _stages(result)
    assert result["status"] == "partial"
    assert "no camera frame ids" in result["reason"]
    assert "frame-anchored stages are empty" in result["reason"]
    # What the frame ids anchor is empty, and what they do not is still measured.
    for name in ("arrival_to_queue", "sensor_to_pose", "sensor_to_3d", "sensor_to_gui"):
        assert stages[name]["n"] == 0, name
    for name in ("queue_wait", "predict", "predict_to_monitor"):
        assert stages[name]["n"] == 2, name


def test_recording_loop_joins_writes_by_first_frame():
    batches = np.zeros(1, dtype=RECORD_BATCH_DTYPE)
    batches[0] = (0, 59, 60, 1.0, 1.0001, 2, False)
    writes = np.zeros(1, dtype=RECORD_WRITE_DTYPE)
    writes[0] = (0, 59, 60, 1.003, 1.0031, 1.0051)

    result = analysis.recording_loop({"left": (batches, writes)})

    stages = _stages(result)
    assert result["status"] == "complete"
    assert stages["left.recorder_queue"]["p50"] == pytest.approx(0.0029)
    assert stages["left.write"]["p50"] == pytest.approx(0.002)


def test_recording_loop_lost_rows_do_not_join_retried_batch():
    batches = np.zeros(2, dtype=RECORD_BATCH_DTYPE)
    batches[0] = (0, 59, 60, 1.0, 4.0, 128, True)  # Lost row with long put_block
    batches[1] = (0, 89, 90, 4.1, 4.1001, 128, False)  # Recovered row (same first_frame_id)
    writes = np.zeros(1, dtype=RECORD_WRITE_DTYPE)
    writes[0] = (0, 89, 90, 4.2, 4.201, 4.21)

    result = analysis.recording_loop({"left": (batches, writes)})

    stages = _stages(result)
    # Only the recovered batch (batch 1) contributes to recorder_queue
    assert stages["left.recorder_queue"]["n"] == 1
    assert stages["left.recorder_queue"]["p50"] == pytest.approx(0.0999)
    assert stages["left.write"]["n"] == 1
    # Lost row's first_frame_id 0 is recovered by batch 1's first_frame_id 0
    assert result["summary"]["left"]["failedPuts"] == 1
    assert result["summary"]["left"]["unrecoveredBatches"] == 0


def test_recording_loop_unrecovered_lost_row():
    batches = np.zeros(2, dtype=RECORD_BATCH_DTYPE)
    batches[0] = (0, 59, 60, 1.0, 4.0, 128, True)  # Lost row
    batches[1] = (10, 89, 90, 4.1, 4.1001, 128, False)  # Recovered row with different first_frame_id
    writes = np.zeros(1, dtype=RECORD_WRITE_DTYPE)
    writes[0] = (10, 89, 90, 4.2, 4.201, 4.21)

    result = analysis.recording_loop({"left": (batches, writes)})

    stages = _stages(result)
    # Only the second batch contributes to recorder_queue
    assert stages["left.recorder_queue"]["n"] == 1
    # Lost row's first_frame_id 0 is NOT recovered (batch 1 has first_frame_id 10)
    assert result["summary"]["left"]["failedPuts"] == 1
    assert result["summary"]["left"]["unrecoveredBatches"] == 1


def test_can_loop_joins_the_ack_by_uuid_after_the_send():
    events = np.zeros(6, dtype=CAN_EVENT_DTYPE)
    events[0] = (b"trial_send", b"", b"SEND_PELLET", -1, 0.990, np.nan, np.nan)
    events[1] = (b"token", b"t1", b"SEND_PELLET", -1, 1.000, np.nan, np.nan)
    events[2] = (b"enqueue", b"t1", b"SEND_PELLET", -1, 1.001, np.nan, np.nan)
    events[3] = (b"dequeue", b"t1", b"", -1, 1.004, np.nan, np.nan)
    events[4] = (b"send", b"t1", b"SEND_PELLET", 9, 1.005, 1.006, np.nan)
    events[5] = (b"ack", b"", b"", 9, 1.016, np.nan, np.nan)

    result = analysis.can_loop(can_events=events, wall_map=None)

    stages = _stages(result)
    assert stages["trial_to_token"]["p50"] == pytest.approx(0.010)
    assert stages["command_queue"]["p50"] == pytest.approx(0.003)
    assert stages["round_trip"]["p50"] == pytest.approx(0.011)
    assert result["summary"]["acked"] == 1


def test_an_ack_that_lands_before_the_send_hook_returns_still_joins():
    events = np.zeros(6, dtype=CAN_EVENT_DTYPE)
    events[0] = (b"trial_send", b"", b"SEND_PELLET", -1, 0.990, np.nan, np.nan)
    events[1] = (b"token", b"t1", b"SEND_PELLET", -1, 1.000, np.nan, np.nan)
    events[2] = (b"enqueue", b"t1", b"SEND_PELLET", -1, 1.001, np.nan, np.nan)
    events[3] = (b"dequeue", b"t1", b"", -1, 1.004, np.nan, np.nan)
    events[4] = (b"send", b"t1", b"SEND_PELLET", 9, 1.005, 1.010, np.nan)
    events[5] = (b"ack", b"", b"", 9, 1.006, np.nan, np.nan)

    result = analysis.can_loop(can_events=events, wall_map=None)

    stages = _stages(result)
    assert stages["round_trip"]["p50"] == pytest.approx(0.001)
    assert result["summary"]["acked"] == 1


def _sent_command(events, first, token, uuid, send_perf):
    events[first] = (b"token", token, b"SET_RGB_LED", -1, send_perf - 0.005, np.nan, np.nan)
    events[first + 1] = (b"enqueue", token, b"SET_RGB_LED", -1, send_perf - 0.004, np.nan, np.nan)
    events[first + 2] = (b"dequeue", token, b"", -1, send_perf - 0.001, np.nan, np.nan)
    events[first + 3] = (b"send", token, b"SET_RGB_LED", uuid, send_perf, send_perf + 0.001, np.nan)


def test_can_loop_with_a_token_but_no_send_is_absent():
    # A CAN-disabled run still records the command's token row, but nothing
    # is sent. That is no CAN traffic, not a complete loop that measured none.
    events = np.zeros(1, dtype=CAN_EVENT_DTYPE)
    events[0] = (b"token", b"t1", b"SET_RGB_LED", -1, 1.000, np.nan, np.nan)

    result = analysis.can_loop(can_events=events, wall_map=None)

    assert result["status"] == "absent"
    assert result["reason"] == "no CAN sends recorded"


def test_can_loop_with_sends_and_no_acks_is_partial():
    events = np.zeros(9, dtype=CAN_EVENT_DTYPE)
    _sent_command(events, 0, b"t1", 3, 1.005)
    _sent_command(events, 4, b"t2", 4, 2.005)
    # An ack for a uuid nobody sent does not count for either command.
    events[8] = (b"ack", b"", b"", 99, 2.010, np.nan, np.nan)

    result = analysis.can_loop(can_events=events, wall_map=None)

    assert result["status"] == "partial"
    assert result["reason"] == "2 sends, none acknowledged"
    assert result["summary"] == {"commands": 2, "acked": 0}


def test_can_loop_with_one_ack_among_the_sends_stays_complete():
    events = np.zeros(9, dtype=CAN_EVENT_DTYPE)
    _sent_command(events, 0, b"t1", 3, 1.005)
    _sent_command(events, 4, b"t2", 4, 2.005)
    events[8] = (b"ack", b"", b"", 4, 2.016, np.nan, np.nan)

    result = analysis.can_loop(can_events=events, wall_map=None)

    assert result["status"] == "complete"
    assert result["reason"] == ""
    assert result["summary"] == {"commands": 2, "acked": 1}


def test_hardware_loop_without_output_edges_is_absent():
    result = analysis.hardware_loop(command_edges={}, diode_edges={}, trigger_edges={},
                                    unusable={"laser1_diode": "swing 0.0001 against noise"})
    assert result["status"] == "absent"
    assert "laser1_diode" in result["reason"]
