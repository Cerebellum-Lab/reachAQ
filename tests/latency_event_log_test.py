import logging
import math
from types import SimpleNamespace

import h5py
import pytest

from autotrainer.core.latency.schema import EVENT_TABLES
from tools.acquisition.model.latency_event_log import LatencyEventLog, _BlockTable

_LOG_NAME = "tools.acquisition.model.latency_event_log"


def _response(sequence):
    return SimpleNamespace(sequence=sequence, perf_c=2.0, live_recv_perf_c=1.5,
                           live_put_perf_c=2.1)


def test_rows_are_kept_only_while_a_recording_is_active():
    log = LatencyEventLog(block_rows=2)
    log.record_live_pose(_response(99), 2.2)  # before begin: ignored
    log.begin()
    for sequence in range(5):  # spans three blocks
        log.record_live_pose(_response(sequence), 2.2)
    tables = log.end()
    log.record_live_pose(_response(98), 2.2)  # after end: ignored

    assert tables["live_poses"]["live_sequence"].tolist() == [0, 1, 2, 3, 4]
    assert tables["live_poses"]["gui_recv_perf"][0] == 2.2
    assert tables["live_poses"]["triangulated_perf"][0] == 2.0


def test_a_direct_route_payload_becomes_one_dispatch_row():
    log = LatencyEventLog()
    log.begin()
    log.record_stim_dispatch({
        "operation_id": "op-1", "trigger_route": "direct_ni_software", "stim_frame_id": 9,
        "frame_perf_time": 1.0, "decision_perf_time": 1.001,
        "evidence_done_perf_time": 1.0012, "clip_done_perf_time": 1.0013,
        "ipc_send_perf_time": 1.0014, "ipc_receive_perf_time": 1.002,
        "validated_perf_time": 1.0021, "daqmx_start_entry_perf_time": 1.0022,
        "daqmx_start_return_perf_time": 1.0030, "accepted": True,
    })
    row = log.end()["stim_dispatch"][0]

    assert row["operation_id"] == b"op-1"
    assert row["route"] == b"direct_ni_software"
    assert row["send_perf"] == 1.0014
    assert row["gui_recv_perf"] == 1.002
    assert row["accepted"] == 1


def test_a_stim3_message_uses_its_own_send_and_receive_stamps():
    log = LatencyEventLog()
    log.begin()
    log.record_stim_dispatch({"operation_id": "op-2", "trigger_route": "hardware_stim3",
                              "stim_frame_id": 3, "msg_send_perf_time": 2.0},
                             gui_recv_perf=2.004)
    row = log.end()["stim_dispatch"][0]

    assert row["send_perf"] == 2.0
    assert row["gui_recv_perf"] == 2.004
    assert row["accepted"] == -1
    assert math.isnan(row["start_entry_perf"])


def test_a_gate_observation_without_a_sample_is_recorded_as_unknown():
    log = LatencyEventLog()
    log.begin()
    log.record_gate_observation(5.0, None, None)
    log.record_gate_observation(6.0, SimpleNamespace(sequence=8, processing_perf=5.9), True)
    rows = log.end()["gate_observations"]

    assert rows["live_sequence"].tolist() == [-1, 8]
    assert rows["reaching"].tolist() == [-1, 1]
    assert math.isnan(rows["triangulated_perf"][0])


def test_can_rows_keep_stage_token_and_uuid():
    log = LatencyEventLog()
    log.begin()
    log.record_can("send", {"token": "t-1", "kind": "SEND_PELLET", "can_uuid": 12,
                            "perf": 1.0, "perf_end": 1.001})
    log.record_can("ack", {"can_uuid": 12, "perf": 1.01, "kernel_wall": None})
    rows = log.end()["can_events"]

    assert rows["stage"].tolist() == [b"send", b"ack"]
    assert rows["can_uuid"].tolist() == [12, 12]
    assert math.isnan(rows["kernel_wall"][1])


def test_write_creates_every_table_even_when_empty(tmp_path):
    log = LatencyEventLog()
    log.begin()
    log.record_stim3_pulse("op-2", "tok", 1.0, 1.002)
    path = tmp_path / "events.h5"
    LatencyEventLog.write(path, log.end())

    with h5py.File(path, "r") as store:
        assert set(store) == set(EVENT_TABLES)
        assert len(store["stim3_pulses"]) == 1
        assert len(store["live_poses"]) == 0


def test_a_disabled_record_keeps_nothing(monkeypatch):
    monkeypatch.setenv("REACHAQ_LATENCY_RECORD", "0")
    log = LatencyEventLog()
    log.begin()
    log.record_live_pose(_response(1), 2.2)
    assert len(log.end()["live_poses"]) == 0


def test_a_disabled_record_reports_itself_as_not_enabled(monkeypatch):
    monkeypatch.setenv("REACHAQ_LATENCY_RECORD", "0")
    log = LatencyEventLog()
    log.begin()
    assert not log.enabled
    assert not log.is_active
    log.end()
    assert not log.enabled


def test_an_enabled_record_stays_enabled_after_it_ends(monkeypatch):
    monkeypatch.delenv("REACHAQ_LATENCY_RECORD", raising=False)
    log = LatencyEventLog()
    assert not log.enabled  # nothing has begun yet
    log.begin()
    assert log.enabled and log.is_active
    log.end()
    assert log.enabled and not log.is_active


@pytest.mark.parametrize("payload", [
    {"operation_id": "bad", "stim_frame_id": None},
    {"operation_id": "bad", "stim_frame_id": float("nan")},
    {"operation_id": "bad", "stim_frame_id": "frame"},
    None,
    "not a mapping",
    7,
])
def test_a_stim_payload_that_cannot_become_a_row_is_counted_not_raised(payload):
    log = LatencyEventLog()
    log.begin()
    log.record_stim_dispatch(payload)
    log.record_stim_dispatch({"operation_id": "good", "stim_frame_id": 4})

    assert log.rows_rejected == 1
    assert log.end()["stim_dispatch"]["operation_id"].tolist() == [b"good"]


@pytest.mark.parametrize("fields", [
    None,
    {"can_uuid": 10**6},
    {"can_uuid": 40000},  # numpy 1.x would wrap this to a wrong, valid-looking id
    {"can_uuid": -32769},
    {"can_uuid": "abc"},
])
def test_a_can_row_that_cannot_be_stored_is_counted_not_raised(fields):
    log = LatencyEventLog()
    log.begin()
    log.record_can("send", fields)
    log.record_can("ack", {"can_uuid": 12, "perf": 1.0})

    assert log.rows_rejected == 1
    rows = log.end()["can_events"]
    assert rows["stage"].tolist() == [b"ack"]
    assert rows["can_uuid"].tolist() == [12]


class _Unprintable:
    def __str__(self):
        raise RuntimeError("no text form")


def test_every_other_row_kind_is_counted_not_raised_too():
    log = LatencyEventLog()
    log.begin()
    log.record_live_pose(SimpleNamespace(sequence=None, perf_c=1.0), 2.0)
    log.record_live_pose(SimpleNamespace(sequence=1), 2.0)  # no perf_c
    log.record_gate_observation(1.0, SimpleNamespace(sequence=None, processing_perf=1.0), True)
    log.record_stim3_pulse(_Unprintable(), "tok", 1.0, 2.0)

    assert log.rows_rejected == 4
    tables = log.end()
    assert all(len(rows) == 0 for rows in tables.values())


def test_a_row_the_block_refuses_is_counted_and_later_rows_still_land(monkeypatch):
    original = _BlockTable.append

    def refuse(self, row):
        if row[0] == 666:
            raise ValueError("refused")
        original(self, row)

    monkeypatch.setattr(_BlockTable, "append", refuse)
    log = LatencyEventLog(block_rows=2)
    log.begin()
    for sequence in (1, 666, 2, 3):
        log.record_live_pose(_response(sequence), 2.2)

    assert log.rows_rejected == 1
    assert log.end()["live_poses"]["live_sequence"].tolist() == [1, 2, 3]


def test_a_row_offered_while_inactive_is_ignored_not_rejected():
    log = LatencyEventLog()
    log.record_can("send", None)  # before begin()
    log.begin()
    log.end()
    log.record_stim_dispatch(None)  # after end()
    assert log.rows_rejected == 0


def test_only_the_first_rejected_row_of_a_recording_is_logged(caplog):
    log = LatencyEventLog()
    log.begin()
    with caplog.at_level(logging.ERROR, logger=_LOG_NAME):
        log.record_can("send", None)
        log.record_can("send", None)
    assert log.rows_rejected == 2
    assert len([r for r in caplog.records if r.name == _LOG_NAME]) == 1

    caplog.clear()
    log.begin()  # a new recording: the count and the once-only log both reset
    assert log.rows_rejected == 0
    with caplog.at_level(logging.ERROR, logger=_LOG_NAME):
        log.record_can("send", None)
    assert len([r for r in caplog.records if r.name == _LOG_NAME]) == 1


def test_the_reject_is_logged_with_the_log_lock_released():
    # The session log handler takes the recorder's lock, and the recorder holds
    # it while it calls begin() and end() here: logging under this lock would
    # be a lock-order cycle.
    log = LatencyEventLog()
    log.begin()
    lock_free = []

    class Probe(logging.Handler):
        def emit(self, record):
            got = log._lock.acquire(blocking=False)
            if got:
                log._lock.release()
            lock_free.append(got)

    target = logging.getLogger(_LOG_NAME)
    probe = Probe()
    target.addHandler(probe)
    try:
        log.record_can("send", None)
    finally:
        target.removeHandler(probe)

    assert lock_free == [True]
