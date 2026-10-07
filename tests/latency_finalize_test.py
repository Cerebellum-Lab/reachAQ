import json

import h5py
import numpy as np
import pytest

from autotrainer.core.latency.schema import (
    CAMERA_FRAME_DTYPE,
    CLOCK_PAIR_DTYPE,
    EVENT_TABLES,
    LIVE_POSE_DTYPE,
    POSE_FORWARD_DTYPE,
    RECORD_BATCH_DTYPE,
    RECORD_WRITE_DTYPE,
    STIM_DISPATCH_DTYPE,
    pose_batch_dtype,
)
from tools.acquisition.model.latency_event_log import LatencyEventLog
from tools.latency import finalize
from tools.latency import rebuild
from tools.latency.finalize import finalize_session_latency

RATE = 10_000.0
PERIOD = 1 / 150
HOST = 1000.0          # host perf = HOST + NI seconds (no drift, to keep the arithmetic readable)
CAMERA_OFFSET = 5.0    # camera clock = NI seconds + 5 s
FRAMES = 300
ARRIVAL_LAG = 0.002
COMMAND_DELAY = 0.0003
DIODE_DELAY = 0.0001
START_ENTRY_NI = 1.0


def _write(path, datasets, attrs=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as store:
        store.attrs["schema_version"] = 1
        for key, value in (attrs or {}).items():
            store.attrs[key] = value
        for name, rows in datasets.items():
            store.create_dataset(name, data=rows)


def _session(tmp_path):
    rng = np.random.default_rng(7)
    session = tmp_path / "session001"
    streams = session / "streams"
    raw = streams / "latency"

    samples = int(2.5 * RATE)
    exposure_ni = 0.05 + np.arange(FRAMES) * PERIOD
    transition_samples = np.round(exposure_ni * RATE).astype(np.int64)
    cam = np.zeros(samples, dtype=np.float32)
    level = 0.0
    for start, stop in zip(transition_samples, list(transition_samples[1:]) + [samples]):
        level = 1.0 - level
        cam[start:stop] = level
    command = rng.normal(0, 0.002, samples).astype(np.float32)
    diode = rng.normal(0, 0.002, samples).astype(np.float32)
    edge = int(round((START_ENTRY_NI + COMMAND_DELAY) * RATE))
    command[edge:edge + 100] += 5.0
    diode[edge + 1:edge + 101] += 2.0
    block = np.arange(samples) // 100
    block_last = np.arange(samples // 100) * 100 + 99
    seen_per_block = HOST + block_last / RATE + 0.0002 + rng.uniform(0, 0.003, block_last.size)
    _write(streams / "nidaq.h5", {
        "sample_index": np.arange(samples, dtype=np.int64),
        "block_observation_perf_time": seen_per_block[block].astype(np.float64),
        "epoch": np.ones(samples, dtype=np.uint64),
        "values": np.vstack([cam, command, diode]).astype(np.float32),
    }, {"channel_names": np.array([b"cam_frames", b"laser1_command_copy", b"laser1_diode"]),
        "sample_rate_hz": RATE})

    ids = np.arange(FRAMES)
    exposure_host = HOST + exposure_ni
    frames = np.zeros(FRAMES, dtype=CAMERA_FRAME_DTYPE)
    frames["frame_id"] = ids
    frames["camera_ts_ns"] = np.round((exposure_ni + CAMERA_OFFSET) * 1e9).astype(np.int64)
    frames["arrival_perf"] = exposure_host + ARRIVAL_LAG
    frames["poll_perf"] = frames["arrival_perf"] - 0.0005
    frames["capture_return_perf"] = frames["arrival_perf"] + 0.0001
    batches = np.zeros(2, dtype=RECORD_BATCH_DTYPE)
    batches["first_frame_id"] = [0, 150]
    batches["last_frame_id"] = [149, 299]
    batches["frame_count"] = 150
    batches["put_entry_perf"] = [HOST + 1.0, HOST + 2.0]
    batches["put_return_perf"] = batches["put_entry_perf"] + 0.0001
    batches["queue_depth"] = 1
    writes = np.zeros(2, dtype=RECORD_WRITE_DTYPE)
    writes["first_frame_id"] = [0, 150]
    writes["last_frame_id"] = [149, 299]
    writes["frame_count"] = 150
    writes["dequeue_perf"] = batches["put_return_perf"] + 0.001
    writes["write_entry_perf"] = writes["dequeue_perf"] + 0.0001
    writes["write_return_perf"] = writes["write_entry_perf"] + 0.002
    pairs = np.array([(1.7e9 + t, HOST + t) for t in (0.0, 1.0, 2.0)], dtype=CLOCK_PAIR_DTYPE)
    for name in ("left", "right"):
        _write(raw / f"camera_{name}.h5",
               {"frames": frames, "record_batches": batches, "clock_pairs": pairs})
        _write(raw / f"record_{name}.h5", {"record_writes": writes, "clock_pairs": pairs})

    posed = ids[::2]
    count = posed.size
    pose = np.zeros(count, dtype=pose_batch_dtype(2))
    pose["pose_seq"] = np.arange(count)
    pose["frame_ids"] = np.stack([posed, posed], axis=1)
    put = exposure_host[posed] + ARRIVAL_LAG + 0.0002
    pose["put_perf"] = np.stack([put, put], axis=1)
    pose["dequeue_perf"] = put + 0.001
    pose["predict_start_perf"] = pose["dequeue_perf"] + 0.0001
    pose["predict_done_perf"] = pose["predict_start_perf"] + 0.004
    pose["data_put_perf"] = pose["predict_done_perf"] + 0.0001
    pose["monitor_recv_perf"] = pose["data_put_perf"] + 0.0005
    forwards = np.zeros(count, dtype=POSE_FORWARD_DTYPE)
    forwards["pose_seq"] = np.arange(count)
    forwards["live_sequence"] = np.arange(count) + 100
    forwards["live_put_perf"] = pose["monitor_recv_perf"] + 0.0001
    _write(raw / "pose.h5", {"batches": pose, "forwards": forwards, "clock_pairs": pairs},
           {"cameras": json.dumps(["left", "right"])})

    live = np.zeros(count, dtype=LIVE_POSE_DTYPE)
    live["live_sequence"] = forwards["live_sequence"]
    live["live_recv_perf"] = forwards["live_put_perf"] + 0.001
    live["triangulated_perf"] = live["live_recv_perf"] + 0.002
    live["live_put_perf"] = live["triangulated_perf"] + 0.0001
    live["gui_recv_perf"] = live["live_put_perf"] + 0.003
    dispatch = np.zeros(1, dtype=STIM_DISPATCH_DTYPE)
    dispatch[0] = (b"op-1", b"direct_ni_software", 7, HOST + 0.995, HOST + 0.9952,
                   HOST + 0.9953, HOST + 0.9954, HOST + 0.9955, HOST + 0.998, HOST + 0.9985,
                   HOST + START_ENTRY_NI, HOST + START_ENTRY_NI + 0.0006, 1)
    tables = {name: np.zeros(0, dtype=dtype) for name, dtype in EVENT_TABLES.items()}
    tables["live_poses"] = live
    tables["stim_dispatch"] = dispatch
    LatencyEventLog.write(raw / "events.h5", tables)
    return session


def _stages(store, loop):
    return {row["stage"].decode(): row for row in store[f"{loop}/stages"][:]}


def test_a_synthetic_session_recovers_its_known_delays(tmp_path):
    session = _session(tmp_path)

    status = finalize_session_latency(session, primary_camera="left")

    assert status["status"] == "complete", status
    assert status["clocks"]["cameraToNi"]["valid"] is True
    assert status["clocks"]["cameraToNi"]["pairing"]["ambiguous"] is False
    json.dumps(status)  # metadata-safe
    with h5py.File(session / "streams" / "latency.h5", "r") as store:
        pose = _stages(store, "pose")
        stim = _stages(store, "stim")
        hardware = _stages(store, "hardware")
    # Mapped NI times are late by the minimum delivery delay (0.2 ms here) plus fit noise.
    assert pose["exposure_to_arrival"]["confidence"] == b"mixed"
    assert ARRIVAL_LAG - 0.0015 <= pose["exposure_to_arrival"]["p50"] <= ARRIVAL_LAG
    assert pose["predict"]["p50"] == pytest.approx(0.004, abs=1e-9)
    assert pose["sensor_to_pose"]["n"] == FRAMES // 2
    assert hardware["laser1.command_to_diode"]["p50"] == pytest.approx(DIODE_DELAY, abs=1.5 / RATE)
    assert COMMAND_DELAY <= stim["direct.start_to_output_edge"]["p50"] <= COMMAND_DELAY + 0.0015


def test_a_session_without_latency_streams_is_absent(tmp_path):
    (tmp_path / "streams").mkdir()
    status = finalize_session_latency(tmp_path, primary_camera="left")
    assert status["status"] == "absent"
    assert not (tmp_path / "streams" / "latency.h5").exists()


def test_an_unreadable_stream_fails_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize, "OPEN_RETRY_SECONDS", 0.2)
    raw = tmp_path / "streams" / "latency"
    raw.mkdir(parents=True)
    (raw / "camera_left.h5").write_bytes(b"not hdf5")

    status = finalize_session_latency(tmp_path, primary_camera="left")

    assert status["status"] == "failed"
    assert "unreadable" in status["reasons"][0]


def test_rebuild_writes_beside_the_original_and_leaves_it_alone(tmp_path, capsys):
    session = _session(tmp_path)
    (session / "streams" / "alignment.json").write_text(
        json.dumps({"canonicalBoundary": {"primaryCamera": "left"}}))

    assert rebuild.main([str(session)]) == 0

    rebuilt = list((session / "streams").glob("latency.rebuilt-*.h5"))
    assert len(rebuilt) == 1
    assert not (session / "streams" / "latency.h5").exists()
    assert '"status": "complete"' in capsys.readouterr().out
