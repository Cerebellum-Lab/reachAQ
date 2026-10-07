import hashlib
import json
from pathlib import Path

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
from tools.autotrainer_version import __version__ as app_version
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


def _session(tmp_path, *, first_sample=0, first_frame=0, primary="left"):
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
        "sample_index": first_sample + np.arange(samples, dtype=np.int64),
        "block_observation_perf_time": seen_per_block[block].astype(np.float64),
        "epoch": np.ones(samples, dtype=np.uint64),
        "values": np.vstack([cam, command, diode]).astype(np.float32),
    }, {"channel_names": np.array([b"cam_frames", b"laser1_command_copy", b"laser1_diode"]),
        "sample_rate_hz": RATE})

    ids = np.arange(FRAMES)
    exposure_host = HOST + exposure_ni
    frames = np.zeros(FRAMES, dtype=CAMERA_FRAME_DTYPE)
    frames["frame_id"] = ids + first_frame
    frames["camera_ts_ns"] = np.round((exposure_ni + CAMERA_OFFSET) * 1e9).astype(np.int64)
    frames["arrival_perf"] = exposure_host + ARRIVAL_LAG
    frames["poll_perf"] = frames["arrival_perf"] - 0.0005
    frames["capture_return_perf"] = frames["arrival_perf"] + 0.0001
    batches = np.zeros(2, dtype=RECORD_BATCH_DTYPE)
    batches["first_frame_id"] = [first_frame, first_frame + 150]
    batches["last_frame_id"] = [first_frame + 149, first_frame + 299]
    batches["frame_count"] = 150
    batches["put_entry_perf"] = [HOST + 1.0, HOST + 2.0]
    batches["put_return_perf"] = batches["put_entry_perf"] + 0.0001
    batches["queue_depth"] = 1
    writes = np.zeros(2, dtype=RECORD_WRITE_DTYPE)
    writes["first_frame_id"] = [first_frame, first_frame + 150]
    writes["last_frame_id"] = [first_frame + 149, first_frame + 299]
    writes["frame_count"] = 150
    writes["dequeue_perf"] = batches["put_return_perf"] + 0.001
    writes["write_entry_perf"] = writes["dequeue_perf"] + 0.0001
    writes["write_return_perf"] = writes["write_entry_perf"] + 0.002
    pairs = np.array([(1.7e9 + t, HOST + t) for t in (0.0, 1.0, 2.0)], dtype=CLOCK_PAIR_DTYPE)
    for number, name in enumerate(("left", "right")):
        _write(raw / f"camera_{name}.h5",
               {"frames": frames, "record_batches": batches, "clock_pairs": pairs},
               {"camera": name, "camera_index": number, "is_primary": name == primary})
        _write(raw / f"record_{name}.h5", {"record_writes": writes, "clock_pairs": pairs})

    posed = ids[::2]
    count = posed.size
    pose = np.zeros(count, dtype=pose_batch_dtype(2))
    pose["pose_seq"] = np.arange(count)
    pose["frame_ids"] = np.stack([posed + first_frame, posed + first_frame], axis=1)
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


@pytest.mark.parametrize("start", [
    {},
    # The real cropped nidaq.h5 starts at a nonzero sample index, and frame ids
    # do not start at zero either: nothing may assume position == index == id.
    {"first_sample": 123456, "first_frame": 1000},
], ids=["from_zero", "offset_ids"])
def test_a_synthetic_session_recovers_its_known_delays(tmp_path, start):
    session = _session(tmp_path, **start)

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
    assert finalize_session_latency(session, primary_camera="left")["status"] == "complete"
    original = session / "streams" / "latency.h5"
    before = hashlib.sha256(original.read_bytes()).hexdigest()

    assert rebuild.main([str(session)]) == 0

    rebuilt = list((session / "streams").glob("latency.rebuilt-*.h5"))
    assert len(rebuilt) == 1
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    # The top-level status, not a substring a per-loop status would also match.
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_two_rebuilds_in_one_second_do_not_overwrite_each_other(tmp_path, capsys):
    session = _session(tmp_path)

    assert rebuild.main([str(session), "--primary-camera", "left"]) == 0
    assert rebuild.main([str(session), "--primary-camera", "left"]) == 0

    assert len(list((session / "streams").glob("latency.rebuilt-*.h5"))) == 2


@pytest.mark.parametrize("alignment", [
    {"canonicalBoundary": None}, [], {"canonicalBoundary": {"primaryCamera": None}}, "{",
], ids=["null_boundary", "list", "null_camera", "not_json"])
def test_rebuild_survives_an_alignment_file_of_the_wrong_shape(tmp_path, capsys, alignment):
    session = _session(tmp_path)
    text = alignment if isinstance(alignment, str) else json.dumps(alignment)
    (session / "streams" / "alignment.json").write_text(text)

    assert rebuild.main([str(session)]) == 0

    # No primary camera was named, so the stream that says it is primary is used.
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "partial"
    assert status["reasons"][0].startswith("no primary camera given; using 'left'")


def test_rebuild_takes_the_session_id_from_the_manifest(tmp_path, capsys):
    session = _session(tmp_path)
    (session / "streams" / "stream_manifest.json").write_text(json.dumps({"sessionId": "s-17"}))

    rebuild.main([str(session), "--primary-camera", "left"])

    rebuilt = next((session / "streams").glob("latency.rebuilt-*.h5"))
    with h5py.File(rebuilt, "r") as store:
        assert store.attrs["session_id"] == "s-17"


def test_the_file_records_its_clock_fits_session_and_app_version(tmp_path):
    session = _session(tmp_path)

    status = finalize_session_latency(session, primary_camera="left", session_id="abc-1")

    assert status["output"] == "latency.h5"
    with h5py.File(session / "streams" / "latency.h5", "r") as store:
        assert store.attrs["session_id"] == "abc-1"
        assert store.attrs["app_version"] == app_version
        assert json.loads(store.attrs["status"])["output"] == "latency.h5"
        assert set(store["clock"]) == {"wall_to_host", "ni_to_host", "camera_to_ni"}
        ni = dict(store["clock/ni_to_host"].attrs)
        camera = dict(store["clock/camera_to_ni"].attrs)
        wall = dict(store["clock/wall_to_host"].attrs)
    assert ni["valid"] and ni["points"] == 20 and ni["residual_rms"] < 0.0005
    assert "minimum delivery delay" in ni["note"]
    assert "unmeasured" in ni["envelope_bias_floor"]
    assert "unmeasured" in ni["secondary_trigger_delay"]
    assert camera["valid"] and camera["points"] == FRAMES
    assert camera["pairing_shift"] == 0 and not camera["pairing_ambiguous"]
    assert wall["valid"] and wall["segments"] == 1


def test_one_truncated_stream_costs_only_what_depends_on_it(tmp_path, monkeypatch):
    whole = finalize_session_latency(_session(tmp_path / "whole"), primary_camera="left")
    session = _session(tmp_path / "cut")
    pose = session / "streams" / "latency" / "pose.h5"
    pose.write_bytes(pose.read_bytes()[:2000])
    monkeypatch.setattr(finalize, "OPEN_RETRY_SECONDS", 0.2)

    status = finalize_session_latency(session, primary_camera="left")

    assert [item["path"] for item in status["unreadable"]] == ["streams/latency/pose.h5"]
    assert status["unreadable"][0]["reason"].startswith("OSError")
    assert status["status"] == "partial"
    assert "pose.h5 unreadable" in status["loops"]["pose"]["reason"]
    for loop in ("recording", "stim", "hardware"):
        assert status["loops"][loop]["status"] == "complete"
        assert status["loops"][loop] == whole["loops"][loop]
    assert status["clocks"]["cameraToNi"]["valid"] is True
    with h5py.File(session / "streams" / "latency.h5", "r") as store:
        assert _stages(store, "hardware")["laser1.command_to_diode"]["n"] == 1


@pytest.mark.parametrize("broken, loops", [
    ("fit_ni_to_host", {"stim": "partial", "hardware": "complete"}),
    ("choose_pairing", {"pose": "partial", "hardware": "complete"}),
    ("fit_camera_to_ni", {"pose": "partial", "hardware": "complete"}),
    ("fit_wall_to_host", {"pose": "complete", "hardware": "complete"}),
])
def test_a_clock_fit_that_raises_is_an_invalid_fit_not_a_failure(tmp_path, monkeypatch, broken, loops):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(finalize.clocks, broken, explode)

    status = finalize_session_latency(_session(tmp_path), primary_camera="left")

    key = {"fit_ni_to_host": "niToHost", "choose_pairing": "cameraToNi",
           "fit_camera_to_ni": "cameraToNi", "fit_wall_to_host": "wallToHost"}[broken]
    assert status["clocks"][key]["valid"] is False
    assert "boom" in status["clocks"][key]["reason"]
    assert status["status"] in {"partial", "complete"}
    for loop, expected in loops.items():
        assert status["loops"][loop]["status"] == expected, status["loops"][loop]
    assert (tmp_path / "session001" / "streams" / "latency.h5").is_file()


def test_a_session_whose_loops_all_find_nothing_is_absent(tmp_path):
    raw = tmp_path / "streams" / "latency"
    raw.mkdir(parents=True)
    LatencyEventLog.write(raw / "events.h5",
                          {name: np.zeros(0, dtype=dtype) for name, dtype in EVENT_TABLES.items()})

    status = finalize_session_latency(tmp_path, primary_camera="left")

    assert status["status"] == "absent", status
    assert {entry["status"] for entry in status["loops"].values()} == {"absent"}


def test_a_session_whose_only_loop_fails_is_failed(tmp_path):
    _write(tmp_path / "streams" / "latency" / "pose.h5", {}, {"cameras": "not json"})

    status = finalize_session_latency(tmp_path, primary_camera="left")

    assert status["loops"]["pose"]["status"] == "failed"
    assert status["status"] == "failed", status


def test_a_reason_caps_an_otherwise_complete_session_at_partial(tmp_path):
    status = finalize_session_latency(_session(tmp_path), primary_camera="zzz")

    assert {entry["status"] for entry in status["loops"].values()} <= {"complete", "absent"}
    assert status["reasons"] == ["primary camera 'zzz' has no latency stream; using 'left' "
                                 "(its stream marks it primary)"]
    assert status["status"] == "partial"


@pytest.mark.parametrize("stats, expected", [
    ({"rowsDropped": {"frames": 3, "clock_pairs": 1}}, "rowsDropped=4, rowsRejected=0, failed=False"),
    ({"rowsRejected": 2}, "rowsDropped=0, rowsRejected=2, failed=False"),
    ({"failed": True, "firstError": "disk full"}, "failed=True (disk full)"),
])
def test_a_writer_that_lost_rows_caps_the_session_at_partial(tmp_path, stats, expected):
    session = _session(tmp_path)
    with h5py.File(session / "streams" / "latency" / "camera_left.h5", "r+") as store:
        store.attrs["writer_stats"] = json.dumps(stats)

    status = finalize_session_latency(session, primary_camera="left")

    assert status["status"] == "partial"
    assert len(status["reasons"]) == 1
    assert status["reasons"][0].startswith("streams/latency/camera_left.h5: writer lost rows")
    assert expected in status["reasons"][0]
    assert {entry["status"] for entry in status["loops"].values()} <= {"complete", "absent"}


def test_a_writer_that_lost_nothing_does_not_cap_the_session(tmp_path):
    session = _session(tmp_path)
    clean = {"rowsWritten": {"frames": 300}, "rowsDropped": {"frames": 0}, "rowsRejected": 0,
             "failed": False, "firstError": None, "queueHighWater": 1}
    with h5py.File(session / "streams" / "latency" / "camera_left.h5", "r+") as store:
        store.attrs["writer_stats"] = json.dumps(clean)

    assert finalize_session_latency(session, primary_camera="left")["status"] == "complete"


def test_the_stream_that_says_it_is_primary_is_preferred_to_the_alphabetical_first(tmp_path):
    session = _session(tmp_path, primary="right")

    named = finalize_session_latency(session, primary_camera="")
    absent = finalize_session_latency(session, primary_camera="zzz")

    assert named["reasons"] == ["no primary camera given; using 'right' "
                                "(its stream marks it primary)"]
    assert absent["reasons"][0].startswith("primary camera 'zzz' has no latency stream; "
                                           "using 'right'")
    assert named["clocks"]["cameraToNi"]["valid"] is True


def test_a_failed_write_leaves_no_partial_file(tmp_path):
    session = _session(tmp_path)
    (session / "streams" / "latency.h5").mkdir()  # the final path cannot be replaced

    status = finalize_session_latency(session, primary_camera="left")

    assert status["status"] == "failed"
    assert status["output"] is None
    assert "latency.h5 not written" in status["reasons"][-1]
    assert not list((session / "streams").glob("*.partial"))


def test_finalization_never_raises_whatever_it_is_given(tmp_path):
    status = finalize_session_latency(None, primary_camera="left")

    assert status["status"] == "failed"
    assert "TypeError" in status["reasons"][0]


def test_the_status_is_json_safe_for_arrays_and_unknown_types():
    class Odd:
        def __str__(self):
            return "odd"

    safe = finalize._json_safe({"a": np.arange(3, dtype=np.int64), "b": Odd(),
                                "c": np.float32("nan"), "d": (np.bool_(True), b"x")})

    assert safe == {"a": [0, 1, 2], "b": "odd", "c": None, "d": [True, "x"]}
    json.dumps(safe)


def test_raw_streams_are_opened_again_while_a_writer_still_holds_them(tmp_path, monkeypatch):
    session = _session(tmp_path)
    real = h5py.File
    attempts = {}

    def held_twice(path, mode="r", *args, **kwargs):
        if mode == "r" and Path(path).parent.name == "latency":
            attempts[str(path)] = attempts.get(str(path), 0) + 1
            if attempts[str(path)] <= 2:
                raise OSError("unable to lock file")
        return real(path, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", held_twice)
    monkeypatch.setattr(finalize.time, "sleep", lambda seconds: None)

    status = finalize_session_latency(session, primary_camera="left")

    assert status["status"] == "complete", status
    assert status["unreadable"] == []
    assert set(attempts.values()) == {3}


def test_a_consecutive_sample_index_is_derived_and_a_gap_uses_the_stored_one(tmp_path):
    values = np.array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.float32)
    attrs = {"channel_names": np.array([b"cam_frames"]), "sample_rate_hz": 1000.0}
    seen = np.arange(10, dtype=np.float64)
    for index, expected in (
        (1000 + np.arange(10), [1002, 1004, 1006, 1008]),
        (np.array([5, 6, 7, 8, 9, 20, 21, 22, 23, 24]), [7, 9, 21, 23]),
    ):
        path = tmp_path / f"nidaq_{expected[0]}.h5"
        _write(path, {"sample_index": index.astype(np.int64),
                      "block_observation_perf_time": seen, "values": values}, attrs)

        ni = finalize._read_nidaq(path)

        assert ni["transitions"].tolist() == expected
        assert "values" not in ni and "channels" not in ni  # no samples are retained
