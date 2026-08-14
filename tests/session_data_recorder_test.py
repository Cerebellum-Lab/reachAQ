import csv
import json
import logging
import time as stdlib_time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from autotrainer.core import (
    ObservableObject,
    ProjectInfo,
    SystemCommandKind,
    SystemStatusMessageKind,
)
from tools.acquisition.model import session_data_recorder
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


class _EventSource(ObservableObject):
    LAST_FEEDBACK_SAMPLE = "last_feedback_sample"

    def __init__(self, *event_names):
        super().__init__(event_names)


def test_inactive_session_log_handler_does_not_consume_the_runtime_clock(
    monkeypatch,
):
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    monkeypatch.setattr(
        session_data_recorder,
        "time",
        SimpleNamespace(
            perf_counter=lambda: (_ for _ in ()).throw(
                AssertionError("clock must not be read")
            ),
            time=stdlib_time.time,
        ),
    )
    try:
        recorder.add_current_log(100.0, "outside a recording")
    finally:
        recorder.close()


def test_session_log_handler_filters_raw_can_frames_but_keeps_can_warnings():
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    recorder._armed = True
    try:
        for level, message in (
            (9, "raw frame"),
            (logging.WARNING, "CAN warning"),
        ):
            recorder._log_handler.emit(
                logging.LogRecord(
                    "can.bus",
                    level,
                    __file__,
                    1,
                    message,
                    (),
                    None,
                )
            )

        messages = [row[2] for row in recorder._log_rows]

        assert all("raw frame" not in message for message in messages)
        assert any("CAN warning" in message for message in messages)
    finally:
        recorder.close()


def test_device_event_does_not_serialize_container_index_method():
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    try:
        recorder._append_device_event(
            10.0,
            100.0,
            "inbound",
            SystemStatusMessageKind.STIMULUS_INPUTS,
            [False, False, False, False],
            None,
            None,
        )

        assert recorder._device_rows[0][7] is None
    finally:
        recorder.close()


def test_pellet_stimulus_lines_are_labeled_for_tone_confirmation():
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    try:
        recorder._on_device_message(
            SystemStatusMessageKind.STIMULUS_INPUTS,
            [True, False, True, False],
            10.0,
            100.0,
        )

        payload = json.loads(recorder._device_rows[0][-1])
        assert payload == {
            "tone1": True,
            "tone2": False,
            "stim2": True,
            "stim3": False,
        }
    finally:
        recorder.close()


def test_structured_device_ledger_captures_decoded_input_and_output():
    handler = _EventSource("decoded_message_received")
    hardware = _EventSource("device_event")
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(
        object(),
        laser,
        system_message_handler=handler,
        hardware_model=hardware,
    )
    recorder._armed = True
    try:
        handler.decoded_message_received(
            SystemStatusMessageKind.STIMULUS_INPUTS,
            {"tone1": True},
            10.0,
            100.0,
        )
        hardware.device_event(
            "outbound",
            SystemCommandKind.PLAY_TONE,
            (7000, 100),
            "token-1",
            "PELLET_DEVICE",
            10.1,
            100.1,
        )
        handler.decoded_message_received(
            SystemStatusMessageKind.ACKNOWLEDGE,
            ("token-1", 10.15),
            10.2,
            100.2,
        )

        rows = tuple(recorder._device_rows)

        assert len(rows) == 3
        assert rows[0][2:5] == ("inbound", "STIMULUS_INPUTS", "")
        assert json.loads(rows[0][-1]) == {"tone1": True}
        assert rows[1][2:6] == (
            "outbound",
            "PLAY_TONE",
            "PELLET_DEVICE",
            "token-1",
        )
        assert json.loads(rows[1][-1]) == [7000, 100]
        assert rows[2][2:6] == (
            "inbound",
            "ACKNOWLEDGE",
            "",
            "token-1",
        )
    finally:
        recorder.close()


def test_device_events_are_buffered_before_record_is_armed():
    handler = _EventSource("decoded_message_received")
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(
        object(),
        laser,
        system_message_handler=handler,
    )
    try:
        handler.decoded_message_received(
            SystemStatusMessageKind.STIMULUS_INPUTS,
            {"tone1": True},
            10.0,
            100.0,
        )

        assert len(recorder._device_rows) == 1
        assert recorder._device_rows[0][3] == "STIMULUS_INPUTS"
    finally:
        recorder.close()


def test_laser_output_state_is_preserved_as_a_structured_event():
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    recorder._armed = True
    try:
        laser.trace_received(SimpleNamespace(
            x_values=(0.0,),
            command_volts=(),
            diode_volts=(),
            command_copy_volts=(),
            channel_id=SimpleNamespace(value=1),
            source="output state",
            output_name="shutter_open",
            output_value=1.0,
        ))

        assert recorder._laser_rows[0][-2:] == ("shutter_open", 1.0)
    finally:
        recorder.close()


def test_session_outputs_are_clipped_to_camera_boundaries(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=1,
    )
    start_perf = 10.0
    end_perf = 12.0
    nidaq_chunk = (
        np.arange(4, dtype=np.int64),
        np.array([9.9, 10.0, 11.0, 12.1]),
        np.array([99.9, 100.0, 101.0, 102.1]),
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        ("force",),
        1000.0,
        1,
        0,
        0,
    )

    SessionDataRecorder._write_session(
        project,
        start_perf,
        100.0,
        end_perf,
        (
            (9.9, 99.9, 0, 1, 2, 3),
            (10.0, 100.0, 1, 4, 5, 6),
            (12.1, 102.1, 0, 7, 8, 9),
        ),
        (
            (11.0, 101.0, "feedback", 0, "", 1.0, 2.0, 3.0),
        ),
        (
            (9.9, 99.9, "before"),
            (10.5, 100.5, "inside"),
            (12.1, 102.1, "after"),
        ),
        (nidaq_chunk,),
    )

    session_dir = tmp_path / "20260102" / "test" / "session001"
    with (session_dir / "streams" / "device.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["offset_seconds"]) == 0.0
    assert rows[0]["direction"] == "inbound"
    assert rows[0]["kind"] == "MEASUREMENT_SAMPLE"
    assert json.loads(rows[0]["payload_json"])["switch"] == 1

    with (session_dir / "streams" / "laser.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["offset_seconds"]) == 1.0

    with h5py.File(session_dir / "streams" / "nidaq.h5") as stream:
        assert stream.attrs["alignment"] == "first sample at or after primary camera first frame"
        assert stream["sample_index"][:].tolist() == [1, 2]
        assert stream["offset_seconds"][:].tolist() == [0.0, 1.0]
        assert stream["values"][:].tolist() == [[2.0, 3.0]]

    text = (session_dir / "logs" / "session.log").read_text()
    assert "inside" in text
    assert "before" not in text
    assert "after" not in text

    alignment = json.loads(
        (session_dir / "streams" / "alignment.json").read_text()
    )
    assert alignment["canonicalBoundary"]["startPerfTime"] == start_perf
    assert alignment["canonicalBoundary"]["endPerfTime"] == end_perf
    assert alignment["streams"]["nidaq"]["sampleCount"] == 2
    assert alignment["streams"]["nidaq"]["firstOffsetSeconds"] == 0.0
    assert alignment["streams"]["device"]["sampleCount"] == 1
    assert alignment["streams"]["laser"]["lastOffsetSeconds"] == 1.0
    assert alignment["streams"]["logs"]["sampleCount"] == 1


def test_empty_session_streams_still_have_alignment_metadata(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=2,
    )
    SessionDataRecorder._write_session(
        project,
        20.0,
        200.0,
        21.0,
        (),
        (),
        (),
        (),
    )

    streams = tmp_path / "20260102" / "test" / "session002" / "streams"
    alignment = json.loads((streams / "alignment.json").read_text())
    for stream in alignment["streams"].values():
        assert stream["sampleCount"] == 0
        assert stream["firstPerfTime"] is None
        assert stream["lastPerfTime"] is None
    with h5py.File(streams / "nidaq.h5") as output:
        assert output["sample_index"].size == 0
        assert output.attrs["recording_start_perf"] == 20.0
        assert output.attrs["recording_end_perf"] == 21.0


def test_camera_and_tone_edges_are_correlated_on_nidaq_timeline(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=3,
    )
    sample_rate = 1000.0
    indices = np.arange(100, 108, dtype=np.int64)
    perf = 9.999 + np.arange(8, dtype=np.float64) / sample_rate
    chunk = (
        indices,
        perf,
        100.0 + (perf - 10.0),
        np.array(
            (
                (0, 1, 0, 0, 1, 0, 0, 0),
                (0, 0, 0, 0, 1, 1, 0, 0),
            ),
            dtype=np.float32,
        ),
        ("cam_frames", "tone1"),
        sample_rate,
        1,
        0,
        0,
    )
    boundary = SessionBoundary(
        session_id=project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    )
    device_rows = (
        (
            10.0025,
            100.0025,
            "inbound",
            "STIMULUS_INPUTS",
            "PELLET_DEVICE",
            "status-1",
            None,
            None,
            json.dumps({"tone1": True}),
        ),
    )

    result = SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        10.006,
        device_rows,
        (),
        (),
        (chunk,),
        boundary=boundary,
    )

    camera = result["cameraNidaqAlignment"]
    assert camera["status"] == "matched"
    assert camera["confidence"] == "hardware_edge"
    assert camera["matchedSampleIndex"] == 101
    assert camera["signedOffsetSeconds"] == 0.0
    assert camera["edgePolarity"] == "rising"
    tone = result["toneConfirmation"]
    assert tone["status"] == "complete"
    assert tone["matched"][0]["channel"] == "tone1"
    assert tone["matched"][0]["sampleIndex"] == 104
    assert tone["matched"][0]["latencySeconds"] == pytest.approx(0.0005)
    assert tone["matched"][0]["physicalEventPerfTime"] == pytest.approx(10.003)
    assert tone["matched"][0]["recordingOffsetSeconds"] == pytest.approx(0.003)
    assert tone["matched"][0]["wallTimeUnixSeconds"] == pytest.approx(100.003)
    assert tone["matched"][0]["wallTimeUtc"] == "1970-01-01T00:01:40.003000Z"
    observation = tone["matched"][0]["observations"][0]
    assert observation["eventRecordingOffsetSeconds"] == pytest.approx(0.0025)
    assert observation["eventWallTimeUnixSeconds"] == pytest.approx(100.0025)
    frame = tone["matched"][0]["frameAssociation"]
    assert frame["status"] == "matched"
    assert frame["frameId"] == 44
    assert frame["recordedFrameIndex"] == 2
    assert frame["relation"] == "first_recorded_frame_at_or_after_event"
    assert frame["frameStartPerfTime"] == pytest.approx(10.003)
    assert frame["frameStartRecordingOffsetSeconds"] == pytest.approx(0.003)
    assert frame["frameStartWallTimeUnixSeconds"] == pytest.approx(100.003)
    assert frame["eventToFrameStartSeconds"] == pytest.approx(0.0)
    assert frame["nearestFrameId"] == 44
    assert frame["nearestFrameSignedOffsetSeconds"] == pytest.approx(0.0)
    assert frame["method"].startswith("nidaq_same_task_tone_edge")

    alignment = json.loads(
        (
            tmp_path
            / "20260102"
            / "test"
            / "session003"
            / "streams"
            / "alignment.json"
        ).read_text()
    )
    assert alignment["canonicalBoundary"]["primaryFrameId"] == 42
    assert alignment["canonicalBoundary"]["nidaqSampleIndex"] == 101
    assert alignment["schemaVersion"] == 2


def test_all_structured_events_receive_recorded_frame_alignment(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=8,
    )
    session_dir = Path(project.get_session_path().location)
    session_dir.mkdir(parents=True, exist_ok=True)
    timing_path = Path(project.get_frame_timing_path())
    timing_path.write_text(
        "frame_id,frame_when,frame_present_primary,"
        "frame_present_secondary,utc_when\n"
        "100,1,1,1,100.000\n"
        # The writer timestamp deliberately precedes the hardware exposure.
        # Generic events must use the NI exposure timeline when it is present.
        "101,2,1,1,100.004\n"
        "102,3,1,1,100.020\n",
        encoding="utf-8",
    )
    tracking_dir = session_dir / "streams" / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = tracking_dir / "trial_000001_attempt_001.json"
    tracking_path.write_text(
        json.dumps({
            "schemaVersion": 1,
            "metadataGenerationId": project.short_id,
            "identity": {},
            "window": {"startPerf": 10.005, "endPerf": 10.015},
            "pelletState": {},
        }),
        encoding="utf-8",
    )
    boundary = SessionBoundary(
        session_id=project.short_id,
        primary_camera="left",
        primary_frame_id=100,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1.0,
    )
    device_rows = ((
        10.005,
        100.005,
        "inbound",
        "PELLET_LOAD",
        "PELLET_DEVICE",
        "device-1",
        None,
        None,
        json.dumps(104.0),
    ),)
    laser_rows = ((
        10.015,
        100.015,
        "state",
        "laser_1",
        "device",
        None,
        None,
        None,
        "enabled",
        1.0,
    ),)
    trial_records = ({
        "send_perf_time": 10.005,
        "send_wall_time": 100.005,
        "send_ack_perf_time": 10.015,
        "send_ack_wall_time": 100.015,
    },)
    event_rows = ((
        10.005,
        100.005,
        "pelletSendBegin",
        1205,
        10_005_000_000,
        json.dumps({"trial": 1}),
        "event_info_perf_counter_ns",
    ),)
    sample_rate = 1000.0
    perf = 9.999 + np.arange(25, dtype=np.float64) / sample_rate
    indices = np.arange(25, dtype=np.int64)
    cam_frames = np.zeros(25, dtype=np.float32)
    cam_frames[1:7] = 1.0
    cam_frames[14:21] = 1.0
    nidaq_chunks = ((
        indices,
        perf,
        100.0 + (perf - 10.0),
        cam_frames[np.newaxis, :],
        ("cam_frames",),
        sample_rate,
        1,
        0,
        0,
    ),)

    SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        10.02,
        device_rows,
        laser_rows,
        (),
        nidaq_chunks,
        boundary=boundary,
        trial_records=trial_records,
        event_rows=event_rows,
    )

    with (session_dir / "streams" / "device.csv").open(newline="") as stream:
        device = next(csv.DictReader(stream))
    assert device["frame_id"] == "101"
    assert device["recorded_frame_index"] == "1"
    assert float(device["event_to_frame_start_seconds"]) == pytest.approx(0.001)
    assert device["alignment_confidence"] == "host_timestamp"
    assert "nidaq_camera_exposure_timeline" in device["alignment_method"]

    with (session_dir / "streams" / "events.csv").open(newline="") as stream:
        event = next(csv.DictReader(stream))
    assert event["event_name"] == "pelletSendBegin"
    assert event["frame_id"] == "101"
    assert event["recorded_frame_index"] == "1"
    assert event["timestamp_method"] == "event_info_perf_counter_ns"

    with (session_dir / "streams" / "laser.csv").open(newline="") as stream:
        laser = next(csv.DictReader(stream))
    assert laser["frame_id"] == "102"
    assert laser["recorded_frame_index"] == "2"

    trial = json.loads(
        (session_dir / "streams" / "trials.jsonl").read_text().strip()
    )
    assert trial["event_alignment"]["send"]["frameAssociation"]["frameId"] == 101
    assert (
        trial["event_alignment"]["sendAcknowledged"]["frameAssociation"]["frameId"]
        == 102
    )

    tracking = json.loads(tracking_path.read_text())
    assert tracking["eventAlignment"]["windowStart"]["frameAssociation"]["frameId"] == 101
    assert tracking["eventAlignment"]["windowEnd"]["frameAssociation"]["frameId"] == 102

    alignment = json.loads(
        (session_dir / "streams" / "alignment.json").read_text()
    )
    assert "all discrete structured" in alignment["eventAlignmentContract"]["scope"]


def test_tone_correlation_uses_session_pulses_and_groups_can_observations():
    sample_rate = 1000.0
    perf = 9.998 + np.arange(155, dtype=np.float64) / sample_rate
    indices = np.arange(perf.size, dtype=np.int64)
    tone1 = np.zeros(perf.size, dtype=np.float32)
    tone1[1] = 1.0  # Short artifact before Record.
    tone1[5:8] = 1.0  # Valid 3 ms pulse beginning at 10.003.
    tone1[10] = 1.0  # Short artifact within the saved session.
    tone1[153] = 1.0  # Short artifact after Stop.
    chunks = ((
        indices,
        perf,
        100.0 + (perf - 10.0),
        tone1[np.newaxis, :],
        ("tone1",),
        sample_rate,
        1,
        0,
        0,
    ),)
    rows = (
        (
            10.0025, 100.0025, "outbound", "PLAY_TONE",
            "PELLET_DEVICE", "sequence-1", None, None,
            json.dumps([5000, 300]),
        ),
        (
            10.004, 100.004, "inbound", "TONE_STATUS",
            "PELLET_DEVICE", "tone-1", None, None,
            json.dumps({"frequency_hz": 5000, "time_remaining_ms": 300}),
        ),
        (
            10.108, 100.108, "inbound", "STIMULUS_INPUTS",
            "PELLET_DEVICE", "gpio-1", None, None,
            json.dumps({"tone1": True}),
        ),
    )

    result = SessionDataRecorder._correlate_tone_confirmations(
        rows,
        chunks,
        start_perf=10.0,
        end_perf=10.15,
    )

    assert result["status"] == "complete"
    assert result["signalQuality"] == "artifacts_detected"
    assert result["artifactCount"] == 1
    assert result["artifacts"][0]["perfTime"] == pytest.approx(10.008)
    assert result["unmatchedEdges"] == []
    assert result["unmatchedEvents"] == []
    assert len(result["matched"]) == 1
    match = result["matched"][0]
    assert match["kind"] == "TONE_STATUS"
    assert match["edgePerfTime"] == pytest.approx(10.003)
    assert match["alignedEventPerfTime"] == pytest.approx(10.003)
    assert match["pulseDurationSeconds"] == pytest.approx(0.003)
    assert {item["kind"] for item in match["observations"]} == {
        "PLAY_TONE",
        "TONE_STATUS",
        "STIMULUS_INPUTS",
    }


def test_idle_tone_status_rearms_same_frequency_correlation():
    sample_rate = 1000.0
    perf = 10.0 + np.arange(20, dtype=np.float64) / sample_rate
    tone2 = np.zeros(perf.size, dtype=np.float32)
    tone2[2:5] = 1.0
    tone2[12:15] = 1.0
    chunks = ((
        np.arange(perf.size, dtype=np.int64),
        perf,
        100.0 + (perf - 10.0),
        tone2[np.newaxis, :],
        ("tone2",),
        sample_rate,
        1,
        0,
        0,
    ),)
    tone_status = lambda when, frequency, remaining: (
        when,
        100.0 + (when - 10.0),
        "inbound",
        "TONE_STATUS",
        "PELLET_DEVICE",
        "tone",
        None,
        None,
        json.dumps({
            "frequency_hz": frequency,
            "time_remaining_ms": remaining,
        }),
    )
    rows = (
        tone_status(10.003, 6000, 300),
        tone_status(10.006, 0, 0),
        tone_status(10.013, 6000, 300),
    )

    result = SessionDataRecorder._correlate_tone_confirmations(
        rows,
        chunks,
        start_perf=10.0,
        end_perf=10.019,
    )

    assert result["status"] == "complete"
    assert len(result["matched"]) == 2
    assert all(match["kind"] == "TONE_STATUS" for match in result["matched"])


def test_camera_alignment_matches_falling_square_wave_transition():
    sample_rate = 1000.0
    indices = np.arange(200, 207, dtype=np.int64)
    perf = 19.998 + np.arange(7, dtype=np.float64) / sample_rate
    chunk = (
        indices,
        perf,
        200.0 + (perf - 20.0),
        np.array(((1, 1, 0, 0, 1, 1, 0),), dtype=np.float32),
        ("cam_frames",),
        sample_rate,
        2,
        0,
        0,
    )
    boundary = SessionBoundary(
        session_id="trial-square-wave",
        primary_camera="left",
        primary_frame_id=44,
        start_perf_time=20.0,
        start_wall_time=200.0,
        camera_when=2_000_000.0,
    )

    alignment = SessionDataRecorder._match_camera_nidaq_edge(
        boundary,
        20.0,
        (chunk,),
    )

    assert alignment["status"] == "matched"
    assert alignment["confidence"] == "hardware_edge"
    assert alignment["matchedSampleIndex"] == 202
    assert alignment["signedOffsetSeconds"] == 0.0
    assert alignment["edgePolarity"] == "falling"


def test_missing_cam_frames_is_explicitly_host_estimated():
    chunk = (
        np.arange(3, dtype=np.int64),
        np.array((9.999, 10.0, 10.001)),
        np.array((99.999, 100.0, 100.001)),
        np.zeros((1, 3), dtype=np.float32),
        ("barcode",),
        1000.0,
        1,
        0,
        0,
    )

    alignment = SessionDataRecorder._match_camera_nidaq_edge(
        None,
        10.0,
        (chunk,),
    )

    assert alignment["status"] == "host_estimated"
    assert alignment["confidence"] == "host_estimated"
    assert alignment["matchedSampleIndex"] == 1


def test_device_event_overrun_marks_session_incomplete(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=4,
    )

    result = SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        11.0,
        (),
        (),
        (),
        (),
        device_event_overruns=3,
    )

    assert result["sessionComplete"] is False
    assert "overran by 3 event(s)" in result["incompleteReasons"][0]
    alignment = json.loads(
        (
            tmp_path
            / "20260102"
            / "test"
            / "session004"
            / "streams"
            / "alignment.json"
        ).read_text()
    )
    assert alignment["sessionComplete"] is False
    assert alignment["deviceEventOverruns"] == 3


def test_enabled_source_manifest_contains_final_paths_counts_and_health(
    tmp_path,
):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=5,
    )
    session_dir = Path(project.get_session_path().location)
    session_dir.mkdir(parents=True, exist_ok=True)
    camera_path = session_dir / "session005_left.mp4"
    camera_path.touch()
    chunk = (
        np.arange(3, dtype=np.int64),
        np.array((10.0, 10.001, 10.002)),
        np.array((100.0, 100.001, 100.002)),
        np.zeros((1, 3), dtype=np.float32),
        ("barcode",),
        1000.0,
        1,
        2,
        4,
    )
    sources = (
        {
            "id": "camera.left",
            "kind": "camera",
            "path": "placeholder",
            "runtimeState": "ready",
        },
        {
            "id": "nidaq.barcode",
            "kind": "nidaq_digital",
            "path": "streams/nidaq.h5",
            "runtimeState": "ready",
        },
        {
            "id": "device",
            "kind": "decoded_can_and_device_events",
            "path": "streams/device.csv",
            "runtimeState": "ready",
        },
    )

    result = SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        10.002,
        (),
        (),
        (),
        (chunk,),
        source_manifest=sources,
        source_results={
            "camera.left": {
                "sampleCount": 3,
                "path": camera_path.relative_to(session_dir).as_posix(),
                "failure": "",
            },
        },
    )

    manifest = {
        source["id"]: source
        for source in result["enabledSources"]
    }
    assert manifest["camera.left"]["path"] == "session005_left.mp4"
    assert manifest["camera.left"]["sampleCount"] == 3
    assert manifest["camera.left"]["firstOffsetSeconds"] == 0.0
    assert manifest["camera.left"]["lastOffsetSeconds"] == pytest.approx(0.002)
    assert (
        manifest["camera.left"]["timingSource"]
        == "canonical_camera_boundary"
    )
    assert manifest["camera.left"]["persistenceStatus"] == "written"
    assert manifest["nidaq.barcode"]["sampleCount"] == 3
    assert manifest["nidaq.barcode"]["firstOffsetSeconds"] == 0.0
    assert manifest["nidaq.barcode"]["gapCount"] == 2
    assert manifest["nidaq.barcode"]["overrunCount"] == 4
    assert len(manifest["nidaq.barcode"]["warnings"]) == 2
    assert result["sessionComplete"] is True
    assert manifest["device"]["sampleCount"] == 0
    assert manifest["device"]["persistenceStatus"] == "written"


def test_trial_ledger_is_written_on_the_canonical_session_timeline(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=6,
    )
    records = (
        {
            "session_id": project.short_id,
            "operation_id": "send-1",
            "trial_id": 10,
            "attempt_id": 1,
            "attempt_label": "10.1",
            "send_perf_time": 10.25,
            "send_wall_time": 100.25,
            "send_ack_perf_time": 10.5,
            "outcome": "success",
        },
        {
            "session_id": project.short_id,
            "operation_id": "outside",
            "trial_id": 11,
            "attempt_id": 1,
            "attempt_label": "11.1",
            "send_perf_time": 12.1,
            "send_wall_time": 102.1,
            "outcome": "incomplete",
        },
    )
    summary = {
        "physical_attempts": 1,
        "trials_completed": 1,
    }

    result = SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        12.0,
        (),
        (),
        (),
        (),
        source_manifest=({
            "id": "trials",
            "kind": "pellet_trial_ledger",
            "path": "streams/trials.jsonl",
            "runtimeState": "ready",
        },),
        trial_records=records,
        trial_summary=summary,
    )

    streams = tmp_path / "20260102" / "test" / "session006" / "streams"
    written = tuple(
        json.loads(line)
        for line in (streams / "trials.jsonl").read_text().splitlines()
    )
    assert len(written) == 1
    assert written[0]["attempt_label"] == "10.1"
    assert written[0]["send_offset_seconds"] == 0.25
    written_summary = json.loads((streams / "trial_summary.json").read_text())
    assert written_summary == {
        **summary,
        "metadata_generation_id": f"{project.short_id}-legacy",
    }

    alignment = json.loads((streams / "alignment.json").read_text())
    assert alignment["streams"]["trials"]["sampleCount"] == 1
    assert alignment["streams"]["trials"]["firstOffsetSeconds"] == 0.25
    trial_source = result["enabledSources"][0]
    assert trial_source["id"] == "trials"
    assert trial_source["sampleCount"] == 1
    assert trial_source["persistenceStatus"] == "written"


def test_post_analysis_trial_ledger_replaces_pending_records_atomically(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=7,
    )
    SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        12.0,
        (), (), (), (),
        trial_records=({
            "session_id": project.short_id,
            "operation_id": "send-1",
            "trial_id": 1,
            "attempt_id": 1,
            "attempt_label": "1.1",
            "send_perf_time": 10.25,
            "outcome": "pending_analysis",
        },),
        trial_summary={"pending_analysis_attempts": 1},
    )

    SessionDataRecorder.update_persisted_trial_ledger(
        project,
        ({
            "session_id": project.short_id,
            "operation_id": "send-1",
            "trial_id": 1,
            "attempt_id": 1,
            "attempt_label": "1.1",
            "send_perf_time": 10.25,
            "outcome": "success",
            "reach_count": 1,
        },),
        {"pending_analysis_attempts": 0, "scored_trials": 1},
    )

    streams = tmp_path / "20260102" / "test" / "session007" / "streams"
    record = json.loads((streams / "trials.jsonl").read_text().strip())
    assert record["outcome"] == "success"
    assert record["reach_count"] == 1
    assert record["send_offset_seconds"] == 0.25
    assert json.loads((streams / "trial_summary.json").read_text()) == {
        "metadata_generation_id": f"{project.short_id}-legacy",
        "pending_analysis_attempts": 0,
        "scored_trials": 1,
    }
    assert not tuple(streams.glob("*.tmp"))


def test_auxiliary_generation_is_shared_by_alignment_trials_and_manifest(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=8,
    )
    generation_id = f"{project.short_id}-g4"

    result = SessionDataRecorder._write_session(
        project,
        10.0,
        100.0,
        11.0,
        (), (), (), (),
        trial_records=({
            "session_id": project.short_id,
            "operation_id": "send-1",
            "trial_id": 1,
            "attempt_id": 1,
            "attempt_label": "1.1",
            "send_perf_time": 10.25,
            "outcome": "success",
        },),
        trial_summary={"scored_trials": 1},
        metadata_generation_id=generation_id,
    )

    streams = Path(project.get_session_path().location) / "streams"
    alignment = json.loads((streams / "alignment.json").read_text())
    trial = json.loads((streams / "trials.jsonl").read_text())
    summary = json.loads((streams / "trial_summary.json").read_text())
    manifest = json.loads((streams / "stream_manifest.json").read_text())

    assert result["metadataGenerationId"] == generation_id
    assert alignment["metadataGenerationId"] == generation_id
    assert trial["metadata_generation_id"] == generation_id
    assert summary["metadata_generation_id"] == generation_id
    assert manifest["metadataGenerationId"] == generation_id


def test_failed_auxiliary_finalization_retains_snapshot_for_retry(
    monkeypatch,
    tmp_path,
):
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=9,
    )
    generation_id = f"{project.short_id}-g1"
    staging = (
        Path(project.get_session_path().location)
        / ".staging"
        / generation_id
        / "streams"
    )
    staging.mkdir(parents=True)
    snapshot = {
        "project": project,
        "metadata_generation_id": generation_id,
    }
    recorder._pending_finalization = snapshot
    attempts = []

    def fail_then_succeed(**received):
        attempts.append(received)
        if len(attempts) == 1:
            raise OSError("temporary write failure")
        return {"sessionComplete": True}

    monkeypatch.setattr(recorder, "_write_session", fail_then_succeed)
    try:
        with pytest.raises(RuntimeError, match="retained for retry"):
            recorder._publish_pending_finalization(max_attempts=1)
        assert recorder._pending_finalization is snapshot

        assert recorder.retry_pending_finalization() == {"sessionComplete": True}
        assert recorder._pending_finalization is None
        assert not staging.parent.parent.exists()
    finally:
        recorder.close()


def test_nidaq_stop_timeout_remains_retryable(monkeypatch, tmp_path):
    laser = _EventSource("trace_received")
    laser.configuration = SimpleNamespace(backend="disabled")
    recorder = SessionDataRecorder(
        SimpleNamespace(timing_plan=None),
        laser,
    )
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=10,
    )
    recorder._armed = True
    recorder._project = project
    recorder._start_perf = 10.0
    recorder._start_wall = 100.0
    recorder._metadata_generation_id = f"{project.short_id}-g1"
    stop_calls = 0

    def stop_nidaq():
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            raise RuntimeError("NI-DAQ recorder thread did not stop")

    monkeypatch.setattr(recorder, "_stop_nidaq_thread", stop_nidaq)
    monkeypatch.setattr(
        recorder,
        "_write_session",
        lambda **_snapshot: {"sessionComplete": True},
    )
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            recorder.stop(12.0)

        assert recorder.has_pending_finalization
        assert recorder.pending_finalization_session_id == project.short_id
        assert recorder.retry_pending_finalization() == {"sessionComplete": True}
        assert not recorder.has_pending_finalization
        assert stop_calls == 2
    finally:
        recorder.close()


def test_nidaq_poll_reuses_preallocated_ring_scratch():
    destinations = []

    class Ring:
        channel_names = ("barcode",)
        capacity = 16

        @staticmethod
        def copy_since(_last, destination):
            destinations.append(destination)
            return None

    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(SimpleNamespace(sample_ring=Ring()), laser)
    try:
        recorder._copy_nidaq_once()
        recorder._copy_nidaq_once()
        assert destinations[0] is destinations[1]
    finally:
        recorder.close()


def test_live_tone_edge_rejects_one_sample_glitch_and_keeps_exact_onset():
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    recorder._start_perf = 10.0
    sample_rate = 10_000.0
    try:
        glitch = np.zeros((2, 30), dtype=np.float32)
        glitch[1, 5] = 1.0
        assert recorder._validated_live_tone_edges(
            np.arange(30),
            10.0 + np.arange(30) / sample_rate,
            100.0 + np.arange(30) / sample_rate,
            glitch,
            ("tone1", "tone2"),
            sample_rate,
        ) == ()

        pulse = np.zeros((2, 40), dtype=np.float32)
        pulse[1, 4:30] = 1.0
        edges = recorder._validated_live_tone_edges(
            np.arange(30, 70),
            10.003 + np.arange(40) / sample_rate,
            100.003 + np.arange(40) / sample_rate,
            pulse,
            ("tone1", "tone2"),
            sample_rate,
        )

        assert edges == ({
            "channel": "tone2",
            "perf_time": pytest.approx(10.0034),
            "wall_time": pytest.approx(100.0034),
            "sample_index": 34,
        },)
    finally:
        recorder.close()


def test_live_tone_edge_confirmation_can_span_nidaq_chunks():
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(object(), laser)
    recorder._start_perf = 10.0
    sample_rate = 10_000.0
    try:
        first = np.ones((1, 12), dtype=np.float32)
        assert recorder._validated_live_tone_edges(
            np.arange(100, 112),
            10.0 + np.arange(12) / sample_rate,
            100.0 + np.arange(12) / sample_rate,
            first,
            ("tone2",),
            sample_rate,
        ) == ()

        second = np.ones((1, 12), dtype=np.float32)
        edges = recorder._validated_live_tone_edges(
            np.arange(112, 124),
            10.0012 + np.arange(12) / sample_rate,
            100.0012 + np.arange(12) / sample_rate,
            second,
            ("tone2",),
            sample_rate,
        )

        assert edges[0]["sample_index"] == 100
        assert edges[0]["perf_time"] == pytest.approx(10.0)
        assert recorder._validated_live_tone_edges(
            np.arange(124, 130),
            10.0024 + np.arange(6) / sample_rate,
            100.0024 + np.arange(6) / sample_rate,
            np.ones((1, 6), dtype=np.float32),
            ("tone2",),
            sample_rate,
        ) == ()
    finally:
        recorder.close()


def test_nidaq_spool_is_incremental_and_final_output_is_boundary_clipped(tmp_path):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=9,
    )
    ring = SimpleNamespace(
        channel_names=("barcode",),
        capacity=16,
        sample_rate_hz=1000.0,
    )
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(SimpleNamespace(sample_ring=ring), laser)
    try:
        with recorder._lock:
            recorder._project = project
            recorder._metadata_generation_id = f"{project.short_id}-g1"
            recorder._open_nidaq_spool_locked()
            recorder._append_nidaq_spool_locked(
                np.arange(4, dtype=np.int64),
                np.array((9.999, 10.0, 10.001, 10.002)),
                np.array((99.999, 100.0, 100.001, 100.002)),
                np.array(((0.0, 1.0, 0.0, 1.0),), dtype=np.float32),
                ring.channel_names,
                SimpleNamespace(
                    epoch=1,
                    source_perf_time=10.002,
                    source_wall_time=100.002,
                    gap_count=0,
                    overrun_samples=0,
                ),
            )
            recorder._close_nidaq_spool_locked()
            snapshot = recorder._snapshot_nidaq_locked()

        result = SessionDataRecorder._write_session(
            project,
            10.0,
            100.0,
            10.001,
            (), (), (), snapshot,
            source_manifest=({
                "id": "nidaq.barcode",
                "kind": "nidaq_digital",
                "path": "streams/nidaq.h5",
                "runtimeState": "ready",
            },),
            metadata_generation_id=f"{project.short_id}-g1",
        )

        with h5py.File(
            Path(project.get_session_path().location) / "streams" / "nidaq.h5",
            "r",
        ) as output:
            assert output["sample_index"][:].tolist() == [1, 2]
            assert output["values"][:].tolist() == [[1.0, 0.0]]
            assert output.attrs["collection_error_count"] == 0
        assert result["sessionComplete"] is True
        assert recorder._nidaq_chunks == []
    finally:
        recorder.abort()


@pytest.mark.parametrize(
    ("perf", "end_perf", "expected_complete", "warning_fragment"),
    (
        (np.array((10.1, 10.9)), 11.0, True, "coverage missing"),
        (np.array((16.0, 16.1)), 17.0, False, "start boundary coverage"),
        (np.empty(0), 11.0, False, "zero samples"),
    ),
)
def test_nidaq_coverage_classifies_warning_and_critical_boundaries(
    tmp_path, perf, end_perf, expected_complete, warning_fragment,
):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=10,
    )
    count = len(perf)
    chunk = (
        np.arange(count, dtype=np.int64),
        perf,
        100.0 + perf - 10.0,
        np.zeros((1, count), dtype=np.float32),
        ("barcode",),
        1000.0,
        1,
        0,
        0,
    )
    result = SessionDataRecorder._write_session(
        project, 10.0, 100.0, end_perf, (), (), (), (chunk,),
        source_manifest=({
            "id": "nidaq.barcode",
            "kind": "nidaq_digital",
            "path": "streams/nidaq.h5",
            "runtimeState": "ready",
        },),
    )
    source = result["enabledSources"][0]
    messages = " ".join((*source["warnings"], source["failure"] or ""))
    assert result["sessionComplete"] is expected_complete
    assert warning_fragment in messages
