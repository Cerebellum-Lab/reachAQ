import csv
import json
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
    recorder = SessionDataRecorder(_EventSource(), object(), laser)
    monkeypatch.setattr(
        session_data_recorder.time,
        "perf_counter",
        lambda: (_ for _ in ()).throw(AssertionError("clock must not be read")),
    )
    try:
        recorder.add_current_log(100.0, "outside a recording")
    finally:
        recorder.close()


def test_structured_device_ledger_captures_decoded_input_and_output():
    analysis = _EventSource()
    handler = _EventSource("decoded_message_received")
    hardware = _EventSource("device_event")
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(
        analysis,
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
        _EventSource(),
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
    analysis = _EventSource()
    laser = _EventSource("trace_received")
    recorder = SessionDataRecorder(analysis, object(), laser)
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
                (0, 1, 0, 0, 0, 0, 0, 0),
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
    assert json.loads((streams / "trial_summary.json").read_text()) == summary

    alignment = json.loads((streams / "alignment.json").read_text())
    assert alignment["streams"]["trials"]["sampleCount"] == 1
    assert alignment["streams"]["trials"]["firstOffsetSeconds"] == 0.25
    trial_source = result["enabledSources"][0]
    assert trial_source["id"] == "trials"
    assert trial_source["sampleCount"] == 1
    assert trial_source["persistenceStatus"] == "written"
