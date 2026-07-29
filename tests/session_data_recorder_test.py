import csv
import json
from datetime import datetime
from types import SimpleNamespace

import h5py
import numpy as np

from autotrainer.core import (
    ObservableObject,
    ProjectInfo,
    SystemCommandKind,
    SystemStatusMessageKind,
)
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


class _EventSource(ObservableObject):
    LAST_FEEDBACK_SAMPLE = "last_feedback_sample"

    def __init__(self, *event_names):
        super().__init__(event_names)


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

        rows = tuple(recorder._device_rows)

        assert len(rows) == 2
        assert rows[0][2:5] == ("inbound", "STIMULUS_INPUTS", "")
        assert json.loads(rows[0][-1]) == {"tone1": True}
        assert rows[1][2:6] == (
            "outbound",
            "PLAY_TONE",
            "PELLET_DEVICE",
            "token-1",
        )
        assert json.loads(rows[1][-1]) == [7000, 100]
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

    session_dir = tmp_path / "20260102" / "test" / "trial001"
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

    streams = tmp_path / "20260102" / "test" / "trial002" / "streams"
    alignment = json.loads((streams / "alignment.json").read_text())
    for stream in alignment["streams"].values():
        assert stream["sampleCount"] == 0
        assert stream["firstPerfTime"] is None
        assert stream["lastPerfTime"] is None
    with h5py.File(streams / "nidaq.h5") as output:
        assert output["sample_index"].size == 0
        assert output.attrs["recording_start_perf"] == 20.0
        assert output.attrs["recording_end_perf"] == 21.0
