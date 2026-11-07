"""A manual Run Pulse fired while a session records is kept in it.

Ben, 2026-09-30: Run Pulse stays allowed while recording, and each one is
written into the session as a marked manual laser event, which the session
validation tool accepts. It was kept only as its waveform: unmarked, and
stamped as that was told, after the train had ended. It goes the one way
every laser trace does, LaserModel.trace_received into the recorder, which
keeps nothing unless it is armed.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from autotrainer.core import ProjectInfo
from autotrainer.device import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserPulseTrain,
    LaserSystemConfiguration,
)
from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.session_validation import ValidationProfile, validate_session


MANUAL = {"profile_id": "burst", "profile_revision": 2, "trigger_mode": "internal"}
TRAIN = LaserPulseTrain(
    channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=2.0,
    baseline_ms=1.0)


def _model():
    model = LaserModel()
    model.configure_null(LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=1,
            analog_output="Dev1/ao0",
            diode_input="Dev1/ai0",
            shutter_output="Dev1/port0/line0",
        ),
    ), backend="null", hardware_timed=True, sample_rate_hz=10_000))
    return model


def _recorder(model, *, armed):
    recorder = SessionDataRecorder(object(), model)
    recorder._armed = armed
    return recorder


def test_a_manual_run_pulse_while_recording_is_kept_as_a_marked_event(tmp_path):
    model = _model()
    recorder = _recorder(model, armed=True)
    try:
        start = time.perf_counter()
        model.run_pulse_train(TRAIN, manual_context=MANUAL)
        end = time.perf_counter()
        rows = tuple(recorder._laser_rows)
    finally:
        recorder.close()

    project = ProjectInfo(
        root=str(tmp_path), device_id="test", when=datetime(2026, 9, 30, 12, 0, 0),
        session=1)
    # NI samples every millisecond, from before the recording to after it.
    nidaq_perf = start - 0.5 + np.arange(int((end - start + 1.0) * 1000)) / 1000.0
    nidaq_chunk = (
        1000 + np.arange(nidaq_perf.size, dtype=np.int64),
        nidaq_perf,
        100.0 + nidaq_perf,
        np.zeros((1, nidaq_perf.size), dtype=np.float32),
        ("force",),
        1000.0,
        1,
        0,
        0,
    )
    SessionDataRecorder._write_session(
        project, start - 0.25, 100.0, end + 0.25, (), rows, (), (nidaq_chunk,))
    session_dir = Path(project.get_session_path().location)
    with (session_dir / "streams" / "laser.csv").open(newline="") as stream:
        written = list(csv.DictReader(stream))

    perf = [float(row["perf_time"]) for row in written]
    assert perf == sorted(perf)
    requested, completed = [row for row in written if row["source"] == "manual pulse"]
    assert (requested["event"], completed["event"]) == ("requested", "completed")
    assert requested["operation_id"].startswith("manual-")
    assert completed["operation_id"] == requested["operation_id"]
    assert json.loads(requested["context_json"]) == {
        "manual": True,
        "laser_channel_id": 1,
        "profile_id": "burst",
        "profile_revision": 2,
        "amplitude_volts": 2.5,
        "trigger_mode": "internal",
        "route": {"trigger_terminal": "", "trigger_edge": None, "stim_line": None},
    }
    # The waveform starts at the request's time: they can be joined on it.
    waveform = [row for row in written if row["source"] == "internal pulse"]
    assert waveform and waveform[0]["perf_time"] == requested["perf_time"]
    assert {(row["timestamp_method"], row["timing_confidence"]) for row in waveform} == {
        ("manual_pulse_call_perf_counter", "before_output")}
    # The NI sample the request landed within.
    landed = int(np.searchsorted(nidaq_perf, float(requested["perf_time"]), side="right")) - 1
    assert requested["nidaq_sample_index"] == str(1000 + landed)

    report = validate_session(
        session_dir, profile=ValidationProfile.FAST, selected_rules=("events.laser",))
    laser = next(item for item in report.results if item.rule_id == "events.laser")
    # A warning only: this session has no camera frames to associate.
    assert laser.status.value == "warning"
    assert laser.message.endswith("; 1 manual Run Pulse event(s) (1 completed)")


def test_nothing_is_kept_of_a_manual_run_pulse_when_no_session_records():
    model = _model()
    recorder = _recorder(model, armed=False)
    try:
        model.run_pulse_train(TRAIN, manual_context=MANUAL)

        assert recorder._laser_rows == []
    finally:
        recorder.close()
