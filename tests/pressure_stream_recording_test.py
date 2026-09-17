"""The pellet-board FSR stream: where its samples go and how they are saved.

Pressure arrives about 166 times a second across the two sensors, two orders
of magnitude above every other decoded device message.  These tests pin the
two things that follow from that: it must not share the device ring, and it
must land in a file that carries the master clock and the NI-DAQ index.
"""

import csv
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from autotrainer.core import (
    ObservableObject,
    ProjectInfo,
    SystemStatusMessageKind,
)
from tools.acquisition.model.session_data_recorder import (
    SessionDataRecorder,
    _PressureRing,
)


class _EventSource(ObservableObject):
    def __init__(self, *event_names):
        super().__init__(event_names)


def _reading(instance, counts, *, event_perf=None, sequence=3, board_us=5):
    return SimpleNamespace(
        instance=instance,
        pressure=counts,
        event_perf_time=event_perf,
        board_sequence=sequence,
        board_time_us=board_us,
        board_aligned_perf_time=event_perf,
    )


def test_pressure_readings_stay_out_of_the_shared_device_ring():
    """98% of a session's device rows were pressure, which overran the ring."""
    recorder = SessionDataRecorder(object(), _EventSource("trace_received"))
    try:
        recorder._on_device_message(
            SystemStatusMessageKind.PRESSURE_READING,
            _reading(0, 2048, event_perf=10.1),
            10.25,
            100.25,
        )
        recorder._on_device_message(
            SystemStatusMessageKind.TONE_STATUS,
            _reading(0, 0, event_perf=10.2),
            10.30,
            100.30,
        )

        kinds = [row[3] for row in recorder._device_rows]
        assert kinds == ["TONE_STATUS"]
        assert recorder._pressure.written == 1
    finally:
        recorder.close()


def test_pressure_capture_keeps_the_board_event_clock():
    recorder = SessionDataRecorder(object(), _EventSource("trace_received"))
    try:
        recorder._on_device_message(
            SystemStatusMessageKind.PRESSURE_READING,
            _reading(1, 4095, event_perf=10.1, sequence=12, board_us=4_000_000),
            10.25,
            100.25,
        )
        columns = recorder._pressure.snapshot()
        assert columns["perf_time"][0] == pytest.approx(10.1)
        assert columns["instance"][0] == 1
        assert columns["counts"][0] == 4095
        assert columns["board_sequence"][0] == 12
        assert columns["board_time_us"][0] == 4_000_000
    finally:
        recorder.close()


def test_pressure_capture_falls_back_to_host_receipt_without_a_board_clock():
    recorder = SessionDataRecorder(object(), _EventSource("trace_received"))
    try:
        recorder._on_device_message(
            SystemStatusMessageKind.PRESSURE_READING,
            _reading(0, 100, event_perf=None),
            10.25,
            100.25,
        )
        columns = recorder._pressure.snapshot()
        assert columns["perf_time"][0] == pytest.approx(10.25)
        assert columns["event_perf_time"][0] == pytest.approx(10.25)
    finally:
        recorder.close()


def test_pressure_buffer_starts_small_and_grows_only_as_needed():
    """Sizing the five-hour cap up front would cost 162 MB per app start."""
    ring = _PressureRing(3_000_000)
    assert ring.allocated == _PressureRing.INITIAL_CAPACITY

    for index in range(_PressureRing.INITIAL_CAPACITY + 1):
        ring.append((float(index), 0.0, 0, index, float(index), -1.0, -1, -1))

    assert ring.allocated == _PressureRing.INITIAL_CAPACITY * 2
    assert ring.capacity == 3_000_000
    # Growing must not disturb what was already held.
    columns = ring.snapshot()
    assert columns["counts"].size == _PressureRing.INITIAL_CAPACITY + 1
    assert list(columns["counts"][:3]) == [0, 1, 2]
    assert columns["counts"][-1] == _PressureRing.INITIAL_CAPACITY


def test_pressure_buffer_never_grows_past_its_cap():
    ring = _PressureRing(4)
    for index in range(10):
        ring.append((float(index), 0.0, 0, index, float(index), -1.0, -1, -1))
    assert ring.allocated == 4
    assert list(ring.snapshot()["counts"]) == [6, 7, 8, 9]


def test_pressure_ring_keeps_the_newest_samples_when_it_wraps():
    ring = _PressureRing(3)
    for index in range(5):
        ring.append((float(index), 0.0, 0, index, float(index), -1.0, -1, -1))

    columns = ring.snapshot()
    assert list(columns["counts"]) == [2, 3, 4]
    assert list(columns["perf_time"]) == [2.0, 3.0, 4.0]
    assert ring.written == 5


def test_pressure_nidaq_index_places_each_sample_in_its_ni_sample():
    chunks = (
        (
            np.array([100, 101, 102], dtype=np.int64),
            np.array([10.0, 10.5, 11.0]),
            np.array([100.0, 100.5, 101.0]),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            ("force",),
            2.0,
            1,
            0,
            0,
        ),
    )
    perf = np.array([9.9, 10.0, 10.4, 10.6, 11.4])

    indices = SessionDataRecorder._pressure_nidaq_indices(perf, chunks)

    # Before the first NI sample there is nothing to match, so -1 rather than a
    # guess; every later sample takes the NI sample it fell within.
    assert list(indices) == [-1, 100, 100, 101, 102]


def _columns(samples):
    names = ("perf_time", "wall_time", "instance", "counts",
             "event_perf_time", "board_aligned_perf_time",
             "board_sequence", "board_time_us")
    return {
        name: np.array([sample[position] for sample in samples])
        for position, name in enumerate(names)
    }


def _write_session_with_pressure(tmp_path, pressure_columns):
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=1,
    )
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
    result = SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (nidaq_chunk,),
        pressure_columns=pressure_columns,
    )
    session_dir = tmp_path / "20260102" / "test" / "session001"
    with (session_dir / "streams" / "pressure.csv").open(newline="") as stream:
        return result, list(csv.DictReader(stream)), session_dir


def test_pressure_csv_carries_the_master_clock_and_the_nidaq_index(tmp_path):
    columns = _columns([
        (9.9, 99.9, 0, 100, 9.9, -1.0, -1, -1),      # before the boundary
        (10.0, 100.0, 0, 2048, 10.0, 10.0, 7, 900),
        (11.0, 101.0, 1, 4095, 11.0, 11.0, 8, 901),
        (12.1, 102.1, 0, 300, 12.1, -1.0, -1, -1),   # after the boundary
    ])

    _, rows, _ = _write_session_with_pressure(tmp_path, columns)

    assert len(rows) == 2, "samples outside the camera boundary are clipped"
    assert float(rows[0]["offset_seconds"]) == pytest.approx(0.0)
    assert float(rows[1]["offset_seconds"]) == pytest.approx(1.0)
    assert rows[0]["instance"] == "0"
    assert rows[1]["instance"] == "1"
    # Raw counts are preserved; volts is the board's 12-bit/3V3 conversion.
    assert rows[1]["counts"] == "4095"
    assert float(rows[1]["volts"]) == pytest.approx(3.3)
    assert float(rows[0]["volts"]) == pytest.approx(2048 / 4095 * 3.3)
    # Each sample names the NI-DAQ sample it landed within.
    assert rows[0]["nidaq_sample_index"] == "1"
    assert rows[1]["nidaq_sample_index"] == "2"


def test_pressure_stream_is_registered_in_the_session_alignment(tmp_path):
    columns = _columns([(10.0, 100.0, 0, 2048, 10.0, 10.0, 7, 900)])
    _, _, session_dir = _write_session_with_pressure(tmp_path, columns)

    alignment = json.loads(
        (session_dir / "streams" / "alignment.json").read_text()
    )
    entry = alignment["streams"]["pressure"]
    assert entry["path"] == "pressure.csv"
    assert entry["sampleCount"] == 1
    assert entry["firstOffsetSeconds"] == pytest.approx(0.0)


def test_a_rolled_pressure_ring_is_retention_not_an_incomplete_session(tmp_path):
    """A long session drops the oldest samples; that is not a data fault."""
    project = ProjectInfo(
        root=str(tmp_path),
        device_id="test",
        when=datetime(2026, 1, 2, 3, 4, 5),
        session=1,
    )
    result = SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (),
        pressure_overruns=2580,
        device_event_overruns=2580,
    )

    assert result["sessionComplete"] is True
    assert result["incompleteReasons"] == ()
    dropped = {
        item["source"]: item["dropped"]
        for item in result["retentionLimited"]
    }
    assert dropped == {"device_events": 2580, "pressure_samples": 2580}
