import csv
from datetime import datetime

import h5py
import numpy as np

from autotrainer.core import ProjectInfo
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


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
    assert rows[0]["switch"] == "1"

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
