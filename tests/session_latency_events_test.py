from datetime import datetime

import h5py
import numpy as np

from autotrainer.core import ProjectInfo
from autotrainer.core.latency.schema import LIVE_POSE_DTYPE
from tools.acquisition.model.session_data_recorder import SessionDataRecorder


def _nidaq_chunk():
    return (
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


def test_write_session_writes_the_gui_latency_rows(tmp_path):
    project = ProjectInfo(root=str(tmp_path), device_id="test",
                          when=datetime(2026, 1, 2, 3, 4, 5), session=1)
    tables = {"live_poses": np.array([(1, 1.0, 1.1, 1.2, 1.3)], dtype=LIVE_POSE_DTYPE)}

    SessionDataRecorder._write_session(
        project, 10.0, 100.0, 12.0, (), (), (), (_nidaq_chunk(),),
        latency_events=tables,
    )

    path = tmp_path / "20260102" / "test" / "session001" / "streams" / "latency" / "events.h5"
    with h5py.File(path, "r") as store:
        assert store["live_poses"]["live_sequence"].tolist() == [1]
