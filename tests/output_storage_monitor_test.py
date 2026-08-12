import os
import time
from pathlib import Path

import pytest

from tools.acquisition.model import output_storage_monitor as storage_module
from tools.acquisition.model.output_storage_monitor import (
    RecordingStorageMonitor,
    preflight_storage,
)


def test_preflight_performs_fsync_probe_and_reports_capacity(tmp_path, monkeypatch):
    fsync_calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        storage_module.os,
        "fsync",
        lambda descriptor: (fsync_calls.append(descriptor), real_fsync(descriptor))[1],
    )

    result = preflight_storage(
        tmp_path / "session005",
        estimated_bytes_per_second=1024,
        configured_duration_seconds=60,
    )

    assert fsync_calls
    assert result.free_bytes > 0
    assert result.projected_maximum_minutes > 0
    assert result.duration_fits_projection
    assert not tuple((tmp_path / "session005").glob(".reachaq-write-probe-*"))


def test_preflight_removes_probe_after_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage_module.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(OSError, match="fsync failed"):
        preflight_storage(
            tmp_path,
            estimated_bytes_per_second=1,
            configured_duration_seconds=None,
        )

    assert not tuple(tmp_path.glob(".reachaq-write-probe-*"))


def test_monitor_reports_thresholds_once_and_uses_observed_rate(tmp_path, monkeypatch):
    usage = type("Usage", (), {"free": 29 * 60 * 100, "total": 0, "used": 0})
    monkeypatch.setattr(storage_module.shutil, "disk_usage", lambda _path: usage)
    sizes = iter((0, 1_000, 2_000, 3_000))
    monkeypatch.setattr(storage_module, "_directory_size", lambda _path: next(sizes))
    times = iter((100.0, 110.0, 120.0, 130.0))
    monkeypatch.setattr(storage_module.time, "time", lambda: next(times))
    calls = []
    monitor = RecordingStorageMonitor(interval_seconds=999)

    monitor.start(
        tmp_path,
        estimated_bytes_per_second=100,
        callback=lambda snapshot, threshold: calls.append((snapshot, threshold)),
    )
    monitor.sample_now()
    monitor.sample_now()
    result = monitor.stop()

    assert [threshold for _snapshot, threshold in calls] == [30, None, None, None]
    assert result["observedBytesPerSecond"] == 100
    assert [warning["thresholdMinutes"] for warning in result["warnings"]] == [30]


def test_monitor_abort_discards_session_telemetry(tmp_path):
    monitor = RecordingStorageMonitor(interval_seconds=999)
    monitor.start(
        tmp_path,
        estimated_bytes_per_second=1,
        callback=lambda *_args: None,
    )
    monitor.abort()

    assert monitor.snapshot()["sampleCount"] == 0
    assert monitor.snapshot()["targetDirectory"] is None
