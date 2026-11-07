from pathlib import Path

import pytest

from autotrainer.device import (
    CanChannelInUseError,
    CanChannelOwnership,
    can_lock_path,
)


def test_channel_ownership_is_exclusive_and_reusable(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(tmp_path))
    first = CanChannelOwnership("can0")
    second = CanChannelOwnership("can0")

    first.acquire()
    with pytest.raises(CanChannelInUseError, match="already owned"):
        second.acquire()

    first.release()
    second.acquire()
    assert second.acquired is True
    second.release()


def test_channel_lock_path_sanitizes_untrusted_channel(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(tmp_path))

    path = can_lock_path("../../can 0")

    assert path.parent == tmp_path
    assert path.name == "reachaq-can-can_0.lock"


def test_channel_lock_uses_shared_runtime_directory_by_default(monkeypatch):
    monkeypatch.delenv("REACHAQ_CAN_LOCK_DIRECTORY", raising=False)

    path = can_lock_path("can0")

    assert path == Path("/run/lock/reachaq/reachaq-can-can0.lock")


def test_missing_runtime_lock_directory_has_installation_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(missing))

    with pytest.raises(RuntimeError, match="tmpfiles configuration"):
        CanChannelOwnership("can0").acquire()
