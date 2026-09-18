"""A refused CAN channel must say what is holding it.

The lock is advisory and the kernel releases it when its owner dies, so a
refusal always means something is still running. Saying only "owned by another
reachAQ process" leaves the operator to find it by hand, and the holder is
usually an orphaned worker of a session that was closed - which from the
outside looks the same as a broken rig.
"""

import os

import pytest

from autotrainer.device.can_ownership import (
    CanChannelInUseError,
    CanChannelOwnership,
    describe_lock_holders,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX lock only")


@posix_only
def test_a_second_owner_is_refused_and_the_first_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(tmp_path))

    first = CanChannelOwnership("can0")
    first.acquire()
    try:
        with pytest.raises(CanChannelInUseError) as refused:
            CanChannelOwnership("can0").acquire()
        message = str(refused.value)
        assert "can0" in message
        # This process is the holder, so it must be the one named.
        assert f"pid {os.getpid()}" in message
    finally:
        first.release()


@posix_only
def test_the_channel_is_free_again_once_released(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(tmp_path))

    first = CanChannelOwnership("can0")
    first.acquire()
    first.release()

    second = CanChannelOwnership("can0")
    second.acquire()          # must not raise
    second.release()


@posix_only
def test_the_holder_listing_finds_this_process(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(tmp_path))
    owner = CanChannelOwnership("can0")
    owner.acquire()
    try:
        holders = describe_lock_holders(owner.path)
        assert os.getpid() in [pid for pid, _command in holders]
    finally:
        owner.release()


@posix_only
def test_an_unheld_lock_lists_nobody(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHAQ_CAN_LOCK_DIRECTORY", str(tmp_path))
    path = tmp_path / "reachaq-can-can0.lock"
    path.touch()

    assert describe_lock_holders(path) == []


def test_describing_a_missing_lock_is_harmless(tmp_path):
    """Identifying the holder must never replace the real error with a worse one."""
    assert describe_lock_holders(tmp_path / "nothing-here.lock") == []
