"""Process-wide ownership for a physical CAN channel.

The kernel permits more than one SocketCAN socket on an interface, but reachAQ
does not: two applications could otherwise issue conflicting motor commands.
The advisory lock is held for the lifetime of the hardware interface and is
also observed by the privileged reset helper.
"""

from __future__ import annotations

import errno
import os
import re
from pathlib import Path
from typing import Optional


_SAFE_CHANNEL = re.compile(r"[^A-Za-z0-9_.-]+")


def can_lock_path(channel: str) -> Path:
    lock_root = Path(
        os.environ.get("REACHAQ_CAN_LOCK_DIRECTORY", "/run/lock/reachaq")
    )
    safe_channel = _SAFE_CHANNEL.sub("_", channel).strip("._") or "default"
    return lock_root / f"reachaq-can-{safe_channel}.lock"


class CanChannelInUseError(RuntimeError):
    """Raised when another process already owns a CAN channel."""


def describe_lock_holders(path: Path):
    """Which live processes hold this lock, as (pid, command) pairs.

    The lock is advisory and released by the kernel when its owner dies, so
    a refusal always means something is still running - and saying only
    "owned by another reachAQ process" leaves the operator to find it by
    hand. That has cost real time: the holder is usually an orphaned worker
    of an app that was closed, which looks identical to a rig that is simply
    broken.

    Best effort by design. Reading /proc can race with processes exiting,
    and a failure to identify the holder must never replace the original
    error with a worse one.
    """
    holders = []
    proc = Path("/proc")
    if os.name != "posix" or not proc.is_dir():
        return holders
    try:
        target = os.path.realpath(path)
    except OSError:
        return holders
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            descriptors = (entry / "fd").iterdir()
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        try:
            for descriptor in descriptors:
                try:
                    if os.path.realpath(descriptor) != target:
                        continue
                except OSError:
                    continue
                try:
                    command = (entry / "cmdline").read_bytes().replace(
                        bytes(1), b" ").decode("utf-8", "replace").strip()
                except OSError:
                    command = ""
                holders.append((int(entry.name), command or "(unknown)"))
                break
        except (PermissionError, FileNotFoundError):
            continue
    return holders


class CanChannelOwnership:
    """Non-blocking advisory lock for one configured CAN channel."""

    def __init__(self, channel: str):
        self.channel = channel
        self.path = can_lock_path(channel)
        self._fd: Optional[int] = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            return
        if os.name != "posix":
            # Physical SocketCAN/pyjerrycan deployments are POSIX-only today.
            # Keep the abstraction harmless for a future Windows PCAN owner.
            return

        import fcntl

        flags = os.O_RDONLY | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags, 0o660)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"CAN lock directory is missing: {self.path.parent}; reinstall "
                "the reachaq-can tmpfiles configuration"
            ) from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                holders = describe_lock_holders(self.path)
                detail = "; ".join(
                    f"pid {pid}: {command}" for pid, command in holders)
                raise CanChannelInUseError(
                    f"CAN channel {self.channel!r} is already owned by another "
                    "reachAQ process"
                    + (f" ({detail})" if detail else
                       " (holder could not be identified; check for orphaned"
                       " workers of a closed session)")
                ) from exc
            raise
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        if os.name == "posix":
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> "CanChannelOwnership":
        self.acquire()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
