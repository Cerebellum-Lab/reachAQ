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
    lock_root = Path(os.environ.get("REACHAQ_CAN_LOCK_DIRECTORY", "/tmp"))
    safe_channel = _SAFE_CHANNEL.sub("_", channel).strip("._") or "default"
    return lock_root / f"reachaq-can-{safe_channel}.lock"


class CanChannelInUseError(RuntimeError):
    """Raised when another process already owns a CAN channel."""


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
        fd = os.open(self.path, flags, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise CanChannelInUseError(
                    f"CAN channel {self.channel!r} is already owned by another "
                    "reachAQ process"
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

