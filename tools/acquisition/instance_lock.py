"""One reachAQ acquisition process per user at a time.

Two instances would both open the same cameras, NI-DAQ boards and CAN bus, and
fail in ways that look like hardware faults. The desktop launcher and
~/.local/bin/reachaq already refuse a second start, but only for starts that go
through them; the Conda environment's own `reachaq` command, `python -m
reachAQ.app` and `auto-trainer-headless` did not. So the application holds a
lock of its own for as long as it runs.

It is a separate file from the launchers' lock on purpose. The launcher's
`conda run` process keeps that one while the application runs under it, so an
application taking the same lock would refuse itself every time it was
started from the launcher.

The lock is an flock, released by the kernel when the process ends however it
ends, so it cannot go stale. The file records the holder's PID so a refusal can
name it. Standard library only: both entry points take the lock before
importing anything else.
"""

import os
import tempfile
from pathlib import Path
from typing import IO, Optional

try:
    import fcntl
except ImportError:  # Windows: no flock, and no rig runs there.
    fcntl = None

#: The exit status for a refused second start, matching the launchers.
ALREADY_RUNNING_EXIT = 3


class AlreadyRunning(RuntimeError):
    def __init__(self, path: Path, holder: str):
        self.path = path
        self.holder = holder
        detail = f" (pid {holder})" if holder else ""
        super().__init__(f"reachAQ is already running for this user{detail}; lock {path}")


def default_lock_path() -> Path:
    """Beside the launchers' lock: <runtime dir>/reachaq/app.lock.

    The runtime directory is XDG_RUNTIME_DIR when it is a real directory owned
    by this user, otherwise a per-user directory under the temporary directory,
    which is the same choice the launchers make.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if runtime:
        path = Path(runtime)
        try:
            if path.is_dir() and not path.is_symlink() and path.stat().st_uid == os.getuid():
                return path / "reachaq" / "app.lock"
        except OSError:
            pass
    return Path(tempfile.gettempdir()) / f"reachaq-runtime-{os.getuid()}" / "reachaq" / "app.lock"


def acquire_instance_lock(path: Optional[Path] = None) -> Optional[IO[str]]:
    """Take the lock, or raise AlreadyRunning. Keep the result for the process's life.

    Returns None where flock does not exist, which is not a rig platform.
    """
    if fcntl is None:
        return None
    path = Path(path) if path is not None else default_lock_path()
    for directory in (path.parent.parent, path.parent):
        if directory.is_symlink():
            raise RuntimeError(f"Refusing a symbolic link in the reachAQ lock path: {directory}")
        directory.mkdir(mode=0o700, exist_ok=True)
    descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = handle.read().strip()
        handle.close()
        raise AlreadyRunning(path, holder) from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle
