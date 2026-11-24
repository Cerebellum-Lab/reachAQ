"""Finding running reachAQ instances and the local API port each one got.

Each instance binds a port the OS picks (port 0), so two instances on one host,
a ``--simulate`` run beside the rig's for example, never fight over a fixed
port. A running service writes ``instance-<pid>.json`` to the discovery
directory and removes it when it stops:

    {"schemaVersion": 1, "pid": 1234, "processStartTime": 1763560000.12,
     "host": "127.0.0.1", "port": 41877, "simulated": false}

A record whose process has gone, or whose pid now belongs to another process
(``processStartTime`` differs), is ignored and removed, so a crash leaves no
false entry behind. The directory is the per-user runtime directory the
application lock uses, so a reboot clears it.
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

import psutil

from ..logging import get_verbose_logger

logger = get_verbose_logger(__name__)

SCHEMA_VERSION = 1
_START_TIME_TOLERANCE_S = 1.0


class ApiInstanceNotFound(LookupError):
    """No running instance, or more than one, matched the request."""


@dataclass(frozen=True)
class ApiInstance:
    pid: int
    host: str
    port: int
    simulated: bool
    path: Path


def default_discovery_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """<runtime dir>/reachaq/api, beside the application lock (instance_lock.py).

    The runtime directory is XDG_RUNTIME_DIR when it is a real directory owned
    by this user, otherwise a per-user directory under the temporary directory.
    On Windows it is %LOCALAPPDATA%/reachAQ/api.
    """
    env = os.environ if environ is None else environ
    if sys.platform == "win32":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "reachAQ" / "api"
    runtime = env.get("XDG_RUNTIME_DIR", "")
    if runtime:
        path = Path(runtime)
        try:
            if path.is_dir() and not path.is_symlink() and path.stat().st_uid == os.getuid():
                return path / "reachaq" / "api"
        except OSError:
            pass
    return Path(tempfile.gettempdir()) / f"reachaq-runtime-{os.getuid()}" / "reachaq" / "api"


def _process_start_time(pid: int) -> Optional[float]:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError, ValueError, OverflowError):
        return None


def publish_instance(directory: Path, host: str, port: int, simulated: bool) -> Path:
    """Write this process's record atomically and return its path."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    pid = os.getpid()
    record = {
        "schemaVersion": SCHEMA_VERSION,
        "pid": pid,
        "processStartTime": _process_start_time(pid),
        "host": host,
        "port": int(port),
        "simulated": bool(simulated),
    }
    path = directory / f"instance-{pid}.json"
    temporary = directory / f".instance-{pid}.json.tmp"
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, path)
    return path


def withdraw_instance(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError as err:
        logger.warning("Could not remove the local API record %s: %s", path, err)


def _read(path: Path) -> Optional[ApiInstance]:
    """The record's instance, None when its process is not running; raises on a bad file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pid = int(data["pid"])
    started = _process_start_time(pid)
    recorded = data.get("processStartTime")
    if started is None:
        return None
    if recorded is not None and abs(float(recorded) - started) > _START_TIME_TOLERANCE_S:
        return None
    return ApiInstance(
        pid=pid, host=str(data["host"]), port=int(data["port"]),
        simulated=bool(data.get("simulated", False)), path=path,
    )


def list_api_instances(directory: Optional[Path] = None) -> List[ApiInstance]:
    """The running instances with a published API, by pid. Removes stale records."""
    directory = default_discovery_dir() if directory is None else Path(directory)
    instances = []
    for path in sorted(directory.glob("instance-*.json")):
        try:
            instance = _read(path)
        except (OSError, ValueError, KeyError, TypeError) as err:
            logger.info("Skipping unreadable local API record %s: %s", path, err)
            continue
        if instance is None:
            withdraw_instance(path)
            continue
        instances.append(instance)
    return sorted(instances, key=lambda item: (item.pid, item.path.name))


def find_api_instance(simulated: Optional[bool] = None,
                      directory: Optional[Path] = None) -> ApiInstance:
    """The one running instance, or the one with this ``simulated`` flag.

    Raises ApiInstanceNotFound when none matches or several do; the message
    lists them, so the caller can pass a port instead.
    """
    instances = list_api_instances(directory)
    matching = [item for item in instances if simulated is None or item.simulated == simulated]
    if len(matching) == 1:
        return matching[0]
    kind = "" if simulated is None else (" simulated" if simulated else " non-simulated")
    where = default_discovery_dir() if directory is None else directory
    if not matching:
        raise ApiInstanceNotFound(f"No{kind} reachAQ instance has published a local API in {where}")
    listed = ", ".join(
        f"pid {item.pid} port {item.port}{' (simulated)' if item.simulated else ''}" for item in matching)
    raise ApiInstanceNotFound(
        f"{len(matching)} reachAQ instances{kind} are running ({listed}); "
        "pass simulated= or a port")
