"""Diagnostic bundle: one zip an operator can send to support.

Contents: recent app logs, the configuration and preferences (redacted), the
software versions, a hardware inventory, the subsystem statuses and operator
alerts when built from the running app, and the small metadata files of the
most recent sessions. No video and no HDF5.

Secrets stay out by construction, and then the result is checked:

- only files this module names are read; no directory is copied wholesale;
- the OS keyring is never opened, and credential files are never added;
- every text member is redacted: values of password/token/secret/key-like
  fields, URL credentials and bearer tokens are replaced;
- values in the local credential file (the keyring's fallback) are read only
  to search the finished bundle for them; a hit fails the export.

Run from a terminal when the app will not start::

    python -m reachaq.app.model.diagnostic_bundle [-o OUT.zip]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import io
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REDACTED = "<redacted>"

#: A field whose name contains one of these words has its value redacted.
SECRET_FIELD_PATTERN = (
    r"passw(?:or)?d|passphrase|secret|token|api[_-]?key|access[_-]?key|private[_-]?key"
    r"|credential|authorization|auth[_-]?key|cookie"
)
# A field name starts the text or follows a separator: space, punctuation, a
# query string's ? or &, a cookie's ;, or the backslash of escaped JSON.
_FIELD_RE = re.compile(
    r"(?P<lead>(?:^|[\s,{(\"'?&;\\])(?:[\w.\-/]*?)(?:" + SECRET_FIELD_PATTERN + r")[\w.\-]*(?:\\?[\"'])?)"
    r"(?P<sep>\s*[:=]\s*)"
    # An unquoted value ends at a separator. "&" ends it only before the next
    # query parameter and "\" only before an escaped quote, so a secret that
    # contains either is not cut short.
    r"(?P<value>\\\"(?:[^\"\\]|\\(?!\"))*\\\"|\"(?:[^\"\\]|\\.)*\"|'[^']*'"
    r"|(?:[^\s,;&}\])\\]|&(?!\w+=)|\\(?!\"))+)",
    re.IGNORECASE,
)
# Every cookie on a Cookie or Set-Cookie line, whatever its name.
_COOKIE_LINE_RE = re.compile(r"(?P<lead>\b(?:set-)?cookie\s*:\s*)[^\r\n]+", re.IGNORECASE)
_URL_CREDENTIALS_RE = re.compile(r"(?P<scheme>\b[a-z][a-z0-9+.\-]*://)[^/\s:@]+:[^/\s@]+@", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?P<lead>\b(?:bearer|basic)\s+)[A-Za-z0-9._~+/=\-]{8,}", re.IGNORECASE)

#: Files never added, whatever directory they turn up in.
_NEVER_ADD_RE = re.compile(
    r"credential|secret|keyring|\.netrc$|id_rsa|id_ed25519|\.pem$|\.key$|\.p12$|\.pfx$|token",
    re.IGNORECASE,
)

SESSION_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".csv", ".md", ".log", ".ini", ".toml"}
SESSION_FILE_LIMIT = 2 * 1024 * 1024
SESSION_TOTAL_LIMIT = 20 * 1024 * 1024
LOG_TOTAL_LIMIT = 50 * 1024 * 1024
LOG_DAYS = 7
SESSION_COUNT = 5

PACKAGES = (
    "reachaq", "reachaq", "reachaq",
    "reachaq", "reachaq", "reachaq",
    "reachaq", "reachaq",
    "PySide6", "numpy", "scipy", "h5py", "tables", "opencv-python",
    "opencv-python-headless", "nidaqmx", "python-can", "pyserial", "spinnaker-python",
    "torch", "tensorflow", "ultralytics", "deeplabcut", "keyring",
)


def redact_text(text: str) -> Tuple[str, int]:
    """Return the text with secrets replaced, and how many were replaced."""
    count = 0

    def field(match: re.Match) -> str:
        nonlocal count
        value = match.group("value")
        if value.strip("\\\"'") in ("", REDACTED):
            return match.group(0)
        count += 1
        if value.startswith('\\"'):
            quote = '\\"'
        else:
            quote = value[0] if value[:1] in ("'", '"') else ""
        return f"{match.group('lead')}{match.group('sep')}{quote}{REDACTED}{quote}"

    def url(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{match.group('scheme')}{REDACTED}@"

    def bearer(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{match.group('lead')}{REDACTED}"

    def cookie_line(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{match.group('lead')}{REDACTED}"

    # Bearer first: "Authorization: Bearer x" would otherwise lose only "Bearer".
    text = _BEARER_RE.sub(bearer, text)
    text = _COOKIE_LINE_RE.sub(cookie_line, text)
    text = _FIELD_RE.sub(field, text)
    text = _URL_CREDENTIALS_RE.sub(url, text)
    return text, count


def _redact_json_value(value, count: List[int]):
    secret = re.compile(SECRET_FIELD_PATTERN, re.IGNORECASE)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if secret.search(str(key)) and item not in (None, "", [], {}):
                out[key] = REDACTED
                count[0] += 1
            else:
                out[key] = _redact_json_value(item, count)
        return out
    if isinstance(value, list):
        return [_redact_json_value(item, count) for item in value]
    if isinstance(value, str):
        text, n = redact_text(value)
        count[0] += n
        return text
    return value


def known_secret_values(credential_files: Iterable[Path]) -> Tuple[str, ...]:
    """Secret values from local credential files, for the final leak check only."""
    values = set()
    for path in credential_files:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, str) and len(item) >= 4:
                values.add(item)
    return tuple(sorted(values))


def default_credential_files() -> Tuple[Path, ...]:
    try:
        from reachaq.app.model.animal_sources.credentials import (
            FALLBACK_FILE_NAME, user_config_directory)
    except Exception:
        return ()
    return (user_config_directory() / FALLBACK_FILE_NAME,)


@dataclasses.dataclass
class BundleSources:
    configuration_path: Optional[Path] = None
    preferences_path: Optional[Path] = None
    #: Roots searched for ``*.log`` files and session directories.
    output_roots: Sequence[Path] = ()
    log_dirs: Sequence[Path] = ()
    #: Supplied by the running app; absent from a command-line bundle.
    subsystem_snapshot: Optional[Mapping] = None
    alerts: Optional[Sequence[Mapping]] = None
    app_state: Optional[Mapping] = None
    credential_files: Sequence[Path] = dataclasses.field(default_factory=default_credential_files)


@dataclasses.dataclass
class BundleResult:
    path: Path
    members: Tuple[str, ...]
    skipped: Tuple[str, ...]
    redactions: int
    size_bytes: int


class SecretLeakError(RuntimeError):
    pass


class _Writer:
    def __init__(self, archive: zipfile.ZipFile, secrets: Sequence[str]):
        self.archive = archive
        self.secrets = tuple(secrets)
        self.members: List[str] = []
        self.skipped: List[str] = []
        self.redactions = 0

    def text(self, name: str, text: str, *, redact: bool = True) -> None:
        if redact:
            text, n = redact_text(text)
            self.redactions += n
        for secret in self.secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
                self.redactions += 1
        self.archive.writestr(name, text)
        self.members.append(name)

    def json(self, name: str, value) -> None:
        count = [0]
        value = _redact_json_value(value, count)
        self.redactions += count[0]
        self.text(name, json.dumps(value, indent=2, sort_keys=True, default=str), redact=False)

    def file(self, name: str, path: Path, *, limit: Optional[int] = None) -> int:
        """Add a text file, redacted. Over ``limit`` bytes, only its tail."""
        if _NEVER_ADD_RE.search(path.name):
            self.skipped.append(f"{path}: never added (credential-like name)")
            return 0
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if limit is not None and size > limit:
                    handle.seek(size - limit)
                    data = handle.read()
                    data = data[data.find(b"\n") + 1:]
                    header = f"[first {size - len(data)} bytes omitted]\n"
                else:
                    data = handle.read()
                    header = ""
        except OSError as error:
            self.skipped.append(f"{path}: {error}")
            return 0
        self.text(name, header + data.decode("utf-8", errors="replace"))
        return len(data)


def _run(command: Sequence[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return f"{command[0]}: not found"
    except subprocess.TimeoutExpired:
        return f"{command[0]}: timed out after {timeout:g} s"
    except OSError as error:
        return f"{command[0]}: {error}"
    output = (completed.stdout or "") + (completed.stderr or "")
    return output.strip() or f"{command[0]}: exit {completed.returncode}"


def collect_versions() -> Dict[str, object]:
    from importlib import metadata

    versions: Dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        from reachaq.version import __version__
        versions["reachaq"] = __version__
    except Exception as error:
        versions["reachaq"] = f"unknown ({error})"
    repo = Path(__file__).resolve().parents[4]
    versions["git_commit"] = _run(("git", "-C", str(repo), "rev-parse", "HEAD"))
    versions["git_describe"] = _run(("git", "-C", str(repo), "describe", "--always", "--dirty", "--tags"))
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception as error:
            packages[name] = f"error: {error}"
    versions["packages"] = packages
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        try:
            versions["os_release"] = os_release.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        versions["kernel"] = _run(("uname", "-a"))
    return versions


def collect_hardware_inventory(*, query_gpu: bool = True) -> Dict[str, object]:
    """Read-only probes that open no device: nothing here touches the CAN bus,
    an NI task or a camera stream. ``query_gpu=False`` skips nvidia-smi, whose
    every call stalls live pose inference by 16-19 ms; it is off while a
    recording runs."""
    inventory: Dict[str, object] = {}
    try:
        from reachaq.app.model.hardware_scan import scan_can_adapters, scan_gpus
        scans = [("can_adapters", scan_can_adapters)]
        if query_gpu:
            scans.append(("gpus", scan_gpus))
        else:
            inventory["gpus"] = "not queried: a recording was running"
        for name, scan in scans:
            try:
                entry = scan()
                inventory[name] = {"state": entry.state, "info": entry.info}
            except Exception as error:
                inventory[name] = {"state": "error", "info": str(error)}
    except Exception as error:
        inventory["scan_error"] = str(error)
    if sys.platform.startswith("linux"):
        if query_gpu:
            inventory["nvidia_smi"] = _run(("nvidia-smi",))
        # 1093 is National Instruments; 1e10 is Teledyne FLIR.
        inventory["ni_pci_devices"] = _run(("lspci", "-nn", "-d", "1093:"))
        inventory["flir_usb_devices"] = _run(("lsusb", "-d", "1e10:"))
        inventory["usb_devices"] = _run(("lsusb",))
        inventory["block_devices"] = _run(("lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL"))
        inventory["disk_free"] = _run(("df", "-h"))
    return inventory


def _iter_dirs(root: Path, depth: int) -> Iterable[Path]:
    if depth < 0:
        return
    try:
        children = [child for child in root.iterdir() if child.is_dir() and not child.is_symlink()]
    except OSError:
        return
    for child in children:
        yield child
        yield from _iter_dirs(child, depth - 1)


def find_recent_logs(roots: Sequence[Path], *, days: int, now: float) -> List[Path]:
    cutoff = now - days * 86400
    found: Dict[Path, float] = {}
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for directory in (root, *_iter_dirs(root, 2)):
            try:
                entries = list(directory.glob("*.log"))
            except OSError:
                continue
            for path in entries:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime >= cutoff and path.is_file():
                    found[path.resolve()] = mtime
    return sorted(found, key=lambda item: found[item], reverse=True)


def find_recent_sessions(roots: Sequence[Path], *, count: int) -> List[Path]:
    found: Dict[Path, float] = {}
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for directory in _iter_dirs(root, 3):
            if directory.name.lower().startswith("session"):
                try:
                    found[directory.resolve()] = directory.stat().st_mtime
                except OSError:
                    continue
    return sorted(found, key=lambda item: found[item], reverse=True)[:count]


def _member_name(prefix: str, path: Path, roots: Sequence[Path]) -> str:
    for root in roots:
        try:
            relative = path.resolve().relative_to(Path(root).expanduser().resolve())
        except ValueError:
            continue
        return f"{prefix}/{relative.as_posix()}"
    return f"{prefix}/{path.name}"


def build_diagnostic_bundle(
    destination: Path,
    sources: BundleSources,
    *,
    now: Optional[float] = None,
    versions: Optional[Callable[[], Mapping]] = None,
    inventory: Optional[Callable[[], Mapping]] = None,
) -> BundleResult:
    import time as _time

    now = _time.time() if now is None else float(now)
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    secrets = known_secret_values(sources.credential_files)
    credential_paths = set()
    for path in sources.credential_files:
        try:
            credential_paths.add(Path(path).expanduser().resolve())
        except OSError:
            pass
    partial = destination.with_name(destination.name + ".partial")
    try:
        writer = _write_bundle(partial, sources, secrets, credential_paths, now, versions, inventory)
        _verify_no_secrets(partial, secrets)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, destination)
    return BundleResult(
        path=destination,
        members=tuple(writer.members),
        skipped=tuple(writer.skipped),
        redactions=writer.redactions,
        size_bytes=destination.stat().st_size,
    )


def _write_bundle(partial, sources, secrets, credential_paths, now, versions, inventory) -> "_Writer":
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        writer = _Writer(archive, secrets)

        writer.json("versions.json", (versions or collect_versions)())
        writer.json("hardware_inventory.json", (inventory or collect_hardware_inventory)())
        if sources.subsystem_snapshot is not None:
            writer.json("subsystem_statuses.json", sources.subsystem_snapshot)
        if sources.alerts is not None:
            writer.json("operator_alerts.json", list(sources.alerts))
        if sources.app_state is not None:
            writer.json("app_state.json", sources.app_state)

        for label, path in (
            ("configuration", sources.configuration_path),
            ("preferences", sources.preferences_path),
        ):
            if path is None:
                continue
            path = Path(path).expanduser()
            if not path.is_file():
                writer.skipped.append(f"{label}: {path} does not exist")
                continue
            writer.file(f"{label}/{path.name}", path)

        log_roots = [Path(root) for root in (*sources.log_dirs, *sources.output_roots)]
        budget = LOG_TOTAL_LIMIT
        for path in find_recent_logs(log_roots, days=LOG_DAYS, now=now):
            if path in credential_paths:
                continue
            if budget <= 0:
                writer.skipped.append(f"{path}: log size limit reached")
                continue
            budget -= writer.file(_member_name("logs", path, log_roots), path, limit=budget)

        session_budget = SESSION_TOTAL_LIMIT
        roots = [Path(root) for root in sources.output_roots]
        for session in find_recent_sessions(roots, count=SESSION_COUNT):
            for path in sorted(session.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                if path.suffix.lower() not in SESSION_SUFFIXES or path.resolve() in credential_paths:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size > SESSION_FILE_LIMIT or size > session_budget:
                    writer.skipped.append(f"{path}: {size} bytes, over the session file limit")
                    continue
                session_budget -= writer.file(_member_name("sessions", path, roots), path)

        manifest = {
            "created": _dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "host": platform.node(),
            "members": writer.members,
            "skipped": writer.skipped,
            "redactions": writer.redactions,
            "secret_handling": (
                "The OS keyring was not read. Credential files were not added. Values of "
                "password, token, secret and key fields, cookies, URL credentials and bearer "
                "tokens were replaced with " + REDACTED + ". Values in the local credential "
                "file were also searched for and removed. A secret held only in the OS "
                "keyring is protected by the field redaction alone, since its value is "
                "never read."
            ),
        }
        writer.text("manifest.json", json.dumps(manifest, indent=2), redact=False)
    return writer


def _verify_no_secrets(path: Path, secrets: Sequence[str]) -> None:
    if not secrets:
        return
    encoded = [secret.encode("utf-8") for secret in secrets]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            if any(secret in data for secret in encoded):
                raise SecretLeakError(f"a stored credential appeared in {name}; no bundle was written")


def default_bundle_name(now: Optional[_dt.datetime] = None) -> str:
    now = now or _dt.datetime.now()
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", platform.node() or "host")
    return f"reachaq-diagnostics-{host}-{now:%Y%m%d-%H%M%S}.zip"


def sources_from_paths(
    configuration_path: Optional[Path],
    preferences_path: Optional[Path] = None,
    log_location: Optional[Path] = None,
) -> BundleSources:
    """Sources for a bundle from file locations alone, without a running app."""
    from reachaq.core.configuration.persistence_configuration import PersistenceConfiguration

    output_roots: List[Path] = []
    if configuration_path is not None and Path(configuration_path).is_file():
        output_roots.extend(_configured_output_roots(Path(configuration_path)))
    if not output_roots:
        output_roots.append(PersistenceConfiguration.get_default_output_path())
    return BundleSources(
        configuration_path=configuration_path,
        preferences_path=preferences_path,
        output_roots=tuple(output_roots),
        log_dirs=() if not log_location else (Path(log_location),),
    )


def _configured_output_roots(configuration_path: Path) -> List[Path]:
    """The outputLocation line of a configuration file, read as text: a file
    the app refuses to load is still worth bundling."""
    try:
        text = Path(configuration_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    from reachaq.app.model.first_run import output_location_from_text

    value = output_location_from_text(text)
    return [Path(value).expanduser()] if value else []


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reachaq-diagnostics",
        description="Write a reachAQ diagnostic bundle (logs, configuration, versions, "
                    "hardware inventory, recent session metadata) with secrets removed.")
    parser.add_argument("-o", "--output", type=Path, help="zip file to write (default: ~/<name>.zip)")
    parser.add_argument("-c", "--configuration", type=Path, help="configuration file")
    parser.add_argument("--preferences-file", type=Path, help="preferences ini file")
    parser.add_argument("--data-dir", type=Path, action="append", default=[],
                        help="data folder to search for logs and sessions (repeatable)")
    parser.add_argument("--log-dir", type=Path, action="append", default=[],
                        help="extra folder to search for logs (repeatable)")
    # Used by the running app, which builds the bundle in this separate
    # process so its work never holds the app's interpreter: the app's
    # statuses and alerts in, a result file out.
    parser.add_argument("--app-snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--no-gpu-query", action="store_true",
                        help="skip nvidia-smi (it briefly stalls live pose inference)")
    args = parser.parse_args(argv)

    from reachaq.core.configuration.system_configuration import SystemConfiguration

    configuration = args.configuration
    if configuration is None:
        candidate = SystemConfiguration.DEFAULT_PATH.expanduser()
        configuration = candidate if candidate.is_file() else None
    sources = sources_from_paths(
        configuration, args.preferences_file or _default_preferences_path())
    if args.data_dir:
        sources.output_roots = tuple(args.data_dir)
    if args.log_dir:
        sources.log_dirs = tuple(args.log_dir)
    if args.app_snapshot is not None:
        snapshot = json.loads(args.app_snapshot.read_text(encoding="utf-8"))
        sources.subsystem_snapshot = snapshot.get("subsystem_statuses")
        sources.alerts = snapshot.get("alerts")
        sources.app_state = snapshot.get("app_state")
    output = args.output or (Path.home() / default_bundle_name())
    try:
        result = build_diagnostic_bundle(
            output, sources,
            inventory=lambda: collect_hardware_inventory(query_gpu=not args.no_gpu_query))
    except Exception as error:
        refused = isinstance(error, SecretLeakError)
        label = "Diagnostic bundle refused" if refused else "Diagnostic bundle failed"
        print(f"{label}: {error}", file=sys.stderr)
        if args.result_file is not None:
            args.result_file.write_text(json.dumps({"error": str(error) or type(error).__name__}))
        return 2 if refused else 1
    if args.result_file is not None:
        args.result_file.write_text(json.dumps({
            "path": str(result.path), "members": len(result.members), "skipped": len(result.skipped),
            "redactions": result.redactions, "size_bytes": result.size_bytes}))
    print(f"Wrote {result.path} ({result.size_bytes / 1e6:.1f} MB, {len(result.members)} files, "
          f"{result.redactions} values redacted)")
    for line in result.skipped:
        print(f"  skipped: {line}")
    return 0


def _default_preferences_path() -> Optional[Path]:
    """Where QSettings keeps the app's preferences, without starting Qt."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        path = base / "Colorado" / "Auto Trainer.ini"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        path = base / "Colorado" / "Auto Trainer.conf"
    return path if path.is_file() else None


if __name__ == "__main__":
    sys.exit(main())
