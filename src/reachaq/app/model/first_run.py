"""The first-run check: is there a configuration, and can data be written?

Before this step a new install got a default configuration written silently
on the first load, and a configured data folder on a drive that was not
mounted surfaced only when Record refused. ``assess`` decides whether the
first-run dialog is needed; the hardware checks are read-only and open no
device.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from reachaq.core.configuration.persistence_configuration import PersistenceConfiguration
from reachaq.core.configuration.system_configuration import SystemConfiguration

#: Below this the data folder is reported as low on space (about 1 h of 3-camera video).
LOW_SPACE_BYTES = 50 * 1024 ** 3

# One line only: "[ \t]" so an empty value never reaches into the next line.
_OUTPUT_LOCATION_RE = re.compile(
    r"^(?P<lead>[ \t]*(?:outputLocation|output_location):[ \t]*)(?P<value>[^\r\n]*?)[ \t]*$",
    re.MULTILINE)


def _yaml_scalar(raw: str) -> str:
    """A plain or quoted YAML scalar on one line, without a trailing comment."""
    raw = raw.strip()
    if raw.startswith("'"):
        end = 1
        while True:
            end = raw.find("'", end)
            if end < 0 or raw[end + 1:end + 2] != "'":
                break
            end += 2
        return raw[1:end if end > 0 else None].replace("''", "'")
    if raw.startswith('"'):
        end = raw.find('"', 1)
        return raw[1:end if end > 0 else None].replace('\\"', '"')
    comment = re.search(r"[ \t]#", raw)
    return (raw[:comment.start()] if comment else raw).strip()


def output_location_from_text(text: str) -> Optional[str]:
    """The ``outputLocation`` value of a configuration's text, or None."""
    match = _OUTPUT_LOCATION_RE.search(text)
    if match is None:
        return None
    return _yaml_scalar(match.group("value")) or None


@dataclasses.dataclass(frozen=True)
class FolderCheck:
    path: Path
    ok: bool
    problem: str = ""
    free_bytes: Optional[int] = None

    @property
    def summary(self) -> str:
        if not self.ok:
            return self.problem
        if self.free_bytes is None:
            return "Writable"
        text = f"Writable · {self.free_bytes / 1024 ** 3:.0f} GB free"
        if self.free_bytes < LOW_SPACE_BYTES:
            text += " (low)"
        return text


@dataclasses.dataclass(frozen=True)
class FirstRunAssessment:
    configuration_path: Path
    configuration_exists: bool
    data_folder: Path
    data_check: FolderCheck
    #: Set when the configuration exists but cannot be read for its data folder.
    configuration_problem: str = ""

    @property
    def needed(self) -> bool:
        return not self.configuration_exists or not self.data_check.ok


@dataclasses.dataclass(frozen=True)
class HardwareCheck:
    name: str
    found: bool
    detail: str
    remedy: str = ""


def check_folder(path: Path, *, create: bool = False) -> FolderCheck:
    """Whether data can be written under ``path``. With ``create`` it is made first."""
    path = Path(path).expanduser()
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return FolderCheck(path, False, f"Cannot create the folder: {error.strerror or error}")
    if not path.exists():
        return FolderCheck(path, False, "The folder does not exist (is its drive mounted?)")
    if not path.is_dir():
        return FolderCheck(path, False, "This is a file, not a folder")
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".reachaq-write-check-"):
            pass
    except OSError as error:
        return FolderCheck(path, False, f"The folder is not writable: {error.strerror or error}")
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        free = None
    return FolderCheck(path, True, free_bytes=free)


def configured_data_folder(configuration_path: Path) -> Tuple[Optional[Path], str]:
    """The configuration's ``persistence.outputLocation``, read as text so a
    file from another version still answers. ``(None, problem)`` if unreadable."""
    try:
        text = Path(configuration_path).read_text(encoding="utf-8")
    except OSError as error:
        return None, f"Cannot read the configuration: {error}"
    value = output_location_from_text(text)
    if value is None:
        return PersistenceConfiguration.get_default_output_path(), ""
    return Path(value).expanduser(), ""


def assess(configuration_path: Path) -> FirstRunAssessment:
    configuration_path = Path(configuration_path).expanduser()
    if not configuration_path.is_file():
        folder = PersistenceConfiguration.get_default_output_path()
        return FirstRunAssessment(
            configuration_path, False, folder, check_folder(folder))
    folder, problem = configured_data_folder(configuration_path)
    if folder is None:
        # Not this step's to fix: the app's own load reports an unreadable file.
        default = PersistenceConfiguration.get_default_output_path()
        return FirstRunAssessment(
            configuration_path, True, default, FolderCheck(default, True), problem)
    return FirstRunAssessment(configuration_path, True, folder, check_folder(folder))


def create_configuration(configuration_path: Path, data_folder: Path) -> Path:
    """Write a default configuration that records into ``data_folder``."""
    configuration_path = Path(configuration_path).expanduser()
    configuration = SystemConfiguration()
    configuration.persistence.output_location = Path(data_folder).expanduser().as_posix()
    configuration.save_file(configuration_path, as_yaml=True)
    return configuration_path.with_suffix(".yaml")


def set_data_folder(configuration_path: Path, data_folder: Path) -> Optional[Path]:
    """Point an existing configuration at ``data_folder``, changing only that
    line. The previous file is kept beside it as ``*.before-first-run`` (an
    earlier backup is never overwritten). Returns the backup, or None when the
    folder was already the configured one and nothing was written."""
    configuration_path = Path(configuration_path).expanduser()
    text = configuration_path.read_text(encoding="utf-8")
    value = Path(data_folder).expanduser().as_posix()
    match = _OUTPUT_LOCATION_RE.search(text)
    if match is None:
        raise ValueError(
            f"{configuration_path} has no outputLocation line; set Preferences ▸ Data location")
    current = _yaml_scalar(match.group("value"))
    if current and Path(current).expanduser().as_posix() == value:
        return None
    new_text = text[:match.start("value")] + _yaml_string(value) + text[match.end("value"):]
    backup = configuration_path.with_name(configuration_path.name + ".before-first-run")
    index = 1
    while backup.exists():
        index += 1
        backup = configuration_path.with_name(f"{configuration_path.name}.before-first-run.{index}")
    shutil.copy2(configuration_path, backup)
    partial = configuration_path.with_name(configuration_path.name + ".partial")
    partial.write_text(new_text, encoding="utf-8")
    os.replace(partial, configuration_path)
    return backup


def _yaml_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run(command, timeout: float = 3.0) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return -1, str(error)
    return completed.returncode, (completed.stdout or "").strip()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_cameras(run=_run) -> HardwareCheck:
    sdk = _module_available("PySpin")
    if sys.platform.startswith("linux"):
        code, out = run(("lsusb", "-d", "1e10:"))
        devices = [line for line in out.splitlines() if line.strip()] if code == 0 else []
    else:
        devices = []
    if devices:
        detail = f"{len(devices)} FLIR USB device(s)"
        if not sdk:
            return HardwareCheck("Cameras", False, detail + ", but the Spinnaker SDK is not installed",
                                 "Install the Spinnaker SDK and its Python package (see INSTALL.md).")
        return HardwareCheck("Cameras", True, detail)
    return HardwareCheck(
        "Cameras", False,
        "No FLIR camera found" + ("" if sdk else "; the Spinnaker SDK is not installed"),
        "Connect the cameras to USB3 ports. You can add them later in Edit ▸ Edit Camera Settings.")


def check_nidaq(run=_run) -> HardwareCheck:
    driver = _module_available("nidaqmx")
    devices = []
    if sys.platform.startswith("linux"):
        code, out = run(("lspci", "-d", "1093:"))
        devices = [line for line in out.splitlines() if line.strip()] if code == 0 else []
    if devices:
        if not driver:
            return HardwareCheck("NI-DAQ", False, "NI hardware found, but nidaqmx is not installed",
                                 "Install NI-DAQmx and the nidaqmx Python package.")
        return HardwareCheck("NI-DAQ", True, f"{len(devices)} NI PCI device(s)")
    return HardwareCheck(
        "NI-DAQ", False, "No NI chassis found",
        "Turn the NI chassis on before the computer; it is detected only at boot.")


def check_can() -> HardwareCheck:
    from reachaq.app.model.hardware_scan import scan_can_adapters

    entry = scan_can_adapters()
    found = entry.state == "ok"
    return HardwareCheck(
        "Pellet board (CAN)", found, entry.info.replace("\n", " "),
        "" if found else "Connect the CAN adapter; the pellet board can be enabled later in File ▸ Hardware.")


def check_gpu() -> HardwareCheck:
    from reachaq.app.model.hardware_scan import scan_gpus

    entry = scan_gpus()
    found = entry.state == "ok"
    return HardwareCheck(
        "GPU (live pose)", found, entry.info.replace("\n", " "),
        "" if found else "Live pose needs an NVIDIA GPU and driver; recording works without one.")


HARDWARE_CHECKS: Tuple[Callable[[], HardwareCheck], ...] = (
    check_cameras, check_nidaq, check_can, check_gpu)


def run_hardware_checks(checks=HARDWARE_CHECKS) -> List[HardwareCheck]:
    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as error:
            name = getattr(check, "__name__", "check").replace("check_", "").upper()
            results.append(HardwareCheck(name, False, f"Check failed: {error}"))
    return results
