"""Resolve the bundled Spinnaker wheel for a platform and Python ABI."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Optional


class SpinnakerArtifactError(RuntimeError):
    """Raised when reachAQ has no bundled artifact for a requested platform."""


_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "x64": "x86_64",
    "arm64": "aarch64",
}


def _normalize_machine(value: str) -> str:
    value = value.strip().lower()
    return _MACHINE_ALIASES.get(value, value)


def select_spinnaker_wheel(
    *,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    python_tag: Optional[str] = None,
    repository_root: Optional[Path] = None,
) -> Path:
    """Return the matching bundled wheel or report the unsupported tuple."""
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    vendor_dir = root / "vendor" / "spinnaker"
    manifest = json.loads((vendor_dir / "manifest.json").read_text())
    requested = {
        "system": (platform.system() if system is None else system).lower(),
        "machine": _normalize_machine(
            platform.machine() if machine is None else machine
        ),
        "python": (
            f"cp{sys.version_info.major}{sys.version_info.minor}"
            if python_tag is None
            else python_tag.lower()
        ),
    }
    for artifact in manifest["artifacts"]:
        candidate = dict(artifact)
        candidate["machine"] = _normalize_machine(candidate["machine"])
        if all(candidate[key].lower() == value for key, value in requested.items()):
            wheel = vendor_dir / artifact["path"]
            if not wheel.is_file():
                raise SpinnakerArtifactError(
                    f"Spinnaker artifact is declared but missing: {wheel}"
                )
            return wheel
    supported = ", ".join(
        f"{item['system']}/{item['machine']}/{item['python']}"
        for item in manifest["artifacts"]
    )
    requested_text = "/".join(requested.values())
    raise SpinnakerArtifactError(
        f"No bundled Spinnaker wheel for {requested_text}; supported: {supported}"
    )
