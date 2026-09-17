"""Exact pellet-firmware compatibility policy and capability assessment."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Mapping, Optional, Tuple

import yaml


CAPABILITY_BITS = {
    "timing_trailer": 1 << 0,
    "time_sync": 1 << 1,
    "finite_stim3_pulse": 1 << 2,
    # Reserved for the pre-cue pellet authorization path. No released
    # firmware advertises it yet, so precheck methods stay unavailable
    # until a board reports this bit.
    "pellet_precheck": 1 << 3,
}


@dataclasses.dataclass(frozen=True)
class FirmwareCompatibility:
    version: str
    supported: bool
    board_type: str = "pellet_module"
    wire_schema_version: Optional[int] = None
    reported_capabilities: Tuple[str, ...] = ()
    required_capabilities: Tuple[str, ...] = ()
    missing_capabilities: Tuple[str, ...] = ()
    optional_capabilities: Tuple[str, ...] = ()
    qualification_status: str = "unknown"
    hardware_release: str = ""
    limitations: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def commands_allowed(self) -> bool:
        return self.supported and not self.missing_capabilities

    def to_record(self):
        record = dataclasses.asdict(self)
        record["commands_allowed"] = self.commands_allowed
        return record


class FirmwareCompatibilityPolicy:
    def __init__(self, entries: Mapping[str, dict]):
        self._entries = dict(entries)

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else (
            Path(__file__).resolve().parents[3]
            / "config/pellet-firmware-compatibility.yaml"
        )
        with path.open("r", encoding="utf-8") as stream:
            record = yaml.safe_load(stream) or {}
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported pellet firmware compatibility schema")
        entries = {}
        for item in record.get("versions", ()):
            version = str(item.get("version", "")).strip()
            if not version or version in entries:
                raise ValueError("Firmware policy versions must be nonempty and unique")
            for name in (
                *item.get("required_capabilities", ()),
                *item.get("optional_capabilities", ()),
            ):
                if name not in CAPABILITY_BITS:
                    raise ValueError(f"Unknown firmware capability in policy: {name}")
            entries[version] = dict(item)
        if not entries:
            raise ValueError("Firmware compatibility policy is empty")
        return cls(entries)

    def evaluate(
        self,
        version: str,
        *,
        wire_schema_version: Optional[int] = None,
        capabilities: int = 0,
        emulation: bool = False,
    ) -> FirmwareCompatibility:
        reported_version = str(version or "").strip()
        match = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", reported_version)
        version = match.group(1) if match else reported_version
        entry = self._entries.get(version)
        if entry is None:
            return FirmwareCompatibility(
                version=version,
                supported=False,
                reason=(
                    "Firmware version has not been explicitly qualified"
                    if version else "Firmware version has not been reported"
                ),
            )
        board_type = str(entry.get("board_type", "pellet_module"))
        if (board_type == "pellet_emulator") != bool(emulation):
            return FirmwareCompatibility(
                version=version,
                supported=False,
                board_type=board_type,
                reason="Emulator/physical-board policy type does not match the connection",
            )
        reported = tuple(
            name for name, bit in CAPABILITY_BITS.items() if int(capabilities) & bit
        )
        required = tuple(entry.get("required_capabilities", ()))
        missing = tuple(name for name in required if name not in reported)
        expected_wire = int(entry.get("wire_schema_version", 0))
        if wire_schema_version not in (None, 0, expected_wire):
            return FirmwareCompatibility(
                version=version,
                supported=False,
                board_type=board_type,
                wire_schema_version=wire_schema_version,
                reported_capabilities=reported,
                required_capabilities=required,
                missing_capabilities=missing,
                reason=(
                    f"Wire schema {wire_schema_version} does not match qualified "
                    f"schema {expected_wire}"
                ),
            )
        return FirmwareCompatibility(
            version=version,
            supported=True,
            board_type=board_type,
            wire_schema_version=wire_schema_version,
            reported_capabilities=reported,
            required_capabilities=required,
            missing_capabilities=missing,
            optional_capabilities=tuple(entry.get("optional_capabilities", ())),
            qualification_status=str(entry.get("qualification_status", "unknown")),
            hardware_release=str(entry.get("hardware_release", "")),
            limitations=tuple(entry.get("limitations", ())),
            reason=(
                "Required capability mismatch" if missing
                else "Exact firmware version is listed in the compatibility policy"
            ),
        )
