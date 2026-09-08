"""Operator setup profiles for manual mouse positioning.

When an operator seats an animal, they need to know where its landmarks should
sit in each camera view.  A setup profile stores those expected positions in
normalized image coordinates per camera role, so one profile applies across
resolutions, crops, and sensor offsets.

reachAQ already owns stereo calibration, triangulation, and the DCS coordinate
model, so unlike the reach-training reference this module deliberately does not
carry a second calibration pipeline.  It covers only what reachAQ lacks: the
reusable profile of expected marker positions, its provenance, and the
conversion from normalized coordinates to pixels for a given camera geometry.

Provenance is enforced rather than advisory.  A profile whose markers came from
a placeholder rather than a calibrated session is marked as such and refuses to
be used for comparison, because expected landmark positions that look
authoritative but are invented would quietly bias how every animal is seated.
No placeholder coordinates ship here; the profile is supplied by the operator.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MOUSE_SETUP_SCHEMA_VERSION = 1

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class SetupProfileStatus(str, enum.Enum):
    """Whether a profile's marker positions may be trusted."""

    PLACEHOLDER = "placeholder_not_for_scientific_use"
    CALIBRATED = "calibrated"

    @property
    def is_usable(self) -> bool:
        return self is SetupProfileStatus.CALIBRATED


class MouseSetupError(RuntimeError):
    """Raised when a profile cannot be used as asked."""


@dataclasses.dataclass(frozen=True)
class CameraGeometry:
    """Enough of a camera's image geometry to place a normalized point.

    ``offset_x`` and ``offset_y`` are the sensor origin of a cropped capture, so
    a profile authored against a full frame still lands correctly when the
    camera is running a region of interest.
    """

    width: int
    height: int
    offset_x: int = 0
    offset_y: int = 0

    def __post_init__(self):
        width, height = int(self.width), int(self.height)
        if width < 1 or height < 1:
            raise ValueError("Camera geometry requires a positive width and height")
        offset_x, offset_y = int(self.offset_x), int(self.offset_y)
        if offset_x < 0 or offset_y < 0:
            raise ValueError("Camera geometry offsets cannot be negative")
        for name, value in (
            ("width", width),
            ("height", height),
            ("offset_x", offset_x),
            ("offset_y", offset_y),
        ):
            object.__setattr__(self, name, value)

    def to_pixels(self, normalized: Tuple[float, float]) -> Tuple[float, float]:
        """Convert a normalized point to absolute sensor pixels."""

        x, y = _validated_normalized(normalized)
        return (
            self.offset_x + x * self.width,
            self.offset_y + y * self.height,
        )

    def to_record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SetupMarker:
    """One landmark and where it should sit in each camera role."""

    marker_id: str
    label: str = ""
    color: str = "#ffffff"
    normalized_by_role: Mapping[str, Tuple[float, float]] = (
        dataclasses.field(default_factory=dict)
    )

    def __post_init__(self):
        marker_id = str(self.marker_id).strip().lower()
        if not _IDENTIFIER.fullmatch(marker_id):
            raise ValueError(
                "Marker ID must start with a lowercase letter or number and "
                "contain only lowercase letters, numbers, '.', '_' or '-'"
            )
        color = str(self.color).strip()
        if not _COLOR.fullmatch(color):
            raise ValueError(f"Marker {marker_id!r} has an invalid color {color!r}")
        positions = {}
        for role, point in dict(self.normalized_by_role).items():
            role = str(role).strip().lower()
            if not _IDENTIFIER.fullmatch(role):
                raise ValueError(f"Invalid camera role {role!r}")
            positions[role] = _validated_normalized(point)
        if not positions:
            raise ValueError(
                f"Marker {marker_id!r} needs a position for at least one camera role"
            )
        object.__setattr__(self, "marker_id", marker_id)
        object.__setattr__(self, "label", str(self.label).strip() or marker_id)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "normalized_by_role", positions)

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(sorted(self.normalized_by_role))

    def position_for(self, role: str) -> Optional[Tuple[float, float]]:
        return self.normalized_by_role.get(str(role).strip().lower())

    def to_record(self) -> Dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "label": self.label,
            "color": self.color,
            "normalized_by_role": {
                role: list(point)
                for role, point in sorted(self.normalized_by_role.items())
            },
        }


@dataclasses.dataclass(frozen=True)
class MouseSetupProfile:
    """A named, provenance-carrying set of expected marker positions."""

    profile_id: str
    revision: int = 1
    status: SetupProfileStatus = SetupProfileStatus.PLACEHOLDER
    markers: Tuple[SetupMarker, ...] = ()
    description: str = ""
    provenance: str = ""
    authored_at: str = ""
    schema_version: int = MOUSE_SETUP_SCHEMA_VERSION

    def __post_init__(self):
        profile_id = str(self.profile_id).strip().lower()
        if not _IDENTIFIER.fullmatch(profile_id):
            raise ValueError("Setup profile ID must be a lowercase identifier")
        if int(self.revision) < 1:
            raise ValueError("Setup profile revision must be positive")
        if self.schema_version != MOUSE_SETUP_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported mouse setup schema {self.schema_version}"
            )
        markers = tuple(
            item if isinstance(item, SetupMarker) else SetupMarker(**dict(item))
            for item in self.markers
        )
        if not markers:
            raise ValueError("A setup profile requires at least one marker")
        seen = set()
        for marker in markers:
            if marker.marker_id in seen:
                raise ValueError(f"Duplicate marker ID {marker.marker_id!r}")
            seen.add(marker.marker_id)
        status = SetupProfileStatus(self.status)
        if status.is_usable and not str(self.provenance).strip():
            raise ValueError(
                "A calibrated setup profile must record where its positions "
                "came from"
            )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "markers", markers)

    @property
    def is_usable(self) -> bool:
        return self.status.is_usable

    @property
    def roles(self) -> Tuple[str, ...]:
        roles = set()
        for marker in self.markers:
            roles.update(marker.roles)
        return tuple(sorted(roles))

    def require_usable(self) -> None:
        """Raise unless the profile may be used for comparison."""

        if not self.is_usable:
            raise MouseSetupError(
                f"Setup profile {self.profile_id!r} is marked "
                f"{self.status.value} and cannot be used for comparison"
            )

    def overlay_for(
        self,
        role: str,
        geometry: CameraGeometry,
        *,
        require_usable: bool = True,
    ) -> Tuple[Dict[str, Any], ...]:
        """Return pixel-space markers for one camera role."""

        if require_usable:
            self.require_usable()
        role = str(role).strip().lower()
        if role not in self.roles:
            raise MouseSetupError(
                f"Setup profile {self.profile_id!r} has no positions for "
                f"camera role {role!r}"
            )
        overlay = []
        for marker in self.markers:
            point = marker.position_for(role)
            if point is None:
                continue
            x, y = geometry.to_pixels(point)
            overlay.append({
                "marker_id": marker.marker_id,
                "label": marker.label,
                "color": marker.color,
                "normalized": list(point),
                "pixel_x": x,
                "pixel_y": y,
            })
        return tuple(overlay)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "MouseSetupProfile":
        markers = record.get("markers", ())
        if isinstance(markers, Mapping):
            # Accept the reference's keyed form as well as a plain sequence.
            markers = [
                {"marker_id": marker_id, **dict(value)}
                for marker_id, value in markers.items()
            ]
        return cls(
            profile_id=str(record["profile_id"]),
            revision=int(record.get("revision", 1)),
            status=record.get("status", SetupProfileStatus.PLACEHOLDER),
            markers=tuple(markers),
            description=str(record.get("description", "")),
            provenance=str(record.get("provenance", "")),
            authored_at=str(record.get("authored_at", "")),
            schema_version=int(
                record.get("schema_version", MOUSE_SETUP_SCHEMA_VERSION)
            ),
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "revision": self.revision,
            "status": self.status.value,
            "description": self.description,
            "provenance": self.provenance,
            "authored_at": self.authored_at,
            "markers": [marker.to_record() for marker in self.markers],
        }


def _validated_normalized(point: Iterable[float]) -> Tuple[float, float]:
    values = tuple(point)
    if len(values) != 2:
        raise ValueError("A normalized position needs exactly two coordinates")
    x, y = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError("Normalized positions must be finite")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("Normalized positions must be within 0.0 to 1.0")
    return (x, y)
