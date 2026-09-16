"""Demo playback source specification.

Maps camera names to pre-recorded video files so the acquisition application can
run its real pipeline against recorded frames when no animal is present.

Pure data: no Qt, no hardware, no application imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DEMO_SOURCES_PATH = Path("~/Autotrainer/demo/demo_sources.yaml")

#: Canonical camera names, matching ``str(CameraId)``.
CANONICAL_CAMERA_NAMES = ("left", "right", "camera3", "camera4", "camera5", "camera6")

#: The shipped rig configuration names camera id 3 "stimCam". Accept both spellings.
CAMERA_NAME_ALIASES = {"stimcam": "camera3"}


class DemoSourcesError(ValueError):
    """A demo source specification could not be loaded or is not usable."""


def _canonical(name: str) -> Optional[str]:
    lowered = str(name).strip().lower()
    lowered = CAMERA_NAME_ALIASES.get(lowered, lowered)
    if lowered in CANONICAL_CAMERA_NAMES:
        return lowered
    return None


@dataclass
class DemoSources:
    """Which video plays for which camera, and how it is paced."""

    fps: float
    loop: bool = True
    cameras: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        resolved: Dict[str, Path] = {}
        for name, video in self.cameras.items():
            canonical = _canonical(name)
            if canonical is None:
                raise DemoSourcesError(
                    f"Unknown demo camera name {name!r}; "
                    f"expected one of {', '.join(CANONICAL_CAMERA_NAMES)} (or stimCam)"
                )
            resolved[canonical] = Path(video)
        self.cameras = resolved

    def video_for(self, camera_name: str) -> Optional[Path]:
        """Return the video for a camera name, or None if the spec has none."""
        canonical = _canonical(camera_name)
        if canonical is None:
            return None
        return self.cameras.get(canonical)

    def enabled_camera_names(self) -> Set[str]:
        """Canonical names of every camera this spec supplies a video for."""
        return set(self.cameras)


def load_demo_sources(path: Path) -> DemoSources:
    """Load and validate a demo source specification.

    Raises DemoSourcesError naming the offending path or field. A demo must never
    start partially configured: a spec that cannot be fully satisfied is an error,
    not a reason to fall back to physical cameras.
    """

    path = Path(path).expanduser()
    if not path.is_file():
        raise DemoSourcesError(f"Demo source specification not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as err:
        raise DemoSourcesError(f"Demo source specification {path} is not valid YAML: {err}") from err

    if not isinstance(raw, dict):
        raise DemoSourcesError(f"Demo source specification {path} must be a mapping")

    try:
        fps = float(raw.get("fps", 0))
    except (TypeError, ValueError) as err:
        raise DemoSourcesError(f"Demo source specification {path} has a non-numeric fps") from err
    if fps <= 0:
        raise DemoSourcesError(f"Demo source specification {path} needs a positive fps, got {fps!r}")

    cameras_raw = raw.get("cameras") or {}
    if not isinstance(cameras_raw, dict):
        raise DemoSourcesError(f"Demo source specification {path} needs a 'cameras' mapping")
    if not cameras_raw:
        raise DemoSourcesError(
            f"Demo source specification {path} must name at least one camera"
        )

    cameras: Dict[str, Path] = {}
    for name, video in cameras_raw.items():
        video_path = Path(str(video)).expanduser()
        if not video_path.is_file():
            raise DemoSourcesError(
                f"Demo video for camera {name!r} not found: {video_path}"
            )
        cameras[str(name)] = video_path

    sources = DemoSources(fps=fps, loop=bool(raw.get("loop", True)), cameras=cameras)
    logger.info(
        "Loaded demo sources from %s: fps=%s loop=%s cameras=%s",
        path, sources.fps, sources.loop, sorted(sources.enabled_camera_names()),
    )
    return sources
