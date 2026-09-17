"""Demo playback source specification.

Maps camera names to pre-recorded video files so the acquisition application can
run its real pipeline against recorded frames when no animal is present.

Pure data: no Qt, no hardware, no application imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DEMO_SOURCES_PATH = Path("~/Autotrainer/demo/demo_sources.yaml")

#: The event kind the navigation controls step through by default.
DEFAULT_EVENT_KIND = "pellet_delivery"

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
    """Which video plays for which camera, how it is paced, and what is in it."""

    fps: float
    loop: bool = True
    cameras: Dict[str, Path] = field(default_factory=dict)
    #: Frame numbers worth jumping to, by event kind. Optional: a clip staged
    #: without its session's event file simply has none, and the navigation
    #: controls stay disabled rather than pretending to work.
    events: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

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

    def primary_event_kind(self) -> Optional[str]:
        """The event kind navigation should step through, or None.

        Prefers the default kind when the clip has it, otherwise takes whatever
        single kind is there. A clip staged from a session carries pellet
        deliveries; a reel of trials carries trial starts; the caller should not
        have to know which it is looking at.
        """
        if DEFAULT_EVENT_KIND in self.events:
            return DEFAULT_EVENT_KIND
        for kind in sorted(self.events):
            if self.events[kind]:
                return kind
        return None

    def event_label(self, plural: bool = False) -> str:
        """The primary kind as words, for a menu item or a status message."""
        kind = self.primary_event_kind()
        if kind is None:
            return "events" if plural else "event"
        label = kind.replace("_", " ")
        return f"{label}s" if plural else label

    def event_frames(self, kind: Optional[str] = None) -> Tuple[int, ...]:
        """Frames for one event kind, ascending, or empty when there are none.

        Defaults to the primary kind rather than a fixed name, so a reel of
        trials navigates as readily as a session recording.
        """
        if kind is None:
            kind = self.primary_event_kind()
        return () if kind is None else self.events.get(kind, ())

    def next_event_frame(self, frame: int,
                         kind: Optional[str] = None) -> Optional[int]:
        """The first event strictly after `frame`, or None past the last one.

        Strictly after, so repeatedly pressing next always advances rather than
        sticking on the event the playhead is already sitting on.
        """
        return next((f for f in self.event_frames(kind) if f > frame), None)

    def previous_event_frame(self, frame: int,
                             kind: Optional[str] = None) -> Optional[int]:
        """The last event strictly before `frame`, or None before the first."""
        return next((f for f in reversed(self.event_frames(kind)) if f < frame),
                    None)


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

    events = _load_events(raw.get("events"), path)

    sources = DemoSources(fps=fps, loop=bool(raw.get("loop", True)),
                          cameras=cameras, events=events)
    logger.info(
        "Loaded demo sources from %s: fps=%s loop=%s cameras=%s events=%s",
        path, sources.fps, sources.loop, sorted(sources.enabled_camera_names()),
        {kind: len(frames) for kind, frames in sorted(events.items())} or "none",
    )
    return sources


def _load_events(raw, path: Path) -> Dict[str, Tuple[int, ...]]:
    """Frame numbers per event kind, sorted and de-duplicated.

    Absent is fine and common - a clip staged without its session's event file
    has nothing to navigate to. Present but malformed is not: a spec that says
    it has events and cannot deliver them would leave the controls enabled and
    doing nothing.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise DemoSourcesError(
            f"Demo source specification {path} needs 'events' to be a mapping "
            f"of event kind to frame numbers")
    events: Dict[str, Tuple[int, ...]] = {}
    for kind, frames in raw.items():
        if not isinstance(frames, (list, tuple)):
            raise DemoSourcesError(
                f"Demo source specification {path}: events[{kind!r}] must be a "
                f"list of frame numbers")
        try:
            numbers = sorted({int(frame) for frame in frames})
        except (TypeError, ValueError) as err:
            raise DemoSourcesError(
                f"Demo source specification {path}: events[{kind!r}] has a "
                f"frame number that is not an integer") from err
        if any(number < 0 for number in numbers):
            raise DemoSourcesError(
                f"Demo source specification {path}: events[{kind!r}] has a "
                f"negative frame number")
        events[str(kind)] = tuple(numbers)
    return events
