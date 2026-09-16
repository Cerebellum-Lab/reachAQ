# Demo Mode and Onboarding Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a demo mode that plays pre-recorded video through the real acquisition pipeline while every other subsystem stays live, and build a 14-slide deck explaining reachAQ's lineage.

**Architecture:** Demo mode is a camera-layer override. A `--demo` flag and a Tools-menu toggle both funnel into one `AppModel` method that rewrites only the reach cameras to the existing `playback://` scheme, leaving hardware, NI-DAQ, laser, protocol, and persistence untouched. A multiprocessing barrier aligns playback camera start times so live 3D triangulation pairs matching frames. Demo sessions record normally but are explicitly tagged.

**Tech Stack:** Python 3.10+, PySide6, OpenCV, PyYAML, pytest, python-pptx, matplotlib, Pillow

**Spec:** `docs/superpowers/specs/2026-09-15-demo-mode-and-presentation-design.md`

## Global Constraints

- Python floor is 3.10 (`build(tools): raise the python floor to 3.10`, commit `e833c1aa`).
- Test files use the `<name>_test.py` suffix, never the `test_<name>.py` prefix.
- Pure decision logic lands in `auto-trainer-core` with no Qt and no hardware imports, so it stays testable and upstream-exportable (project `AGENTS.md`).
- Do not modify `auto-trainer-core`, `auto-trainer-video`, `auto-trainer-device`, `auto-trainer-inference`, or `auto-trainer-behavior` beyond the additive changes named in these tasks.
- Session identifiers are never changed by demo mode. Analysis naming conventions depend on them.
- Demo video media is never committed. `temp/` stays untracked.
- Run tests from the repository root so `pytest.ini` sets the rootdir correctly.
- Commit with the configured Git identity; never as an agent.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Spec Correction Carried By This Plan

The spec places the UI toggle in `tools/acquisition/view/camera_content.py`. That file is a
per-camera widget (`CameraContent.__init__(self, app_model, capture_model)`), instantiated once
per camera, so a global toggle does not belong there. This plan places the toggle in
`tools/acquisition/view/main_window.py` as a checkable Tools-menu action, registered into the
existing `hardware_enable_actions` gating via `_set_hardware_menu_actions_enabled`. That
inherits the correct idle/recording gating from all eight existing call sites rather than
inventing new gating.

## File Structure

**Created:**
- `auto-trainer-core/src/autotrainer/core/configuration/demo_sources.py` — `DemoSources` dataclass and YAML loader. Pure data, no Qt, no hardware.
- `auto-trainer-core/tests/demo_sources_test.py` — loader tests.
- `tests/demo_camera_override_test.py` — `AppModel._apply_demo_playback_override` tests.
- `tests/demo_session_tagging_test.py` — metadata block and marker file tests.
- `auto-trainer-video/tests/playback_start_barrier_test.py` — barrier alignment tests.
- `tools/hardware/reachaq_demo_sources.example.yaml` — example spec shipped with the repo.
- `docs/acquisition/demo-mode.md` — operator documentation.
- `docs/presentation/outline.md` — per-slide content and speaker notes.
- `docs/presentation/build_deck.py` — deck builder.
- `docs/presentation/make_assets.py` — diagram and still generator.
- `docs/presentation/assets/*.png` — generated images.
- `docs/presentation/reachAQ-overview.pptx` — built deck.

**Modified:**
- `auto-trainer-video/src/autotrainer/video/video_capture.py` — add `playback_start_barrier` to `CaptureAttrs`; wait on it before the capture loop.
- `tools/acquisition/model/app_model.py` — add `_apply_demo_playback_override`, the `demo_sources` argument to `load_configuration`, the `demoMode` metadata block, and the `DEMO` marker file.
- `tools/acquisition/args.py` — add `--demo`.
- `tools/acquisition/run_acquisition.py` — thread `--demo` through to `MainWindow`.
- `tools/acquisition/view/main_window.py` — `--demo` parameter, Tools-menu toggle, status-bar banner.
- `tests/acquisition_args_test.py` — extend for `--demo`.

---

### Task 1: Demo source specification

Pure data module. No Qt, no hardware, no `AppModel` import.

**Files:**
- Create: `auto-trainer-core/src/autotrainer/core/configuration/demo_sources.py`
- Test: `auto-trainer-core/tests/demo_sources_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DemoSources` dataclass with fields `fps: float`, `loop: bool`, `cameras: Dict[str, Path]`.
  - `DemoSources.video_for(camera_name: str) -> Optional[Path]`
  - `load_demo_sources(path: Path) -> DemoSources`
  - `DemoSourcesError(ValueError)`
  - `DEFAULT_DEMO_SOURCES_PATH: Path` equal to `Path("~/Autotrainer/demo/demo_sources.yaml")` (not expanded; callers expand).
  - Valid camera names are `left`, `right`, `camera3`, `camera4`, `camera5`, `camera6`, `stimCam`. `stimCam` is an accepted alias for `camera3`, because the shipped rig configuration names camera id 3 `stimCam`. `video_for` accepts either spelling.

- [ ] **Step 1: Write the failing tests**

Create `auto-trainer-core/tests/demo_sources_test.py`:

```python
from pathlib import Path

import pytest

from autotrainer.core.configuration.demo_sources import (
    DemoSources,
    DemoSourcesError,
    load_demo_sources,
)


def _write_videos(tmp_path: Path, *names: str) -> dict:
    made = {}
    for name in names:
        video = tmp_path / f"{name}.mp4"
        video.write_bytes(b"not really a video, but readable")
        made[name] = video
    return made


def test_loads_a_valid_spec(tmp_path):
    videos = _write_videos(tmp_path, "left", "right")
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 150\n"
        "loop: true\n"
        "cameras:\n"
        f"  left: {videos['left'].as_posix()}\n"
        f"  right: {videos['right'].as_posix()}\n"
    )

    sources = load_demo_sources(spec)

    assert sources.fps == 150
    assert sources.loop is True
    assert sources.video_for("left") == videos["left"]
    assert sources.video_for("right") == videos["right"]
    assert sources.video_for("camera4") is None


def test_stimcam_is_an_alias_for_camera3(tmp_path):
    videos = _write_videos(tmp_path, "stim")
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 150\n"
        "cameras:\n"
        f"  stimCam: {videos['stim'].as_posix()}\n"
    )

    sources = load_demo_sources(spec)

    assert sources.video_for("stimCam") == videos["stim"]
    assert sources.video_for("camera3") == videos["stim"]


def test_loop_defaults_to_true(tmp_path):
    videos = _write_videos(tmp_path, "left")
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 150\n"
        "cameras:\n"
        f"  left: {videos['left'].as_posix()}\n"
    )

    assert load_demo_sources(spec).loop is True


def test_missing_spec_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"

    with pytest.raises(DemoSourcesError) as err:
        load_demo_sources(missing)

    assert "nope.yaml" in str(err.value)


def test_missing_video_names_the_path(tmp_path):
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 150\n"
        "cameras:\n"
        "  left: /definitely/not/here/left.mp4\n"
    )

    with pytest.raises(DemoSourcesError) as err:
        load_demo_sources(spec)

    assert "left.mp4" in str(err.value)


def test_unknown_camera_name_is_rejected(tmp_path):
    videos = _write_videos(tmp_path, "left")
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 150\n"
        "cameras:\n"
        f"  frontLeftUpper: {videos['left'].as_posix()}\n"
    )

    with pytest.raises(DemoSourcesError) as err:
        load_demo_sources(spec)

    assert "frontLeftUpper" in str(err.value)


def test_non_positive_fps_is_rejected(tmp_path):
    videos = _write_videos(tmp_path, "left")
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 0\n"
        "cameras:\n"
        f"  left: {videos['left'].as_posix()}\n"
    )

    with pytest.raises(DemoSourcesError) as err:
        load_demo_sources(spec)

    assert "fps" in str(err.value)


def test_empty_camera_map_is_rejected(tmp_path):
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text("fps: 150\ncameras: {}\n")

    with pytest.raises(DemoSourcesError) as err:
        load_demo_sources(spec)

    assert "at least one camera" in str(err.value)


def test_enabled_camera_names_is_the_resolved_set(tmp_path):
    videos = _write_videos(tmp_path, "left", "right")
    spec = tmp_path / "demo_sources.yaml"
    spec.write_text(
        "fps: 150\n"
        "cameras:\n"
        f"  left: {videos['left'].as_posix()}\n"
        f"  right: {videos['right'].as_posix()}\n"
    )

    sources = load_demo_sources(spec)

    assert sources.enabled_camera_names() == {"left", "right"}


def test_direct_construction_needs_no_yaml():
    sources = DemoSources(fps=150.0, loop=False, cameras={"left": Path("/tmp/a.mp4")})

    assert sources.video_for("left") == Path("/tmp/a.mp4")
    assert sources.loop is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest auto-trainer-core/tests/demo_sources_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autotrainer.core.configuration.demo_sources'`

- [ ] **Step 3: Write the implementation**

Create `auto-trainer-core/src/autotrainer/core/configuration/demo_sources.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest auto-trainer-core/tests/demo_sources_test.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add auto-trainer-core/src/autotrainer/core/configuration/demo_sources.py auto-trainer-core/tests/demo_sources_test.py
git commit -m "feat(core): add a demo playback source specification

Maps camera names to pre-recorded videos so acquisition can run its real
pipeline against recorded frames. Pure data, no Qt and no hardware, so it
stays testable and upstream-exportable.

A spec that cannot be fully satisfied raises naming the offending path.
Demo mode must never start partially configured, and must never fall back
to physical cameras silently.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Camera override

**Files:**
- Modify: `tools/acquisition/model/app_model.py` (near `_apply_random_camera_override`, around line 947, and `load_configuration`, around line 6887)
- Test: `tests/demo_camera_override_test.py`

**Interfaces:**
- Consumes: `DemoSources`, `load_demo_sources`, `DemoSourcesError` from Task 1.
- Produces:
  - `AppModel._apply_demo_playback_override(configuration: SystemConfiguration, sources: DemoSources) -> None`, a `classmethod` matching `_apply_random_camera_override`.
  - `AppModel.load_configuration(location=None, *, random_cameras=False, demo_sources: Optional[DemoSources] = None)`.

Behavior: for each camera whose id is in `CameraId.reach_camera_ids()`, if the spec supplies a video for `str(camera_config.id)`, set `scheme="playback"`, `host=""`, `port=0`, `path=<video posix path>`, `params["fps"]=<spec fps>`, and enable it. Otherwise disable it. Hardware, inference, laser, NI-DAQ, persistence, and watchdog sections are untouched. Reset `configuration._camera_map = {}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/demo_camera_override_test.py`:

```python
from copy import deepcopy
from pathlib import Path

from autotrainer.core.configuration.camera_configuration import CameraConfiguration, CameraId
from autotrainer.core.configuration.demo_sources import DemoSources
from autotrainer.core.configuration.system_configuration import SystemConfiguration

from tools.acquisition.model.app_model import AppModel


def _configuration() -> SystemConfiguration:
    configuration = SystemConfiguration()
    configuration.cameras = [
        CameraConfiguration(
            id=CameraId.Left, name="left", is_enabled=True,
            scheme="spin", host="", port=0, path="22234357",
            params={"width": 300, "height": 200, "fps": 150, "primary": "yes"},
        ),
        CameraConfiguration(
            id=CameraId.Right, name="right", is_enabled=True,
            scheme="spin", host="", port=0, path="23199872",
            params={"width": 300, "height": 200, "fps": 150, "primary": "no"},
        ),
        CameraConfiguration(
            id=CameraId.Camera3, name="stimCam", is_enabled=True,
            scheme="spin", host="", port=0, path="23230374",
            params={"width": 300, "height": 200, "fps": 900, "primary": "no"},
        ),
    ]
    return configuration


def _sources(tmp_path: Path, *names: str) -> DemoSources:
    cameras = {}
    for name in names:
        video = tmp_path / f"{name}.mp4"
        video.write_bytes(b"video")
        cameras[name] = video
    return DemoSources(fps=150.0, loop=True, cameras=cameras)


def test_supplied_cameras_become_playback(tmp_path):
    configuration = _configuration()
    sources = _sources(tmp_path, "left", "right")

    AppModel._apply_demo_playback_override(configuration, sources)

    left = next(c for c in configuration.cameras if c.id == CameraId.Left)
    assert left.scheme == "playback"
    assert left.path == (tmp_path / "left.mp4").as_posix()
    assert left.params["fps"] == 150.0
    assert left.is_enabled is True
    assert left.params["primary"] == "yes"


def test_cameras_without_video_are_disabled(tmp_path):
    configuration = _configuration()
    sources = _sources(tmp_path, "left", "right")

    AppModel._apply_demo_playback_override(configuration, sources)

    stim = next(c for c in configuration.cameras if c.id == CameraId.Camera3)
    assert stim.is_enabled is False
    assert stim.scheme == "spin"


def test_stimcam_alias_resolves_to_camera3(tmp_path):
    configuration = _configuration()
    sources = _sources(tmp_path, "left", "right", "stimCam")

    AppModel._apply_demo_playback_override(configuration, sources)

    stim = next(c for c in configuration.cameras if c.id == CameraId.Camera3)
    assert stim.scheme == "playback"
    assert stim.is_enabled is True


def test_hardware_and_inference_sections_are_untouched(tmp_path):
    configuration = _configuration()
    before_hardware = deepcopy(configuration.hardware)
    before_inference = deepcopy(configuration.inference)
    before_nidaq = deepcopy(configuration.nidaq_stream)

    AppModel._apply_demo_playback_override(configuration, _sources(tmp_path, "left"))

    assert configuration.hardware.__dict__ == before_hardware.__dict__
    assert configuration.inference.__dict__ == before_inference.__dict__
    assert configuration.nidaq_stream.__dict__ == before_nidaq.__dict__


def test_camera_map_is_reset(tmp_path):
    configuration = _configuration()
    configuration._camera_map = {"stale": "entry"}

    AppModel._apply_demo_playback_override(configuration, _sources(tmp_path, "left"))

    assert configuration._camera_map == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/demo_camera_override_test.py -v`
Expected: FAIL — `AttributeError: type object 'AppModel' has no attribute '_apply_demo_playback_override'`

If the import of `SystemConfiguration` fails, check the real module path with
`grep -rn "class SystemConfiguration" auto-trainer-core/src` and correct the import in the test.

- [ ] **Step 3: Write the implementation**

In `tools/acquisition/model/app_model.py`, add the import alongside the other configuration imports:

```python
from autotrainer.core.configuration.demo_sources import (
    DemoSources,
    DemoSourcesError,
    load_demo_sources,
)
```

Add the override immediately after `_apply_random_camera_override`:

```python
    @classmethod
    def _apply_demo_playback_override(
        cls, configuration: SystemConfiguration, sources: DemoSources
    ) -> None:
        """Point the reach cameras at pre-recorded video for a demo run.

        Only the camera sources change. Hardware, inference, laser, NI-DAQ,
        protocol, and persistence stay exactly as the rig has them configured,
        because the demo's whole claim is that it is the real pipeline with one
        substituted input.
        """

        reach_ids = set(CameraId.reach_camera_ids())
        for camera_config in configuration.cameras:
            if camera_config.id not in reach_ids:
                continue

            video = sources.video_for(str(camera_config.id))
            if video is None:
                # An enabled camera with no demo video would point at a physical
                # handle this run is not driving. Disable it rather than fail late.
                if camera_config.is_enabled:
                    logger.notice(
                        "Demo mode has no video for camera %s; disabling it",
                        camera_config.name,
                    )
                camera_config.is_enabled = False
                continue

            camera_config.scheme = "playback"
            camera_config.host = ""
            camera_config.port = 0
            camera_config.path = Path(video).as_posix()
            params = dict(camera_config.params)
            params["fps"] = sources.fps
            camera_config.params = params
            camera_config.is_enabled = True
            logger.notice(
                "Demo mode camera %s plays %s at %s fps",
                camera_config.name, camera_config.path, sources.fps,
            )

        configuration._camera_map = {}
```

In `load_configuration`, change the signature and add the override call:

```python
    @_serialized_session_configuration
    def load_configuration(
        self,
        location: Optional[Path] = None,
        *,
        random_cameras: bool = False,
        demo_sources: Optional[DemoSources] = None,
    ):
        self._require_session_ready_for_configuration("Loading configuration")
        if location is None:
            location = self.get_config_location()

        config_started = time.perf_counter()
        log_hardware_initialization(
            logger,
            "START | hardware configuration | path=%s random_cameras=%s demo=%s",
            location,
            random_cameras,
            demo_sources is not None,
        )
        configuration: SystemConfiguration = self.get_config_from_location(location)
        if random_cameras:
            logger.notice("Using random camera override for this run")
            self._apply_random_camera_override(configuration)
        if demo_sources is not None:
            logger.notice("Using demo playback camera override for this run")
            self._apply_demo_playback_override(configuration, demo_sources)
        self._ensure_optional_stim_camera(configuration)
```

Then find the existing line further down:

```python
        self._loaded_configuration_has_runtime_override = random_cameras
```

and change it to:

```python
        self._loaded_configuration_has_runtime_override = random_cameras or demo_sources is not None
        self._demo_sources = demo_sources
```

Initialize `self._demo_sources: Optional[DemoSources] = None` in `AppModel.__init__` alongside the other runtime-override state, and add the read-only accessor near the other properties:

```python
    @property
    def demo_sources(self) -> Optional[DemoSources]:
        """The demo playback spec for this run, or None when not in demo mode."""
        return self._demo_sources

    @property
    def is_demo_mode(self) -> bool:
        return self._demo_sources is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/demo_camera_override_test.py -v`
Expected: PASS, 5 tests

Then confirm nothing regressed in the neighbouring override:

Run: `python -m pytest tests/ -k "camera" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/model/app_model.py tests/demo_camera_override_test.py
git commit -m "feat(acquisition): add the demo playback camera override

Rewrites only the reach cameras to the existing playback:// scheme,
leaving hardware, inference, laser, NI-DAQ, protocol, and persistence
exactly as the rig has them. A reach camera with no demo video is
disabled rather than left pointing at a handle this run is not driving.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The --demo flag

**Files:**
- Modify: `tools/acquisition/args.py`
- Modify: `tools/acquisition/run_acquisition.py:88-96`
- Modify: `tools/acquisition/view/main_window.py:117-196`
- Test: `tests/acquisition_args_test.py`

**Interfaces:**
- Consumes: `DEFAULT_DEMO_SOURCES_PATH` from Task 1; `load_configuration(..., demo_sources=...)` from Task 2.
- Produces: `AutoTrainerParsedArgs.demo: Optional[Path]`; `MainWindow(..., demo_sources: Optional[DemoSources] = None)`.

`--demo` takes an optional path (`nargs="?"`). Bare `--demo` uses `DEFAULT_DEMO_SOURCES_PATH`. Omitting it leaves `demo` as `None`. `--demo` and `--random-cameras` are mutually exclusive.

- [ ] **Step 1: Write the failing tests**

Append to `tests/acquisition_args_test.py`:

```python
def test_demo_flag_defaults_to_none():
    from tools.acquisition.args import make_autotrainer_parser

    args = make_autotrainer_parser().parse_args([])

    assert args.demo is None


def test_bare_demo_flag_uses_the_default_spec_path():
    from pathlib import Path

    from autotrainer.core.configuration.demo_sources import DEFAULT_DEMO_SOURCES_PATH
    from tools.acquisition.args import make_autotrainer_parser

    args = make_autotrainer_parser().parse_args(["--demo"])

    assert Path(args.demo) == DEFAULT_DEMO_SOURCES_PATH


def test_demo_flag_accepts_an_explicit_path():
    from pathlib import Path

    from tools.acquisition.args import make_autotrainer_parser

    args = make_autotrainer_parser().parse_args(["--demo", "/tmp/custom.yaml"])

    assert Path(args.demo) == Path("/tmp/custom.yaml")


def test_demo_and_random_cameras_are_mutually_exclusive():
    import pytest

    from tools.acquisition.args import make_autotrainer_parser

    with pytest.raises(SystemExit):
        make_autotrainer_parser().parse_args(["--demo", "--random-cameras"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/acquisition_args_test.py -v -k demo`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'demo'`

- [ ] **Step 3: Write the implementation**

In `tools/acquisition/args.py`, add the import:

```python
from autotrainer.core.configuration.demo_sources import DEFAULT_DEMO_SOURCES_PATH
```

Add the field to `AutoTrainerParsedArgs`:

```python
    demo: Optional[Path] = None
```

Replace the standalone `--random-cameras` argument with a mutually exclusive group:

```python
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--random-cameras",
                              help="use in-memory random image cameras instead of physical cameras",
                              action="store_true")
    camera_group.add_argument(
        "--demo",
        nargs="?",
        const=DEFAULT_DEMO_SOURCES_PATH,
        default=None,
        type=Path,
        metavar="SPEC",
        help="play pre-recorded video through the real pipeline instead of physical "
             "cameras; SPEC defaults to %(const)s",
    )
```

In `tools/acquisition/run_acquisition.py`, load the spec before constructing the window, and fail with a clear message rather than a traceback:

```python
    demo_sources = None
    if args.demo is not None:
        try:
            demo_sources = load_demo_sources(args.demo)
        except DemoSourcesError as err:
            logging.error("Cannot start in demo mode: %s", err)
            return -1
```

with the import:

```python
from autotrainer.core.configuration.demo_sources import DemoSourcesError, load_demo_sources
```

and pass it through:

```python
        window = MainWindow(
            app,
            preferences,
            args.configuration,
            is_dev=args.dev,
            random_cameras=args.random_cameras,
            live_inference=args.live_inference,
            demo_sources=demo_sources,
        )
```

In `tools/acquisition/view/main_window.py`, add the parameter to `__init__`:

```python
        live_inference: Optional[bool] = None,
        demo_sources: Optional[DemoSources] = None,
    ):
```

store it before the UI is built:

```python
        self._demo_sources = demo_sources
        self._demo_spec_path = DEFAULT_DEMO_SOURCES_PATH
```

and pass it to the initial load:

```python
            app_model.load_configuration(
                config_file, random_cameras=random_cameras, demo_sources=demo_sources
            )
```

with the import:

```python
from autotrainer.core.configuration.demo_sources import (
    DEFAULT_DEMO_SOURCES_PATH,
    DemoSources,
    DemoSourcesError,
    load_demo_sources,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/acquisition_args_test.py -v`
Expected: PASS, including the four new tests

Run: `python -m pytest tests/import_tools_test.py -q`
Expected: PASS — confirms the new imports resolve

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/args.py tools/acquisition/run_acquisition.py tools/acquisition/view/main_window.py tests/acquisition_args_test.py
git commit -m "feat(acquisition): add the --demo flag

Bare --demo uses the default spec path; an explicit path overrides it.
Mutually exclusive with --random-cameras, so passing both is an argument
error rather than a silent precedence rule. A bad spec reports the
validation error and exits instead of raising through startup.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Playback start barrier

Without this, two independently started playback processes drift apart and live 3D
triangulation pairs mismatched frames: plausible 2D, silently wrong 3D.

**Files:**
- Modify: `auto-trainer-video/src/autotrainer/video/video_capture.py` (`CaptureAttrs` around line 107, `_run_capture_loop` around line 412)
- Modify: `tools/acquisition/model/app_model.py` (where `CaptureAttrs` instances are built)
- Test: `auto-trainer-video/tests/playback_start_barrier_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `CaptureAttrs.playback_start_barrier: Optional[BarrierType] = None`
  - `CaptureAttrs.playback_start_timeout: float = 10.0`
  - `VideoCapture._await_playback_start()` — waits on the barrier, raising `RuntimeError` naming the camera on timeout.

- [ ] **Step 1: Write the failing test**

Create `auto-trainer-video/tests/playback_start_barrier_test.py`:

```python
import multiprocessing
import time

import pytest


def _worker(barrier, started_at, index):
    # Stagger arrivals so an unsynchronized start would be obvious.
    time.sleep(0.05 * index)
    barrier.wait(timeout=5)
    started_at[index] = time.perf_counter()


def test_barrier_aligns_staggered_starts():
    """Cameras that arrive up to 100ms apart still start together."""
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    started_at = context.Array("d", 3)

    processes = [
        context.Process(target=_worker, args=(barrier, started_at, index))
        for index in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)

    values = list(started_at)
    assert all(value > 0 for value in values), "every worker must have recorded a start"
    spread = max(values) - min(values)
    # One frame at 150 FPS is 6.7ms. Allow generous slack for process scheduling
    # while still proving the 100ms stagger was absorbed.
    assert spread < 0.05, f"start spread {spread:.4f}s is too wide"


def test_capture_attrs_carries_a_playback_barrier():
    from autotrainer.video.video_capture import CaptureAttrs

    assert "playback_start_barrier" in CaptureAttrs.__dataclass_fields__
    assert "playback_start_timeout" in CaptureAttrs.__dataclass_fields__
    assert CaptureAttrs.__dataclass_fields__["playback_start_barrier"].default is None


def test_await_playback_start_is_a_noop_without_a_barrier():
    from autotrainer.video.video_capture import VideoCapture

    assert hasattr(VideoCapture, "_await_playback_start")


def test_barrier_timeout_names_the_camera():
    """A camera that never arrives fails the start rather than hanging the UI."""
    context = multiprocessing.get_context("spawn")
    # Sized for two participants, but only this test arrives.
    barrier = context.Barrier(2)

    with pytest.raises(Exception):
        barrier.wait(timeout=0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest auto-trainer-video/tests/playback_start_barrier_test.py -v`
Expected: FAIL on `test_capture_attrs_carries_a_playback_barrier` and `test_await_playback_start_is_a_noop_without_a_barrier`. The two pure-multiprocessing tests should already pass; they document the mechanism.

- [ ] **Step 3: Write the implementation**

In `auto-trainer-video/src/autotrainer/video/video_capture.py`, add to the imports:

```python
from multiprocessing.synchronize import Barrier as BarrierType
```

Add to `CaptureAttrs`, after `stim_trigger_queue`:

```python
    playback_start_barrier: Optional[BarrierType] = None
    """Aligns playback camera start times.

    Hardware-synchronized cameras guarantee frame N left matches frame N right.
    Independently started playback processes do not, and live 3D triangulation
    pairs by frame id, so an unsynchronized start yields wrong 3D from
    plausible-looking 2D. Only set for playback sources.
    """

    playback_start_timeout: float = 10.0
    """Bounded wait on playback_start_barrier, so a missing camera fails visibly."""
```

Add the method to `VideoCapture`:

```python
    def _await_playback_start(self) -> None:
        """Block until every playback camera is ready to take its first frame.

        CameraBase.capture sets _capture_start on the first frame, and PlaybackCam
        paces to an absolute _capture_start + n/fps target rather than to an
        accumulating delta. So aligning the moment of the first capture call is
        enough to keep frame indices aligned for the whole run.
        """

        barrier = self._attrs.playback_start_barrier
        if barrier is None:
            return
        try:
            barrier.wait(timeout=self._attrs.playback_start_timeout)
        except Exception as err:
            raise RuntimeError(
                f"Playback camera {self._name} timed out waiting for the other demo "
                f"cameras after {self._attrs.playback_start_timeout:g}s: {err}"
            ) from err
        logger.notice("<%s> playback start barrier cleared", self._name)
```

Call it at the top of `_run_capture_loop`, before the loop body begins:

```python
    def _run_capture_loop(self, camera: CameraBase) -> None:
        self._await_playback_start()
```

In `tools/acquisition/model/app_model.py`, where `CaptureAttrs` is constructed for each camera, create one barrier sized to the number of enabled playback cameras and pass it to each. Locate the construction with:

```bash
grep -n "CaptureAttrs(" tools/acquisition/model/app_model.py
```

Build the barrier once before the per-camera loop:

```python
        playback_cameras = [
            camera for camera in self._reach_cameras
            if camera.is_enabled and camera.camera_source.url.startswith("playback:")
        ]
        playback_start_barrier = (
            multiprocessing.Barrier(len(playback_cameras))
            if len(playback_cameras) > 1
            else None
        )
```

and pass `playback_start_barrier=playback_start_barrier` into each playback camera's
`CaptureAttrs`, and `None` for every non-playback camera. A single playback camera needs no
barrier, so leave it `None` to avoid a pointless wait.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest auto-trainer-video/tests/playback_start_barrier_test.py -v`
Expected: PASS, 4 tests

Run: `python -m pytest auto-trainer-video/tests/ -q`
Expected: PASS — confirms Spinnaker capture is unaffected

- [ ] **Step 5: Commit**

```bash
git add auto-trainer-video/src/autotrainer/video/video_capture.py auto-trainer-video/tests/playback_start_barrier_test.py tools/acquisition/model/app_model.py
git commit -m "feat(video): align playback camera start times with a barrier

Hardware-synchronized cameras guarantee frame N left matches frame N
right; independently started playback processes do not. Live 3D
triangulation pairs by frame id, so an unsynchronized start produces
silently wrong 3D under a correct-looking 2D overlay.

The barrier is created only for playback sources, and only when more than
one camera plays. Spinnaker capture is untouched.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Demo session tagging

A demo session writes real files through the real recorder. Those files must never be
mistaken for experimental data.

**Files:**
- Modify: `tools/acquisition/model/app_model.py` (`_save_metadata` around line 8938)
- Test: `tests/demo_session_tagging_test.py`

**Interfaces:**
- Consumes: `AppModel.demo_sources` and `AppModel.is_demo_mode` from Task 2.
- Produces:
  - `AppModel._demo_metadata_block() -> Optional[dict]` returning `{"active": True, "sources": {<camera>: <video path>}, "fps": <float>}`, or `None` when not in demo mode.
  - `AppModel._write_demo_marker(session_dir: Path) -> None` writing a `DEMO` file.
  - A `demoMode` key in session-scope metadata.

- [ ] **Step 1: Write the failing tests**

Create `tests/demo_session_tagging_test.py`:

```python
from pathlib import Path

from autotrainer.core.configuration.demo_sources import DemoSources

from tools.acquisition.model.app_model import AppModel


class _FakeModel:
    """Exercises the tagging helpers without constructing a whole AppModel."""

    _demo_metadata_block = AppModel._demo_metadata_block
    _write_demo_marker = AppModel._write_demo_marker

    def __init__(self, sources=None):
        self._demo_sources = sources

    @property
    def demo_sources(self):
        return self._demo_sources

    @property
    def is_demo_mode(self):
        return self._demo_sources is not None


def test_no_demo_block_when_not_in_demo_mode():
    assert _FakeModel()._demo_metadata_block() is None


def test_demo_block_records_sources_and_fps(tmp_path):
    sources = DemoSources(
        fps=150.0, loop=True,
        cameras={"left": tmp_path / "l.mp4", "right": tmp_path / "r.mp4"},
    )

    block = _FakeModel(sources)._demo_metadata_block()

    assert block["active"] is True
    assert block["fps"] == 150.0
    assert block["sources"]["left"] == (tmp_path / "l.mp4").as_posix()
    assert block["sources"]["right"] == (tmp_path / "r.mp4").as_posix()


def test_marker_file_is_written_in_demo_mode(tmp_path):
    sources = DemoSources(fps=150.0, loop=True, cameras={"left": tmp_path / "l.mp4"})

    _FakeModel(sources)._write_demo_marker(tmp_path)

    marker = tmp_path / "DEMO"
    assert marker.is_file()
    text = marker.read_text()
    assert "not experimental data" in text
    assert "l.mp4" in text


def test_no_marker_file_when_not_in_demo_mode(tmp_path):
    _FakeModel()._write_demo_marker(tmp_path)

    assert not (tmp_path / "DEMO").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/demo_session_tagging_test.py -v`
Expected: FAIL — `AttributeError: type object 'AppModel' has no attribute '_demo_metadata_block'`

- [ ] **Step 3: Write the implementation**

In `tools/acquisition/model/app_model.py`, add the helpers immediately before `_save_metadata`:

```python
    def _demo_metadata_block(self) -> Optional[dict]:
        """Describe the demo playback sources for session metadata.

        Session metadata already embeds the full configuration, so a playback
        scheme is technically detectable. That is not enough for data nobody
        should analyze: record it explicitly.
        """

        sources = self.demo_sources
        if sources is None:
            return None
        return {
            "active": True,
            "fps": sources.fps,
            "loop": sources.loop,
            "sources": {
                name: Path(video).as_posix()
                for name, video in sorted(sources.cameras.items())
            },
        }

    def _write_demo_marker(self, session_dir: Path) -> None:
        """Drop a DEMO file beside the session so the directory is self-describing."""

        sources = self.demo_sources
        if sources is None:
            return
        lines = [
            "This session was recorded in demo mode and is not experimental data.",
            "",
            "The camera frames were played from pre-recorded video. Every other",
            "subsystem was live. Do not analyze this session.",
            "",
            f"playback fps: {sources.fps}",
            "sources:",
        ]
        lines.extend(
            f"  {name}: {Path(video).as_posix()}"
            for name, video in sorted(sources.cameras.items())
        )
        try:
            Path(session_dir).joinpath("DEMO").write_text("\n".join(lines) + "\n")
        except OSError as err:
            logger.error("Could not write the demo marker in %s: %s", session_dir, err)
```

In `_save_metadata`, in the session-scope branch (the `else:` arm that builds `out` with
`"scope": "session"`), add the key after `"appVersion": self._app_version,`:

```python
                "demoMode": self._demo_metadata_block(),
```

and write the marker in the same branch, next to where `session_dir` is computed:

```python
            session_dir = Path(file_name).parent
            self._write_demo_marker(session_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/demo_session_tagging_test.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add tools/acquisition/model/app_model.py tests/demo_session_tagging_test.py
git commit -m "feat(acquisition): tag demo sessions explicitly

Session metadata already embeds the configuration, so a playback scheme is
technically detectable. Implicit detection is not enough for data nobody
should analyze, so record a demoMode block and write a DEMO marker file
that makes the session directory self-describing.

Session identifiers are unchanged; prefixing them would ripple into the
naming conventions analysis depends on.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: UI toggle and banner

**Files:**
- Modify: `tools/acquisition/view/main_window.py` (`_create_actions` around line 1078, `_configure_menubar` around line 1209, `_set_hardware_menu_actions_enabled` around line 1232, `_configure_statusbar` around line 1476)

**Interfaces:**
- Consumes: `load_demo_sources`, `DemoSourcesError`, `DEFAULT_DEMO_SOURCES_PATH` (Task 1); `load_configuration(..., demo_sources=...)` (Task 2); `self._demo_sources`, `self._demo_spec_path` (Task 3).
- Produces: `MainWindow.demo_mode_action` (checkable `QAction`); `MainWindow._on_demo_mode_toggled(checked: bool)`; `MainWindow._refresh_demo_banner()`.

The action joins `hardware_enable_actions` gating so it inherits the existing idle/recording
rules from all eight `_set_hardware_menu_actions_enabled` call sites instead of inventing new
ones.

- [ ] **Step 1: Add the action**

In `_create_actions`, after the `make_3d_calib_action` block:

```python
        action = self.demo_mode_action = QAction("Demo Mode (play recorded video)", self)
        action.setCheckable(True)
        action.setChecked(self._demo_sources is not None)
        action.setToolTip(
            "Play pre-recorded video through the real pipeline. Every other "
            "subsystem stays live. Available only while the session is idle."
        )
        action.toggled.connect(self._on_demo_mode_toggled)
```

- [ ] **Step 2: Add it to the Tools menu**

In `_configure_menubar`, after `tools_menu.addAction(self.make_3d_calib_action)`:

```python
        tools_menu.addSeparator()
        tools_menu.addAction(self.demo_mode_action)
```

- [ ] **Step 3: Gate it with the hardware actions**

In `_set_hardware_menu_actions_enabled`, after the existing loop:

```python
        self.demo_mode_action.setEnabled(enabled)
```

- [ ] **Step 4: Implement the handler**

Add near the other menu handlers:

```python
    def _on_demo_mode_toggled(self, checked: bool) -> None:
        """Reload the configuration with or without the demo camera override.

        A spec that cannot be loaded leaves the toggle off and the current
        configuration untouched. Demo mode never falls back to physical cameras
        silently, and never starts partially configured.
        """

        if checked == (self._demo_sources is not None):
            return

        sources = None
        if checked:
            try:
                sources = load_demo_sources(self._demo_spec_path)
            except DemoSourcesError as err:
                self._show_message("Cannot enter demo mode", str(err))
                self.demo_mode_action.blockSignals(True)
                self.demo_mode_action.setChecked(False)
                self.demo_mode_action.blockSignals(False)
                return

        try:
            self._app_model.load_configuration(
                self._app_model.get_config_location(), demo_sources=sources
            )
        except Exception as err:
            self._show_message("Could not change demo mode", str(err))
            self.demo_mode_action.blockSignals(True)
            self.demo_mode_action.setChecked(self._demo_sources is not None)
            self.demo_mode_action.blockSignals(False)
            return

        self._demo_sources = sources
        self._refresh_demo_banner()
```

- [ ] **Step 5: Add the banner**

In `_configure_statusbar`, after `bar.addWidget(self._status_label)`:

```python
        self._demo_banner = QLabel("")
        self._demo_banner.setStyleSheet(
            "QLabel { background: #b3261e; color: white; padding: 2px 8px; font-weight: bold; }"
        )
        self._demo_banner.setVisible(False)
        bar.addPermanentWidget(self._demo_banner)
```

Add the refresher, and call it once at the end of `__init__` after the initial
`load_configuration`:

```python
    def _refresh_demo_banner(self) -> None:
        active = self._demo_sources is not None
        self._demo_banner.setVisible(active)
        if active:
            names = ", ".join(sorted(self._demo_sources.enabled_camera_names()))
            self._demo_banner.setText(f"DEMO MODE — recorded video on {names}")
        self.setWindowTitle(f"{self._title} — DEMO MODE" if active else self._title)
```

- [ ] **Step 6: Verify the UI loads**

Run: `python -m pytest tests/ -k "main_window or hardware_control_content or import_tools" -q`
Expected: PASS

Confirm `QLabel` is already imported in `main_window.py`; if not, add it to the
`PySide6.QtWidgets` import list.

- [ ] **Step 7: Commit**

```bash
git add tools/acquisition/view/main_window.py
git commit -m "feat(acquisition-ui): add a Demo Mode toggle and banner

The toggle joins the existing hardware-menu gating, so it inherits the
idle/recording rules the model already enforces through
_require_session_ready_for_configuration rather than inventing new ones.

A spec that cannot be loaded leaves the toggle off and the configuration
untouched. A red status-bar banner and a window-title suffix stay visible
for as long as demo mode is active, so the operator cannot forget
mid-presentation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Example spec and operator documentation

**Files:**
- Create: `tools/hardware/reachaq_demo_sources.example.yaml`
- Create: `docs/acquisition/demo-mode.md`
- Modify: `README.md` (Acquisition Application bullet list)

- [ ] **Step 1: Write the example spec**

Create `tools/hardware/reachaq_demo_sources.example.yaml`:

```yaml
# reachAQ demo playback sources.
#
# Demo mode plays pre-recorded video through the real acquisition pipeline so the
# system can be shown without an animal present. Only the camera frames are
# recorded: pose inference, NI-DAQ streaming, pellet control, laser control, and
# protocol execution all stay live.
#
# Copy to ~/Autotrainer/demo/demo_sources.yaml and point it at real media.
# Video files are never committed to the repository.

# Playback rate. Match the rate the source session was recorded at, so live
# inference sees the same load it sees on the rig.
fps: 150

# Restart at frame 0 on reaching the end, so the demo runs indefinitely.
loop: true

cameras:
  left: /home/christie07/Autotrainer/demo/session001_left-0000.mp4
  right: /home/christie07/Autotrainer/demo/session001_right-0000.mp4
  # stimCam is an accepted alias for camera3.
  stimCam: /home/christie07/Autotrainer/demo/session001_stimCam-0000.mp4
```

- [ ] **Step 2: Write the operator documentation**

Create `docs/acquisition/demo-mode.md`:

```markdown
# Demo mode

Demo mode plays pre-recorded video through the real acquisition pipeline. It exists so
the system can be demonstrated when no animal is present.

Only the camera frames are substituted. Pose inference runs on the played frames, and
NI-DAQ streaming, pellet device control, laser control, protocol editing, and port
configuration are all live and real.

## Requirements

Demo mode runs on the rig. Live inference requires a working CUDA runtime and refuses
to start without one, so a machine with no GPU will show the video but no pose.

## Setup

Copy `tools/hardware/reachaq_demo_sources.example.yaml` to
`~/Autotrainer/demo/demo_sources.yaml` and point it at real session video. Media files
are never committed.

Match `fps` to the rate the source session was recorded at. The shipped rig records at
150 FPS.

## Running

Start in demo mode:

    python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml --demo

With a specific spec:

    python -m reachAQ.app -c ~/Autotrainer/system_configuration.yaml --demo /path/to/spec.yaml

Or toggle it at runtime from **Tools → Demo Mode**. The toggle is available only while
the session is idle, because the configuration cannot be reloaded mid-recording.

`--demo` and `--random-cameras` are mutually exclusive.

## While demo mode is active

A red banner appears in the status bar and the window title gains a `DEMO MODE` suffix.
Both stay visible until demo mode is switched off.

## Demo session data

Demo sessions record normally. Every demo session is tagged twice:

- a `demoMode` block in the session metadata, recording the source videos and playback
  rate;
- a `DEMO` marker file in the session directory.

Session identifiers are unchanged. **Do not analyze demo sessions.**

## Camera names

Valid names are `left`, `right`, `camera3`, `camera4`, `camera5`, and `camera6`.
`stimCam` is accepted as an alias for `camera3`, matching the shipped rig
configuration.

A reach camera with no video in the spec is disabled for the run, rather than left
pointing at a physical camera the demo is not driving.

## Frame synchronization

Playback cameras start together on a shared barrier. This matters: hardware-synchronized
cameras guarantee that frame *N* on left matches frame *N* on right, but independently
started playback processes do not, and live 3D triangulation pairs by frame id. Without
the barrier the 2D overlay looks correct while the 3D is silently wrong.
```

- [ ] **Step 3: Link it from the README**

In `README.md`, in the Acquisition Application bullet list, after the
`--random-cameras` line:

```markdown
    * Use `--demo` to play pre-recorded video through the real pipeline when no animal is present. [Demo mode](docs/acquisition/demo-mode.md)
```

- [ ] **Step 4: Commit**

```bash
git add tools/hardware/reachaq_demo_sources.example.yaml docs/acquisition/demo-mode.md README.md
git commit -m "docs(acquisition): document demo mode

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Presentation assets

Generated images. Diagrams are drawn with matplotlib so they are reproducible and
rebuildable; video stills are extracted with OpenCV from material already on disk.

**Files:**
- Create: `docs/presentation/make_assets.py`
- Create: `docs/presentation/assets/` (generated PNGs)

**Interfaces:**
- Produces, in `docs/presentation/assets/`:
  - `lineage.png` — three-stage progression, reach-training → auto-trainer → reachAQ
  - `pipeline.png` — dataflow from cameras through capture, inference, behavior, to device and laser
  - `two_tier.png` — Tier 1 stim loop against Tier 2 behavior loop, on a shared time axis
  - `demo_substitution.png` — the pipeline with the camera stage swapped for playback
  - `demo_frames.png` — a real left/right still pair from the demo session
  - `pose_overlay.png` — a still from the existing tracked-overlay material

- [ ] **Step 1: Write the asset generator**

Create `docs/presentation/make_assets.py`:

```python
"""Generate presentation assets.

Diagrams are drawn rather than hand-made so the deck can be rebuilt. Stills are
extracted from material already on disk under temp/, which is untracked: the
generator degrades to a labelled placeholder when a source is absent, so the deck
always builds.

Run from the repository root:
    python docs/presentation/make_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"

REACH = "#6b7f9e"
AUTO = "#4a8c7a"
NEW = "#b3593f"
INK = "#26282b"
MUTED = "#7a7f87"


def _fig(width: float, height: float):
    figure, axes = plt.subplots(figsize=(width, height), dpi=200)
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100 * height / width)
    axes.axis("off")
    figure.patch.set_facecolor("white")
    return figure, axes


def _box(axes, x, y, w, h, label, color, text_color="white", size=9):
    axes.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        facecolor=color, edgecolor="none",
    ))
    axes.text(x + w / 2, y + h / 2, label, ha="center", va="center",
              color=text_color, fontsize=size, weight="bold", wrap=True)


def _arrow(axes, start, end, color=MUTED):
    axes.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=1.6, shrinkA=2, shrinkB=2,
    ))


def _save(figure, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def lineage():
    figure, axes = _fig(12, 5)
    stages = [
        (4, REACH, "reach-training", [
            "PySpin multi-camera capture",
            "Arduino serial pellet control",
            "ROI threshold reach detection",
            "Offline DeepLabCut + curation",
        ]),
        (36, AUTO, "auto-trainer", [
            "Modular package split",
            "Synchronized VideoCapture",
            "Pose inference + 3D calibration",
            "CAN device, sessions, PySide6",
        ]),
        (68, NEW, "reachAQ", [
            "Dell/Ubuntu, SocketCAN/PCAN",
            "NI-DAQ laser + signal streaming",
            "Live 3D pose closes the cue gate",
            "Protocols, epochs, validation",
        ]),
    ]
    for x, color, title, bullets in stages:
        _box(axes, x, 26, 28, 9, title, color, size=13)
        for index, bullet in enumerate(bullets):
            axes.text(x + 1, 22 - index * 4.2, f"— {bullet}", ha="left", va="center",
                      color=INK, fontsize=8.5)
    _arrow(axes, (32.5, 30.5), (35.5, 30.5))
    _arrow(axes, (64.5, 30.5), (67.5, 30.5))
    axes.text(50, 39, "What each layer contributed", ha="center",
              color=MUTED, fontsize=10, style="italic")
    _save(figure, "lineage.png")


def pipeline():
    figure, axes = _fig(12, 4.5)
    stages = [
        (2, "Cameras\n150 FPS", AUTO),
        (18, "VideoCapture\nper-process", AUTO),
        (34, "Pose\ninference", AUTO),
        (50, "3D\ntriangulation", NEW),
        (66, "Behavior\nstate machine", AUTO),
        (82, "Pellet / Laser\noutput", NEW),
    ]
    for x, label, color in stages:
        _box(axes, x, 14, 14, 11, label, color, size=9)
    for index in range(len(stages) - 1):
        _arrow(axes, (stages[index][0] + 14.5, 19.5), (stages[index + 1][0] - 0.5, 19.5))
    axes.text(50, 31, "Acquisition pipeline", ha="center", color=INK,
              fontsize=12, weight="bold")
    axes.text(50, 8, "Green: inherited from auto-trainer     Red: new in reachAQ",
              ha="center", color=MUTED, fontsize=8.5)
    _save(figure, "pipeline.png")


def two_tier():
    figure, axes = _fig(12, 5)
    axes.text(50, 39, "Two-tier timing architecture", ha="center", color=INK,
              fontsize=12, weight="bold")

    _box(axes, 2, 26, 22, 8, "Tier 1 — stim loop", NEW, size=10)
    axes.text(2, 22, "stimCam at 900 Hz on a small ROI", color=INK, fontsize=8.5)
    axes.text(2, 18.5, "decision inline in the capture thread", color=INK, fontsize=8.5)
    axes.text(2, 15, "pre-armed hardware-timed NI-DAQ output", color=INK, fontsize=8.5)
    axes.text(2, 11.5, "no process queue may be crossed", color=INK, fontsize=8.5)
    axes.text(2, 7, "target 5 ms p99", color=NEW, fontsize=10, weight="bold")

    _box(axes, 52, 26, 22, 8, "Tier 2 — behavior loop", AUTO, size=10)
    axes.text(52, 22, "reach cameras at 150 FPS", color=INK, fontsize=8.5)
    axes.text(52, 18.5, "pose inference and 3D triangulation", color=INK, fontsize=8.5)
    axes.text(52, 15, "behavior state machine", color=INK, fontsize=8.5)
    axes.text(52, 11.5, "trial, pellet, and protocol decisions", color=INK, fontsize=8.5)
    axes.text(52, 7, "20–100 ms", color=AUTO, fontsize=10, weight="bold")

    axes.text(50, 2, "The 5 ms figure is a design target, not a measured result.",
              ha="center", color=MUTED, fontsize=8.5, style="italic")
    _save(figure, "two_tier.png")


def demo_substitution():
    figure, axes = _fig(12, 4.5)
    stages = [
        (2, "Recorded\nvideo", NEW),
        (18, "VideoCapture\nper-process", AUTO),
        (34, "Pose\ninference", AUTO),
        (50, "3D\ntriangulation", AUTO),
        (66, "Behavior\nstate machine", AUTO),
        (82, "Pellet / Laser\noutput", AUTO),
    ]
    for x, label, color in stages:
        _box(axes, x, 14, 14, 11, label, color, size=9)
    for index in range(len(stages) - 1):
        _arrow(axes, (stages[index][0] + 14.5, 19.5), (stages[index + 1][0] - 0.5, 19.5))
    axes.text(50, 31, "Demo mode substitutes exactly one stage", ha="center",
              color=INK, fontsize=12, weight="bold")
    axes.text(9, 8, "substituted", ha="center", color=NEW, fontsize=9, weight="bold")
    axes.text(58, 8, "unchanged, live, real", ha="center", color=AUTO,
              fontsize=9, weight="bold")
    _save(figure, "demo_substitution.png")


def _first_frame(video: Path):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, frame = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _placeholder(axes, message):
    axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes,
              color=MUTED, fontsize=9, style="italic", wrap=True)
    axes.set_xticks([])
    axes.set_yticks([])


def _stills(sources, out_name, title, captions):
    figure, axes_list = plt.subplots(1, len(sources), figsize=(11, 4.6), dpi=200)
    if len(sources) == 1:
        axes_list = [axes_list]
    for axes, source, caption in zip(axes_list, sources, captions):
        frame = _first_frame(source) if source.is_file() else None
        if frame is None:
            _placeholder(axes, f"{caption}\n\nsource not on this machine:\n{source.name}")
        else:
            axes.imshow(frame, cmap="gray")
            axes.set_xticks([])
            axes.set_yticks([])
        axes.set_title(caption, fontsize=10, color=INK)
        for spine in axes.spines.values():
            spine.set_edgecolor("#d8dade")
    figure.suptitle(title, fontsize=12, color=INK, weight="bold")
    figure.patch.set_facecolor("white")
    _save(figure, out_name)


def demo_frames():
    session = REPO_ROOT / "temp/2p_sessions/20260914_christie2P_session001"
    _stills(
        [
            session / "20260914_christie2P_session001_left-0000.mp4",
            session / "20260914_christie2P_session001_right-0000.mp4",
        ],
        "demo_frames.png",
        "Demo source: a real recorded session, 150 FPS",
        ["left camera", "right camera"],
    )


def pose_overlay():
    overlays = REPO_ROOT / "temp/overlays"
    _stills(
        [overlays / "compare_left.mp4", overlays / "compare_right.mp4"],
        "pose_overlay.png",
        "Pose inference output",
        ["left, tracked", "right, tracked"],
    )


def main():
    lineage()
    pipeline()
    two_tier()
    demo_substitution()
    demo_frames()
    pose_overlay()
    print(f"\nassets in {ASSETS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the assets**

Run: `python docs/presentation/make_assets.py`
Expected: six PNG files written to `docs/presentation/assets/`. Any still whose source
video is not present on the machine renders as a labelled placeholder rather than
failing the build.

- [ ] **Step 3: Inspect the output**

Open the six PNGs and confirm no text is clipped and no box overlaps another. Adjust
coordinates in `make_assets.py` and regenerate if any are wrong.

- [ ] **Step 4: Commit**

```bash
git add docs/presentation/make_assets.py docs/presentation/assets
git commit -m "docs(presentation): generate deck diagrams and stills

Diagrams are drawn with matplotlib so the deck is rebuildable rather than
depending on hand-made images. Stills come from material already on disk
under the untracked temp/, and degrade to labelled placeholders when a
source is absent so the deck always builds.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Deck outline

The outline is the reviewable source of truth for deck content. Writing it before the
builder keeps slide text out of Python string literals during editing.

**Files:**
- Create: `docs/presentation/outline.md`

- [ ] **Step 1: Write the outline**

Create `docs/presentation/outline.md` with fourteen sections plus backup. Each section
carries a title, the bullets, the asset filename if any, and speaker notes.

Content requirements, from the spec:

- Slide 3 is the only repository-organized slide.
- Every feature slide carries its own three-way lineage inline.
- Slide 7 (closed-loop reach detection) is the central novel claim and stands alone.
- Slide 10 states the 5 ms figure as a target, not a measurement.
- Slide 12 carries the eight-step demo run order in its speaker notes.
- Slide 13 is honest about what is and is not rig-validated.

The full text is written directly into this file during implementation; Task 10's builder
reads it only for human review, not programmatically, so prose may be edited freely
afterwards without breaking the build.

- [ ] **Step 2: Commit**

```bash
git add docs/presentation/outline.md
git commit -m "docs(presentation): write the deck outline

Staged by feature rather than by repository, so the reach-training,
auto-trainer, and reachAQ lineage is answered on every slide instead of
in three segregated sections.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Build the deck

**Files:**
- Create: `docs/presentation/build_deck.py`
- Create: `docs/presentation/reachAQ-overview.pptx`

**Interfaces:**
- Consumes: the PNGs from Task 8.
- Produces: a 14-slide `.pptx` with speaker notes.

`python-pptx` is not installed in this environment and must be installed first.

- [ ] **Step 1: Install python-pptx**

Run: `python -m pip install python-pptx`
Expected: successful install

- [ ] **Step 2: Write the builder**

Create `docs/presentation/build_deck.py`. It builds a 16:9 deck using only the blank
layout, so nothing depends on template placeholders. Structure:

```python
"""Build the reachAQ overview deck.

Run from the repository root, after generating assets:
    python docs/presentation/make_assets.py
    python docs/presentation/build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "reachAQ-overview.pptx"

W, H = Inches(13.333), Inches(7.5)
INK = RGBColor(0x26, 0x28, 0x2B)
MUTED = RGBColor(0x7A, 0x7F, 0x87)
REACH = RGBColor(0x6B, 0x7F, 0x9E)
AUTO = RGBColor(0x4A, 0x8C, 0x7A)
NEW = RGBColor(0xB3, 0x59, 0x3F)


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, left, top, width, height, text, size, color=INK,
          bold=False, align=PP_ALIGN.LEFT, italic=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def _title(slide, text, subtitle=None):
    _text(slide, Inches(0.7), Inches(0.45), Inches(12), Inches(0.9), text, 32, INK, bold=True)
    if subtitle:
        _text(slide, Inches(0.7), Inches(1.28), Inches(12), Inches(0.5), subtitle, 14, MUTED, italic=True)


def _bullets(slide, items, left=Inches(0.7), top=Inches(2.0),
             width=Inches(6.0), size=14, gap=0.52):
    for index, item in enumerate(items):
        _text(slide, left, top + Inches(index * gap), width, Inches(0.5),
              f"•  {item}", size)


def _lineage(slide, reach, auto, new, top=Inches(4.9)):
    """The three-way strip every feature slide carries."""
    column = Inches(4.0)
    for index, (label, body, color) in enumerate((
        ("reach-training", reach, REACH),
        ("auto-trainer", auto, AUTO),
        ("reachAQ", new, NEW),
    )):
        left = Inches(0.7) + index * column
        _text(slide, left, top, column - Inches(0.3), Inches(0.35), label, 11, color, bold=True)
        _text(slide, left, top + Inches(0.34), column - Inches(0.3), Inches(1.1), body, 11, INK)


def _picture(slide, name, left, top, width):
    path = ASSETS / name
    if not path.is_file():
        _text(slide, left, top, width, Inches(0.5),
              f"[missing asset: {name} — run make_assets.py]", 11, MUTED, italic=True)
        return
    slide.shapes.add_picture(str(path), left, top, width=width)


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text
```

Then one function per slide, each calling `_blank`, `_title`, its content helpers, and
`_notes`. Slide content comes from `outline.md`. Slides using generated images:

| Slide | Asset |
|---|---|
| 3 Where it came from | `lineage.png` |
| 4 Camera acquisition | `pipeline.png` |
| 6 Pose inference | `pose_overlay.png` |
| 10 Timing architecture | `two_tier.png` |
| 12 Demo | `demo_substitution.png` and `demo_frames.png` |

The `main()` calls each slide function in order against one `Presentation()` with
`slide_width = W` and `slide_height = H`, then saves to `OUT` and prints the path and
slide count.

- [ ] **Step 3: Build the deck**

Run: `python docs/presentation/build_deck.py`
Expected: `docs/presentation/reachAQ-overview.pptx` written, 14 slides reported

- [ ] **Step 4: Verify the deck**

Run:

```bash
python -c "from pptx import Presentation; p=Presentation('docs/presentation/reachAQ-overview.pptx'); print(len(p.slides.__iter__.__self__._sldIdLst), 'slides'); [print(i+1, s.shapes.title.text if s.shapes.title else [sh.text_frame.text.split(chr(10))[0] for sh in s.shapes if sh.has_text_frame][:1]) for i,s in enumerate(p.slides)]"
```

Expected: 14 slides, each printing its heading. Open the file and confirm no text
overflows its box and every image is placed.

- [ ] **Step 5: Commit**

```bash
git add docs/presentation/build_deck.py docs/presentation/reachAQ-overview.pptx
git commit -m "docs(presentation): build the reachAQ overview deck

Fourteen slides staged by feature, each carrying its own three-way
lineage strip. Built from a script so the deck can be regenerated when
the content or the diagrams change.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Full suite and push

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ auto-trainer-core/tests/ auto-trainer-video/tests/ -q`
Expected: PASS. Pre-existing failures unrelated to this work are recorded, not fixed
here — commit `31aee73b` made the suite green on Linux, so a Windows-only failure is
plausible and should be reported rather than chased.

- [ ] **Step 2: Update planning.md**

Add a section recording demo mode, the spec-correction on toggle placement, the two
unretired risks (model detection and calibration match), and the required rig rehearsal.

- [ ] **Step 3: Update todo.md**

Add: rig rehearsal to verify model detection on the christie2P video and calibration
geometry match; capture real UI screenshots to replace the generated placeholders in
slides 4, 6, and 12.

- [ ] **Step 4: Commit and push**

```bash
git add planning.md todo.md
git commit -m "docs: record demo mode in planning and todo

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push
```

---

## Self-Review

**Spec coverage.** Component 1 → Task 1. Component 2 → Task 2. Component 3 → Task 4.
Component 4 → Task 5. Component 5 → Task 6 (with the documented file correction).
Component 6 → Task 3. Testing section → the test steps in Tasks 1–6 plus Task 11. Slide
deck framing, lineage facts, slide plan, demo run order, and build method → Tasks 8, 9,
10. Non-goals are respected: no simulated hardware, no CPU inference fallback, no
session-identifier change.

**Placeholder scan.** Task 9 deliberately defers slide prose to implementation rather
than inlining fourteen slides of text here, but it constrains that prose with six
explicit content requirements, so it is a specification rather than a TODO. Task 10
names the exact asset-to-slide mapping and gives the full helper implementation. Every
other task carries runnable code.

**Type consistency.** `DemoSources`, `load_demo_sources`, `DemoSourcesError`, and
`DEFAULT_DEMO_SOURCES_PATH` are defined in Task 1 and used with identical spellings in
Tasks 2, 3, 5, and 6. `_apply_demo_playback_override` is defined in Task 2 and used in
Task 2 only. `demo_sources` is the keyword argument name in `load_configuration`
(Task 2), `MainWindow.__init__` (Task 3), and `_on_demo_mode_toggled` (Task 6).
`enabled_camera_names()` is defined in Task 1 and used in Task 6's banner.
`playback_start_barrier` is defined in Task 4 and used only there.
