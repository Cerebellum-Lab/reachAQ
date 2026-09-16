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


def test_fps_param_is_an_int_the_camera_url_can_parse(tmp_path):
    """CameraBase.set_property parses fps with int(), and int("150.0") raises.

    A float here reaches the capture process as fps=150.0 in the camera URL and
    kills camera creation with "invalid literal for int() with base 10".
    """
    configuration = _configuration()
    sources = DemoSources(
        fps=150.015, loop=True, cameras={"left": tmp_path / "left.mp4"}
    )
    (tmp_path / "left.mp4").write_bytes(b"video")

    AppModel._apply_demo_playback_override(configuration, sources)

    left = next(c for c in configuration.cameras if c.id == CameraId.Left)
    assert isinstance(left.params["fps"], int)
    assert left.params["fps"] == 150
    assert int(str(left.params["fps"])) == 150
