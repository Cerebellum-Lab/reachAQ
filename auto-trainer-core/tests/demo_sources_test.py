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
