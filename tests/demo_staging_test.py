import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load a script module by path; scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_demo_media = _load("stage_demo_media")
verify_demo_readiness = _load("verify_demo_readiness")


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"x")
    return path


def test_finds_each_camera_video(tmp_path):
    _touch(tmp_path, "20260914_x_session001_left-0000.mp4")
    _touch(tmp_path, "20260914_x_session001_right-0000.mp4")
    _touch(tmp_path, "20260914_x_session001_stimCam-0000.mp4")

    found = stage_demo_media.find_camera_videos(tmp_path, ["left", "right", "stimCam"])

    assert set(found) == {"left", "right", "stimCam"}
    assert found["left"].name.endswith("_left-0000.mp4")


def test_absent_camera_is_simply_missing(tmp_path):
    _touch(tmp_path, "20260914_x_session001_left-0000.mp4")

    found = stage_demo_media.find_camera_videos(tmp_path, ["left", "right"])

    assert set(found) == {"left"}


def test_split_recording_stages_only_the_first_part(tmp_path):
    _touch(tmp_path, "s_left-0001.mp4")
    first = _touch(tmp_path, "s_left-0000.mp4")
    _touch(tmp_path, "s_left-0002.mp4")

    found = stage_demo_media.find_camera_videos(tmp_path, ["left"])

    assert found["left"] == first


def test_camera_names_do_not_collide(tmp_path):
    """'camera4' must not be matched by a search for 'camera5'."""
    _touch(tmp_path, "s_camera4-0000.mp4")

    found = stage_demo_media.find_camera_videos(tmp_path, ["camera5", "camera6"])

    assert found == {}


def test_written_spec_loads_through_the_application_loader(tmp_path):
    from autotrainer.core.configuration.demo_sources import load_demo_sources

    left = _touch(tmp_path, "s_left-0000.mp4")
    right = _touch(tmp_path, "s_right-0000.mp4")
    spec = tmp_path / "demo_sources.yaml"

    stage_demo_media.write_spec(spec, 150.0, {"left": left, "right": right}, loop=True)
    sources = load_demo_sources(spec)

    assert sources.fps == 150.0
    assert sources.loop is True
    assert sources.enabled_camera_names() == {"left", "right"}


def test_written_spec_honours_no_loop(tmp_path):
    from autotrainer.core.configuration.demo_sources import load_demo_sources

    left = _touch(tmp_path, "s_left-0000.mp4")
    spec = tmp_path / "demo_sources.yaml"

    stage_demo_media.write_spec(spec, 150.0, {"left": left}, loop=False)

    assert load_demo_sources(spec).loop is False


def test_report_fails_only_on_fail_rows():
    report = verify_demo_readiness.Report()
    report.add(verify_demo_readiness.PASS, "a", "ok")
    report.add(verify_demo_readiness.WARN, "b", "hmm")
    report.add(verify_demo_readiness.SKIP, "c", "skipped")

    assert report.failed is False

    report.add(verify_demo_readiness.FAIL, "d", "broken")

    assert report.failed is True


def test_calibration_geometry_mismatch_is_a_failure(tmp_path):
    """A calibration built at a different size must fail, not warn.

    It does not crash at runtime: it produces wrong 3D under a correct-looking
    2D overlay, which is the whole reason this check exists.
    """
    import pickle

    calib = tmp_path / "calib"
    (calib / "camera_matrix").mkdir(parents=True)
    with (calib / "camera_matrix" / "stereo_params.pickle").open("wb") as handle:
        pickle.dump({"sessionA_left-sessionA_right": {"image_shape": [(256, 256), (256, 256)]}},
                    handle)

    report = verify_demo_readiness.Report()
    verify_demo_readiness.check_calibration(
        report, calib,
        {"left": {"width": 256, "height": 258, "fps": 150.0, "frames": 10},
         "right": {"width": 260, "height": 258, "fps": 150.0, "frames": 10}},
    )

    assert report.failed is True
    failures = [row for row in report.rows if row[0] == verify_demo_readiness.FAIL]
    assert len(failures) == 2


def test_calibration_geometry_match_passes(tmp_path):
    import pickle

    calib = tmp_path / "calib"
    (calib / "camera_matrix").mkdir(parents=True)
    with (calib / "camera_matrix" / "stereo_params.pickle").open("wb") as handle:
        pickle.dump({"sessionA_left-sessionA_right": {"image_shape": [(258, 256), (258, 260)]}},
                    handle)

    report = verify_demo_readiness.Report()
    verify_demo_readiness.check_calibration(
        report, calib,
        {"left": {"width": 256, "height": 258, "fps": 150.0, "frames": 10},
         "right": {"width": 260, "height": 258, "fps": 150.0, "frames": 10}},
    )

    assert report.failed is False


def test_missing_calibration_directory_is_a_failure(tmp_path):
    report = verify_demo_readiness.Report()

    verify_demo_readiness.check_calibration(report, tmp_path / "nope", {})

    assert report.failed is True
