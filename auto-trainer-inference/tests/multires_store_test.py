"""The calibration store: one folder per calibration, a sub-folder per preset, and resolution by id or time.

A folder is a calibration only once its calibration.yaml is written, last, at format 3
(format 2 folders still read), so a crash mid-write never leaves something that loads. Every preset folder holds today's
camera_matrix files, so existing loaders and DeepLabCut read it unchanged.

Installing a calibration renames the active folder to <name>_<install time>, so a backup's
suffix is when it stopped being active; the resolution tests build such folders by hand.

The fit is the synthetic rig turned 35 degrees about a generic axis (ruling R12): on the
symmetric rig the frame transform is diag(signs), and a transposed one would round-trip
unnoticed.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.inference.multires_calibration import pipeline, store
from autotrainer.inference.multires_calibration.fit import CaptureCamera
from autotrainer.inference.multires_calibration.geometry import (
    PresetGeometry,
    PresetSet,
    map_intrinsics,
    presets_for_cameras,
    presets_from_base,
    sensor_to_preset_points,
)
from autotrainer.inference.multires_calibration.products import (
    STEREO_KEY,
    anipose_toml,
    frame_transform,
    motor_to_inference_signs,
    preset_stereo_params,
)
from autotrainer.inference.multires_calibration.report import (
    CameraVerdict,
    CaptureSummary,
    PreviousCalibration,
    QualityReport,
    build_report,
)
from autotrainer.inference.multires_calibration.store import (
    INDEX_FILE,
    CalibrationGeometryError,
    CalibrationResolutionError,
    calibration_candidates,
    calibration_created,
    find_calibration,
    is_multires_calibration,
    load_index,
    load_preset,
    load_preset_set,
    load_sensor_fit,
    resolve_for_time,
    write_calibration,
)
from multires_synthetic import synthetic_rig


NAME = DEFAULT_3D_CALIB_DIR_NAME
BASE = PresetGeometry(4, 256, 256, 52, 6)
PRESETS = presets_from_base(BASE)
CAMERAS = ("left", "right")
LEGACY_FIXTURE = Path(__file__).parent / DEFAULT_3D_CALIB_DIR_NAME
RIG = synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0)
# One parameter the capture did not determine, as fit.py reports it, so fit.yaml must keep inf.
FIT = dataclasses.replace(
    RIG.truth,
    param_std={**RIG.truth.param_std, "left": {**RIG.truth.param_std["left"], "k2": math.inf}},
)
CAPTURE = {
    "gain_db": 1.0,
    "gamma": None,
    "exposure_us": 4500,
    "fps": 40,
    "waypoints_total": 37,
    "waypoints_verified": 37,
    "corner_refinement": "subpix",
}


@pytest.fixture(scope="module")
def report():
    n = len(RIG.observations)
    capture = CaptureSummary(
        waypoints_total=n,
        waypoints_verified=n,
        waypoints_unreached=0,
        views_kept={cam: n for cam in CAMERAS},
        rejections={cam: {} for cam in CAMERAS},
    )
    return build_report(FIT, RIG.observations, RIG.board, BASE.sensor_roi(), capture, [], None)


def write(folder, report):
    return write_calibration(folder, fit=FIT, presets=PRESETS, board=RIG.board, report=report, capture=CAPTURE)


def test_write_then_load_round_trip(tmp_path, report):
    folder = tmp_path / NAME
    before = datetime.now().replace(microsecond=0)
    written = write(folder, report)
    after = datetime.now()

    index = load_index(folder)
    assert index == written
    assert [g.name for g in index.presets] == ["256x256_bin4", "512x512_bin2", "1024x1024_bin1"]
    assert index.presets == tuple(PRESETS)
    assert index.board == RIG.board
    assert index.verdict == report.verdict
    assert index.override is None
    assert index.motor_scale == FIT.motor_scale
    assert index.capture == CAPTURE
    assert uuid.UUID(hex=index.id).hex == index.id
    # Local, naive and to the second, like the backup suffixes it is compared with.
    assert index.created.tzinfo is None and index.created.microsecond == 0
    assert before <= index.created <= after

    signs = motor_to_inference_signs()
    for g in PRESETS:
        loaded = load_preset(folder, g)
        expected = preset_stereo_params(FIT, g)
        assert loaded.geometry == g
        assert loaded.stereo_params.key_name == STEREO_KEY
        assert set(loaded.stereo_params.matrix) == set(expected.matrix)
        for key, value in expected.matrix.items():
            np.testing.assert_array_equal(np.asarray(loaded.stereo_params.matrix[key]), np.asarray(value), err_msg=key)
        np.testing.assert_array_equal(loaded.frame_transform, frame_transform(FIT, expected, signs))
    np.testing.assert_array_equal(
        load_preset(folder, BASE).stereo_params.matrix["cameraMatrix1"], preset_stereo_params(FIT, BASE).matrix["cameraMatrix1"]
    )


def test_sensor_fit_round_trips_exactly_with_undetermined_parameters(tmp_path, report):
    folder = tmp_path / NAME
    write(folder, report)
    assert ".inf" in (folder / "sensor" / "fit.yaml").read_text()

    loaded = load_sensor_fit(folder)
    assert list(loaded.cameras) == list(FIT.cameras)
    for cam, model in FIT.cameras.items():
        for field in ("K", "dist", "rvec", "tvec"):
            got, want = getattr(loaded.cameras[cam], field), np.asarray(getattr(model, field))
            assert got.shape == want.shape, f"{cam}.{field}"
            np.testing.assert_array_equal(got, want, err_msg=f"{cam}.{field}")
    np.testing.assert_array_equal(loaded.board_rvec, FIT.board_rvec)
    np.testing.assert_array_equal(loaded.board_tvec, FIT.board_tvec)
    assert loaded.motor_scale == FIT.motor_scale
    assert loaded.image_size == FIT.image_size
    assert loaded.rms_px == FIT.rms_px
    assert loaded.per_view_rms_px == FIT.per_view_rms_px
    assert math.isinf(loaded.param_std["left"]["k2"])
    assert loaded.param_std == FIT.param_std


def test_each_preset_folder_is_a_deeplabcut_camera_matrix_with_an_anipose_file(tmp_path, report):
    folder = tmp_path / NAME
    index = write(folder, report)
    signs = motor_to_inference_signs()
    for g in PRESETS:
        preset_dir = folder / "presets" / g.name
        for cam in CAMERAS:
            with (preset_dir / "camera_matrix" / f"{cam}_intrinsic_params.pickle").open("rb") as fh:
                intrinsics = pickle.load(fh)
            # Keyed by camera name, like the {cam}_intrinsic_params.pickle DLC and calibration_FLIR write.
            assert list(intrinsics) == [cam]
            assert set(intrinsics[cam]) == {"mtx", "dist"}
            np.testing.assert_array_equal(intrinsics[cam]["mtx"], map_intrinsics(FIT.cameras[cam].K, g))
            np.testing.assert_array_equal(intrinsics[cam]["dist"], np.reshape(FIT.cameras[cam].dist, (1, -1)))
        assert (preset_dir / "calibration.toml").read_text() == anipose_toml(
            FIT, PresetSet(g.name, {cam: g for cam in CAMERAS}), calibration_id=index.id,
            cameras=[CaptureCamera(cam, None, None) for cam in CAMERAS],
        )
        frame = yaml.safe_load((preset_dir / "frame.yaml").read_text())
        assert frame["units"] == "mm"
        assert frame["signs"] == list(signs)
        assert np.shape(frame["transform"]) == (3, 3)
        readme = (preset_dir / "README.txt").read_text()
        assert 'camera_names: ["left", "right"]' in readme
        assert "calibration.toml" in readme
    assert (folder / "quality_report.txt").read_text().rstrip("\n") == report.to_text()
    assert QualityReport.from_json((folder / "quality_report.json").read_text()).to_json() == report.to_json()


def test_extra_files_are_copied_in_before_the_index(tmp_path, report):
    image = tmp_path / "coverage_left.png"
    image.write_bytes(b"\x89PNG not really")
    folder = tmp_path / NAME
    write_calibration(
        folder, fit=FIT, presets=PRESETS, board=RIG.board, report=report, capture=CAPTURE,
        extra_files={"coverage_left.png": image},
    )
    assert (folder / "coverage_left.png").read_bytes() == image.read_bytes()


def test_unknown_geometry_raises(tmp_path, report):
    folder = tmp_path / NAME
    write(folder, report)
    with pytest.raises(CalibrationGeometryError) as raised:
        load_preset(folder, PresetGeometry(4, 256, 256, 0, 0))
    message = str(raised.value)
    assert "256x256_bin4" in message
    # The request and the available presets differ only in their offsets, so both must be named.
    assert "(0, 0)" in message and "(52, 6)" in message
    assert "512x512_bin2" in message and "1024x1024_bin1" in message


def test_partial_folder_is_not_a_calibration(tmp_path, report):
    folder = tmp_path / NAME
    write(folder, report)
    assert is_multires_calibration(folder)
    (folder / INDEX_FILE).unlink()
    assert not is_multires_calibration(folder)
    with pytest.raises(ValueError):
        load_preset(folder, BASE)
    with pytest.raises(ValueError):
        load_sensor_fit(folder)
    (folder / INDEX_FILE).write_text(yaml.safe_dump({"format": 1}))
    assert not is_multires_calibration(folder)


def test_a_write_that_fails_midway_leaves_no_calibration(tmp_path, report, monkeypatch):
    folder = tmp_path / NAME
    partial = tmp_path / f"{NAME}.partial"
    written = []

    def failing_toml(fit, preset_set, **kwargs):
        # As a crash would: after the sensor file and the first preset, partway through the second.
        if len(written) == 2:
            raise OSError("disk full")
        written.append(preset_set.name)
        return anipose_toml(fit, preset_set, **kwargs)

    monkeypatch.setattr(store, "anipose_toml", failing_toml)
    with pytest.raises(OSError, match="disk full"):
        write(folder, report)
    assert not folder.exists()
    assert (partial / "calibration.toml").is_file()
    assert (partial / "presets" / written[1] / "calibration.toml").is_file()
    assert not is_multires_calibration(partial)
    # Nor is the .partial folder a candidate for an old session.
    with pytest.raises(CalibrationResolutionError):
        resolve_for_time(tmp_path, datetime(2026, 5, 1))

    monkeypatch.undo()
    write(folder, report)
    assert is_multires_calibration(folder)
    assert not partial.exists()


def test_write_never_overwrites_and_clears_a_stale_partial(tmp_path, report):
    folder = tmp_path / NAME
    stale = tmp_path / f"{NAME}.partial"
    (stale / "presets" / "left over").mkdir(parents=True)
    first = write(folder, report)
    assert not stale.exists()
    assert is_multires_calibration(folder)

    with pytest.raises(FileExistsError):
        write(folder, report)
    assert load_index(folder) == first
    assert not stale.exists()


def test_existing_fixture_is_legacy():
    assert (LEGACY_FIXTURE / "camera_matrix" / "stereo_params.pickle").is_file()
    assert not is_multires_calibration(LEGACY_FIXTURE)


def test_calibration_created_reads_the_index_or_the_legacy_pickle_time(tmp_path, report):
    folder = tmp_path / NAME
    written = write(folder, report)
    assert calibration_created(folder) == written.created

    legacy_pickle = tmp_path / "legacy" / "camera_matrix" / "stereo_params.pickle"
    legacy_pickle.parent.mkdir(parents=True)
    legacy_pickle.write_bytes(b"not read, only dated")
    made = datetime(2026, 3, 3, 9, 8, 15)
    stamp = made.timestamp() + 0.25
    os.utime(legacy_pickle, (stamp, stamp))
    assert calibration_created(tmp_path / "legacy") == made


def test_resolve_for_time_with_backups(tmp_path):
    first = tmp_path / f"{NAME}_20260303_090815"
    second = tmp_path / f"{NAME}_20260701_120000"
    active = tmp_path / NAME
    for path in (first, second, active):
        path.mkdir()
    assert resolve_for_time(tmp_path, datetime(2026, 2, 1)) == first
    assert resolve_for_time(tmp_path, datetime(2026, 5, 1)) == second
    assert resolve_for_time(tmp_path, datetime(2026, 8, 1)) == active
    # A backup's suffix is the moment it was replaced, so from then on the next one was active.
    assert resolve_for_time(tmp_path, datetime(2026, 3, 3, 9, 8, 15)) == second
    # An aware time is compared in local time, as the suffixes were written.
    assert resolve_for_time(tmp_path, datetime(2026, 5, 1, 12, tzinfo=timezone.utc)) == second


def test_resolve_without_a_backup_uses_the_active_folder_or_refuses(tmp_path):
    with pytest.raises(CalibrationResolutionError):
        resolve_for_time(tmp_path, datetime(2026, 5, 1))
    (tmp_path / f"{NAME}_20260303_090815").mkdir()
    # The only backup ended before the session and there is no active folder.
    with pytest.raises(CalibrationResolutionError):
        resolve_for_time(tmp_path, datetime(2026, 5, 1))
    (tmp_path / NAME).mkdir()
    assert resolve_for_time(tmp_path, datetime(2026, 5, 1)) == tmp_path / NAME


def _made(index, folder: Path, when: datetime):
    """`index`, the calibration in `folder`, as if it had been made at `when`."""
    data = yaml.safe_load((folder / INDEX_FILE).read_text())
    data["created"] = when.isoformat(timespec="seconds")
    (folder / INDEX_FILE).write_text(yaml.safe_dump(data, sort_keys=False))
    made = load_index(folder)
    assert made == dataclasses.replace(index, created=when)
    return made


def _dated_legacy(folder: Path, made: datetime) -> Path:
    """A legacy folder whose stereo_params.pickle is dated `made`, as copytree keeps it."""
    pickle_path = folder / "camera_matrix" / "stereo_params.pickle"
    pickle_path.parent.mkdir(parents=True)
    pickle_path.write_bytes(b"not read, only dated")
    os.utime(pickle_path, (made.timestamp(), made.timestamp()))
    return folder


def test_resolve_refuses_a_time_before_every_kept_calibration_was_made(tmp_path, report):
    # Spec, Old sessions (ruling R49): the calibration such a session was recorded with is
    # not kept, and the oldest one that is cannot stand in for it.
    backup = _dated_legacy(tmp_path / f"{NAME}_20260303_090815", datetime(2026, 1, 10, 9, 30))
    with pytest.raises(CalibrationResolutionError) as refused:
        resolve_for_time(tmp_path, datetime(2026, 1, 5, 8))
    assert "2026-01-05 08:00:00" in str(refused.value) and "2026-01-10 09:30:00" in str(refused.value)
    assert resolve_for_time(tmp_path, datetime(2026, 1, 10, 9, 31)) == backup

    # The same for a format-2 active folder, dated by its index.
    only = tmp_path / "format2"
    only.mkdir()
    written = write(only / NAME, report)
    with pytest.raises(CalibrationResolutionError, match=str(written.created)):
        resolve_for_time(only, written.created.replace(year=written.created.year - 1))
    assert resolve_for_time(only, written.created) == only / NAME


def test_resolve_refuses_duplicate_windows(tmp_path):
    # Folder names are unique, but DATE_TIME_FORMAT also parses an hour typed without its
    # leading zero, so two backups can claim the same end time.
    (tmp_path / f"{NAME}_20260303_090815").mkdir()
    (tmp_path / f"{NAME}_20260303_90815").mkdir()
    (tmp_path / NAME).mkdir()
    with pytest.raises(CalibrationResolutionError, match="2026-03-03 09:08:15"):
        resolve_for_time(tmp_path, datetime(2026, 2, 1))


def test_backup_names_that_do_not_parse_are_ignored_with_a_warning(tmp_path, caplog):
    for name in (NAME, f"{NAME}_20260303_090815", f"{NAME}_crop256", f"{NAME}.partial", f"{NAME}.new", "other"):
        (tmp_path / name).mkdir()
    (tmp_path / f"{NAME}_20260401_000000").write_text("a file, not a calibration folder")
    with caplog.at_level(logging.WARNING, logger=store.__name__):
        candidates = calibration_candidates(tmp_path)
    assert [(c.path.name, c.active_until) for c in candidates] == [
        (f"{NAME}_20260303_090815", datetime(2026, 3, 3, 9, 8, 15)),
        (NAME, None),
    ]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and f"{NAME}_crop256" in warnings[0]


def test_a_folder_named_like_a_backup_is_warned_about_once_per_process(tmp_path, caplog):
    # Every lookup lists the folders again (the rig's 4mm_6r_8c_4x_crop256, on every session's
    # offline 3D): one warning says it, later lookups log it at debug level.
    for name in (NAME, f"{NAME}_crop256"):
        (tmp_path / name).mkdir()
    with caplog.at_level(logging.DEBUG, logger=store.__name__):
        for _ in range(3):
            calibration_candidates(tmp_path)
    about = [r.levelno for r in caplog.records if f"{NAME}_crop256" in r.getMessage()]
    assert about == [logging.WARNING, logging.DEBUG, logging.DEBUG]


def test_find_by_id(tmp_path, report):
    active = tmp_path / NAME
    backup = tmp_path / f"{NAME}_20260303_090815"
    legacy_backup = tmp_path / f"{NAME}_20260101_000000"
    index = write(active, report)
    backup_index = write(backup, report)
    legacy_backup.mkdir()

    assert find_calibration(tmp_path, index.id) == active
    assert find_calibration(tmp_path, backup_index.id) == backup
    ids = {c.path.name: c.calibration_id for c in calibration_candidates(tmp_path)}
    assert ids == {legacy_backup.name: None, backup.name: backup_index.id, NAME: index.id}
    with pytest.raises(CalibrationResolutionError):
        find_calibration(tmp_path, uuid.uuid4().hex)


def corrupt_yaml(path):
    # As a file cut short would read: an unclosed flow mapping.
    path.write_text("format: 2\npresets:\n- {name: 256x256_bin4, binning: 4, wid\n")


def drop_id(path):
    data = yaml.safe_load(path.read_text())
    del data["id"]
    path.write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.mark.parametrize("corrupt", [corrupt_yaml, drop_id], ids=["invalid-yaml", "no-id"])
def test_a_corrupt_backup_index_blocks_only_its_own_folder(tmp_path, report, caplog, corrupt):
    damaged = tmp_path / f"{NAME}_20260303_090815"
    second = tmp_path / f"{NAME}_20260701_120000"
    active = tmp_path / NAME
    write(damaged, report)
    # Each made before it became active, as installs leave them: resolve_for_time refuses a
    # time before the calibration it would pick was made.
    second_index = _made(write(second, report), second, datetime(2026, 3, 3, 9))
    active_index = _made(write(active, report), active, datetime(2026, 7, 1, 11))
    corrupt(damaged / INDEX_FILE)

    with caplog.at_level(logging.WARNING, logger=store.__name__):
        assert not is_multires_calibration(damaged)
        # Still placed in time by its folder name; it just has no id.
        assert resolve_for_time(tmp_path, datetime(2026, 2, 1)) == damaged
        assert resolve_for_time(tmp_path, datetime(2026, 5, 1)) == second
        assert resolve_for_time(tmp_path, datetime(2026, 8, 1)) == active
        assert find_calibration(tmp_path, second_index.id) == second
        assert find_calibration(tmp_path, active_index.id) == active
        ids = {c.path.name: c.calibration_id for c in calibration_candidates(tmp_path)}
    assert ids == {damaged.name: None, second.name: second_index.id, NAME: active_index.id}
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings and all(str(damaged) in w for w in warnings)

    # Everything that would use the calibration still refuses it loudly.
    loaders = {
        "load_index": load_index,
        "load_preset": lambda folder: load_preset(folder, BASE),
        "load_sensor_fit": load_sensor_fit,
        "calibration_created": calibration_created,
    }
    for name, load in loaders.items():
        with pytest.raises(ValueError, match=INDEX_FILE):
            load(damaged)
            pytest.fail(f"{name} accepted a corrupt index")


# --- Format 3: any 2-6 cameras ----------------------------------------------------------

FORMAT2_FIXTURE = Path(__file__).parent / "multires_format2_calibration"
RIG3 = synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0, cameras=("left", "right", "camera4"))
# stimCam was in the capture but saw too little: it is not in the fit.
CAPTURE3 = [
    CaptureCamera("left", "24152533", 0),
    CaptureCamera("right", "24152534", 1),
    CaptureCamera("camera4", "24152535", 3),
    CaptureCamera("stimCam", "20142630", 2),
]
STIM_REASON = "too few board views: 4 of 12 needed"
LIVE3 = {"left": BASE, "right": BASE, "camera4": PresetGeometry(2, 300, 200, 10, 40),
         "stimCam": PresetGeometry(1, 720, 540, 0, 0)}
CHANGE_METRICS = {"camera_rotation_change_deg", "focal_change_rel", "baseline_change_mm"}


@pytest.fixture(scope="module")
def report3():
    n = len(RIG3.observations)
    names = [c.name for c in CAPTURE3]
    capture = CaptureSummary(
        waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
        views_kept={cam: (4 if cam == "stimCam" else n) for cam in names}, rejections={cam: {} for cam in names},
    )
    verdicts = {cam: CameraVerdict(cam, cam != "stimCam", STIM_REASON if cam == "stimCam" else "") for cam in names}
    return build_report(RIG3.truth, RIG3.observations, RIG3.board, BASE.sensor_roi(), capture, [], None,
                        cameras=verdicts)


def write3(folder, report3, fit=RIG3.truth, cameras=CAPTURE3, live=LIVE3):
    return write_calibration(folder, fit=fit, preset_sets=presets_for_cameras(live), board=RIG3.board,
                             report=report3, capture=CAPTURE, cameras=cameras)


def toml_loads():
    """tomllib where Python has it (3.11+), else its backport tomli, which the rig's 3.10 has."""
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads


def tree_hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_format_3_entry_layout_and_index(tmp_path, report3):
    folder = tmp_path / NAME
    written = write3(folder, report3)
    index = load_index(folder)
    assert index == written
    assert index.format == 3 and yaml.safe_load((folder / INDEX_FILE).read_text())["format"] == 3
    assert index.cameras == [
        {"name": "left", "serial": "24152533", "camera_id": 0, "calibrated": True, "reason": ""},
        {"name": "right", "serial": "24152534", "camera_id": 1, "calibrated": True, "reason": ""},
        {"name": "camera4", "serial": "24152535", "camera_id": 3, "calibrated": True, "reason": ""},
        {"name": "stimCam", "serial": "20142630", "camera_id": 2, "calibrated": False, "reason": STIM_REASON},
    ]
    # One geometry per calibrated camera: stimCam's live geometry is dropped with it.
    assert [p["name"] for p in index.preset_sets] == ["live", "bin1", "bin2", "bin4"]
    for preset_set in index.preset_sets:
        assert list(preset_set["cameras"]) == ["left", "right", "camera4"]
    assert index.preset_sets[0]["cameras"]["camera4"] == {
        "binning": 2, "width": 300, "height": 200, "offset_x": 10, "offset_y": 40}
    # The geometries load_preset serves: those at which left and right share a set, once each.
    assert index.presets == (BASE, PresetGeometry(1, 1024, 1024, 208, 24), PresetGeometry(2, 512, 512, 104, 12))

    loads = toml_loads()
    sensor = loads((folder / "calibration.toml").read_text())
    assert [sensor[f"cam_{i}"]["name"] for i in range(3)] == ["left", "right", "camera4"]
    assert all(sensor[f"cam_{i}"]["size"] == list(RIG3.truth.image_size) for i in range(3))
    assert sensor["metadata"]["calibration_id"] == index.id
    for preset_set in presets_for_cameras(LIVE3):
        data = loads((folder / "presets" / preset_set.name / "calibration.toml").read_text())
        assert sorted(data) == ["cam_0", "cam_1", "cam_2", "metadata"]
        for i, cam in enumerate(["left", "right", "camera4"]):
            g = preset_set.cameras[cam]
            assert data[f"cam_{i}"]["name"] == cam and data[f"cam_{i}"]["size"] == [g.width, g.height]
            np.testing.assert_array_equal(data[f"cam_{i}"]["matrix"], map_intrinsics(RIG3.truth.cameras[cam].K, g))
    # The reloaded fit is the 3 calibrated cameras.
    assert list(load_sensor_fit(folder).cameras) == ["left", "right", "camera4"]


def test_left_right_extras_only_when_both_present(tmp_path, report3):
    with_pair = tmp_path / "with_pair"
    write3(with_pair, report3)
    for name in ("live", "bin1", "bin2"):
        matrix = with_pair / "presets" / name / "camera_matrix"
        assert (matrix / "stereo_params.pickle").is_file(), name
        assert (matrix / "left_intrinsic_params.pickle").is_file() and (matrix / "right_intrinsic_params.pickle").is_file()
        assert (with_pair / "presets" / name / "frame.yaml").is_file()
    signs = motor_to_inference_signs()
    loaded = load_preset(with_pair, BASE)
    expected = preset_stereo_params(RIG3.truth, BASE)
    for key, value in expected.matrix.items():
        np.testing.assert_array_equal(np.asarray(loaded.stereo_params.matrix[key]), np.asarray(value), err_msg=key)
    np.testing.assert_array_equal(loaded.frame_transform, frame_transform(RIG3.truth, expected, signs))

    # left and right in a set at different geometries: no stereo pair to describe.
    split = dict(LIVE3, right=PresetGeometry(2, 512, 512, 104, 12))
    apart = tmp_path / "apart"
    write3(apart, report3, live=split)
    assert not (apart / "presets" / "live" / "camera_matrix").exists()
    assert not (apart / "presets" / "live" / "frame.yaml").exists()
    assert (apart / "presets" / "bin1" / "camera_matrix" / "stereo_params.pickle").is_file()

    # Neither left nor right: no pickle and no frame anywhere, and nothing for load_preset to serve.
    rig = synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0, cameras=("camera4", "stimCam"))
    n = len(rig.observations)
    cameras = [CaptureCamera("camera4", "24152535", 0), CaptureCamera("stimCam", "20142630", 1)]
    summary = CaptureSummary(waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
                             views_kept={c.name: n for c in cameras}, rejections={c.name: {} for c in cameras})
    report = build_report(rig.truth, rig.observations, rig.board, BASE.sensor_roi(), summary, [], None)
    none = tmp_path / "none"
    write_calibration(none, fit=rig.truth, preset_sets=presets_for_cameras({"camera4": BASE, "stimCam": BASE}),
                      board=rig.board, report=report, capture=CAPTURE, cameras=cameras)
    assert not list(none.rglob("*.pickle")) and not list(none.rglob("frame.yaml"))
    assert (none / "calibration.toml").is_file() and (none / "presets" / "live" / "calibration.toml").is_file()
    assert load_index(none).presets == ()
    with pytest.raises(CalibrationGeometryError, match="left and right"):
        load_preset(none, BASE)


def test_write_refuses_a_set_without_every_calibrated_camera_and_a_fitted_camera_not_in_the_capture(tmp_path, report3):
    with pytest.raises(ValueError, match="camera4"):
        write_calibration(tmp_path / "a", fit=RIG3.truth, preset_sets=[PresetSet("live", {"left": BASE, "right": BASE})],
                          board=RIG3.board, report=report3, capture=CAPTURE, cameras=CAPTURE3)
    with pytest.raises(ValueError, match="camera4"):
        write3(tmp_path / "b", report3, cameras=CAPTURE3[:2])
    assert not list(tmp_path.iterdir())


def _format2_copy(tmp_path: Path) -> Path:
    """The format-2 fixture, written by the format-2 writer, with its pickles put back.

    *.pickle files are in git LFS, which the rig's test clone does not fetch, so the fixture
    holds the text files only. The pickles are what that writer wrote: the stereo parameters
    of the saved fit, through the same write_pickle."""
    from autotrainer.inference.calibration_FLIR import write_pickle

    folder = tmp_path / "format2" / NAME
    shutil.copytree(FORMAT2_FIXTURE, folder)
    fit = load_sensor_fit(folder)
    for g in load_index(folder).presets:
        stereo = preset_stereo_params(fit, g)
        matrix = folder / "presets" / g.name / "camera_matrix"
        matrix.mkdir(exist_ok=True)
        write_pickle(matrix / "stereo_params.pickle", stereo.as_pickle_dict())
    return folder


def test_format_2_index_reads_as_left_right(tmp_path):
    assert yaml.safe_load((FORMAT2_FIXTURE / INDEX_FILE).read_text())["format"] == 2
    folder = _format2_copy(tmp_path)
    assert is_multires_calibration(folder)
    index = load_index(folder)
    assert index.format == 2
    assert index.id == "2692b0b23fa2499f8189447f9631aa4a"
    assert index.cameras == [
        {"name": "left", "serial": None, "camera_id": None, "calibrated": True, "reason": ""},
        {"name": "right", "serial": None, "camera_id": None, "calibrated": True, "reason": ""},
    ]
    assert index.presets == tuple(PRESETS)
    assert index.preset_sets == [
        {"name": g.name, "cameras": {cam: dataclasses.asdict(g) for cam in CAMERAS}} for g in PRESETS
    ]
    # Still found by id and by time, and its presets still load as before.
    assert find_calibration(folder.parent, index.id) == folder
    assert resolve_for_time(folder.parent, index.created) == folder
    fit = load_sensor_fit(folder)
    loaded = load_preset(folder, BASE)
    np.testing.assert_array_equal(loaded.stereo_params.matrix["P2"], preset_stereo_params(fit, BASE).matrix["P2"])
    assert loaded.frame_transform.shape == (3, 3)
    stored = load_preset_set(folder, {"left": BASE, "right": BASE}, derived_root=tmp_path / ".derived")
    assert stored.toml_path == folder / "presets" / BASE.name / "calibration.toml"
    assert stored.stereo is not None and stored.stereo.geometry == BASE
    # A format-2 entry has no sensor-pixel file; that geometry is derived like any other.
    sensor = PresetGeometry(1, *fit.image_size, 0, 0)
    derived = load_preset_set(folder, {"left": sensor}, derived_root=tmp_path / ".derived")
    assert derived.toml_path.parent.parent == tmp_path / ".derived" / index.id


def test_preset_set_lookup_derives_an_unknown_geometry_outside_the_entry(tmp_path, report3):
    folder = tmp_path / "calibrations" / "entry"
    index = write3(folder, report3)
    derived_root = tmp_path / "calibrations" / ".derived"
    before = tree_hashes(folder)

    # A stored set, asked for by a subset of its cameras: the stored file, and the left-right extras.
    stored = load_preset_set(folder, {"left": BASE, "camera4": LIVE3["camera4"]}, derived_root=derived_root)
    assert stored.toml_path == folder / "presets" / "live" / "calibration.toml"
    assert stored.stereo is None
    paired = load_preset_set(folder, {"left": BASE, "right": BASE}, derived_root=derived_root)
    assert paired.toml_path.parent.parent == folder / "presets"
    assert paired.stereo is not None and paired.stereo.geometry == BASE
    sensor = PresetGeometry(1, *RIG3.truth.image_size, 0, 0)
    assert load_preset_set(folder, {"camera4": sensor}, derived_root=derived_root).toml_path == folder / "calibration.toml"
    assert not derived_root.exists()

    # No stored set has camera4 at bin 4 with right at bin 1: derived into the cache, once.
    wanted = {"camera4": PresetGeometry(4, 150, 100, 5, 20), "right": PresetGeometry(1, 1024, 1024, 208, 24)}
    derived = load_preset_set(folder, wanted, derived_root=derived_root)
    assert derived.toml_path.parent.parent == derived_root / index.id
    assert derived.toml_path.name == "calibration.toml" and derived.stereo is None
    data = toml_loads()(derived.toml_path.read_text())
    # In the calibration's camera order, not the request's.
    assert sorted(data) == ["cam_0", "cam_1", "metadata"]
    assert [data["cam_0"]["name"], data["cam_1"]["name"]] == ["right", "camera4"]
    assert data["metadata"]["calibration_id"] == index.id
    X = np.concatenate([RIG3.world_corners(o.motor_mm) for o in RIG3.observations[:5]])
    for key in ("cam_0", "cam_1"):
        table, cam = data[key], data[key]["name"]
        m = RIG3.truth.cameras[cam]
        got = cv2.projectPoints(X, np.array(table["rotation"]), np.array(table["translation"]),
                                np.array(table["matrix"]), np.array(table["distortions"]))[0].reshape(-1, 2)
        sensor_px = cv2.projectPoints(X, m.rvec, m.tvec, m.K, m.dist)[0].reshape(-1, 2)
        np.testing.assert_allclose(got, sensor_to_preset_points(sensor_px, wanted[cam]), rtol=0, atol=1e-9)
    mtime = derived.toml_path.stat().st_mtime_ns
    again = load_preset_set(folder, dict(reversed(list(wanted.items()))), derived_root=derived_root)
    assert again.toml_path == derived.toml_path and again.toml_path.stat().st_mtime_ns == mtime
    assert tree_hashes(folder) == before

    # Never into the entry itself, and never for a camera the calibration did not calibrate.
    with pytest.raises(ValueError, match="inside"):
        load_preset_set(folder, wanted, derived_root=folder / ".derived")
    with pytest.raises(CalibrationGeometryError, match="stimCam"):
        load_preset_set(folder, {"stimCam": LIVE3["stimCam"]}, derived_root=derived_root)
    assert tree_hashes(folder) == before


def test_no_file_in_the_entry_changes_after_write(tmp_path, report3):
    root = tmp_path / "calibrations"
    folder = root / NAME
    index = write3(folder, report3)
    before = tree_hashes(folder)
    assert "calibration.toml" in before and INDEX_FILE in before

    is_multires_calibration(folder)
    load_index(folder)
    load_sensor_fit(folder)
    calibration_created(folder)
    for g in index.presets:
        load_preset(folder, g)
    for preset_set in presets_for_cameras(LIVE3):
        cams = {cam: g for cam, g in preset_set.cameras.items() if cam != "stimCam"}
        load_preset_set(folder, cams, derived_root=root / ".derived")
    load_preset_set(folder, {"left": PresetGeometry(4, 100, 100, 0, 0)}, derived_root=root / ".derived")
    calibration_candidates(root)
    find_calibration(root, index.id)
    resolve_for_time(root, index.created)
    pipeline._previous(folder, [])

    assert tree_hashes(folder) == before


def test_a_format_2_previous_is_compared_with_a_format_3_calibration_on_the_cameras_in_both(tmp_path):
    notes = []
    previous = pipeline._previous(_format2_copy(tmp_path), notes)
    assert notes == []
    assert not previous.legacy and list(previous.fit.cameras) == ["left", "right"]
    n = len(RIG3.observations)
    summary = CaptureSummary(waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
                             views_kept={cam: n for cam in RIG3.truth.cameras},
                             rejections={cam: {} for cam in RIG3.truth.cameras})
    report = build_report(RIG3.truth, RIG3.observations, RIG3.board, BASE.sensor_roi(), summary, [], previous)
    compared = {m.camera for m in report.metrics if m.name in CHANGE_METRICS}
    assert compared == {"left", "right", "left-right"}
    assert any(note.startswith("camera4: not in the previous calibration") for note in report.notes)
