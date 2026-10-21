"""The calibration store: one folder per calibration, a sub-folder per preset, and resolution by id or time.

A folder is a calibration only once its calibration.yaml is written, last, at format 2,
so a crash mid-write never leaves something that loads. Every preset folder holds today's
camera_matrix files, so existing loaders and DeepLabCut read it unchanged.

Installing a calibration renames the active folder to <name>_<install time>, so a backup's
suffix is when it stopped being active; the resolution tests build such folders by hand.

The fit is the synthetic rig turned 35 degrees about a generic axis (ruling R12): on the
symmetric rig the frame transform is diag(signs), and a transposed one would round-trip
unnoticed.
"""

import dataclasses
import json
import logging
import math
import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.inference.multires_calibration import store
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, map_intrinsics, presets_from_base
from autotrainer.inference.multires_calibration.products import (
    STEREO_KEY,
    anipose_toml,
    frame_transform,
    motor_to_inference_signs,
    preset_stereo_params,
)
from autotrainer.inference.multires_calibration.report import CaptureSummary, QualityReport, build_report
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
    load_sensor_fit,
    record_override,
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
    write(folder, report)
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
        assert (preset_dir / "calibration.toml").read_text() == anipose_toml(FIT, g)
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

    def failing_toml(fit, g):
        # As a crash would: after the first preset, partway through the second.
        if written:
            raise OSError("disk full")
        written.append(g.name)
        return anipose_toml(fit, g)

    monkeypatch.setattr(store, "anipose_toml", failing_toml)
    with pytest.raises(OSError, match="disk full"):
        write(folder, report)
    assert not folder.exists()
    assert (partial / "presets" / written[0] / "calibration.toml").is_file()
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
    second_index = write(second, report)
    active_index = write(active, report)
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
        "record_override": lambda folder: record_override(folder, {"confirmed": True}),
    }
    for name, load in loaders.items():
        with pytest.raises(ValueError, match=INDEX_FILE):
            load(damaged)
            pytest.fail(f"{name} accepted a corrupt index")


def test_record_override_persists(tmp_path, report):
    folder = tmp_path / NAME
    written = write(folder, report)
    override = {"confirmed": True, "by": "operator", "at": "2026-10-20T15:04:05.123456"}
    record_override(folder, override)

    index = load_index(folder)
    assert index.override == override
    assert dataclasses.replace(index, override=None) == written
    data = json.loads((folder / "quality_report.json").read_text())
    assert data["override"] == override
    assert QualityReport.from_json((folder / "quality_report.json").read_text()).override == override
    assert "Override recorded" in (folder / "quality_report.txt").read_text()
