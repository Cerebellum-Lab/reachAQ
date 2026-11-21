"""Offline 3D from every calibrated camera a session recorded, and the pose-consistency check.

A ring of synthetic cameras looks at points near the origin. Everything is known, so the
triangulation is checked against the truth and a camera whose 2D points are shifted, as a
knocked camera's would be, is the one the check flags. The session-level tests write a real
format-3 calibration entry and the per-camera *_raw2D_live.h5 files live pose writes.
"""

import hashlib
import importlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from autotrainer.core import ProjectInfo
from autotrainer.inference.analysis import offline_3d
from autotrainer.inference.analysis.intersession_process import intersession_process
from autotrainer.inference.analysis.offline_3d import (
    ConsistencyVerdict,
    PoseConsistencySettings,
    export_session_3d,
    pose_consistency,
    read_pose_consistency_settings,
    select_session_cameras,
    session_offline_3d,
    triangulate_dlt,
)
from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME, _load_saved_offsets
from autotrainer.inference.multires_calibration import library, store
from autotrainer.inference.multires_calibration.fit import CaptureCamera
from autotrainer.inference.multires_calibration.geometry import (
    PresetGeometry,
    presets_for_cameras,
    sensor_to_preset_points,
)
from autotrainer.inference.multires_calibration.report import QualityReport
from autotrainer.inference.multires_calibration.store import CalibrationResolutionError
from multires_offline_resolution_test import (
    BASE,
    IN_SECOND_WINDOW,
    NAME,
    PRESETS,
    _capture_process_raw_data,
    _metadata,
    _project,
    _write_format3_with_live_set,
    _write_metadata,
    _Stop,
)
from multires_synthetic import synthetic_rig

intersession_module = importlib.import_module("autotrainer.inference.analysis.intersession_process")

PARTS = ["Pellet", "Nose", "Diamond", "Star"]
K = np.array([[680.0, 0.0, 128.0], [0.0, 680.0, 128.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.2, 0.05, 0.0005, -0.0003, 0.0])
LIVE_3D_OFF = "live 3D is off"
_REPORT = QualityReport(created="2026-11-20T12:00:00-07:00", verdict="pass", metrics=[], thresholds={}, notes=[])


# --- a synthetic ring of cameras ----------------------------------------------------------

def _look_at(eye: np.ndarray) -> np.ndarray:
    """The world -> camera rotation of a camera at `eye` looking at the origin, y roughly down."""
    z = -eye / np.linalg.norm(eye)
    down = np.array([0.0, 0.0, -1.0])
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _ring(n: int, *, distance: float = 150.0) -> List[dict]:
    """`n` cameras around the origin, 40 degrees above it, each a {name, rvec, tvec} in mm."""
    cameras = []
    for i in range(n):
        azimuth = np.deg2rad(360.0 * i / n + 10.0)
        elevation = np.deg2rad(40.0)
        eye = distance * np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                                   np.sin(elevation)])
        R = _look_at(eye)
        cameras.append({"name": f"cam{i}", "rvec": cv2.Rodrigues(R)[0].ravel(), "tvec": -R @ eye})
    return cameras


def _pair() -> List[dict]:
    """Two cameras side by side along x, 30 degrees apart: their epipolar lines run along the image x."""
    cameras = []
    for i, azimuth in enumerate((-15.0, 15.0)):
        a = np.deg2rad(azimuth)
        eye = 150.0 * np.array([np.sin(a), -np.cos(a), 0.0])
        R = _look_at(eye)
        cameras.append({"name": f"cam{i}", "rvec": cv2.Rodrigues(R)[0].ravel(), "tvec": -R @ eye})
    return cameras


def _points(frames: int, seed: int = 0) -> np.ndarray:
    """(frames, parts, 3) points within 10 mm of the origin."""
    return np.random.default_rng(seed).uniform(-10.0, 10.0, (frames, len(PARTS), 3))


def _pixels(camera: dict, X: np.ndarray) -> np.ndarray:
    shape = X.shape[:-1]
    px = cv2.projectPoints(X.reshape(-1, 3), camera["rvec"], camera["tvec"], K, DIST)[0]
    return px.reshape(*shape, 2)


def _toml_value(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    array = np.asarray(value, float)
    if array.ndim == 0:
        return repr(float(array))
    return "[" + ", ".join(_toml_value(v) for v in array) + "]"


def _write_toml(path: Path, cameras: Sequence[dict], calibration_id: str = "c0ffee") -> Path:
    lines = []
    for i, camera in enumerate(cameras):
        lines += [f"[cam_{i}]", f"name = {_toml_value(camera['name'])}", "size = [256, 256]",
                  f"matrix = {_toml_value(K)}", f"distortions = {_toml_value(DIST)}",
                  f"rotation = {_toml_value(camera['rvec'])}", f"translation = {_toml_value(camera['tvec'])}", ""]
    lines += ["[metadata]", "adjusted = false", "error = 0.25", f"calibration_id = {_toml_value(calibration_id)}",
              'units = "mm"', 'frame = "motor"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_pose(path: Path, pixels: np.ndarray, likelihood=None, frame_idx=None) -> Path:
    """A *_raw2D_live.h5 as live pose writes it: bodyparts/coords columns plus frame_idx."""
    frames = pixels.shape[0]
    columns = pd.MultiIndex.from_product([PARTS, ["x", "y", "likelihood"]], names=["bodyparts", "coords"])
    p = np.ones(pixels.shape[:2]) if likelihood is None else likelihood
    data = np.concatenate([pixels, p[..., None]], axis=-1).reshape(frames, -1)
    df = pd.DataFrame(data, columns=columns, index=range(frames))
    df["frame_idx"] = list(range(1, frames + 1)) if frame_idx is None else list(frame_idx)
    df.to_hdf(path, key="df_with_missing", format="table", mode="w")
    return path


def _session(tmp_path: Path, cameras: Sequence[dict], X: np.ndarray, shift: Dict[str, Sequence[float]] = None,
             likelihood: Dict[str, np.ndarray] = None):
    session = tmp_path / "session003"
    session.mkdir(exist_ok=True)
    toml = _write_toml(tmp_path / "calibration.toml", cameras)
    files = {}
    for camera in cameras:
        px = _pixels(camera, X) + np.asarray((shift or {}).get(camera["name"], (0.0, 0.0)))
        files[camera["name"]] = _write_pose(session / f"20261120_rig_session003_{camera['name']}_raw2D_live.h5", px,
                                            (likelihood or {}).get(camera["name"]))
    return session, toml, files


def _enabled(**limits) -> PoseConsistencySettings:
    return offline_3d.pose_consistency_settings({"enabled": True, **limits})


# --- triangulation ------------------------------------------------------------------------

def _normalised(camera: dict, X: np.ndarray) -> np.ndarray:
    R = cv2.Rodrigues(camera["rvec"])[0]
    Xc = X @ R.T + camera["tvec"]
    return Xc[..., :2] / Xc[..., 2:]


def _projections(cameras: Sequence[dict]) -> np.ndarray:
    return np.stack([np.hstack([cv2.Rodrigues(c["rvec"])[0], c["tvec"][:, None]]) for c in cameras])


@pytest.mark.parametrize("n", [3, 6])
def test_dlt_recovers_synthetic_points(n):
    cameras = _ring(n)
    X = _points(50)
    points = np.stack([_normalised(c, X) for c in cameras])
    mask = np.ones(points.shape[:3], bool)
    recovered = triangulate_dlt(points, _projections(cameras), mask)
    assert recovered.shape == X.shape
    assert np.abs(recovered - X).max() < 1e-6


def test_fewer_than_two_valid_views_is_nan():
    cameras = _ring(3)
    X = _points(4)
    points = np.stack([_normalised(c, X) for c in cameras])
    mask = np.ones(points.shape[:3], bool)
    mask[1:, 0, :] = False  # frame 0: one camera only
    mask[:, 1, 2] = False  # frame 1, part 2: none
    mask[0, 2, :] = False  # frame 2: two cameras
    points[1, 3, 1] = np.nan  # frame 3, part 1: a NaN point is not a view even when unmasked
    mask[2, 3, 1] = False
    recovered = triangulate_dlt(points, _projections(cameras), mask)
    assert np.isnan(recovered[0]).all()
    assert np.isnan(recovered[1, 2]).all()
    assert np.isnan(recovered[3, 1]).all()
    assert np.abs(recovered[2] - X[2]).max() < 1e-6
    assert np.isfinite(recovered[1, [0, 1, 3]]).all()


def test_low_confidence_points_are_ignored(tmp_path):
    cameras = _ring(3)
    X = _points(20)
    likelihood = np.ones((20, len(PARTS)))
    likelihood[::2] = 0.1
    # cam0's low-confidence points are 40 px off: had they been used, the 3D would be wrong.
    shift = np.zeros((20, len(PARTS), 2))
    shift[::2] = 40.0
    session, toml, files = _session(tmp_path, cameras, X, likelihood={"cam0": likelihood})
    _write_pose(files["cam0"], _pixels(cameras[0], X) + shift, likelihood)
    result = export_session_3d(session, toml, files)
    with h5py.File(result.points3d_path, "r") as fh:
        points3d = fh["points3d"][()]
    assert np.abs(points3d - X).max() < 1e-4
    with h5py.File(result.reprojections_path, "r") as fh:
        error = fh["reprojection_error"][()]
    assert np.isnan(error[0, ::2]).all() and np.isfinite(error[0, 1::2]).all()
    assert np.isfinite(error[1:]).all()


def test_export_writes_expected_datasets(tmp_path):
    cameras = _ring(4)
    X = _points(30)
    session, toml, files = _session(tmp_path, cameras, X)
    result = export_session_3d(session, toml, files)
    assert result.points3d_path == session / "points3d.h5"
    assert result.reprojections_path == session / "reprojections.h5"
    names = [c["name"] for c in cameras]
    with h5py.File(result.points3d_path, "r") as fh:
        assert fh["points3d"].shape == (30, len(PARTS), 3)
        assert [n.decode() for n in fh["node_names"][()]] == PARTS
        assert [n.decode() for n in fh["camera_names"][()]] == names
        assert fh.attrs["calibration_id"] == "c0ffee" and fh.attrs["units"] == "mm"
        np.testing.assert_array_equal(fh["frame_idx"][()], np.arange(1, 31))
        np.testing.assert_allclose(fh["points3d"][()], X, atol=1e-4)
    with h5py.File(result.reprojections_path, "r") as fh:
        assert fh["reprojections"].shape == (4, 30, len(PARTS), 2)
        assert fh["reprojection_error"].shape == (4, 30, len(PARTS))
        assert [n.decode() for n in fh["node_names"][()]] == PARTS
        assert [n.decode() for n in fh["camera_names"][()]] == names
        assert np.nanmax(fh["reprojection_error"][()]) < 1e-3
        np.testing.assert_allclose(fh["reprojections"][0], _pixels(cameras[0], X), atol=1e-3)
    assert set(result.residuals) == set(names)


def test_frames_are_matched_by_frame_index(tmp_path):
    cameras = _ring(3)
    X = _points(10)
    session, toml, files = _session(tmp_path, cameras, X)
    # cam2 dropped frame 3 (index 4): its file has nine rows, and the others' frames still pair.
    keep = [i for i in range(10) if i != 3]
    _write_pose(files["cam2"], _pixels(cameras[2], X)[keep], frame_idx=[i + 1 for i in keep])
    result = export_session_3d(session, toml, files)
    with h5py.File(result.points3d_path, "r") as fh:
        np.testing.assert_allclose(fh["points3d"][()], X, atol=1e-4)
    with h5py.File(result.reprojections_path, "r") as fh:
        assert np.isnan(fh["reprojection_error"][2, 3]).all()


# --- the pose-consistency check -----------------------------------------------------------

def test_moved_camera_is_flagged(tmp_path):
    cameras = _ring(6)
    X = _points(400)
    session, toml, files = _session(tmp_path, cameras, X, shift={"cam2": (12.0, 0.0)})
    result = export_session_3d(session, toml, files)
    verdicts = pose_consistency(result.residuals, _enabled())
    assert verdicts["cam2"].status == "possibly moved"
    assert verdicts["cam2"].median_px > 6.0 and verdicts["cam2"].n_obs == 400 * len(PARTS)
    for name in ("cam0", "cam1", "cam3", "cam4", "cam5"):
        assert verdicts[name].status == "ok", verdicts[name]
    assert "possibly moved: cam2, median residual" in verdicts["cam2"].describe()
    assert "limit 6.0" in verdicts["cam2"].describe()


def test_consistent_cameras_are_ok_and_per_camera_limits_apply(tmp_path):
    cameras = _ring(3)
    X = _points(400)
    session, toml, files = _session(tmp_path, cameras, X, shift={"cam1": (4.0, 0.0)})
    result = export_session_3d(session, toml, files)
    assert {v.status for v in pose_consistency(result.residuals, _enabled()).values()} == {"ok"}
    tight = _enabled(per_camera={"cam1": {"median_px": 1.0}})
    verdicts = pose_consistency(result.residuals, tight)
    assert verdicts["cam1"].status == "possibly moved"
    # The optional p95 limit is a second, independent bar.
    p95 = pose_consistency(result.residuals, _enabled(p95_px=1.0))
    assert p95["cam1"].status == "possibly moved"


def test_two_cameras_report_pair_inconsistent(tmp_path):
    cameras = _pair()
    X = _points(400)
    session, toml, files = _session(tmp_path, cameras, X)
    consistent = export_session_3d(session, toml, files)
    assert {v.status for v in pose_consistency(consistent.residuals, _enabled()).values()} == {"ok"}
    # Off its epipolar line: the pair can no longer agree, and neither camera can be named.
    session, toml, files = _session(tmp_path, cameras, X, shift={"cam1": (0.0, 12.0)})
    result = export_session_3d(session, toml, files)
    verdicts = pose_consistency(result.residuals, _enabled())
    assert {v.status for v in verdicts.values()} == {"pair inconsistent"}
    assert all(v.median_px > 6.0 for v in verdicts.values())


def test_not_enough_data(tmp_path):
    cameras = _ring(3)
    session, toml, files = _session(tmp_path, cameras, _points(50), shift={"cam0": (20.0, 0.0)})
    verdicts = pose_consistency(export_session_3d(session, toml, files).residuals, _enabled())
    assert {v.status for v in verdicts.values()} == {"not enough data"}
    # Enough frames, too few confident observations.
    session, toml, files = _session(tmp_path, cameras, _points(400))
    verdicts = pose_consistency(export_session_3d(session, toml, files).residuals,
                                _enabled(min_observations=5000))
    assert {v.status for v in verdicts.values()} == {"not enough data"}
    assert all(v.n_obs == 400 * len(PARTS) for v in verdicts.values())


def test_settings_are_read_from_settings_yaml_and_never_written(tmp_path):
    root = tmp_path / "calibrations"
    assert read_pose_consistency_settings(root) == PoseConsistencySettings()
    defaults = PoseConsistencySettings()
    assert defaults.enabled is False
    limits = defaults.for_camera("anything")
    assert (limits.median_px, limits.p95_px, limits.min_frames, limits.min_observations, limits.min_confidence) == (
        6.0, None, 300, 1000, 0.5)
    root.mkdir()
    library.update_settings(root, live3d=True, checks={"quality": "warn", "pose_consistency": {
        "enabled": True, "median_px": 5.0, "per_camera": {"camera4": {"median_px": 9.0, "p95_px": 15}}}})
    text = library.settings_path(root).read_bytes()
    settings = read_pose_consistency_settings(root)
    assert settings.enabled is True
    assert settings.for_camera("left").median_px == 5.0 and settings.for_camera("left").p95_px is None
    assert settings.for_camera("camera4").median_px == 9.0 and settings.for_camera("camera4").p95_px == 15.0
    assert settings.for_camera("camera4").min_frames == 300
    assert library.settings_path(root).read_bytes() == text
    # A value of the wrong type is ignored, with the default in its place.
    library.update_settings(root, checks={"pose_consistency": {"enabled": "yes", "median_px": "big"}})
    settings = read_pose_consistency_settings(root)
    assert settings.enabled is False and settings.for_camera("left").median_px == 6.0


# --- which cameras a session triangulates -------------------------------------------------

CALIBRATION_CAMERAS = [
    {"name": "left", "serial": "21000001", "camera_id": 0, "calibrated": True, "reason": ""},
    {"name": "right", "serial": "21000002", "camera_id": 1, "calibrated": True, "reason": ""},
    {"name": "camera4", "serial": "21000004", "camera_id": 3, "calibrated": True, "reason": ""},
    {"name": "stimCam", "serial": "20142630", "camera_id": 2, "calibrated": False,
     "reason": "too few board views: 4 of 12 needed"},
]


def test_swapped_camera_excluded_and_named():
    recorded = {
        "left": {"serial": "21000001", "inCalibration": True},
        "right": {"serial": "21000099", "inCalibration": False,
                  "uncalibratedReason": "swapped: calibrated serial 21000002, now 21000099"},
        # Renamed: the serial is camera4's, so it is camera4.
        "camera5": {"serial": "21000004", "inCalibration": True},
        "stimCam": {"serial": "20142630", "inCalibration": False,
                    "uncalibratedReason": "too few board views: 4 of 12 needed"},
        "camera6": {"serial": "21000006", "inCalibration": False},
    }
    selected = select_session_cameras(CALIBRATION_CAMERAS, recorded)
    assert selected.used == {"left": "left", "camera5": "camera4"}
    assert selected.excluded == {
        "right": "swapped: calibrated serial 21000002, now 21000099",
        "stimCam": "too few board views: 4 of 12 needed",
        "camera6": "not in the session's calibration",
    }
    # The serial decides even when the metadata says the camera was in the calibration.
    recorded["right"] = {"serial": "21000099", "inCalibration": True}
    assert select_session_cameras(CALIBRATION_CAMERAS, recorded).excluded["right"] == (
        "swapped: calibrated serial 21000002, now 21000099")
    # A camera without a serial on either side is matched by name.
    no_serials = [dict(c, serial=None) for c in CALIBRATION_CAMERAS[:2]]
    assert select_session_cameras(no_serials, {"left": {}, "right": {"serial": None}}).used == {
        "left": "left", "right": "right"}


RIG3 = synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0, cameras=("left", "right", "camera4"))
CAPTURE3 = [CaptureCamera("left", "21000001", 0), CaptureCamera("right", "21000002", 1),
            CaptureCamera("camera4", "21000004", 3)]
LIVE3 = {"left": BASE, "right": BASE, "camera4": PresetGeometry(2, 300, 200, 10, 40)}


def _entry(config: Path) -> Path:
    """A format-3 library entry for left, right and camera4."""
    root = library.library_root(config)
    built = config / "built"
    store.write_calibration(built, fit=RIG3.truth, preset_sets=presets_for_cameras(LIVE3), board=RIG3.board,
                            report=_REPORT, capture={}, cameras=CAPTURE3)
    return library.entry_path(root, library.add_entry(root, built))


def _params(g: PresetGeometry) -> dict:
    return {"hbin": g.binning, "vbin": g.binning, "width": g.width, "height": g.height,
            "offsetx": g.offset_x, "offsety": g.offset_y}


def _motor_points(frames: int) -> np.ndarray:
    """Points every camera sees: around the middle of the board's travel."""
    centre = RIG3.world_corners((0.0, 22.0, -28.0)).mean(axis=0)
    return centre + np.random.default_rng(3).uniform(-4.0, 4.0, (frames, len(PARTS), 3))


def _session_pixels(cam: str, X: np.ndarray, g: PresetGeometry) -> np.ndarray:
    m = RIG3.truth.cameras[cam]
    sensor = cv2.projectPoints(X.reshape(-1, 3), m.rvec, m.tvec, m.K, m.dist)[0].reshape(-1, 2)
    return sensor_to_preset_points(sensor, g).reshape(*X.shape[:-1], 2)


def _recorded_session(tmp_path: Path, entry: Path, serials: Dict[str, str], X: np.ndarray, *,
                      live3d_reason=None) -> (Path, dict):
    session = tmp_path / "session003"
    session.mkdir(exist_ok=True)
    index = store.load_index(entry)
    calibration_serials = {c["name"]: c["serial"] for c in index.cameras}
    cameras, configuration = {}, []
    for cam, serial in serials.items():
        g = LIVE3[cam]
        _write_pose(session / f"20261120_rig_session003_{cam}_raw2D_live.h5", _session_pixels(cam, X, g))
        swapped = serial != calibration_serials[cam]
        cameras[cam] = {"serial": serial, "inCalibration": not swapped, "uncalibratedReason": (
            f"swapped: calibrated serial {calibration_serials[cam]}, now {serial}" if swapped else None)}
        configuration.append({"name": cam, "scheme": "spinnaker", "host": serial, "params": _params(g)})
    calibration3d = {"id": index.id, "path": f"calibrations/{index.id}", "format": 3, "preset": None}
    if live3d_reason is not None:
        calibration3d.update(live3dDisabledReason=live3d_reason, live3d={"enabled": False,
                                                                         "disabledReason": live3d_reason})
    metadata = {"metadataSchemaVersion": 3, "scope": "session",
                "hardware": {"configured": {"cameras": cameras, "calibration3d": calibration3d}},
                "configuration": {"cameras": configuration}}
    return session, metadata


def test_session_offline_3d_uses_every_calibrated_camera_but_the_swapped_one(tmp_path):
    config = tmp_path / "Autotrainer"
    entry = _entry(config)
    before = _tree_hashes(entry)
    X = _motor_points(60)
    session, metadata = _recorded_session(
        tmp_path, entry, {"left": "21000001", "right": "21000099", "camera4": "21000004"}, X)
    result = session_offline_3d(session, entry, metadata, library_dir=entry.parent)
    assert result.excluded == {"right": "swapped: calibrated serial 21000002, now 21000099"}
    with h5py.File(result.points3d_path, "r") as fh:
        assert [n.decode() for n in fh["camera_names"][()]] == ["left", "camera4"]
        assert fh.attrs["calibration_id"] == entry.name
        np.testing.assert_allclose(fh["points3d"][()], X, atol=1e-3)
    report = json.loads((session / "offline3d_report.json").read_text(encoding="utf-8"))
    assert report["calibrationId"] == entry.name
    assert report["cameras"] == {"left": "left", "camera4": "camera4"}
    assert report["excluded"] == {"right": "swapped: calibrated serial 21000002, now 21000099"}
    assert report["poseConsistency"] is None, "the check is off by default"
    # camera4's 3D used its own geometry (bin 2, its own ROI), derived beside the entry.
    assert _tree_hashes(entry) == before


def test_session_offline_3d_with_fewer_than_two_cameras_is_2d_only(tmp_path):
    config = tmp_path / "Autotrainer"
    entry = _entry(config)
    session, metadata = _recorded_session(tmp_path, entry, {"left": "21000001", "right": "21000099"},
                                          _motor_points(10))
    assert session_offline_3d(session, entry, metadata, library_dir=entry.parent) is None
    assert not (session / "points3d.h5").exists()
    report = json.loads((session / "offline3d_report.json").read_text(encoding="utf-8"))
    assert report["twoDOnlyReason"] == "fewer than 2 calibrated cameras: left (right: swapped: calibrated serial" \
                                       " 21000002, now 21000099)"


def test_session_offline_3d_runs_the_check_when_enabled(tmp_path):
    config = tmp_path / "Autotrainer"
    entry = _entry(config)
    library.update_settings(entry.parent, checks={"pose_consistency": {
        "enabled": True, "min_frames": 10, "min_observations": 10}})
    session, metadata = _recorded_session(
        tmp_path, entry, {"left": "21000001", "right": "21000002", "camera4": "21000004"}, _motor_points(40))
    result = session_offline_3d(session, entry, metadata, library_dir=entry.parent)
    assert {v.status for v in result.consistency.values()} == {"ok"}
    report = json.loads((session / "offline3d_report.json").read_text(encoding="utf-8"))
    assert set(report["poseConsistency"]) == {"left", "right", "camera4"}
    assert report["poseConsistency"]["camera4"]["status"] == "ok"


def _tree_hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


# --- intersession processing --------------------------------------------------------------

def test_intersession_live_3d_switched_off_is_not_refused(tmp_path, monkeypatch):
    root = tmp_path / "Autotrainer"
    folder = root / NAME
    index = _write_format3_with_live_set(folder)
    project = _project(tmp_path)
    _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=index.id, preset="512x512_bin2",
                                       live3dDisabledReason=LIVE_3D_OFF,
                                       live3d={"enabled": False, "disabledReason": LIVE_3D_OFF}))
    seen = _capture_process_raw_data(monkeypatch)
    with pytest.raises(_Stop):
        intersession_process(project, calib_root=root)
    assert seen["preset"].geometry == PRESETS[1]


def test_intersession_live_3d_off_without_a_preset_reads_the_recorded_geometry(tmp_path, monkeypatch):
    root = tmp_path / "Autotrainer"
    folder = root / NAME
    index = _write_format3_with_live_set(folder)
    project = _project(tmp_path)
    data = _metadata(IN_SECOND_WINDOW, id=index.id, preset=None, live3dDisabledReason=LIVE_3D_OFF)
    data["configuration"] = {"cameras": [{"name": cam, "params": _params(PRESETS[1])} for cam in ("left", "right")]}
    _write_metadata(project, data)
    seen = _capture_process_raw_data(monkeypatch)
    with pytest.raises(_Stop):
        intersession_process(project, calib_root=root)
    assert seen["preset"].geometry == PRESETS[1]


def test_intersession_live_3d_off_with_a_swapped_left_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "Autotrainer"
    index = _write_format3_with_live_set(root / NAME)
    project = _project(tmp_path)
    data = _metadata(IN_SECOND_WINDOW, id=index.id, preset=None, live3dDisabledReason=LIVE_3D_OFF)
    reason = "swapped: calibrated serial 21000001, now 21000099"
    data["hardware"]["configured"]["cameras"] = {
        "left": {"serial": "21000099", "inCalibration": False, "uncalibratedReason": reason},
        "right": {"serial": "21000002", "inCalibration": True, "uncalibratedReason": None},
    }
    _write_metadata(project, data)
    seen = _capture_process_raw_data(monkeypatch)
    with pytest.raises(CalibrationResolutionError, match="left: swapped"):
        intersession_process(project, calib_root=root)
    assert seen == {}


def test_intersession_runs_offline_3d_after_2d_processing(tmp_path, monkeypatch):
    config = tmp_path / "Autotrainer"
    entry = _entry(config)
    library.set_active(entry.parent, entry.name, by="test")
    project = _project(tmp_path)
    session, metadata = _recorded_session(
        tmp_path, entry, {"left": "21000001", "right": "21000002", "camera4": "21000004"}, _motor_points(5),
        live3d_reason=LIVE_3D_OFF)
    metadata["hardware"]["configured"]["calibration3d"]["preset"] = BASE.name
    metadata["boundary"] = {"startWallTime": IN_SECOND_WINDOW.timestamp()}
    _write_metadata(project, metadata)
    calls = []
    monkeypatch.setattr(intersession_module, "process_raw_data",
                        lambda *args, **kwargs: calls.append("2d") or (None, None))
    monkeypatch.setattr(intersession_module, "session_offline_3d",
                        lambda session_dir, folder, data, *, library_dir: calls.append(
                            ("3d", Path(session_dir), folder, data["hardware"]["configured"]["calibration3d"]["id"],
                             library_dir)))

    def stop(**kwargs):
        raise _Stop

    monkeypatch.setattr(intersession_module, "segment_reaches", stop)
    with pytest.raises(_Stop):
        intersession_process(project, calib_root=config)
    location = Path(project.get_session_path()[0])
    assert calls == ["2d", ("3d", location, entry, entry.name, entry.parent)]


def test_intersession_offline_3d_failure_does_not_stop_the_session(tmp_path, monkeypatch, caplog):
    config = tmp_path / "Autotrainer"
    entry = _entry(config)
    project = _project(tmp_path)
    _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=entry.name, preset=BASE.name))
    monkeypatch.setattr(intersession_module, "process_raw_data", lambda *args, **kwargs: (None, None))

    def broken(*args, **kwargs):
        raise ValueError("a broken pose file")

    monkeypatch.setattr(intersession_module, "session_offline_3d", broken)

    def stop(**kwargs):
        raise _Stop

    monkeypatch.setattr(intersession_module, "segment_reaches", stop)
    with pytest.raises(_Stop):
        intersession_process(project, calib_dir=entry)
    assert "offline 3D of session" in caplog.text and "a broken pose file" in caplog.text


# --- camera offsets of a library entry ----------------------------------------------------

def test_intersession_offsets_of_a_library_entry_come_from_the_offsets_store(tmp_path):
    config = tmp_path / "Autotrainer"
    entry = _entry(config)
    store_dir = entry.parent / ".offsets" / entry.name
    # None yet: generated for this run, never saved, and nothing is written in the entry.
    before = _tree_hashes(entry)
    assert _load_saved_offsets(entry.as_posix()) == (None, False)
    assert _tree_hashes(entry) == before
    # A plain camera_offsets.pkl in the store, as the app writes it today.
    store_dir.mkdir(parents=True)
    (store_dir / DEFAULT_CAM_OFFSET_FILE_NAME).write_bytes(pickle.dumps({"x_off": 1.0}))
    assert _load_saved_offsets(entry.as_posix()) == ({"x_off": 1.0}, False)
    # Timestamped DCS results: the newest by name wins over the plain file.
    (store_dir / "camera_offsets-20261119T100000Z.pkl").write_bytes(pickle.dumps({"x_off": 2.0}))
    (store_dir / "camera_offsets-20261120T090000Z.pkl").write_bytes(pickle.dumps({"x_off": 3.0}))
    assert _load_saved_offsets(entry.as_posix()) == ({"x_off": 3.0}, False)


def test_intersession_offsets_fall_back_to_a_legacy_entry_s_own_file(tmp_path):
    config = tmp_path / "Autotrainer"
    legacy = config / NAME
    (legacy / "camera_matrix").mkdir(parents=True)
    (legacy / "camera_matrix" / "stereo_params.pickle").write_bytes(b"not read")
    (legacy / DEFAULT_CAM_OFFSET_FILE_NAME).write_bytes(pickle.dumps({"x_off": 7.0}))
    legacy_id, = library.import_existing(config)
    entry = library.library_root(config) / legacy_id
    assert _load_saved_offsets(entry.as_posix()) == ({"x_off": 7.0}, False)
    # Outside the library a legacy folder keeps today's behaviour: missing offsets are saved there.
    (legacy / DEFAULT_CAM_OFFSET_FILE_NAME).unlink()
    assert _load_saved_offsets(legacy.as_posix()) == (None, True)


def test_recorded_geometry_reads_params_as_the_app_does():
    calibration_geometry = pytest.importorskip("tools.acquisition.model.calibration_geometry")
    assert offline_3d.SPINCAM_DEFAULT_GEOMETRY == calibration_geometry.SPINCAM_DEFAULT_GEOMETRY
    for params in ({}, {"hbin": 2, "vbin": 2, "width": 512, "height": 400, "offset_x": 30, "offsety": "12"},
                   {"offsetx": 10, "offset_x": 20, "width": " none "}):
        assert offline_3d.recorded_geometry(params) == calibration_geometry.live_inference_geometry(params), params
    with pytest.raises(ValueError):
        offline_3d.recorded_geometry({"hbin": 2, "vbin": 4})
