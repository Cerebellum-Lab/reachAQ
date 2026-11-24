"""Offline 3D finds the calibration a session was recorded with, and reads a format-2 one at its preset.

A session whose metadata names a calibration id uses that calibration, wherever it now sits
under the calibration root: the active folder or a backup. A session without one, recorded
with a legacy calibration or before the id was written, uses the calibration that was active
when it started. When that match is ambiguous or missing, processing refuses and says to pass
--calib-dir. An explicit calib_dir always wins.

A format-2 calibration has no calibration_userset.yaml and no board in its folder name, so
offline 3D triangulates with the session's preset and turns the points into the motor frame
with that preset's measured frame, as live 3D does.

Backups are built by hand, as in the store tests: installing a calibration renames the active
folder to <name>_<install time>, so a backup's suffix is when it stopped being active.
"""

import importlib
import json
import logging
import os
import pickle
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from reachaq.core import ProjectInfo
from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.core.pose_elements import SceneElement
from reachaq.inference.analysis import prepare_jetson_data
from reachaq.inference.analysis.intersession_process import intersession_process, resolve_session_calibration
from reachaq.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from reachaq.inference.multires_calibration import store
from reachaq.inference.multires_calibration.geometry import (
    PresetGeometry,
    PresetSet,
    presets_from_base,
    sensor_to_preset_points,
)
from reachaq.inference.multires_calibration.products import motor_to_inference_signs
from reachaq.inference.multires_calibration.report import QualityReport
from reachaq.inference.multires_calibration.store import CalibrationGeometryError, CalibrationResolutionError
from multires_synthetic import synthetic_rig


# The module itself: the package's attribute of the same name is the function.
intersession_module = importlib.import_module("reachaq.inference.analysis.intersession_process")

NAME = DEFAULT_3D_CALIB_DIR_NAME
BASE = PresetGeometry(4, 256, 256, 52, 6)
PRESETS = presets_from_base(BASE)
# The scene turned 35 degrees about a generic axis (ruling R12), so a transposed frame shows.
RIG = synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0)
_REPORT = QualityReport(created="2026-10-20T12:00:00-06:00", verdict="pass", metrics=[], thresholds={}, notes=[])
# The parts extract_tracking_data gives, in its order.
PARTS = ['R_Hand', 'L_Hand', 'Pellet', 'Nose', 'Mouth', 'Tongue_mid', 'Tongue_tip', 'Star', 'Triangle', 'Diamond']
# Inside the second of the legacy windows built by _legacy_root: the morning of the day that
# calibration was replaced, at noon. A test that loses the start time and falls back to the
# date alone therefore finds the match ambiguous instead of passing by luck.
IN_SECOND_WINDOW = datetime(2026, 7, 1, 10, 30)
SESSION_DAY = datetime(2026, 7, 1)


class _Stop(Exception):
    """Raised by a stand-in for process_raw_data once it has seen its arguments."""


def _write_format3_with_live_set(folder: Path, made: Optional[datetime] = None) -> store.CalibrationIndex:
    """A format-3 calibration in `folder` with a live set (left and right at BASE) and one set
    per preset, made now or, for a session resolved by time that must not predate it
    (resolve_for_time refuses that), at `made`."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    # The live set comes first, as the capture tool writes it: the cameras recorded at BASE.
    sets = [PresetSet("live", {cam: BASE for cam in RIG.truth.cameras})]
    sets += [PresetSet(g.name, {cam: g for cam in RIG.truth.cameras}) for g in PRESETS]
    index = store.write_calibration(folder, fit=RIG.truth, preset_sets=sets, board=RIG.board, report=_REPORT,
                                    capture={})
    if made is not None:
        index_path = folder / store.INDEX_FILE
        data = yaml.safe_load(index_path.read_text())
        data["created"] = made.isoformat(timespec="seconds")
        index_path.write_text(yaml.safe_dump(data, sort_keys=False))
        index = store.load_index(folder)
    return index


def _write_true_format2(folder: Path, made: datetime) -> None:
    """A format-2 folder, as calibrations made before 2026-11-20 are: one presets/<name>/ per
    preset with left and right at it, and an index listing the presets, no cameras or sets."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    store.write_calibration(folder, fit=RIG.truth, presets=PRESETS, board=RIG.board, report=_REPORT, capture={})
    index_path = folder / store.INDEX_FILE
    data = yaml.safe_load(index_path.read_text())
    data.update(format=2, created=made.isoformat(timespec="seconds"),
                presets=[dict(s["cameras"]["left"], name=s["name"]) for s in data.pop("preset_sets")])
    data.pop("cameras")
    index_path.write_text(yaml.safe_dump(data, sort_keys=False))
    assert store.load_index(folder).format == 2 and store.load_index(folder).presets == tuple(PRESETS)


def _write_legacy(folder: Path, made: datetime) -> Path:
    """A legacy folder whose stereo_params.pickle is dated `made`, as copytree keeps it."""
    pickle_path = folder / "camera_matrix" / "stereo_params.pickle"
    pickle_path.parent.mkdir(parents=True)
    pickle_path.write_bytes(b"not read, only dated")
    os.utime(pickle_path, (made.timestamp(), made.timestamp()))
    return folder


def _legacy_root(tmp_path: Path) -> Path:
    """Two legacy backups and the active legacy folder, each made when the one before it ended."""
    root = tmp_path / "reachAQ"
    _write_legacy(root / f"{NAME}_20260303_090815", datetime(2026, 1, 10))
    _write_legacy(root / f"{NAME}_20260701_120000", datetime(2026, 3, 3, 9, 8, 15))
    _write_legacy(root / NAME, datetime(2026, 7, 1, 12))
    return root


def _project(tmp_path: Path, when: datetime = SESSION_DAY) -> ProjectInfo:
    # Built as the CLI builds it: the date alone, and no record start.
    return ProjectInfo(root=(tmp_path / "Data").as_posix(), device_id="rig", when=when, session=3)


def _write_metadata(project: ProjectInfo, data: dict) -> Path:
    path = Path(project.get_metadata_file() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _metadata(start: datetime, **calibration3d) -> dict:
    """Session metadata at schema 2, started at `start`, naming `calibration3d` when given."""
    data = {"metadataSchemaVersion": 2, "scope": "session", "boundary": {"startWallTime": start.timestamp()}}
    if calibration3d:
        data["hardware"] = {"configured": {"calibration3d": calibration3d}}
    return data


def _capture_process_raw_data(monkeypatch) -> dict:
    """Stand in for process_raw_data: record its calibration arguments, then stop the run."""
    seen = {}

    def fake(project, session, vid_tag, dlc_seg, calib_src_dir, center_method, *, frame_rate, preset=None):
        seen.update(calib_src_dir=calib_src_dir, preset=preset)
        raise _Stop

    monkeypatch.setattr(intersession_module, "process_raw_data", fake)
    return seen


def test_resolves_by_id(tmp_path, caplog):
    root = tmp_path / "reachAQ"
    _write_legacy(root / f"{NAME}_20260303_090815", datetime(2026, 1, 10))
    backup = root / f"{NAME}_20260701_120000"
    backup_index = _write_format3_with_live_set(backup)
    _write_format3_with_live_set(root / NAME)
    # By time this session would get the active folder: every backup ended before it began.
    start = backup_index.created + timedelta(days=1)
    project = _project(tmp_path, start.replace(hour=0, minute=0, second=0))
    _write_metadata(project, _metadata(start, id=backup_index.id, preset=BASE.name))
    with caplog.at_level(logging.WARNING, logger=intersession_module.__name__):
        assert resolve_session_calibration(project, root) == backup
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize("source", [
    "metadata boundary", "metadata naming no id", "schema 1 boundary", "schema 1 record start", "createdUtc",
    "project record start", "project time", "project date only",
])
def test_resolves_legacy_by_date(tmp_path, caplog, source):
    root = _legacy_root(tmp_path)
    project = _project(tmp_path)
    start = IN_SECOND_WINDOW.timestamp()
    if source == "metadata boundary":
        _write_metadata(project, _metadata(IN_SECOND_WINDOW))
    elif source == "metadata naming no id":
        _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=None, created="2026-03-03T09:08:15", preset=None,
                                           folder=str(root / NAME), override=None))
    elif source == "schema 1 boundary":
        _write_metadata(project, {"sessionBoundary": {"startWallTime": start}, "start_record_timestamp": None})
    elif source == "schema 1 record start":
        _write_metadata(project, {"sessionBoundary": None, "start_record_timestamp": start})
    elif source == "createdUtc":
        _write_metadata(project, {"metadataSchemaVersion": 2, "boundary": None, "createdUtc": start})
    elif source == "project record start":
        project.start_record_timestamp = start
    elif source == "project time":
        project.when = IN_SECOND_WINDOW
    else:
        # The date alone, of a day on which no calibration changed.
        project = _project(tmp_path, datetime(2026, 5, 1))
    with caplog.at_level(logging.WARNING, logger=intersession_module.__name__):
        assert resolve_session_calibration(project, root) == root / f"{NAME}_20260701_120000"
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize("case", [
    "two backups end at once", "unknown id", "calibration changed that day", "no calibration",
])
def test_ambiguous_raises_with_hint(tmp_path, case):
    root = tmp_path / "reachAQ"
    project = _project(tmp_path)
    if case == "two backups end at once":
        # DATE_TIME_FORMAT also parses an hour typed without its leading zero.
        _write_legacy(root / f"{NAME}_20260601_090815", datetime(2026, 1, 1))
        _write_legacy(root / f"{NAME}_20260601_90815", datetime(2026, 1, 1))
        _write_legacy(root / NAME, datetime(2026, 6, 1))
        _write_metadata(project, _metadata(IN_SECOND_WINDOW))
    elif case == "unknown id":
        _write_format3_with_live_set(root / NAME)
        _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=uuid.uuid4().hex, preset=BASE.name))
    elif case == "calibration changed that day":
        root = _legacy_root(tmp_path)
        # Only the date is known, and a backup ended at noon on it.
        project = _project(tmp_path, SESSION_DAY)
    else:
        root.mkdir(parents=True)
        _write_metadata(project, _metadata(IN_SECOND_WINDOW))
    with pytest.raises(CalibrationResolutionError, match="pass --calib-dir"):
        resolve_session_calibration(project, root)


def test_explicit_calib_dir_wins(tmp_path, monkeypatch):
    root = tmp_path / "reachAQ"
    index = _write_format3_with_live_set(root / NAME)
    project = _project(tmp_path)
    _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=index.id, preset=BASE.name))
    explicit = _write_legacy(tmp_path / "chosen" / NAME, datetime(2026, 1, 1))
    seen = _capture_process_raw_data(monkeypatch)

    def not_consulted(*args, **kwargs):
        raise AssertionError("an explicit calib_dir is used as given")

    monkeypatch.setattr(intersession_module, "resolve_session_calibration", not_consulted)
    with pytest.raises(_Stop):
        intersession_process(project, calib_dir=explicit, calib_root=root)
    # A legacy folder takes the legacy path, unchanged.
    assert seen == {"calib_src_dir": explicit.as_posix(), "preset": None}


def test_without_a_calib_dir_the_session_resolves_under_the_root(tmp_path, monkeypatch):
    root = _legacy_root(tmp_path)
    project = _project(tmp_path)
    _write_metadata(project, _metadata(IN_SECOND_WINDOW))
    seen = _capture_process_raw_data(monkeypatch)
    with pytest.raises(_Stop):
        intersession_process(project, calib_root=root)
    assert seen == {"calib_src_dir": (root / f"{NAME}_20260701_120000").as_posix(), "preset": None}

    roots = []
    monkeypatch.setattr(intersession_module, "resolve_session_calibration",
                        lambda project, calib_root: roots.append(calib_root) or root / NAME)
    with pytest.raises(_Stop):
        intersession_process(project)
    assert roots == [Path.home() / "reachAQ"]


@pytest.mark.parametrize("named, expected", [
    ("512x512_bin2", PRESETS[1]),
    # Recorded before calibration3d existed: the 2D points are in inference space, the base.
    ("no calibration3d", BASE),
], ids=["named preset", "no calibration3d"])
def test_a_format3_session_loads_its_preset(tmp_path, monkeypatch, named, expected):
    root = tmp_path / "reachAQ"
    folder = root / NAME
    # Made before the session, which the one without calibration3d is resolved against by time.
    index = _write_format3_with_live_set(folder, made=datetime(2026, 6, 1))
    project = _project(tmp_path)
    if named == "no calibration3d":
        _write_metadata(project, _metadata(IN_SECOND_WINDOW))
    else:
        _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=index.id, preset=named))
    seen = _capture_process_raw_data(monkeypatch)
    with pytest.raises(_Stop):
        intersession_process(project, calib_root=root)
    assert seen["calib_src_dir"] == folder.as_posix()
    preset = seen["preset"]
    assert preset.geometry == expected
    reference = store.load_preset(folder, expected)
    np.testing.assert_array_equal(preset.frame_transform, reference.frame_transform)
    np.testing.assert_array_equal(preset.stereo_params.matrix["P1"], reference.stereo_params.matrix["P1"])


def test_a_true_format2_session_without_calibration3d_is_read_at_the_base_without_a_warning(
        tmp_path, monkeypatch, caplog):
    # Recorded before calibration3d existed, on a format-2 calibration: its 2D points are in
    # inference space, the base preset, and nothing about it is worth a warning.
    root = tmp_path / "reachAQ"
    folder = root / NAME
    _write_true_format2(folder, made=datetime(2026, 6, 1))
    project = _project(tmp_path)
    _write_metadata(project, _metadata(IN_SECOND_WINDOW))
    seen = _capture_process_raw_data(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=intersession_module.__name__), pytest.raises(_Stop):
        intersession_process(project, calib_root=root)
    assert seen["calib_src_dir"] == folder.as_posix() and seen["preset"].geometry == BASE
    assert [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == intersession_module.__name__] == []


def test_a_session_recorded_with_live_3d_off_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "reachAQ"
    folder = root / NAME
    index = _write_format3_with_live_set(folder)
    project = _project(tmp_path)
    # The operator had moved the ROI, so live 3D matched no preset of this calibration.
    reason = f"the calibration in {folder} has no preset for 256x256_bin4 at ROI offset (60, 6)"
    _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=index.id, created=index.created.isoformat(),
                                       preset=None, folder=str(folder), override=None,
                                       live3dDisabledReason=reason))
    seen = _capture_process_raw_data(monkeypatch)
    with pytest.raises(CalibrationResolutionError, match="live 3D off") as raised:
        intersession_process(project, calib_root=root)
    assert "pass --calib-dir" in str(raised.value) and "(60, 6)" in str(raised.value)
    # Refused before any 3D: never read at the base preset as a guess.
    assert seen == {}


@pytest.mark.parametrize("route", ["explicit calib-dir", "legacy backup deleted"])
def test_a_legacy_live_session_reaching_format2_uses_the_base_preset(tmp_path, monkeypatch, caplog, route):
    root = tmp_path / "reachAQ"
    folder = root / NAME
    # A real format-2 folder: a format-3 one is read at its live set, never at the base preset.
    # Made before the session: by time, a format-2 calibration made after it would be refused
    # (test_a_session_older_than_every_kept_calibration_is_refused).
    _write_true_format2(folder, made=datetime(2026, 6, 1))
    project = _project(tmp_path)
    # Live 3D ran on a legacy calibration: no id, no preset, and it was not off.
    _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=None, created="2026-03-03T09:08:15", preset=None,
                                       folder=str(root / f"{NAME}_old"), override=None))
    seen = _capture_process_raw_data(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=intersession_module.__name__), pytest.raises(_Stop):
        if route == "explicit calib-dir":
            intersession_process(project, calib_dir=folder)
        else:
            # The legacy backup is gone, so by time the session lands on this format-2 folder.
            intersession_process(project, calib_root=root)
    assert seen["calib_src_dir"] == folder.as_posix()
    assert seen["preset"].geometry == BASE
    np.testing.assert_array_equal(seen["preset"].frame_transform, store.load_preset(folder, BASE).frame_transform)
    legacy_warnings = [r.getMessage() for r in caplog.records
                       if r.levelno == logging.WARNING and "legacy calibration" in r.getMessage()]
    assert len(legacy_warnings) == 1 and str(folder) in legacy_warnings[0]


def test_a_preset_the_calibration_lacks_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "reachAQ"
    index = _write_format3_with_live_set(root / NAME)
    project = _project(tmp_path)
    _write_metadata(project, _metadata(IN_SECOND_WINDOW, id=index.id, preset="300x300_bin3"))
    _capture_process_raw_data(monkeypatch)
    with pytest.raises(CalibrationGeometryError, match="300x300_bin3"):
        intersession_process(project, calib_root=root)


def test_an_unreadable_index_is_never_read_as_legacy(tmp_path, monkeypatch):
    folder = _write_legacy(tmp_path / NAME, datetime(2026, 1, 1))
    (folder / store.INDEX_FILE).write_text("format: 2\nid: abc\n")
    _capture_process_raw_data(monkeypatch)
    with pytest.raises(ValueError, match="corrupt"):
        intersession_process(_project(tmp_path), calib_dir=folder)


def test_a_session_older_than_every_kept_calibration_is_refused(tmp_path, monkeypatch):
    # The calibration it was recorded with is not kept, and the oldest one that is cannot
    # stand in for it: processing refuses and asks for --calib-dir (spec, Old sessions;
    # ruling R49).
    root = _legacy_root(tmp_path)
    project = _project(tmp_path)
    _write_metadata(project, _metadata(datetime(2026, 1, 5, 8)))
    with pytest.raises(CalibrationResolutionError, match="pass --calib-dir") as refused:
        resolve_session_calibration(project, root)
    assert "2026-01-05 08:00:00" in str(refused.value) and "2026-01-10 00:00:00" in str(refused.value)

    # An explicit calib_dir still processes it.
    seen = _capture_process_raw_data(monkeypatch)
    folder = root / f"{NAME}_20260303_090815"
    with pytest.raises(_Stop):
        intersession_process(project, calib_dir=folder)
    assert seen["calib_src_dir"] == folder.as_posix()

    # The same with only a format-2 calibration kept, made after the session (its legacy
    # predecessor deleted): the session is not read at that calibration's base preset.
    only = tmp_path / "only_format2"
    index = _write_format3_with_live_set(only / NAME)
    later = _project(tmp_path / "later")
    _write_metadata(later, _metadata(IN_SECOND_WINDOW))
    with pytest.raises(CalibrationResolutionError, match="pass --calib-dir") as refused:
        intersession_process(later, calib_root=only)
    assert str(IN_SECOND_WINDOW) in str(refused.value) and str(index.created) in str(refused.value)


def test_an_aware_calibration_date_is_compared_in_local_time(tmp_path, caplog):
    root = tmp_path / "reachAQ"
    folder = root / NAME
    index = _write_format3_with_live_set(folder)
    # An index whose created carries a UTC offset, as a hand-edited or future one might.
    index_path = folder / store.INDEX_FILE
    data = yaml.safe_load(index_path.read_text())
    made = datetime(2026, 8, 1, 18, 0, tzinfo=timezone.utc)
    data["created"] = made.isoformat()
    index_path.write_text(yaml.safe_dump(data, sort_keys=False))
    assert store.calibration_created(folder).tzinfo is not None
    project = _project(tmp_path)
    # The session (naive local time) began a day before the calibration was made.
    _write_metadata(project, _metadata((made - timedelta(days=1)).astimezone().replace(tzinfo=None), id=index.id,
                                       preset=BASE.name))
    with caplog.at_level(logging.WARNING, logger=intersession_module.__name__):
        assert resolve_session_calibration(project, root) == folder
    message, = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert str(made.astimezone().replace(tzinfo=None)) in message


def test_the_cli_resolves_unless_given_a_calib_dir(monkeypatch, tmp_path):
    from reachaq.inference.analysis import __main__ as cli

    calls = []

    def fake(project, **kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(cli, "intersession_process", fake)
    # main() sets the console handler's level, which only exists once logging is set up.
    monkeypatch.setattr(cli, "get_console_handler", logging.NullHandler)
    base = ["prog", "rig", "20260501", "3", "--data-dir", str(tmp_path)]
    root_level = logging.root.level
    try:
        for argv in (base, base + ["--calib-dir", str(tmp_path / "chosen"), "--calib-root", str(tmp_path / "root")]):
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(_Stop):
                cli.main()
    finally:
        logging.root.setLevel(root_level)
    assert [(c["calib_dir"], c["calib_root"]) for c in calls] == [
        (None, Path.home() / "reachAQ"),
        (tmp_path / "chosen", tmp_path / "root"),
    ]


def _motor_points_in_view(count: int, g: PresetGeometry) -> np.ndarray:
    """`count` motor-frame board corners, spread over the waypoints, inside preset g in both cameras."""
    X = np.concatenate([RIG.world_corners(o.motor_mm) for o in RIG.observations])
    inside = np.ones(len(X), dtype=bool)
    for uv in _preset_pixels(X, g).values():
        inside &= (uv >= 0).all(axis=1) & (uv[:, 0] <= g.width - 1) & (uv[:, 1] <= g.height - 1)
    X = X[inside]
    assert len(X) >= count, "too few points in view to give every part its own"
    return X[np.linspace(0, len(X) - 1, count).round().astype(int)]


def _preset_pixels(X: np.ndarray, g: PresetGeometry) -> dict:
    pixels = {}
    for cam in ("left", "right"):
        m = RIG.truth.cameras[cam]
        sensor = cv2.projectPoints(X, m.rvec, m.tvec, m.K, m.dist)[0].reshape(-1, 2)
        pixels[cam] = sensor_to_preset_points(sensor, g)
    return pixels


@pytest.mark.parametrize("g", [BASE, PRESETS[1]], ids=lambda g: g.name)
def test_format2_offline_3d_is_the_motor_frame(tmp_path, monkeypatch, g):
    folder = tmp_path / "reachAQ" / NAME
    _write_format3_with_live_set(folder)
    preset = store.load_preset(folder, g)
    # Each part held still at its own motor point for twelve frames, seen with p = 1.
    X = _motor_points_in_view(len(PARTS), g)
    pixels = _preset_pixels(X, g)
    columns = pd.MultiIndex.from_product([PARTS, ["x", "y", "likelihood"]], names=["bodyparts", "coordinates"])
    frames = 12
    df_lr = []
    for cam in ("left", "right"):
        row = np.column_stack([pixels[cam], np.ones(len(PARTS))]).ravel()
        df_lr.append(pd.DataFrame(np.tile(row, (frames, 1)), columns=columns))
    monkeypatch.setattr(prepare_jetson_data, "extract_tracking_data", lambda *args: (df_lr, list(PARTS)))
    session = tmp_path / "session003"
    session.mkdir()
    for cam in ("left", "right"):
        (session / f"20260501_rig_session003_{cam}.mp4").touch()

    def run():
        return prepare_jetson_data.process_raw_data(
            _project(tmp_path), session.as_posix(), ".mp4", "_raw2D", folder.as_posix(),
            (1, SceneElement.Diamond), frame_rate=150, preset=preset,
        )[1]

    centered = run()

    # Centred on the Diamond, then the motor frame in mm with the DCS signs: a displacement
    # between two parts is the motor displacement, sign-flipped.
    signs = np.array(motor_to_inference_signs(), dtype=float)
    diamond = X[PARTS.index("Diamond")]
    assert np.abs(centered["Diamond"][["x", "y", "z"]].to_numpy()).max() < 1e-6
    spread = max(np.linalg.norm(x - diamond) for x in X)
    assert spread > 5.0, "the parts are too close together to test the frame"
    for part, x in zip(PARTS, X):
        oriented = centered[part][["x", "y", "z"]].to_numpy()
        np.testing.assert_allclose(oriented, np.tile(signs * (x - diamond), (frames, 1)), atol=0.05, err_msg=part)
    # The folder has neither a userset yaml nor a board in its name, and both were not needed.
    assert not (folder / "calibration_userset.yaml").exists()
    assert (session / "20260501_rig_session003_centered3D.h5").is_file()
    # Only DCS writes a format-2 calibration's offsets (R33): with none there, offline
    # centres on this session alone and saves nothing, unlike a legacy folder.
    offsets_path = folder / DEFAULT_CAM_OFFSET_FILE_NAME
    assert not offsets_path.exists()
    # Offsets that DCS wrote are read, and left exactly as they were.
    offsets_path.write_bytes(pickle.dumps({"x_off": 1.0, "y_off": -2.0, "z_off": 140.0}))
    written = offsets_path.read_bytes()
    run()
    assert offsets_path.read_bytes() == written
