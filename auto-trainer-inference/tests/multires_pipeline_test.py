"""calibrate_capture end to end, on a synthetic paused-waypoint capture.

The capture is what the rig's capture writes: waypoints.yaml, and full-sensor (1440x1080)
videos of both cameras while the stage pauses at each of 16 waypoints. The videos are
rendered from the synthetic rig turned 35 degrees about a generic axis (ruling R12), each
camera from its own true pose: 15 still frames per waypoint and 6 moving frames between
waypoints, written as mp4v at 40 fps, as multires_extract_test renders them.

render_board_frame draws without distortion, so the truth the fit should recover here is
the rig's cameras with zero distortion.

Each calibrate_capture run takes a while, so a run is a module fixture and several tests
read it: the main run (all waypoints verified, capture settings logged, no previous
calibration), a second run (one waypoint unverified and reported 1.7 mm from where it was
commanded, no capture settings logged, the main run as the previous calibration), a run
against the legacy fixture folder, a run whose
right camera labels the board from the other end, a degraded run (the 256 cross-check
fit fails and the previous calibration's fit cannot be read), and a run whose cross-check
maps the sensor corners with the pixel-edge convention.
"""

import filecmp
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest
import yaml

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.inference.multires_calibration import pipeline
from autotrainer.inference.multires_calibration.extract import WaypointAlignmentError, find_stationary_segments
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.pipeline import (
    SOURCE_VIDEOS,
    WAYPOINT_LOG,
    WaypointLog,
    calibrate_capture,
    load_waypoint_log,
    write_waypoint_log,
)
from autotrainer.inference.multires_calibration.report import NO_PREVIOUS_NOTE, Metric, QualityReport
from autotrainer.inference.multires_calibration.store import (
    INDEX_FILE,
    CalibrationIndex,
    is_multires_calibration,
    load_index,
    load_sensor_fit,
)
from multires_synthetic import render_board_frame, synthetic_rig

NAME = DEFAULT_3D_CALIB_DIR_NAME
BASE = PresetGeometry(4, 256, 256, 52, 6)
PRESETS = presets_from_base(BASE)
CAMERAS = ("left", "right")
IMAGE_SIZE = (1440, 1080)
FPS = 40
N_WAYPOINTS = 16
STILL_FRAMES = 15
MOVING_FRAMES = 6
UNVERIFIED = 7  # the waypoint the second run logs as not verified
LEGACY_FIXTURE = Path(__file__).parent / DEFAULT_3D_CALIB_DIR_NAME
RIG = synthetic_rig(frame_rotation_deg=35.0)
MOTOR = [o.motor_mm.copy() for o in RIG.observations[:N_WAYPOINTS]]
# Not the defaults, so the main run shows they are read from waypoints.yaml.
LOGGED_SETTINGS = {"gain_db": 2.5, "gamma": None, "exposure_us": 4000, "fps": 40}
DEFAULT_SETTINGS = {"gain_db": 1.0, "gamma": None, "exposure_us": 4500, "fps": 40}


@dataclass
class Run:
    capture_dir: Path
    output_dir: Path
    index: CalibrationIndex
    report: QualityReport
    seconds: float


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> Dict[str, Path]:
    """left.mp4, right.mp4, and right_turned.mp4: the right camera's frames turned 180 degrees."""
    folder = tmp_path_factory.mktemp("rendered")
    paths = {name: folder / f"{name}.mp4" for name in ("left", "right", "right_turned")}
    outputs = {"left": ["left"], "right": ["right", "right_turned"]}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam, names in outputs.items():
        writers = {name: cv2.VideoWriter(str(paths[name]), fourcc, FPS, IMAGE_SIZE, isColor=False) for name in names}
        assert all(w.isOpened() for w in writers.values())
        for image in capture_frames(RIG.truth.cameras[cam]):
            for name, writer in writers.items():
                writer.write(cv2.rotate(image, cv2.ROTATE_180) if name == "right_turned" else image)
        for writer in writers.values():
            writer.release()
    return paths


def capture_frames(model):
    """The camera's frames: a dwell at each waypoint, with moving frames between dwells."""
    for k, position in enumerate(MOTOR):
        if k > 0:
            previous = MOTOR[k - 1]
            for j in range(1, MOVING_FRAMES + 1):
                between = previous + (position - previous) * j / (MOVING_FRAMES + 1)
                yield render_board_frame(model, RIG.world_corners(between), IMAGE_SIZE, RIG.board)
        still = render_board_frame(model, RIG.world_corners(position), IMAGE_SIZE, RIG.board)
        for _ in range(STILL_FRAMES):
            yield still


def make_log(motor=None, verified=None, capture_settings=None, reported=None) -> WaypointLog:
    motor = [tuple(float(v) for v in p) for p in (MOTOR if motor is None else motor)]
    n = len(motor)
    return WaypointLog(
        motor=motor,
        reported=[tuple(v + 0.01 for v in p) for p in motor] if reported is None else list(reported),
        verified=[True] * n if verified is None else list(verified),
        dcs_moves=[("xyz"[i % 3], float(i)) for i in range(n)],
        times=[(1000.0 + i, 1000.75 + i) for i in range(n)],
        capture_settings={} if capture_settings is None else dict(capture_settings),
    )


def make_capture(folder: Path, left: Path, right: Path, log: WaypointLog) -> Path:
    (folder / SOURCE_VIDEOS).mkdir(parents=True)
    shutil.copyfile(left, folder / SOURCE_VIDEOS / "left.mp4")
    shutil.copyfile(right, folder / SOURCE_VIDEOS / "right.mp4")
    write_waypoint_log(folder / WAYPOINT_LOG, log)
    return folder


def run(capture_dir: Path, output_dir: Path, **kwargs) -> Run:
    start = time.perf_counter()
    index, report = calibrate_capture(capture_dir, output_dir=output_dir, presets=PRESETS, **kwargs)
    return Run(capture_dir, output_dir, index, report, time.perf_counter() - start)


def metrics(report: QualityReport, name: str) -> List[Metric]:
    return [m for m in report.metrics if m.name == name]


@pytest.fixture(scope="module")
def main(videos, tmp_path_factory) -> Run:
    folder = tmp_path_factory.mktemp("main")
    capture = make_capture(folder / "capture", videos["left"], videos["right"], make_log(capture_settings=LOGGED_SETTINGS))
    return run(capture, folder / "out" / NAME)


@pytest.fixture(scope="module")
def second(videos, main, tmp_path_factory) -> Run:
    folder = tmp_path_factory.mktemp("second")
    verified = [True] * N_WAYPOINTS
    verified[UNVERIFIED] = False
    # ... because the motor reported it 1.7 mm short in x, as a stage stopped at its travel
    # limit is (christielab10, 2026-10-26). The videos still show the commanded position.
    reported = [tuple(float(v) + 0.01 for v in p) for p in MOTOR]
    reported[UNVERIFIED] = (float(MOTOR[UNVERIFIED][0]) - 1.7, *reported[UNVERIFIED][1:])
    log = make_log(verified=verified, reported=reported)
    capture = make_capture(folder / "capture", videos["left"], videos["right"], log)
    return run(capture, folder / "out" / NAME, previous_dir=main.output_dir)


@pytest.fixture(scope="module")
def against_legacy(main, tmp_path_factory) -> Run:
    return run(main.capture_dir, tmp_path_factory.mktemp("legacy") / NAME, previous_dir=LEGACY_FIXTURE)


@pytest.fixture(scope="module")
def turned(videos, tmp_path_factory) -> Run:
    folder = tmp_path_factory.mktemp("turned")
    capture = make_capture(folder / "capture", videos["left"], videos["right_turned"], make_log())
    return run(capture, folder / "out" / NAME)


@pytest.fixture(scope="module")
def degraded(main, tmp_path_factory) -> Run:
    """The 256 preset's cross-check fit fails, and the previous calibration's fit cannot be
    read: neither may abort the calibration."""
    folder = tmp_path_factory.mktemp("degraded")
    damaged = folder / "previous" / NAME
    shutil.copytree(main.output_dir, damaged)
    (damaged / "sensor" / "fit.yaml").unlink()
    real = pipeline.fit_motor_frame

    def fit(observations, board, image_size, camera_names=CAMERAS, **kwargs):
        if tuple(image_size) == (256, 256):
            raise RuntimeError("the motor-frame fit did not converge: least_squares status 0 (stand-in)")
        return real(observations, board, image_size, camera_names, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "fit_motor_frame", fit)
        return run(main.capture_dir, folder / "out" / NAME, previous_dir=damaged)


@pytest.fixture(scope="module")
def edge_mapped(main, tmp_path_factory) -> Run:
    """The cross-check maps sensor corners with the pixel-edge convention (no 0.5 terms)."""

    def edge(points, g):
        return np.asarray(points, float) / g.binning - np.array([g.offset_x, g.offset_y], float)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "sensor_to_preset_points", edge)
        return run(main.capture_dir, tmp_path_factory.mktemp("edge_mapped") / NAME)


def test_end_to_end_on_synthetic_videos(main):
    out = main.output_dir
    assert is_multires_calibration(out)
    for g in PRESETS:
        assert (out / "presets" / g.name / "camera_matrix" / "stereo_params.pickle").is_file(), g.name
    assert main.report.verdict != "fail", main.report.to_text()
    # The bin-4 and bin-2 presets are cross-checked; the bin-1 preset is a crop of the fitted frames.
    cross = metrics(main.report, "cross_check_delta_px")
    assert [m.note for m in cross] == ["preset 256x256_bin4", "preset 512x512_bin2"]
    assert all(m.value < 1.5 for m in cross), main.report.to_text()
    # The mapping itself: each preset's corners sit on the mapped sensor corners to the spec's 0.1 px.
    offsets = metrics(main.report, "cross_check_corner_offset_px")
    assert [m.note for m in offsets] == ["preset 256x256_bin4", "preset 512x512_bin2"]
    assert all(m.value < 0.1 and m.status == "pass" for m in offsets), main.report.to_text()
    assert load_index(out) == main.index
    assert main.index.verdict == main.report.verdict
    saved = QualityReport.from_json((out / "quality_report.json").read_text(encoding="utf-8"))
    assert saved.verdict == main.report.verdict and saved.notes == main.report.notes
    (unverified,) = metrics(main.report, "waypoints_unverified")
    assert unverified.value == 0 and unverified.status == "pass"
    # reported 0.01 mm from where commanded on every axis: within 0.1 mm
    (unreached,) = metrics(main.report, "waypoints_unreached")
    assert unreached.value == 0 and unreached.status == "pass"
    assert main.index.capture["waypoints_unreached"] == 0
    assert NO_PREVIOUS_NOTE in main.report.notes


def test_fit_recovers_the_rendered_cameras(main):
    fit = load_sensor_fit(main.output_dir)
    assert fit.image_size == IMAGE_SIZE
    for cam in CAMERAS:
        truth, got = RIG.truth.cameras[cam], fit.cameras[cam]
        np.testing.assert_allclose(np.diag(got.K)[:2], np.diag(truth.K)[:2], rtol=2e-3, err_msg=cam)
        np.testing.assert_allclose(got.K[:2, 2], truth.K[:2, 2], atol=3.0, err_msg=cam)
        assert fit.rms_px[cam] < 0.1, cam
    assert fit.motor_scale == pytest.approx(1.0, abs=2e-3)


def test_capture_info_and_sources_are_kept(main):
    out, capture = main.output_dir, main.index.capture
    assert {key: capture[key] for key in LOGGED_SETTINGS} == LOGGED_SETTINGS
    assert capture["waypoints_total"] == capture["waypoints_distinct"] == capture["waypoints_verified"] == N_WAYPOINTS
    assert capture["corner_refinement"] == "subpix"
    assert capture["corner_order_reversed"] == {"left": False, "right": False}
    assert set(capture["corner_scatter_px"]) == set(CAMERAS)
    assert all(0.0 <= capture["corner_scatter_px"][cam] < 0.3 for cam in CAMERAS)
    assert set(capture["segmentation_stats"]) == set(CAMERAS)
    for cam in CAMERAS:
        stats = capture["segmentation_stats"][cam]
        assert set(stats) == {"p20", "p95", "tau", "moving_fraction", "n_frames"}
        assert all(isinstance(v, float) for v in stats.values())
    # The index is plain YAML: safe_load reads back what load_index returned.
    assert yaml.safe_load((out / INDEX_FILE).read_text(encoding="utf-8"))["capture"] == capture

    for cam in CAMERAS:
        source = main.capture_dir / SOURCE_VIDEOS / f"{cam}.mp4"
        assert filecmp.cmp(source, out / SOURCE_VIDEOS / f"{cam}.mp4", shallow=False), cam
        assert (out / f"coverage_{cam}.png").is_file(), cam
    assert load_waypoint_log(out / WAYPOINT_LOG) == load_waypoint_log(main.capture_dir / WAYPOINT_LOG)


def test_the_still_frames_at_each_waypoint_are_recorded(main):
    # The margin the dwells leave over extraction's shortest segment, for the real capture.
    still = main.index.capture["still_frames"]
    assert set(still) == set(CAMERAS)
    for cam in CAMERAS:
        lengths = [stop - start for start, stop in find_stationary_segments(main.capture_dir / SOURCE_VIDEOS / f"{cam}.mp4")]
        assert len(lengths) == N_WAYPOINTS
        # every waypoint is distinct here, so segment i is waypoint i
        assert still[cam] == {"min": min(lengths), "median": float(np.median(lengths)),
                              "min_waypoint": int(np.argmin(lengths))}, cam
    # The operator sees them in the report too.
    (note,) = [n for n in main.report.notes if n.startswith("still frames per waypoint")]
    for cam in CAMERAS:
        assert f"{cam} min {still[cam]['min']} (waypoint {still[cam]['min_waypoint']})" in note


def test_unverified_waypoints_warn(second):
    (unverified,) = metrics(second.report, "waypoints_unverified")
    assert unverified.value == 1 and unverified.status == "warn"
    assert second.index.capture["waypoints_verified"] == N_WAYPOINTS - 1
    assert second.index.capture["waypoints_total"] == N_WAYPOINTS


def test_a_waypoint_reported_away_from_its_commanded_position_fails(second):
    # The fit tags the view with the commanded position, which the stage did not reach.
    (unreached,) = metrics(second.report, "waypoints_unreached")
    assert unreached.value == 1 and unreached.status == "fail"
    assert second.report.verdict == "fail" and second.index.verdict == "fail"
    assert second.index.capture["waypoints_unreached"] == 1
    assert any(note.startswith(f"waypoint {UNVERIFIED}:") and "1.70 mm" in note for note in second.report.notes), \
        second.report.notes


def _log_reported(*reported):
    motor = [(1.0, 2.0, 3.0)] * len(reported)
    return make_log(motor=motor, reported=reported)


@pytest.mark.parametrize("reported, unreached", [
    ([(1.0, 2.0, 3.0), (1.0 - 1.7, 2.0, 3.0), (1.05, 2.05, 2.95)], [1]),  # one 1.7 mm off in x
    ([None, None, (1.0, 2.0, 3.0)], []),  # no report: no evidence either way
    ([(1.09, 1.91, 3.09), (1.0, 2.0, 3.0), (0.95, 2.0, 3.0)], []),  # all within 0.1 mm
    ([(1.0, 2.0, 3.2), (math.nan, 2.0, 3.0), (1.0, math.inf, 3.0)], [0]),  # z 0.2 off; non-finite: none
], ids=["one_off", "none", "within", "one_axis_and_non_finite"])
def test_unreached_waypoints_are_the_finite_reports_beyond_0_1_mm(reported, unreached):
    assert pipeline.unreached_waypoints(_log_reported(*reported)) == unreached


def test_capture_settings_default_when_the_log_has_none(second):
    assert {key: second.index.capture[key] for key in DEFAULT_SETTINGS} == DEFAULT_SETTINGS


def test_previous_multires_calibration_is_compared(second):
    # The same videos fit the same cameras, so nothing has changed since the main run.
    assert NO_PREVIOUS_NOTE not in second.report.notes
    rotations = metrics(second.report, "camera_rotation_change_deg")
    assert [m.camera for m in rotations] == list(CAMERAS)
    changes = rotations + metrics(second.report, "focal_change_rel") + metrics(second.report, "baseline_change_mm")
    assert len(changes) == 5
    assert all(m.value < 1e-6 and m.status == "pass" for m in changes), second.report.to_text()


def test_previous_legacy_folder_noted(against_legacy):
    assert NO_PREVIOUS_NOTE in against_legacy.report.notes
    assert not metrics(against_legacy.report, "camera_rotation_change_deg")
    assert not is_multires_calibration(LEGACY_FIXTURE)


def test_cameras_labelling_the_board_from_opposite_ends_still_calibrate(turned):
    # The right camera's frames are turned 180 degrees, so it starts the symmetric board from
    # the other end than the left camera does; the pipeline must reverse it before the fit.
    assert turned.index.capture["corner_order_reversed"] == {"left": False, "right": True}
    assert any("right" in note and "reversed" in note for note in turned.report.notes), turned.report.notes
    assert turned.report.verdict != "fail", turned.report.to_text()
    assert all(m.value < 0.1 for m in metrics(turned.report, "reprojection_rms_px")), turned.report.to_text()
    assert all(m.value < 1.5 for m in metrics(turned.report, "cross_check_delta_px")), turned.report.to_text()
    # Turning the frames moves the principal point to (width - 1 - cx, height - 1 - cy).
    truth = RIG.truth.cameras["right"].K
    got = load_sensor_fit(turned.output_dir).cameras["right"].K
    np.testing.assert_allclose(got[:2, 2], [IMAGE_SIZE[0] - 1 - truth[0, 2], IMAGE_SIZE[1] - 1 - truth[1, 2]], atol=3.0)


def test_a_failed_cross_check_fit_is_inf_with_a_note(degraded):
    report = degraded.report
    by_preset = {m.note: m for m in metrics(report, "cross_check_delta_px")}
    assert math.isinf(by_preset["preset 256x256_bin4"].value)
    # The K delta is informational, so an undetermined one warns rather than fails.
    assert by_preset["preset 256x256_bin4"].status == "warn"
    assert by_preset["preset 512x512_bin2"].value < 1.5
    assert any("256x256_bin4" in note and "did not converge" in note for note in report.notes), report.notes
    # The mapping is still checked: the corner offset needs no fit.
    offsets = {m.note: m for m in metrics(report, "cross_check_corner_offset_px")}
    assert offsets["preset 256x256_bin4"].value < 0.1 and offsets["preset 256x256_bin4"].status == "pass"
    # A failed preset fit grades the calibration; it does not abort it.
    assert report.verdict == "warn", report.to_text()
    assert is_multires_calibration(degraded.output_dir)


def test_a_pixel_edge_mapping_fails_the_corner_offset(edge_mapped):
    # u_edge = u_sensor / b - offset leaves out the pixel-centre terms, so every corner the
    # preset detects sits (0.5 / b - 0.5) px from it on each axis: 0.375 at bin 4, 0.25 at bin 2.
    offsets = {m.note: m for m in metrics(edge_mapped.report, "cross_check_corner_offset_px")}
    bin4, bin2 = offsets["preset 256x256_bin4"], offsets["preset 512x512_bin2"]
    assert bin4.value == pytest.approx(0.375 * math.sqrt(2), abs=0.02) and bin4.status == "fail"
    assert bin2.value == pytest.approx(0.25 * math.sqrt(2), abs=0.02) and bin2.status == "fail"
    assert edge_mapped.report.verdict == "fail"


def test_a_previous_calibration_whose_fit_cannot_be_read_is_noted(degraded):
    notes = degraded.report.notes
    assert NO_PREVIOUS_NOTE in notes
    assert any("previous calibration" in note and "could not be read" in note for note in notes), notes
    assert not metrics(degraded.report, "camera_rotation_change_deg")


def test_a_failed_sensor_fit_propagates(main, tmp_path, monkeypatch):
    def fit(observations, board, image_size, camera_names=CAMERAS, **kwargs):
        raise RuntimeError("the motor-frame fit did not converge (stand-in)")

    monkeypatch.setattr(pipeline, "fit_motor_frame", fit)
    out = tmp_path / NAME
    with pytest.raises(RuntimeError, match="stand-in"):
        calibrate_capture(main.capture_dir, output_dir=out, presets=PRESETS)
    assert not out.exists() and not out.with_name(NAME + ".partial").exists()


def test_misaligned_waypoints_raise_and_write_nothing(videos, tmp_path):
    # One more distinct waypoint than the videos have dwells.
    motor = MOTOR + [MOTOR[-1] + np.array([0.0, 0.0, 3.0])]
    capture = make_capture(tmp_path / "capture", videos["left"], videos["right"], make_log(motor=motor))
    out = tmp_path / "out" / NAME
    with pytest.raises(WaypointAlignmentError, match=f"expected {N_WAYPOINTS + 1} stationary segments"):
        calibrate_capture(capture, output_dir=out, presets=PRESETS)
    assert not out.parent.exists()


def test_an_existing_output_folder_is_refused_before_any_work(tmp_path, monkeypatch):
    out = tmp_path / NAME
    out.mkdir()
    (out / "keep.txt").write_text("untouched")

    def extract(*args, **kwargs):
        raise AssertionError("extraction ran for an output folder that already exists")

    monkeypatch.setattr(pipeline, "extract_observations", extract)
    with pytest.raises(FileExistsError):
        calibrate_capture(tmp_path / "no_capture", output_dir=out, presets=PRESETS)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_waypoint_log_round_trips(tmp_path):
    log = make_log(motor=MOTOR[:3], verified=[True, False, True], capture_settings=LOGGED_SETTINGS)
    log.reported[1] = None  # the motor reported no position
    path = tmp_path / WAYPOINT_LOG
    write_waypoint_log(path, log)
    loaded = load_waypoint_log(path)
    assert loaded == log
    assert loaded.motor[0] == tuple(float(v) for v in MOTOR[0]) and isinstance(loaded.motor[0], tuple)
    assert loaded.dcs_moves[1] == ("y", 1.0) and isinstance(loaded.dcs_moves[1], tuple)
    assert loaded.times[2] == (1002.0, 1002.75)
    # Plain YAML, which another tool can read.
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["reported"][1] is None and data["verified"] == [True, False, True]


def test_waypoint_log_missing_a_field_is_refused(tmp_path):
    path = tmp_path / WAYPOINT_LOG
    write_waypoint_log(path, make_log(motor=MOTOR[:2]))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["verified"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not a waypoint log.*verified"):
        load_waypoint_log(path)


def test_waypoint_log_without_capture_settings_loads_empty(tmp_path):
    path = tmp_path / WAYPOINT_LOG
    write_waypoint_log(path, make_log(motor=MOTOR[:2]))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["capture_settings"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_waypoint_log(path).capture_settings == {}


@pytest.mark.parametrize("field", ["reported", "verified", "dcs_moves", "times"])
def test_waypoint_log_with_lists_of_different_lengths_is_refused(tmp_path, field):
    log = make_log(motor=MOTOR[:3])
    getattr(log, field).pop()
    with pytest.raises(ValueError, match=field):
        write_waypoint_log(tmp_path / WAYPOINT_LOG, log)
    # A file written by hand with the same defect is refused on load.
    good = make_log(motor=MOTOR[:3])
    path = tmp_path / "hand.yaml"
    write_waypoint_log(path, good)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data[field].pop()
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_waypoint_log(path)
