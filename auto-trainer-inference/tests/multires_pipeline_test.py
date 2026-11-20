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
fit fails and the previous calibration's fit cannot be read), a run whose cross-check
maps the sensor corners with the pixel-edge convention, and a run whose capture has the
cameras' timestamps files, so its frames come from the logged dwells.
"""

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
from autotrainer.inference.multires_calibration.extract import (
    FROM_DWELLS,
    FROM_MOTION,
    WaypointAlignmentError,
    find_stationary_segments,
)
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.fit import WaypointObservation
from autotrainer.inference.multires_calibration.pipeline import (
    MIN_COVISIBLE,
    MIN_VIEWS,
    SOURCE_VIDEOS,
    TIMESTAMPS_SUFFIX,
    WAYPOINT_LOG,
    CalibrationRefused,
    CaptureCamera,
    WaypointLog,
    calibrate_capture,
    capture_cameras,
    load_frame_times,
    load_waypoint_log,
    per_camera_rule,
    write_waypoint_log,
)
from autotrainer.inference.multires_calibration.report import NO_PREVIOUS_NOTE, CameraVerdict, Metric, QualityReport
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
# A third camera that sees the board at only 4 of the waypoints, too few to calibrate. Its
# rig has the same left and right cameras as RIG.
CAMERA4_SEEN = (0, 4, 8, 12)
CAMERA4 = synthetic_rig(cameras=(*CAMERAS, "camera4"), frame_rotation_deg=35.0).truth.cameras["camera4"]
MOTOR = [o.motor_mm.copy() for o in RIG.observations[:N_WAYPOINTS]]
# Not the defaults, so the main run shows they are read from waypoints.yaml.
LOGGED_SETTINGS = {"gain_db": 2.5, "gamma": None, "exposure_us": 4000, "fps": 40}
DEFAULT_SETTINGS = {"gain_db": 1.0, "gamma": None, "exposure_us": 4500, "fps": 40}
# The wall-clock time of frame 0 in the timestamps files the timed run writes.
T0 = 1_795_134_208.5


@dataclass
class Run:
    capture_dir: Path
    output_dir: Path
    index: CalibrationIndex
    report: QualityReport
    seconds: float


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> Dict[str, Path]:
    """left.mp4, right.mp4, right_turned.mp4 (the right camera's frames turned 180 degrees)
    and camera4.mp4, which shows the board only at the CAMERA4_SEEN waypoints."""
    folder = tmp_path_factory.mktemp("rendered")
    paths = {name: folder / f"{name}.mp4" for name in ("left", "right", "right_turned", "camera4")}
    outputs = {"left": ["left"], "right": ["right", "right_turned"], "camera4": ["camera4"]}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam, names in outputs.items():
        writers = {name: cv2.VideoWriter(str(paths[name]), fourcc, FPS, IMAGE_SIZE, isColor=False) for name in names}
        assert all(w.isOpened() for w in writers.values())
        if cam == "camera4":
            frames = capture_frames(CAMERA4, seen=CAMERA4_SEEN)
        else:
            frames = capture_frames(RIG.truth.cameras[cam])
        for image in frames:
            for name, writer in writers.items():
                writer.write(cv2.rotate(image, cv2.ROTATE_180) if name == "right_turned" else image)
        for writer in writers.values():
            writer.release()
    return paths


def capture_frames(model, seen=None):
    """The camera's frames: a dwell at each waypoint, with moving frames between dwells.

    With `seen`, the board shows only in the dwells at those waypoints; every other frame
    is the bare background, as if the board were out of the camera's view."""
    blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), 100, np.uint8)
    for k, position in enumerate(MOTOR):
        if k > 0:
            previous = MOTOR[k - 1]
            for j in range(1, MOVING_FRAMES + 1):
                between = previous + (position - previous) * j / (MOVING_FRAMES + 1)
                if seen is None:
                    yield render_board_frame(model, RIG.world_corners(between), IMAGE_SIZE, RIG.board)
                else:
                    yield blank
        if seen is not None and k not in seen:
            still = blank
        else:
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
    return make_capture_of(folder, {"left": left, "right": right}, log)


def make_capture_of(folder: Path, videos: Dict[str, Path], log: WaypointLog) -> Path:
    """A capture folder with each camera's video, named for its camera, and `log`."""
    (folder / SOURCE_VIDEOS).mkdir(parents=True)
    for cam, video in videos.items():
        shutil.copyfile(video, folder / SOURCE_VIDEOS / f"{cam}.mp4")
    write_waypoint_log(folder / WAYPOINT_LOG, log)
    return folder


def write_timestamps(capture: Path, frame_times: Dict[str, np.ndarray]) -> None:
    """Each camera's timestamps file as the recorder writes it, with `frame_times` in column 0."""
    for cam, times in frame_times.items():
        rows = [f"{t!r}, {FPS}, {1000 * i}, {500.0 + i / FPS!r}, {i}" for i, t in enumerate(times)]
        (capture / SOURCE_VIDEOS / f"{cam}{TIMESTAMPS_SUFFIX}").write_text("\n".join(rows) + "\n")


def timed_log() -> WaypointLog:
    """make_log, with every waypoint's dwell logged at its still frames' times: waypoint k
    holds still over frames k * (MOVING_FRAMES + STILL_FRAMES) on, for STILL_FRAMES frames."""
    log = make_log()
    step = MOVING_FRAMES + STILL_FRAMES
    log.times = [(T0 + k * step / FPS, T0 + (k * step + STILL_FRAMES - 1) / FPS) for k in range(N_WAYPOINTS)]
    return log


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
def timed(videos, tmp_path_factory) -> Run:
    folder = tmp_path_factory.mktemp("timed")
    capture = make_capture(folder / "capture", videos["left"], videos["right"], timed_log())
    n = N_WAYPOINTS * STILL_FRAMES + (N_WAYPOINTS - 1) * MOVING_FRAMES
    times = T0 + np.arange(n) / FPS
    write_timestamps(capture, {cam: times for cam in CAMERAS})
    return run(capture, folder / "out" / NAME)


def timed_capture_of(folder: Path, videos: Dict[str, Path], cameras) -> Path:
    """A capture of `cameras`, each with its video in `videos`, with timed_log's dwells, the
    cameras in its log, and every camera's timestamps file."""
    log = timed_log()
    log.cameras = [CaptureCamera(cam, None, i) for i, cam in enumerate(cameras)]
    capture = make_capture_of(folder, {cam: videos[cam] for cam in cameras}, log)
    n = N_WAYPOINTS * STILL_FRAMES + (N_WAYPOINTS - 1) * MOVING_FRAMES
    write_timestamps(capture, {cam: T0 + np.arange(n) / FPS for cam in cameras})
    return capture


@pytest.fixture(scope="module")
def with_camera4(videos, tmp_path_factory) -> Run:
    """left, right and camera4, which saw the board at only 4 waypoints."""
    folder = tmp_path_factory.mktemp("camera4")
    capture = timed_capture_of(folder / "capture", videos, ("left", "right", "camera4"))
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
    # Every preset's RMS is reported in its own pixels, for both cameras.
    per_preset = {(m.camera, m.note) for m in metrics(main.report, "preset_reprojection_rms_px")}
    assert per_preset == {(cam, f"preset {g.name}") for cam in CAMERAS for g in PRESETS}
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


def test_capture_info_is_kept_and_the_source_videos_are_named_not_copied(main):
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

    # The videos stay in the capture, which the calibration names, absolute, rather than
    # copied into every calibration and again into every install (M5).
    assert capture["capture_folder"] == str(main.capture_dir.resolve())
    assert Path(capture["capture_folder"]).is_absolute()
    assert not (out / SOURCE_VIDEOS).exists()
    assert not list(out.rglob("*.mp4"))
    for cam in CAMERAS:
        assert (main.capture_dir / SOURCE_VIDEOS / f"{cam}.mp4").is_file(), cam
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


def test_a_capture_without_timestamps_is_segmented_by_motion(main):
    assert main.index.capture["segments_from"] == FROM_MOTION
    assert any(n.startswith("still frames per waypoint") for n in main.report.notes)


def test_a_capture_with_timestamps_takes_each_waypoint_from_its_logged_dwell(timed):
    assert timed.index.capture["segments_from"] == FROM_DWELLS
    # the logged dwells are exactly the still frames, so each holds all of them
    for cam in CAMERAS:
        assert timed.index.capture["still_frames"][cam]["min"] == STILL_FRAMES
        assert timed.index.capture["segmentation_stats"][cam]["motion_segments"] == N_WAYPOINTS
    (note,) = [n for n in timed.report.notes if n.startswith("frames per logged dwell")]
    assert f"left min {STILL_FRAMES}" in note
    assert timed.report.verdict != "fail", timed.report.to_text()


def test_the_logged_dwells_and_column_0_reach_extraction(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    (capture / SOURCE_VIDEOS).mkdir(parents=True)
    for cam in CAMERAS:
        (capture / SOURCE_VIDEOS / f"{cam}.mp4").write_bytes(b"")
    log = timed_log()
    write_waypoint_log(capture / WAYPOINT_LOG, log)
    times = {"left": T0 + np.arange(4) / FPS, "right": T0 + 0.001 + np.arange(4) / FPS}
    write_timestamps(capture, times)
    seen = {}

    class Stop(Exception):
        pass

    def extract(videos, motor, board, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(pipeline, "_video_size", lambda videos: IMAGE_SIZE)
    monkeypatch.setattr(pipeline, "extract_observations", extract)
    with pytest.raises(Stop):
        calibrate_capture(capture, output_dir=tmp_path / "out" / NAME, presets=PRESETS)
    assert seen["dwells"] == log.times
    assert set(seen["frame_times"]) == set(CAMERAS)
    for cam in CAMERAS:
        np.testing.assert_array_equal(seen["frame_times"][cam], times[cam])


def test_frame_times_are_none_without_timestamps_and_refused_for_only_some_cameras(tmp_path):
    (tmp_path / SOURCE_VIDEOS).mkdir()
    assert load_frame_times(tmp_path, CAMERAS) is None
    write_timestamps(tmp_path, {"left": T0 + np.arange(3) / FPS})
    with pytest.raises(FileNotFoundError, match="right_timestamps.txt"):
        load_frame_times(tmp_path, CAMERAS)


def three_camera_log() -> WaypointLog:
    log = make_log()
    log.cameras = [CaptureCamera("a", "111", 0), CaptureCamera("b", None, 1), CaptureCamera("c", "333", 5)]
    return log


def test_waypoint_log_round_trips_its_cameras(tmp_path):
    log = three_camera_log()
    write_waypoint_log(tmp_path / WAYPOINT_LOG, log)
    loaded = load_waypoint_log(tmp_path / WAYPOINT_LOG)
    assert loaded == log
    assert capture_cameras(loaded) == ["a", "b", "c"]
    assert yaml.safe_load((tmp_path / WAYPOINT_LOG).read_text())["cameras"][1] == {
        "name": "b", "serial": None, "camera_id": 1}


def test_a_log_without_cameras_is_left_and_right(tmp_path):
    write_waypoint_log(tmp_path / WAYPOINT_LOG, make_log())
    data = yaml.safe_load((tmp_path / WAYPOINT_LOG).read_text())
    del data["cameras"]
    (tmp_path / WAYPOINT_LOG).write_text(yaml.safe_dump(data))
    loaded = load_waypoint_log(tmp_path / WAYPOINT_LOG)
    assert loaded.cameras == [CaptureCamera("left", None, 0), CaptureCamera("right", None, 1)]
    assert capture_cameras(loaded) == ["left", "right"]


def test_a_capture_names_its_own_cameras(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    (capture / SOURCE_VIDEOS).mkdir(parents=True)
    for cam in "abc":
        (capture / SOURCE_VIDEOS / f"{cam}.mp4").write_bytes(b"")
    write_waypoint_log(capture / WAYPOINT_LOG, three_camera_log())
    write_timestamps(capture, {cam: T0 + np.arange(4) / FPS for cam in "abc"})
    seen = {}

    class Stop(Exception):
        pass

    def extract(videos, motor, board, **kwargs):
        seen["videos"], seen["frame_times"] = list(videos.keys()), kwargs["frame_times"]
        raise Stop

    monkeypatch.setattr(pipeline, "_video_size", lambda videos: IMAGE_SIZE)
    monkeypatch.setattr(pipeline, "extract_observations", extract)
    with pytest.raises(Stop):
        calibrate_capture(capture, output_dir=tmp_path / "out" / NAME, presets=PRESETS)
    assert seen["videos"] == ["a", "b", "c"]
    assert list(seen["frame_times"]) == ["a", "b", "c"]


@pytest.mark.parametrize("count", [1, 7])
def test_more_than_six_or_fewer_than_two_cameras_are_refused(tmp_path, count):
    log = make_log()
    log.cameras = [CaptureCamera(f"cam{i}", None, i) for i in range(count)]
    capture = tmp_path / "capture"
    capture.mkdir()
    write_waypoint_log(capture / WAYPOINT_LOG, log)
    with pytest.raises(ValueError, match="2 to 6 cameras"):
        calibrate_capture(capture, output_dir=tmp_path / "out" / NAME, presets=PRESETS)


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


def test_a_camera_with_four_views_is_uncalibrated_and_the_rest_fit(with_camera4):
    report = with_camera4.report
    assert report.cameras["camera4"] == CameraVerdict("camera4", False, "too few board views: 4 of 12 needed")
    assert report.cameras["left"] == CameraVerdict("left", True, "")
    assert report.cameras["right"] == CameraVerdict("right", True, "")
    assert report.to_text().splitlines()[0].startswith("NOT CALIBRATED (2D only): camera4")
    # The other two are fitted as they are without it, and the 2D-only camera makes it warn.
    fit = load_sensor_fit(with_camera4.output_dir)
    assert set(fit.cameras) == set(CAMERAS)
    for cam in CAMERAS:
        truth, got = RIG.truth.cameras[cam], fit.cameras[cam]
        np.testing.assert_allclose(np.diag(got.K)[:2], np.diag(truth.K)[:2], rtol=2e-3, err_msg=cam)
        assert fit.rms_px[cam] < 0.1, cam
    assert report.verdict == "warn", report.to_text()
    assert {m.camera for m in metrics(report, "reprojection_rms_px")} == set(CAMERAS)
    # What it saw stays on record, and the saved report carries the verdicts.
    assert with_camera4.index.capture["still_frames"]["camera4"]["min"] == STILL_FRAMES
    assert (with_camera4.output_dir / "coverage_camera4.png").is_file()
    saved = QualityReport.from_json((with_camera4.output_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert saved.cameras == report.cameras


def test_one_calibrated_camera_refuses(videos, tmp_path):
    # Without the right camera, the left one has no calibrated camera to share its views with.
    capture = timed_capture_of(tmp_path / "capture", videos, ("left", "camera4"))
    out = tmp_path / "out" / NAME
    with pytest.raises(CalibrationRefused) as raised:
        calibrate_capture(capture, output_dir=out, presets=PRESETS)
    message = str(raised.value)
    assert "left: too few views shared with another camera: 0 of 8 needed" in message
    assert "camera4: too few board views: 4 of 12 needed" in message
    assert isinstance(raised.value, RuntimeError)
    assert raised.value.verdicts["camera4"] == CameraVerdict("camera4", False, "too few board views: 4 of 12 needed")
    assert not out.parent.exists()


# Commanded positions that vary on every axis between any two waypoints, for the rule's tests.
SPREAD_MOTOR = [(float(w), float(7 * w % 11), float(3 * w % 13)) for w in range(37)]


def observations_seen_by(seen, motor=SPREAD_MOTOR):
    """One observation per waypoint of `motor`, holding the cameras in `seen` (camera -> the
    waypoints it saw); the rule does not read the corners."""
    return [
        WaypointObservation(w, np.asarray(m, float), {cam: np.zeros((48, 2)) for cam, ws in seen.items() if w in ws})
        for w, m in enumerate(motor)
    ]


def test_the_rule_thresholds():
    assert (MIN_VIEWS, MIN_COVISIBLE) == (12, 8)


def test_cameras_that_saw_enough_together_are_calibrated():
    verdicts = per_camera_rule(observations_seen_by({"a": range(37), "b": range(37), "c": range(25)}), "abc")
    assert verdicts == {cam: CameraVerdict(cam, True, "") for cam in "abc"}
    assert list(verdicts) == ["a", "b", "c"]


def test_views_that_do_not_span_three_axes_are_refused():
    # Every waypoint at one height: the views lie in one plane of the stage.
    flat = [(x, y, -30.0) for x, y, _ in SPREAD_MOTOR]
    verdicts = per_camera_rule(observations_seen_by({cam: range(37) for cam in "abc"}, motor=flat), "abc")
    assert verdicts == {cam: CameraVerdict(cam, False, "board views do not span all 3 stage axes") for cam in "abc"}


def test_too_few_views_or_too_few_shared_views_are_refused():
    verdicts = per_camera_rule(observations_seen_by({"a": range(37), "b": range(37), "c": range(30, 37)}), "abc")
    assert verdicts["c"] == CameraVerdict("c", False, "too few board views: 7 of 12 needed")
    verdicts = per_camera_rule(observations_seen_by({"a": range(30), "b": range(30), "c": range(23, 37)}), "abc")
    assert verdicts["c"] == CameraVerdict("c", False, "too few views shared with another camera: 7 of 8 needed")
    assert verdicts["a"].calibrated and verdicts["b"].calibrated


def test_a_camera_whose_shared_views_are_with_dropped_cameras_is_dropped_too():
    # a and b see 0-15 together. t sees 12-23: 4 views shared with a, and 4 with s (20-23).
    # s sees 20-31 and shares only those 4 with t, so it goes first; then t has 4 left.
    seen = {"a": range(16), "b": range(12), "t": range(12, 24), "s": range(20, 32)}
    verdicts = per_camera_rule(observations_seen_by(seen), ["a", "b", "t", "s"])
    assert verdicts["s"] == CameraVerdict("s", False, "too few views shared with another camera: 4 of 8 needed")
    assert verdicts["t"] == CameraVerdict("t", False, "too few views shared with another camera: 4 of 8 needed")
    assert verdicts["a"].calibrated and verdicts["b"].calibrated


def test_cameras_outside_the_largest_connected_group_are_refused():
    # a, b and c see 0-17 together, d and e 18-36 together: two groups that never see the
    # board at once. The larger one is calibrated.
    seen = {"a": range(18), "b": range(18), "c": range(18), "d": range(18, 37), "e": range(18, 37)}
    verdicts = per_camera_rule(observations_seen_by(seen), "abcde")
    assert all(verdicts[cam].calibrated for cam in "abc")
    for cam in "de":
        assert verdicts[cam] == CameraVerdict(cam, False, "not connected to the other cameras through shared views")
