"""The ChArUco calibration board, with partial views.

The rig's board is cv2.aruco.CharucoBoard((9, 7), 4.0, 3.0, DICT_4X4_50): 8 x 6 inner
corners, like the 4 mm checkerboard, but every corner carries its id, so a camera that sees
only part of the board still contributes that part. These tests render it with
multires_synthetic and check detection, the per-waypoint median, the fit, the report, the
scout, the store and the library setting on views that hold only some of the corners.
"""

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from reachaq.inference.multires_calibration import store
from reachaq.inference.multires_calibration.extract import detect_board, median_corners
from reachaq.inference.multires_calibration.fit import (
    CHARUCO,
    MIN_CHARUCO_CORNERS,
    RIG_CHARUCO_BOARD,
    BoardSpec,
    CameraModel,
    WaypointObservation,
    fit_motor_frame,
    reconcile_corner_order,
    usable_view,
)
from reachaq.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from reachaq.inference.multires_calibration.pipeline import (
    SOURCE_VIDEOS,
    WAYPOINT_LOG,
    WaypointLog,
    calibrate_capture,
    write_waypoint_log,
)
from reachaq.inference.multires_calibration.report import CaptureSummary, build_report
from reachaq.inference.multires_calibration.scout import configured_board, scout_view
from multires_synthetic import CHARUCO_BOARD, render_board_frame, synthetic_rig

SIZE = (1440, 1080)
CAMERAS = ("left", "right")
RIG = synthetic_rig(frame_rotation_deg=35.0)
BASE = PresetGeometry(4, 256, 256, 52, 6)


def truth_px(model: CameraModel, motor_mm) -> np.ndarray:
    """Where render_board_frame draws the inner corners: the projection without distortion."""
    X = RIG.world_corners(motor_mm)
    return cv2.projectPoints(X, model.rvec, model.tvec, model.K, np.zeros(5))[0].reshape(-1, 2)


def render(model: CameraModel, motor_mm) -> np.ndarray:
    return render_board_frame(model, RIG.world_corners(motor_mm), SIZE, CHARUCO_BOARD)


def test_charuco_ids_are_object_points_order():
    board = CHARUCO_BOARD.charuco_board()
    # cv2 measures from the board's outer corner, one square before inner corner 0.
    cv2_points = np.asarray(board.getChessboardCorners(), float) - [4.0, 4.0, 0.0]
    np.testing.assert_allclose(cv2_points, CHARUCO_BOARD.object_points(), atol=1e-6)
    k = np.arange(48)
    np.testing.assert_allclose(CHARUCO_BOARD.object_points()[:, :2], np.column_stack([k % 8, k // 8]) * 4.0)


def test_charuco_board_is_the_generator_board():
    # The rig's board was made with exactly this, and generateImage((w, h), marginSize=0,
    # borderBits=1), at OpenCV 4.11's default legacy pattern (off).
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    generator = cv2.aruco.CharucoBoard((9, 7), 4.0, 3.0, dictionary)
    spec = BoardSpec(kind=CHARUCO, rows=6, cols=8, square_mm=4.0, marker_mm=3.0, dictionary="DICT_4X4_50")
    ours = spec.charuco_board()
    assert tuple(ours.getChessboardSize()) == tuple(generator.getChessboardSize()) == (9, 7)
    assert ours.getSquareLength() == generator.getSquareLength() == 4.0
    assert ours.getMarkerLength() == generator.getMarkerLength() == 3.0
    np.testing.assert_array_equal(np.asarray(ours.getIds()), np.asarray(generator.getIds()))
    ours_obj, generator_obj = ours.getObjPoints(), generator.getObjPoints()
    assert len(ours_obj) == len(generator_obj) == 31
    for a, b in zip(ours_obj, generator_obj):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert ours.getLegacyPattern() is False
    image = ours.generateImage((900, 700), marginSize=0, borderBits=1)
    expected = generator.generateImage((900, 700), marginSize=0, borderBits=1)
    assert image.dtype == expected.dtype and image.shape == expected.shape
    assert image.tobytes() == expected.tobytes()


def test_charuco_board_pins_the_legacy_pattern_off(monkeypatch):
    # A future OpenCV whose CharucoBoard defaults to the legacy pattern must not change the board.
    real = cv2.aruco.CharucoBoard

    def legacy_by_default(*args, **kwargs):
        board = real(*args, **kwargs)
        board.setLegacyPattern(True)
        return board

    monkeypatch.setattr(cv2.aruco, "CharucoBoard", legacy_by_default)
    assert CHARUCO_BOARD.charuco_board().getLegacyPattern() is False


def test_the_rig_board_has_the_checkerboard_geometry():
    assert (CHARUCO_BOARD.rows, CHARUCO_BOARD.cols, CHARUCO_BOARD.square_mm) == (6, 8, 4.0)
    np.testing.assert_array_equal(CHARUCO_BOARD.object_points(), BoardSpec().object_points())


@pytest.mark.parametrize("cam", CAMERAS)
def test_detect_board_finds_every_charuco_corner(cam):
    model = RIG.truth.cameras[cam]
    motor = RIG.observations[0].motor_mm
    corners = detect_board(render(model, motor), CHARUCO_BOARD)
    assert corners is not None and corners.shape == (48, 2)
    assert np.all(np.isfinite(corners))
    assert np.max(np.linalg.norm(corners - truth_px(model, motor), axis=1)) < 0.1


# A real frame of the rig's ChArUco board: christielab10 session010's scout, left camera,
# frame 1080 (2026-11-23). It is dim (mean grey 58); the checkerboard's contrast stretch
# erases its markers, so the board is found in it only on the raw grey (48 corners there,
# none after the stretch).
REAL_FRAME = Path(__file__).parent / "data" / "charuco" / "session010_left_scout_frame1080.png"


def test_detect_board_finds_the_charuco_board_on_a_real_dim_frame():
    gray = cv2.imread(str(REAL_FRAME), cv2.IMREAD_GRAYSCALE)
    assert gray is not None and gray.shape == (1080, 1440), "is the LFS object checked out?"
    corners = detect_board(gray, RIG_CHARUCO_BOARD)
    assert corners is not None
    assert int(np.isfinite(corners).all(axis=1).sum()) >= MIN_CHARUCO_CORNERS
    view = scout_view("left", gray, RIG_CHARUCO_BOARD)
    assert view.found


def test_a_cropped_board_gives_nan_for_the_hidden_corners():
    model = RIG.truth.cameras["left"]
    motor = RIG.observations[0].motor_mm
    truth = truth_px(model, motor)
    image = render(model, motor)
    # Cut the image half way along the board's rows: the right part shows only background.
    cut = int(round((truth[:, 0].min() + truth[:, 0].max()) / 2))
    image[:, cut:] = 100
    corners = detect_board(image, CHARUCO_BOARD)
    assert corners is not None
    found = np.all(np.isfinite(corners), axis=1)
    pitch = float(np.median(np.linalg.norm(np.diff(truth.reshape(6, 8, 2), axis=1), axis=-1)))
    assert not found[truth[:, 0] >= cut].any(), "a corner outside the image was reported"
    # Every corner a full square inside the cut has both its markers, so it is found.
    assert found[truth[:, 0] < cut - pitch].all()
    assert found.sum() >= MIN_CHARUCO_CORNERS
    assert np.max(np.linalg.norm(corners[found] - truth[found], axis=1)) < 0.1


def test_too_few_or_collinear_corners_are_no_view():
    model = RIG.truth.cameras["left"]
    motor = RIG.observations[0].motor_mm
    truth = truth_px(model, motor)
    image = render(model, motor)
    # Only the first column or so of squares in view: fewer than MIN_CHARUCO_CORNERS corners.
    pitch = float(np.median(np.linalg.norm(np.diff(truth.reshape(6, 8, 2), axis=1), axis=-1)))
    cut = int(round(truth[:, 0].min() + 0.5 * pitch))
    image[:, cut:] = 100
    assert detect_board(image, CHARUCO_BOARD) is None

    full = np.full((48, 2), np.nan)
    full[:16] = 1.0  # two whole rows
    assert usable_view(full, CHARUCO_BOARD)
    one_row = np.full((48, 2), np.nan)
    one_row[:8] = 1.0
    one_row[8:12] = 1.0  # 12 corners, two rows
    assert usable_view(one_row, CHARUCO_BOARD)
    column = np.full((48, 2), np.nan)
    column[0::8] = 1.0
    column[1::8] = 1.0  # 12 corners in two columns
    assert usable_view(column, CHARUCO_BOARD)
    line = np.full((48, 2), np.nan)
    line[0::8] = 1.0  # 6 corners down one column
    assert not usable_view(line, CHARUCO_BOARD)
    eleven = np.full((48, 2), np.nan)
    eleven[:11] = 1.0
    assert not usable_view(eleven, CHARUCO_BOARD)
    # A checkerboard view is always whole.
    assert usable_view(np.zeros((48, 2)), BoardSpec())


def test_the_median_keeps_a_corner_seen_in_half_the_frames():
    stack = np.tile(np.arange(96, dtype=float).reshape(48, 2), (4, 1, 1))
    stack[:2, 0] = np.nan  # seen in 2 of 4 frames: kept
    stack[:3, 1] = np.nan  # seen in 1 of 4: dropped
    stack[0, 2] += 1.0
    median = median_corners(stack, CHARUCO_BOARD)
    np.testing.assert_array_equal(median[0], [0.0, 1.0])
    assert np.all(np.isnan(median[1]))
    np.testing.assert_array_equal(median[2], [4.0, 5.0])
    # The checkerboard's is np.median, unchanged.
    np.testing.assert_array_equal(median_corners(stack[:, 3:], BoardSpec(rows=5, cols=9)),
                                  np.median(stack[:, 3:], axis=0))


def partial_observations(rig, cam, keep):
    """rig's observations with `cam`'s corners NaN outside keep(waypoint index) (a mask of 48)."""
    out = []
    for o in rig.observations:
        corners = dict(o.corners)
        if cam in corners:
            c = np.array(corners[cam], float)
            c[~keep(o.index)] = np.nan
            corners[cam] = c
        out.append(WaypointObservation(index=o.index, motor_mm=o.motor_mm.copy(), corners=corners))
    return out


def _keep(index):
    """The partial views of the fit tests: left half, top half, or (every fifth) a corner of
    8 corners, fewer than MIN_CHARUCO_CORNERS."""
    k = np.arange(48)
    if index % 5 == 4:
        return (k % 8 < 4) & (k // 8 < 2)
    if index % 2 == 0:
        return k % 8 < 4
    return k // 8 < 3


def test_the_fit_recovers_a_camera_that_sees_only_partial_boards():
    rig = synthetic_rig(noise_px=0.1)
    observations = partial_observations(rig, "right", _keep)
    board = CHARUCO_BOARD
    reconciled, flipped = reconcile_corner_order(observations, board, SIZE, CAMERAS)
    assert flipped == {"left": False, "right": False}
    fit = fit_motor_frame(reconciled, board, SIZE, CAMERAS)
    for cam in CAMERAS:
        t, f = rig.truth.cameras[cam], fit.cameras[cam]
        assert f.K[0, 0] == pytest.approx(t.K[0, 0], rel=0.01), cam
        assert f.K[1, 1] == pytest.approx(t.K[1, 1], rel=0.01), cam
        assert abs(f.K[0, 2] - t.K[0, 2]) < 8.0 and abs(f.K[1, 2] - t.K[1, 2]) < 8.0, cam
        assert np.degrees(np.linalg.norm(cv2.Rodrigues(
            cv2.Rodrigues(f.rvec)[0].T @ cv2.Rodrigues(t.rvec)[0])[0])) < 0.3, cam
        assert np.linalg.norm(f.tvec - t.tvec) < 1.0, cam
        # Over the detected corners only: the 0.1 px noise per coordinate, sqrt(2) x 0.1 per corner.
        assert 0.1 < fit.rms_px[cam] < 0.2, cam
    assert fit.motor_scale == pytest.approx(1.0, abs=0.003)
    # Every view with MIN_CHARUCO_CORNERS or more corners is used; the 8-corner ones are not.
    expected = [o.index for o in observations if np.isfinite(o.corners["right"]).all(axis=1).sum() >= 12]
    assert sorted(fit.per_view_rms_px["right"]) == expected
    assert len(fit.per_view_rms_px["left"]) == len(observations)
    assert fit.corner_counts["right"] == {i: 24 for i in expected}
    assert fit.corner_counts["left"] == {o.index: 48 for o in observations}


def test_the_report_counts_partial_views():
    rig = synthetic_rig(noise_px=0.1)
    observations = partial_observations(rig, "right", _keep)
    fit = fit_motor_frame(observations, CHARUCO_BOARD, SIZE, CAMERAS)
    n = len(observations)
    used = len(fit.per_view_rms_px["right"])
    capture = CaptureSummary(waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
                             views_kept={cam: n for cam in CAMERAS}, rejections={cam: {} for cam in CAMERAS})
    report = build_report(fit, observations, CHARUCO_BOARD, BASE.sensor_roi(), capture, [], None,
                          presets=presets_from_base(BASE))
    covisible = {m.camera: m.value for m in report.metrics if m.name == "covisible_views"}
    # A co-visible view needs both cameras to pass the minimum: the 8-corner views do not.
    assert covisible == {"left": used, "right": used}
    assert f"right: views used {used} of {n} waypoints, median 24 corners per view" in report.notes
    assert f"left: views used {n} of {n} waypoints, median 48 corners per view" in report.notes
    for name in ("epipolar_error_px", "square_size_rel_error", "coverage_roi_fraction", "stereo_reprojection_rms_px"):
        assert all(math.isfinite(m.value) for m in report.metrics if m.name == name), name
    square = [m for m in report.metrics if m.name == "square_size_rel_error"]
    assert square[0].value < 0.01


def test_a_checkerboard_report_has_no_views_used_line():
    rig = synthetic_rig(noise_px=0.1)
    fit = fit_motor_frame(rig.observations, rig.board, SIZE, CAMERAS)
    n = len(rig.observations)
    capture = CaptureSummary(waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
                             views_kept={cam: n for cam in CAMERAS}, rejections={cam: {} for cam in CAMERAS})
    report = build_report(fit, rig.observations, rig.board, BASE.sensor_roi(), capture, [], None)
    assert not any("views used" in note for note in report.notes)
    assert fit.corner_counts is None


def test_scout_finds_a_partial_board_at_the_image_edge():
    model = RIG.truth.cameras["left"]
    motor = RIG.observations[0].motor_mm
    # The board's middle on the right edge: half of it runs off the image.
    shifted = _shifted(model, SIZE[0] - float(truth_px(model, motor)[:, 0].mean()))
    truth = truth_px(shifted, motor)
    assert (truth[:, 0] > SIZE[0]).any() and (truth[:, 0] < SIZE[0] - 50).sum() >= 16
    view = scout_view("left", render(shifted, motor), CHARUCO_BOARD)
    assert view.found and view.touches_edge == "right"
    assert np.isnan(view.corners).any() and math.isfinite(view.aspect)


def test_store_round_trips_a_charuco_board():
    plain = store._board_to_plain(CHARUCO_BOARD)
    assert plain == {"square_mm": 4.0, "rows": 6, "cols": 8, "kind": "charuco", "marker_mm": 3.0,
                     "dictionary": "DICT_4X4_50"}
    assert store._board_from_plain(yaml.safe_load(yaml.safe_dump(plain))) == CHARUCO_BOARD
    # A checkerboard is written as it always was, and an entry without a kind reads as one.
    assert store._board_to_plain(BoardSpec()) == {"square_mm": 4.0, "rows": 6, "cols": 8}
    assert store._board_from_plain({"square_mm": 4.0, "rows": 6, "cols": 8}) == BoardSpec()


def test_board_setting_parses():
    # The rig's ChArUco board is the default (Ben, 2026-11-23); BoardSpec() stays the checkerboard.
    assert configured_board({}) == RIG_CHARUCO_BOARD == CHARUCO_BOARD
    assert configured_board({"scoutBounds": None}) == RIG_CHARUCO_BOARD
    assert configured_board({"board": None}) == RIG_CHARUCO_BOARD
    setting = {"kind": "charuco", "square_mm": 4.0, "rows": 6, "cols": 8, "marker_mm": 3.0,
               "dictionary": "DICT_4X4_50"}
    assert configured_board({"board": setting}) == CHARUCO_BOARD
    assert configured_board({"board": {"kind": "checkerboard"}}) == BoardSpec()
    assert configured_board({"board": {"kind": "checkerboard", "square_mm": 4.0, "rows": 6, "cols": 8}}) == BoardSpec()
    # A ChArUco key left out takes the rig board's value.
    assert configured_board({"board": {"kind": "charuco"}}) == RIG_CHARUCO_BOARD
    assert configured_board({"board": {"marker_mm": 2.5}}) == dataclasses.replace(RIG_CHARUCO_BOARD, marker_mm=2.5)


def test_board_spec_default_is_still_the_checkerboard():
    assert BoardSpec().kind == "checkerboard" and BoardSpec().marker_mm is None


@pytest.mark.parametrize("change, key", [
    ({"kind": "circles"}, "kind"),
    ({"dictionary": "DICT_9X9_1"}, "dictionary"),
    ({"dictionary": "getPredefinedDictionary"}, "dictionary"),
    ({"marker_mm": 4.0}, "marker_mm"),
    ({"marker_mm": 0.0}, "marker_mm"),
    ({"kind": "checkerboard", "marker_mm": 3.0}, "marker_mm"),
    ({"dictionary": "DICT_4X4_50", "rows": 10, "cols": 12}, "dictionary"),  # 143 squares, 71 markers
    ({"rows": "six"}, "rows"),
    ({"rows": 6.5}, "rows"),
    ({"square_mm": -1.0}, "square_mm"),
    ({"colour": "white"}, "colour"),
])
def test_a_bad_board_setting_is_refused_naming_the_key(change, key):
    setting = {"kind": "charuco", "square_mm": 4.0, "rows": 6, "cols": 8, "marker_mm": 3.0,
               "dictionary": "DICT_4X4_50", **change}
    with pytest.raises(ValueError, match=rf"board\.{key}"):
        configured_board({"board": setting})


def test_a_board_setting_that_is_not_a_mapping_is_refused():
    with pytest.raises(ValueError, match="board"):
        configured_board({"board": "charuco"})


def test_the_charuco_order_is_not_reversed_for_a_checkerboard_only_rule():
    observations = RIG.observations[:3]
    reconciled, flipped = reconcile_corner_order(observations, CHARUCO_BOARD, SIZE, CAMERAS)
    assert flipped == {"left": False, "right": False}
    for a, b in zip(observations, reconciled):
        for cam in CAMERAS:
            np.testing.assert_array_equal(a.corners[cam], b.corners[cam])
            assert a.corners[cam] is not b.corners[cam]


def test_a_charuco_board_is_refined_with_subpix_only():
    with pytest.raises(ValueError, match="subpix"):
        detect_board(np.zeros((100, 100), np.uint8), CHARUCO_BOARD, refine="legacy")


def test_board_spec_validates():
    with pytest.raises(ValueError, match="kind"):
        BoardSpec(kind="dots")
    with pytest.raises(ValueError, match="marker_mm"):
        BoardSpec(marker_mm=3.0)
    assert dataclasses.replace(CHARUCO_BOARD, marker_mm=2.5).marker_mm == 2.5
    assert CHARUCO_BOARD.kind == CHARUCO


# The end-to-end capture: 16 waypoints, rendered as multires_pipeline_test renders the
# checkerboard, with the right camera's principal point moved 700 px right so the board runs
# off that edge: at 3 waypoints it keeps 3 to 11 identifiable corners, at the others 16 to 45.
E2E_WAYPOINTS = 16
E2E_STILL, E2E_MOVING, E2E_FPS = 15, 6, 40
E2E_SHIFT_PX = 700.0


def _shifted(model: CameraModel, dcx: float) -> CameraModel:
    K = model.K.copy()
    K[0, 2] += dcx
    return CameraModel(K=K, dist=model.dist, rvec=model.rvec, tvec=model.tvec)


E2E_MODELS = {"left": RIG.truth.cameras["left"], "right": _shifted(RIG.truth.cameras["right"], E2E_SHIFT_PX)}
E2E_MOTOR = [o.motor_mm.copy() for o in RIG.observations[:E2E_WAYPOINTS]]


def _identifiable(model: CameraModel, motor_mm) -> np.ndarray:
    """The corners a camera can identify: those a full square pitch inside the image, so
    both markers beside them are in view."""
    truth = truth_px(model, motor_mm)
    pitch = float(np.median(np.linalg.norm(np.diff(truth.reshape(6, 8, 2), axis=1), axis=-1)))
    return ((truth[:, 0] >= pitch) & (truth[:, 0] <= SIZE[0] - 1 - pitch)
            & (truth[:, 1] >= pitch) & (truth[:, 1] <= SIZE[1] - 1 - pitch))


def _e2e_frames(model):
    for k, position in enumerate(E2E_MOTOR):
        if k > 0:
            previous = E2E_MOTOR[k - 1]
            for j in range(1, E2E_MOVING + 1):
                yield render(model, previous + (position - previous) * j / (E2E_MOVING + 1))
        still = render(model, position)
        for _ in range(E2E_STILL):
            yield still


@pytest.fixture(scope="module")
def charuco_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("charuco")
    capture = folder / "capture"
    (capture / SOURCE_VIDEOS).mkdir(parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam, model in E2E_MODELS.items():
        writer = cv2.VideoWriter(str(capture / SOURCE_VIDEOS / f"{cam}.mp4"), fourcc, E2E_FPS, SIZE, isColor=False)
        assert writer.isOpened()
        for image in _e2e_frames(model):
            writer.write(image)
        writer.release()
    motor = [tuple(float(v) for v in p) for p in E2E_MOTOR]
    n = len(motor)
    log = WaypointLog(
        motor=motor,
        reported=[tuple(v + 0.01 for v in p) for p in motor],
        verified=[True] * n,
        dcs_moves=[("xyz"[i % 3], float(i)) for i in range(n)],
        times=[(1000.0 + i, 1000.75 + i) for i in range(n)],
        capture_settings={},
    )
    write_waypoint_log(capture / WAYPOINT_LOG, log)
    output = folder / "out" / "calibration"
    index, report = calibrate_capture(capture, output_dir=output, presets=presets_from_base(BASE),
                                      board=CHARUCO_BOARD)
    return output, index, report


def test_the_pipeline_calibrates_a_charuco_capture_with_partial_views(charuco_run):
    output, index, report = charuco_run
    fit = store.load_sensor_fit(output)
    assert index.board == CHARUCO_BOARD
    assert store.load_index(output).board == CHARUCO_BOARD
    raw = yaml.safe_load((output / store.INDEX_FILE).read_text(encoding="utf-8"))
    assert raw["board"]["kind"] == "charuco" and raw["board"]["dictionary"] == "DICT_4X4_50"
    for cam, model in E2E_MODELS.items():
        assert fit.cameras[cam].K[0, 0] == pytest.approx(model.K[0, 0], rel=0.01), cam
        assert abs(fit.cameras[cam].K[0, 2] - model.K[0, 2]) < 10.0, cam
        assert fit.rms_px[cam] < 0.2, cam
    assert fit.motor_scale == pytest.approx(1.0, abs=3e-3)


def test_every_usable_view_feeds_the_fit(charuco_run):
    output, index, report = charuco_run
    fit = store.load_sensor_fit(output)
    for cam, model in E2E_MODELS.items():
        usable = [k for k, m in enumerate(E2E_MOTOR) if _identifiable(model, m).sum() >= MIN_CHARUCO_CORNERS]
        assert sorted(fit.per_view_rms_px[cam]) == usable, cam
        line = [note for note in report.notes if note.startswith(f"{cam}: views used ")]
        assert len(line) == 1, report.notes
        assert line[0].startswith(f"{cam}: views used {len(usable)} of {E2E_WAYPOINTS} waypoints, median "), line
    # The right camera's views are partial, and counted as such.
    assert 12 <= len(fit.per_view_rms_px["right"]) < E2E_WAYPOINTS
    (right,) = [m for m in report.metrics if m.name == "views_kept" and m.camera == "right"]
    assert right.value == len(fit.per_view_rms_px["right"])
