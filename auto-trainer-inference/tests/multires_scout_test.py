"""The scout pass: a coarse grid inside the hand-written list's box, then a move list every camera sees.

The cameras are the synthetic rig's, each frame rendered at the full 1440x1080 sensor with
render_board_frame and searched with extract.detect_board, as the scout does on the rig.
Three more cameras fail in the ways the 2026-11-19 capture did, or could: one aimed so
the board runs off its right edge, one out of focus, and one that sees the board at a
grazing angle. Each gets its own plain-words advice.

Nothing here touches hardware.
"""

from datetime import datetime

import cv2
import numpy as np
import pytest

from autotrainer.inference.multires_calibration import scout
from autotrainer.inference.multires_calibration.fit import BoardSpec, CameraModel
from autotrainer.inference.multires_calibration.scout import (
    Bounds,
    ScoutView,
    bounds_from_plan,
    diagnose,
    load_move_list,
    order_path,
    save_move_list,
    scout_grid,
    scout_view,
    select_waypoints,
)

from multires_synthetic import _look_at, render_board_frame, synthetic_rig

SIZE = (1440, 1080)
# The synthetic rig's move-list box; every one of its cameras sees the whole board anywhere in it.
BOX = Bounds((-10.0, 10.0, -35.0), (10.0, 35.0, -22.0))
COUNTS = (3, 3, 2)
GOOD = ("left", "right", "camera4")
# Blur that defeats detection on these renders. Sigma 8 does not: their 20/235 squares are
# far crisper than the rig's, and detect_board still finds the board at 15 of 18 points.
FOCUS_SIGMA = 16


def _shifted(model: CameraModel, dcx: float) -> CameraModel:
    K = model.K.copy()
    K[0, 2] += dcx
    return CameraModel(K=K, dist=model.dist, rvec=model.rvec, tvec=model.tvec)


def _off_axis(model: CameraModel, centre, angle_deg: float) -> CameraModel:
    # A camera 150 mm from the board's middle, `angle_deg` from its normal (motor z).
    a = np.radians(angle_deg)
    eye = np.asarray(centre) + 150.0 * np.array([np.sin(a), 0.0, -np.cos(a)])
    rvec, tvec = _look_at(eye, np.asarray(centre), down=np.array([0.0, 1.0, 0.0]))
    return CameraModel(K=model.K, dist=model.dist, rvec=rvec, tvec=tvec)


@pytest.fixture(scope="module")
def scene():
    rig = synthetic_rig(cameras=GOOD)
    board = rig.board
    grid = scout_grid(BOX, counts=COUNTS)
    models = dict(rig.truth.cameras)
    centre = np.mean([rig.world_corners(p) for p in grid], axis=(0, 1))
    # The board's middle sits on the right edge, so it is cut off there at every point.
    models["aimed_off"] = _shifted(models["camera4"], SIZE[0] / 2)
    models["grazing"] = _off_axis(models["left"], centre, 65.0)
    models["blurred"] = models["left"]
    views = {i: {} for i in range(len(grid))}
    for i, point in enumerate(grid):
        X = rig.world_corners(point)
        for name, model in models.items():
            gray = render_board_frame(model, X, SIZE, board)
            if name == "blurred":
                gray = cv2.GaussianBlur(gray, (0, 0), FOCUS_SIGMA)
            views[i][name] = scout_view(name, gray, board)
    return grid, views


def _subset(views, cameras):
    return {i: {c: v[c] for c in cameras} for i, v in views.items()}


def test_bounds_never_exceed_the_plan_box():
    targets = [(0.0, 0.1, 19.8), (17.7, 25.1, 32.8), (5.0, 12.0, 25.0)]
    box = bounds_from_plan(targets, None)
    assert box == Bounds((0.0, 0.1, 19.8), (17.7, 25.1, 32.8))

    # A configured box narrows the plan's, and never widens it.
    narrower = bounds_from_plan(targets, Bounds((2.0, -5.0, 21.0), (40.0, 20.0, 50.0)))
    assert narrower == Bounds((2.0, 0.1, 21.0), (17.7, 20.0, 32.8))
    for got, plan_lo, plan_hi in zip(zip(narrower.lo, narrower.hi), box.lo, box.hi):
        assert plan_lo <= got[0] <= got[1] <= plan_hi

    # The grid stays inside it.
    for point in scout_grid(narrower):
        assert all(lo <= v <= hi for v, lo, hi in zip(point, narrower.lo, narrower.hi))

    with pytest.raises(ValueError, match="no overlap"):
        bounds_from_plan(targets, Bounds((20.0, 0.0, 0.0), (30.0, 30.0, 40.0)))
    with pytest.raises(ValueError):
        bounds_from_plan([], None)


def test_grid_moves_one_axis_at_a_time():
    grid = scout_grid(BOX)
    assert len(grid) == 4 * 4 * 3 == len(set(grid))
    for a, b in zip(grid, grid[1:]):
        assert sum(u != v for u, v in zip(a, b)) == 1
    # Every axis reaches both ends of the box.
    for axis in range(3):
        values = {p[axis] for p in grid}
        assert min(values) == BOX.lo[axis] and max(values) == BOX.hi[axis]


def test_order_path_moves_one_axis_at_a_time_and_keeps_every_point():
    points = [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0), (0.0, 5.0, 0.0), (5.0, 0.0, 5.0)]
    path = order_path(points)
    assert path[0] == points[0]
    assert set(points) <= set(path)
    for a, b in zip(path, path[1:]):
        assert sum(u != v for u, v in zip(a, b)) == 1
    # The intermediate points lie in the box the points span.
    for p in path:
        assert all(0.0 <= v <= 5.0 for v in p)


def test_selection_connects_three_cameras_and_gives_each_twelve_views(scene):
    grid, views = scene
    views = _subset(views, GOOD)
    assert all(views[i][c].found for i in views for c in GOOD)
    # camera4 loses 5 of the 18 points; 14 picks must still give it 12 views.
    missing = {1, 4, 9, 12, 16}
    for i in missing:
        views[i]["camera4"] = ScoutView("camera4", False, None, None, None, False)

    selection = select_waypoints(grid, views, GOOD, max_points=14)

    assert len(selection.points) == 14 and len(set(selection.points)) == 14
    assert selection.connected
    assert all(selection.per_camera[c] >= 12 for c in GOOD), selection.per_camera
    chosen = [grid.index(p) for p in selection.points]
    assert len(missing & set(chosen)) <= 2  # 14 picks, camera4 in 12 or more
    # Each camera's views span all three axes.
    for c in GOOD:
        seen = [grid[i] for i in chosen if views[i][c].found]
        assert all(len({p[axis] for p in seen}) >= 2 for axis in range(3))
    # Deterministic.
    assert select_waypoints(grid, views, GOOD, max_points=14) == selection


def test_selection_keeps_only_points_two_cameras_see():
    grid = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    corners = np.zeros((48, 2))
    seen = lambda c: ScoutView(c, True, corners, None, 0.78, True)
    unseen = lambda c: ScoutView(c, False, None, None, None, False)
    views = {0: {"a": seen("a"), "b": unseen("b")}, 1: {"a": seen("a"), "b": seen("b")}, 2: {"a": unseen("a"), "b": seen("b")}}
    selection = select_waypoints(grid, views, ["a", "b"])
    assert selection.points == [grid[1]]
    assert selection.per_camera == {"a": 1, "b": 1} and selection.connected


def test_a_camera_without_shared_views_leaves_the_selection_unconnected(scene):
    grid, views = scene
    selection = select_waypoints(grid, _subset(views, ("left", "right", "aimed_off")), ("left", "right", "aimed_off"))
    assert not selection.connected
    assert selection.per_camera["aimed_off"] == 0
    assert selection.per_camera["left"] == selection.per_camera["right"] == len(grid)


def test_camera_seeing_board_cut_off_gets_edge_advice(scene):
    grid, views = scene
    cams = (*GOOD, "aimed_off")
    advice = diagnose(_subset(views, cams), cams, {c: SIZE for c in cams})
    assert set(advice) == {"aimed_off"}
    n = sum(views[i]["aimed_off"].touches_edge == "right" for i in views)
    assert n >= len(grid) // 2
    assert advice["aimed_off"] == (
        f"board cut off at the right edge at {n} of {len(grid)} points:"
        " aim the camera towards that side or move the board start")


def test_blurred_camera_gets_focus_advice(scene):
    grid, views = scene
    assert not any(views[i]["blurred"].found for i in views)
    assert all(views[i]["blurred"].board_pixels for i in views)
    cams = (*GOOD, "blurred")
    advice = diagnose(_subset(views, cams), cams, {c: SIZE for c in cams})
    assert advice == {
        "blurred": "board in frame but never detected: check focus at the board distance, then lighting and exposure"}


def test_grazing_camera_gets_angle_advice(scene):
    grid, views = scene
    # It finds the board at every point, but foreshortened.
    assert all(views[i]["grazing"].found for i in views)
    cams = (*GOOD, "grazing")
    advice = diagnose(_subset(views, cams), cams, {c: SIZE for c in cams})
    assert set(advice) == {"grazing"}
    aspect = float(np.median([views[i]["grazing"].aspect for i in views]))
    assert abs(np.log(aspect / scout.BOARD_ASPECT)) > abs(np.log(scout.GRAZING_FORESHORTENING))
    assert advice["grazing"] == (
        f"board seen at a grazing angle (aspect {aspect:.2f} vs 0.78):"
        " aim the camera more face-on; the stage cannot tilt the board")
    # The face-on cameras measure close to the board's own 7:9.
    for c in GOOD:
        assert views[0][c].aspect == pytest.approx(scout.BOARD_ASPECT, rel=0.15)


def test_camera_that_never_sees_the_board_gets_aim_advice():
    blank = np.full((SIZE[1], SIZE[0]), 100, np.uint8)
    view = scout_view("away", blank, BoardSpec())
    assert view == ScoutView("away", False, None, None, None, False)
    views = {i: {"away": view} for i in range(4)}
    assert diagnose(views, ["away"], {"away": SIZE}) == {
        "away": "board never in view: the camera points elsewhere or the scout bounds miss its view"}


def test_saved_list_reused_only_for_same_serials(tmp_path):
    path = tmp_path / "calibrations" / "scout_move_list.yaml"
    points = [(0.0, 0.1, 19.8), (17.7, 0.1, 19.8), (17.7, 25.1, 19.8)]
    save_move_list(path, points, ["20142630", "19230001", "19230002"], datetime(2026, 11, 20, 12, 0, 0))

    assert load_move_list(path, ["19230002", "20142630", "19230001"]) == points
    assert load_move_list(path, ["20142630", "19230001"]) is None
    assert load_move_list(path, ["20142630", "19230001", "19230002", "19239999"]) is None
    assert load_move_list(path, ["20142630", "19230001", "19239999"]) is None
    assert load_move_list(tmp_path / "missing.yaml", ["20142630"]) is None
    path.write_text("points: [oops", encoding="utf-8")
    assert load_move_list(path, ["20142630", "19230001", "19230002"]) is None


def test_scout_move_list_lives_in_the_library(tmp_path):
    assert scout.scout_move_list_path(tmp_path) == tmp_path / "calibrations" / "scout_move_list.yaml"
