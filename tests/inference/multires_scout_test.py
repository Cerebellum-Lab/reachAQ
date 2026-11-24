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

from reachaq.inference.multires_calibration import scout
from reachaq.inference.multires_calibration.fit import BoardSpec, CameraModel
from reachaq.inference.multires_calibration.scout import (
    Bounds,
    ScoutView,
    SCOUT_HOME_MARGIN_MM,
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
def cameras():
    """(the rig, the scout grid, camera name -> model) for the good cameras and the three failing ones."""
    rig = synthetic_rig(cameras=GOOD)
    grid = scout_grid(BOX, counts=COUNTS)
    models = dict(rig.truth.cameras)
    centre = np.mean([rig.world_corners(p) for p in grid], axis=(0, 1))
    # The board's middle sits on the right edge, so it is cut off there at every point.
    models["aimed_off"] = _shifted(models["camera4"], SIZE[0] / 2)
    models["grazing"] = _off_axis(models["left"], centre, 65.0)
    models["blurred"] = models["left"]
    return rig, grid, models


def _render(rig, models, name, point):
    gray = render_board_frame(models[name], rig.world_corners(point), SIZE, rig.board)
    return cv2.GaussianBlur(gray, (0, 0), FOCUS_SIGMA) if name == "blurred" else gray


@pytest.fixture(scope="module")
def scene(cameras):
    rig, grid, models = cameras
    views = {i: {} for i in range(len(grid))}
    for i, point in enumerate(grid):
        for name in models:
            views[i][name] = scout_view(name, _render(rig, models, name, point), rig.board)
    return grid, views


def _subset(views, cameras):
    return {i: {c: v[c] for c in cameras} for i, v in views.items()}


def test_bounds_never_exceed_the_plan_box():
    targets = [(0.0, 0.1, 19.8), (17.7, 25.1, 32.8), (5.0, 12.0, 25.0)]
    box = bounds_from_plan(targets, None)
    # x and y start within SCOUT_HOME_MARGIN_MM of the home stop: raised to it.
    assert box == Bounds((1.0, 1.0, 19.8), (17.7, 25.1, 32.8))

    # A configured box narrows the plan's, and never widens it.
    narrower = bounds_from_plan(targets, Bounds((2.0, -5.0, 21.0), (40.0, 20.0, 50.0)))
    assert narrower == Bounds((2.0, 1.0, 21.0), (17.7, 20.0, 32.8))
    for got, plan_lo, plan_hi in zip(zip(narrower.lo, narrower.hi), box.lo, box.hi):
        assert plan_lo <= got[0] <= got[1] <= plan_hi

    # The grid stays inside it.
    for point in scout_grid(narrower):
        assert all(lo <= v <= hi for v, lo, hi in zip(point, narrower.lo, narrower.hi))

    with pytest.raises(ValueError, match="no overlap"):
        bounds_from_plan(targets, Bounds((20.0, 0.0, 0.0), (30.0, 30.0, 40.0)))
    with pytest.raises(ValueError):
        bounds_from_plan([], None)


def test_the_configured_travel_replaces_the_plan_box():
    targets = [(0.0, 0.1, 19.8), (17.7, 25.1, 32.8), (5.0, 12.0, 25.0)]
    travel = ((0.0, 35.0), (0.0, 35.0), (0.0, 35.0))
    box = bounds_from_plan(targets, None, travel)
    assert box == Bounds((1.0, 1.0, 1.0), (35.0, 35.0, 35.0))
    # Every point of the grid lies inside the travel, and none just off the home stop.
    for point in scout_grid(box):
        assert all(0.0 <= v <= 35.0 for v in point)
        assert not any(0.0 < v < SCOUT_HOME_MARGIN_MM for v in point)

    # scoutBounds still narrows it, and the travel caps a wider scoutBounds.
    narrowed = bounds_from_plan(targets, Bounds((2.0, -5.0, 21.0), (40.0, 20.0, 50.0)), travel)
    assert narrowed == Bounds((2.0, 1.0, 21.0), (35.0, 20.0, 35.0))
    for point in scout_grid(narrowed):
        assert all(0.0 <= v <= 35.0 for v in point)

    # An axis without travel keeps the plan's side; a one-sided travel replaces only its side.
    partial = bounds_from_plan(targets, None, ((None, None), (3.0, None), (None, 30.0)))
    assert partial == Bounds((1.0, 3.0, 19.8), (17.7, 25.1, 30.0))
    assert bounds_from_plan(targets, None, None) == bounds_from_plan(targets, None)

    with pytest.raises(ValueError, match="no overlap"):
        bounds_from_plan(targets, Bounds((40.0, 0.0, 0.0), (50.0, 30.0, 30.0)), travel)


def test_the_scout_box_keeps_clear_of_the_home_stop():
    # 2026-11-23, christielab10: the scout commanded y 0.133 mm from home and the stage
    # stayed at 0.000, so its strict arrival check aborted at the first grid point.
    assert SCOUT_HOME_MARGIN_MM == 1.0
    plan = [(0.0, 0.133, 0.5), (17.7, 25.1, 32.8)]
    for travel in (None, ((0.0, 35.0),) * 3, ((None, None), (0.0, None), (None, 35.0))):
        box = bounds_from_plan(plan, None, travel)
        assert all(lo >= SCOUT_HOME_MARGIN_MM for lo in box.lo), (travel, box)
        for point in scout_grid(box):
            assert not any(0.0 < v < SCOUT_HOME_MARGIN_MM for v in point), (travel, point)
    # A low edge already past the margin, and the upper edges, are left alone.
    assert bounds_from_plan(plan, None, ((2.5, 35.0),) * 3) == Bounds((2.5, 2.5, 2.5), (35.0, 35.0, 35.0))
    # An axis that sits on the home stop stays there: a move to the stop itself is made.
    on_stop = bounds_from_plan([(0.0, 5.0, 20.0), (0.0, 10.0, 30.0)], None)
    assert on_stop == Bounds((0.0, 5.0, 20.0), (0.0, 10.0, 30.0))
    assert {p[0] for p in scout_grid(on_stop)} == {0.0}
    # So does one whose whole box lies within the margin.
    assert bounds_from_plan([(0.0, 5.0, 20.0), (0.6, 10.0, 30.0)], None).lo[0] == 0.0
    assert bounds_from_plan([(0.0, 5.0, 20.0), (0.6, 10.0, 30.0)], None).hi[0] == 0.0
    # Not onto the stop when a configured min_pos above 0 rules it out.
    near = bounds_from_plan([(0.0, 5.0, 20.0), (0.6, 10.0, 30.0)], None, ((0.4, 0.6), (None, None), (None, None)))
    assert (near.lo[0], near.hi[0]) == (0.4, 0.6)


def test_the_home_margin_never_takes_the_box_outside_scout_bounds():
    # scoutBounds y 0.5-0.8 with the travel's min_pos 0: putting y on the stop (0) would leave
    # the operator's bounds, so the box stays as configured.
    plan = [(0.0, 0.0, 20.0), (17.7, 25.1, 32.8)]
    configured = Bounds((0.0, 0.5, 0.0), (35.0, 0.8, 35.0))
    box = bounds_from_plan(plan, configured, ((0.0, 35.0),) * 3)
    assert (box.lo[1], box.hi[1]) == (0.5, 0.8)
    for k in range(3):
        assert configured.lo[k] <= box.lo[k] <= box.hi[k] <= configured.hi[k], (k, box)
    # scoutBounds that reach 0 still allow the stop.
    on_stop = bounds_from_plan(plan, Bounds((0.0, 0.0, 0.0), (35.0, 0.8, 35.0)), ((0.0, 35.0),) * 3)
    assert (on_stop.lo[1], on_stop.hi[1]) == (0.0, 0.0)


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


VIDEO_POINTS = (0, 1, 4, 9)  # grid indices: a few points, each one axis from the one before in the grid
STILL_FRAMES, MOVING_FRAMES, FPS = 6, 2, 40.0


def test_views_from_a_scout_capture_take_each_dwell_by_time(cameras, tmp_path):
    rig, grid, models = cameras
    names = ("left", "right", "aimed_off")
    points = [grid[i] for i in VIDEO_POINTS]
    t0, step = 5000.0, 1.0 / FPS
    videos, frame_times, dwells = {}, {}, []
    for n, name in enumerate(names):
        path = tmp_path / f"{name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE, isColor=False)
        assert writer.isOpened()
        times, k = [], 0
        for p, point in enumerate(points):
            if p > 0:  # the stage moving: half way, so these frames hold another board position
                between = (np.asarray(points[p - 1]) + np.asarray(point)) / 2
                for _ in range(MOVING_FRAMES):
                    writer.write(_render(rig, models, name, between))
                    times.append(t0 + k * step)
                    k += 1
            still = _render(rig, models, name, point)
            first = k
            for _ in range(STILL_FRAMES):
                writer.write(still)
                times.append(t0 + k * step)
                k += 1
            if n == 0:
                dwells.append((t0 + first * step - 0.001, t0 + (k - 1) * step + 0.001))
        writer.release()
        videos[name] = path
        frame_times[name] = times

    views = scout.scout_views_from_capture(videos, frame_times, points, dwells, rig.board)

    assert set(views) == set(range(len(points)))
    for p, i in enumerate(VIDEO_POINTS):
        for name in ("left", "right"):
            assert views[p][name].found, (p, name)
            truth = cv2.projectPoints(rig.world_corners(grid[i]), models[name].rvec, models[name].tvec,
                                      models[name].K, np.zeros(5))[0].reshape(-1, 2)
            # The middle still frame, not a moving one: within a pixel, in either corner order.
            err = min(np.abs(views[p][name].corners - truth).max(), np.abs(views[p][name].corners[::-1] - truth).max())
            assert err < 1.0, (p, name, err)
        assert not views[p]["aimed_off"].found and views[p]["aimed_off"].touches_edge == "right"
    assert scout.video_sizes(videos) == {name: SIZE for name in names}

    # A dwell with no frame in it is a view that found nothing.
    late = dwells[:-1] + [(t0 + 100.0, t0 + 101.0)]
    views = scout.scout_views_from_capture({"left": videos["left"]}, {"left": frame_times["left"]}, points, late,
                                           rig.board)
    assert views[len(points) - 1]["left"] == ScoutView("left", False, None, None, None, False)


def test_the_ordered_path_is_capped_at_60_waypoints():
    # Every point of a 4 x 4 x 3 grid seen by two cameras; a selection taken out of grid order
    # needs inserted one-axis points between its picks.
    grid = scout_grid(BOX)
    rng = np.random.default_rng(1)
    views = {i: {c: ScoutView(c, True, rng.uniform(0, 1000, (48, 2)), None, 0.78, True) for c in ("a", "b")}
             for i in range(len(grid))}

    selection, path = scout.choose_move_list(grid, views, ["a", "b"], max_path=60)
    assert len(path) <= 60
    for a, b in zip(path, path[1:]):
        assert sum(u != v for u, v in zip(a, b)) == 1
    assert set(selection.points) <= set(path) and path == order_path(selection.points)

    selection, path = scout.choose_move_list(grid, views, ["a", "b"], max_path=20)
    assert len(path) <= 20 and len(selection.points) < 20
    uncapped = select_waypoints(grid, views, ["a", "b"])
    assert len(order_path(uncapped.points)) > 20  # so the cap had something to do
    for a, b in zip(path, path[1:]):
        assert sum(u != v for u, v in zip(a, b)) == 1


def test_configured_bounds_come_from_the_library_setting():
    assert scout.configured_bounds({}) is None
    assert scout.configured_bounds({"scoutBounds": {"lo": [0, 1, 2], "hi": [3, 4, 5]}}) == Bounds((0, 1, 2), (3, 4, 5))
    assert scout.configured_bounds({"scoutBounds": {"lo": [0, 1], "hi": [3, 4, 5]}}) is None
    assert scout.configured_bounds({"scoutBounds": "wide"}) is None
    assert scout.configured_bounds({"scoutBounds": {"lo": "012", "hi": "345"}}) is None


def _seen(camera, corners=None):
    return ScoutView(camera, True, np.zeros((48, 2)) if corners is None else corners, None, 0.78, True)


def _unseen(camera):
    return ScoutView(camera, False, None, None, None, False)


def test_a_camera_detected_at_too_few_points_is_told_so_not_never_in_view():
    views = {i: {"few": _seen("few") if i < 5 else _unseen("few")} for i in range(18)}
    assert diagnose(views, ["few"]) == {
        "few": "board seen at only 5 of 18 points, 12 needed: aim the camera at the middle of the board's travel"}


def test_disjoint_camera_groups_leave_the_selection_short():
    grid = scout_grid(BOX, counts=(4, 4, 2))  # 32 points
    views = {i: ({"a": _seen("a"), "b": _seen("b"), "c": _unseen("c"), "d": _unseen("d")} if i < 16 else
                 {"a": _unseen("a"), "b": _unseen("b"), "c": _seen("c"), "d": _seen("d")})
             for i in range(len(grid))}
    selection = select_waypoints(grid, views, ["a", "b", "c", "d"])
    assert not selection.connected and selection.groups == (("a", "b"), ("c", "d"))
    assert all(n >= 12 for n in selection.per_camera.values())
    assert scout.selection_shortfalls(selection, ["a", "b", "c", "d"]) == {
        "c": "not joined to a, b through shared views in the chosen move list",
        "d": "not joined to a, b through shared views in the chosen move list"}


def test_a_cap_that_shrinks_the_list_below_12_views_is_named():
    grid = scout_grid(BOX)
    rng = np.random.default_rng(2)
    views = {i: {c: _seen(c, rng.uniform(0, 1000, (48, 2))) for c in ("a", "b")} for i in range(len(grid))}
    selection, path = scout.choose_move_list(grid, views, ["a", "b"], max_path=10)
    assert len(path) <= 10
    n = selection.per_camera["a"]
    assert n < 12
    assert scout.selection_shortfalls(selection, ["a", "b"]) == {
        "a": f"only {n} of 12 views in the chosen move list", "b": f"only {n} of 12 views in the chosen move list"}
    # The uncapped list passes.
    full, _ = scout.choose_move_list(grid, views, ["a", "b"])
    assert scout.selection_shortfalls(full, ["a", "b"]) == {}


def test_advice_for_one_camera_is_merged():
    assert scout.merge_advice({"a": "x"}, {"a": "y", "b": "z"}, {"a": "x"}) == {"a": "x; y", "b": "z"}
