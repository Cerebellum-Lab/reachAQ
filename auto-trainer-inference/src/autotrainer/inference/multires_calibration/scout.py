"""The scout pass: choose a move list that every ticked camera sees well.

Before a capture the stage visits a coarse grid inside the stage's configured travel, or on
an axis without one the box the hand-written move list already travels (bounds_from_plan),
and each ticked camera looks for the board at every
grid point (scout_view). From what each camera saw, select_waypoints picks up to 36 points
that join the cameras through shared views and give each one 12 or more views, spread over
its image and over depth; order_path turns them into a path of one-axis moves. A camera
that sees too little gets plain-words advice (diagnose) before any capture.

The stage only translates the board, in x, y and z; it never tilts it. Every view is a
parallel plane, so a camera looking at the board edge-on sees it foreshortened wherever
the stage puts it, and only re-aiming the camera helps.

This module moves nothing. tools.acquisition.model.handle_3d_calibration.run_scout records
the grid as a short capture, and scout_views_from_capture reads each point's frames back
from its videos by the logged dwell times.

The chosen list is kept as <config>/calibrations/scout_move_list.yaml with the date and the
cameras' serials, and reused while the same cameras are ticked (load_move_list).
"""

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.multires_calibration.extract import detect_board, distinct_waypoints, dwell_segments
from autotrainer.inference.multires_calibration.fit import BoardSpec
from autotrainer.inference.multires_calibration.library import library_root

logger = get_verbose_logger(__name__)

Point = Tuple[float, float, float]

SCOUT_MOVE_LIST_FILE = "scout_move_list.yaml"
DEFAULT_COUNTS = (4, 4, 3)
MAX_POINTS = 36
MIN_VIEWS = 12
MIN_COVISIBLE = 8
# The ordered path, with its inserted one-axis points, is at most this many waypoints.
MAX_PATH = 60
# The library setting that may narrow the scout's box: {"lo": [x, y, z], "hi": [x, y, z]} in motor mm.
SCOUT_BOUNDS_SETTING = "scoutBounds"
OCCUPANCY_CELLS = 4  # each camera's image is split 4 x 4 to measure how widely its views spread

# The whole board, one square past the inner corners each way, is 9 x 7 squares: seen
# face-on its short side over its long side is 7/9.
BOARD_ASPECT = 7.0 / 9.0
# A camera is at a grazing angle when its views' median square is foreshortened below this,
# shorter one way than the other by more than about a 45-50 degree view does (the synthetic
# 50 degree camera measures 0.72, the 20 degree ones 0.95).
GRAZING_FORESHORTENING = 0.7

# Board pixels without a detection: the frame is shrunk 4 times, and a pixel is board-like
# where the grey levels around it (7 x 7, about 28 sensor pixels) vary by more than
# _TEXTURE_STD. The largest such patch is the board when it covers _MIN_BOARD_AREA of the image.
_TEXTURE_SCALE = 0.25
_TEXTURE_WINDOW = 7
_TEXTURE_STD = 12.0
_MIN_BOARD_AREA = 0.005
_EDGES = ("left", "right", "top", "bottom")
# A detected board whose corners come this close to an image edge is at that edge.
_CORNER_EDGE_MARGIN_PX = 10.0

ADVICE_EDGE = ("board cut off at the {edge} edge at {n} of {m} points:"
               " aim the camera towards that side or move the board start")
ADVICE_FOCUS = "board in frame but never detected: check focus at the board distance, then lighting and exposure"
ADVICE_GRAZING = ("board seen at a grazing angle (aspect {aspect:.2f} vs 0.78):"
                  " aim the camera more face-on; the stage cannot tilt the board")
ADVICE_NEVER = "board never in view: the camera points elsewhere or the scout bounds miss its view"
ADVICE_FEW = ("board seen at only {k} of {m} points, {need} needed:"
              " aim the camera at the middle of the board's travel")
# The chosen move list itself falls short of the rule for a camera.
SHORT_VIEWS = "only {n} of {need} views in the chosen move list"
SHORT_COVISIBLE = "only {n} of {need} views shared with another camera in the chosen move list"
SHORT_JOINED = "not joined to {others} through shared views in the chosen move list"


class ScoutAborted(RuntimeError):
    """The scout stopped at once: a motor fault, a move the motor did not confirm, or a command or camera error."""


@dataclass(frozen=True)
class Bounds:
    """An axis-aligned box in motor mm: lo and hi are (x, y, z)."""

    lo: Point
    hi: Point


@dataclass(frozen=True)
class ScoutView:
    """What one camera saw at one grid point.

    `found` with its (rows*cols, 2) `corners`, and `aspect`, the board's short side over its
    long side as seen (BOARD_ASPECT face-on); `board_pixels` when the frame holds a
    board-like patch, found or not, and `touches_edge`, the image edge that patch reaches,
    or None.
    """

    camera: str
    found: bool
    corners: Optional[np.ndarray]
    touches_edge: Optional[str]
    aspect: Optional[float]
    board_pixels: bool

    def __eq__(self, other):
        if not isinstance(other, ScoutView):
            return NotImplemented
        same_corners = (self.corners is None and other.corners is None) or (
            self.corners is not None and other.corners is not None and np.array_equal(self.corners, other.corners))
        return same_corners and (self.camera, self.found, self.touches_edge, self.aspect, self.board_pixels) == (
            other.camera, other.found, other.touches_edge, other.aspect, other.board_pixels)


@dataclass(frozen=True)
class Selection:
    """The chosen points in the order they were chosen; each camera's views among them;
    whether shared views join every ticked camera; each camera's views shared with another
    camera (`covisible`, as per_camera when not given); and the groups of cameras shared
    views join (`groups`; when not given, one group of all if connected, else one each)."""

    points: List[Point]
    per_camera: Dict[str, int]
    connected: bool
    covisible: Optional[Dict[str, int]] = None
    groups: Optional[Tuple[Tuple[str, ...], ...]] = None


@dataclass
class ScoutResult:
    """`points`, the move list (order_path of the selection); the `selection`; the advice per
    failing camera (`diagnosis`); and every view, grid index -> camera -> ScoutView."""

    points: List[Point]
    selection: Selection
    diagnosis: Dict[str, str]
    views: Dict[int, Dict[str, ScoutView]] = field(default_factory=dict)


# How far the scout box's low edge keeps from the home stop at 0, in motor mm. On
# 2026-11-23 the christielab10 scout commanded y 0.133 mm from home and the stage reported
# 0.000: a sub-millimetre move just off the stop is not reliably made, and the strict arrival
# check aborted the scout at its first grid point. A move to the stop itself is made, so only
# targets in (0, margin) are avoided. The rule is the simplest one: a low edge below the
# margin is raised to it, whether it came from min_pos, the plan or scoutBounds. A min_pos
# above the margin is not a stop and is kept as it is. The upper edges keep no margin: the
# arrival failure is at the home stop, and the far end has no switch for the stage to sit on.
SCOUT_HOME_MARGIN_MM = 1.0

#: Per axis x, y, z, the stage's configured travel (min_pos, max_pos) in mm; None on a side
#: that is not configured (motor_config.yaml, HardwareModel refuses moves outside it).
Travel = Sequence[Tuple[Optional[float], Optional[float]]]


def bounds_from_plan(
    plan_targets: Sequence[Sequence[float]], configured: Optional[Bounds], travel: Optional[Travel] = None
) -> Bounds:
    """The scout box: per axis the configured `travel`, a side without one the box the plan's
    commanded targets span; then intersected with `configured` (scoutBounds) when given.
    Last, a low edge below SCOUT_HOME_MARGIN_MM is raised to it; an axis whose whole box lies
    below the margin is put on the home stop (0, 0) instead, a move the stage does make,
    unless a configured min_pos above 0, or scoutBounds above 0, rules the stop out (it is
    then left as it is).

    Never wider than the travel or `configured`. Without travel it is the plan's box, never
    wider. Raises ValueError when there are no targets or the boxes do not overlap on some axis.
    """
    targets = np.asarray(plan_targets, float).reshape(-1, 3)
    if len(targets) == 0:
        raise ValueError("the scout bounds need the hand-written plan's targets; there are none")
    lo, hi = targets.min(axis=0), targets.max(axis=0)
    for k, (minimum, maximum) in enumerate(travel or ()):
        if minimum is not None:
            lo[k] = minimum
        if maximum is not None:
            hi[k] = maximum
    box = (tuple(float(v) for v in lo), tuple(float(v) for v in hi))
    if configured is not None:
        lo = np.maximum(lo, np.asarray(configured.lo, float))
        hi = np.minimum(hi, np.asarray(configured.hi, float))
    if np.any(lo > hi):
        raise ValueError(f"the configured scout bounds {configured} have no overlap with the box"
                         f" {box[0]} - {box[1]} (the configured travel, else the hand-written list's)")
    minima = [minimum for minimum, _ in travel] if travel else [None] * 3
    for k in range(3):
        if lo[k] < SCOUT_HOME_MARGIN_MM:
            if hi[k] >= SCOUT_HOME_MARGIN_MM:
                lo[k] = SCOUT_HOME_MARGIN_MM
            elif ((minima[k] is None or minima[k] <= 0.0)  # never onto a stop below a configured min
                  and (configured is None or configured.lo[k] <= 0.0)):  # nor outside scoutBounds
                lo[k] = hi[k] = 0.0
    return Bounds(tuple(float(v) for v in lo), tuple(float(v) for v in hi))


def scout_grid(bounds: Bounds, counts: Sequence[int] = DEFAULT_COUNTS) -> List[Point]:
    """An evenly spaced grid of counts[k] values per axis, both ends included, in serpentine order.

    x changes fastest and z slowest, each row and layer walked back the way it came, so
    consecutive points differ in exactly one axis. An axis whose box is a single value, or
    whose count is 1 (its middle), has one value.
    """
    values = []
    for lo, hi, n in zip(bounds.lo, bounds.hi, counts):
        n = int(n)
        if n < 1:
            raise ValueError(f"a grid needs at least one value per axis, not {n}")
        axis = [(lo + hi) / 2.0] if n == 1 else np.linspace(lo, hi, n).tolist()
        values.append(list(dict.fromkeys(float(v) for v in axis)))
    xs, ys, zs = values
    grid = []
    for k, z in enumerate(zs):
        rows = ys if k % 2 == 0 else ys[::-1]
        for j, y in enumerate(rows):
            # The x direction alternates along the whole walk, so the turn at each row's end is one axis.
            row_number = k * len(ys) + j
            for x in (xs if row_number % 2 == 0 else xs[::-1]):
                grid.append((x, y, z))
    return grid


def board_aspect(corners: np.ndarray, board: BoardSpec = BoardSpec()) -> float:
    """The whole board's short side over its long side as seen, from the inner-corner quad.

    The mean square's height over its width, times BOARD_ASPECT: BOARD_ASPECT face-on,
    above it when the view shortens the board's long side, below it when the short side.
    """
    c = np.asarray(corners, float).reshape(board.rows, board.cols, 2)
    width = (np.linalg.norm(c[0, -1] - c[0, 0]) + np.linalg.norm(c[-1, -1] - c[-1, 0])) / 2 / (board.cols - 1)
    height = (np.linalg.norm(c[-1, 0] - c[0, 0]) + np.linalg.norm(c[-1, -1] - c[0, -1])) / 2 / (board.rows - 1)
    return float(height / width * (board.rows + 1) / (board.cols + 1))


def foreshortening(aspect: float) -> float:
    """How much shorter the board looks one way than the other: 1 face-on, towards 0 edge-on."""
    r = aspect / BOARD_ASPECT
    return float(min(r, 1.0 / r))


def scout_view(camera: str, gray: np.ndarray, board: BoardSpec = BoardSpec()) -> ScoutView:
    """Look for the board in one full-sensor frame: detect_board, then the board-like patch."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    corners = detect_board(gray, board)
    board_pixels, edge = _board_patch(gray)
    if corners is None:
        return ScoutView(camera, False, None, edge if board_pixels else None, None, board_pixels)
    return ScoutView(camera, True, corners, edge, board_aspect(corners, board), True)


def _board_patch(gray: np.ndarray) -> Tuple[bool, Optional[str]]:
    """(whether the frame holds a board-like patch, the image edge it reaches or None).

    A heuristic on local contrast, enough to tell "nothing there" from "something with
    squares that the detector could not read". Other high-contrast texture in view reads as
    board too.
    """
    small = cv2.resize(gray, None, fx=_TEXTURE_SCALE, fy=_TEXTURE_SCALE, interpolation=cv2.INTER_AREA)
    f = small.astype(np.float32)
    w = (_TEXTURE_WINDOW, _TEXTURE_WINDOW)
    mean = cv2.blur(f, w)
    std = np.sqrt(np.maximum(cv2.blur(f * f, w) - mean * mean, 0.0))
    textured = (std > _TEXTURE_STD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(textured, connectivity=8)
    if n <= 1:
        return False, None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < _MIN_BOARD_AREA * textured.size:
        return False, None
    patch = labels == biggest
    on_edge = {"left": patch[:, 0].sum(), "right": patch[:, -1].sum(), "top": patch[0, :].sum(),
               "bottom": patch[-1, :].sum()}
    edge = max(_EDGES, key=lambda e: on_edge[e])
    return True, edge if on_edge[edge] > 0 else None


def select_waypoints(
    grid: Sequence[Point],
    views: Mapping[int, Mapping[str, ScoutView]],
    cameras: Sequence[str],
    *,
    max_points: int = MAX_POINTS,
    min_views: int = MIN_VIEWS,
    min_covisible: int = MIN_COVISIBLE,
    image_sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> Selection:
    """Greedily choose up to `max_points` grid points, from those 2 or more cameras found the board at.

    Each pick is the point that, in turn, joins the most separate groups of cameras; then
    gives a view to the most cameras still short of `min_views` (and of `min_covisible`
    views shared with another camera); then adds the most new values on an axis a camera's
    views do not yet span; then reaches the most new cells of each camera's 4 x 4 image
    grid; then lies furthest from the points already chosen. Ties go to the lower grid
    index, so the result is deterministic.

    The cells are of `image_sizes` (camera -> (width, height)) or, without them, of the box
    the camera's corners span over the whole scout.
    """
    cameras = list(cameras)
    found = {i: [c for c in cameras if c in views.get(i, {}) and views[i][c].found] for i in range(len(grid))}
    candidates = [i for i in range(len(grid)) if len(found[i]) >= 2]
    cells = {i: {c: _cells(views[i][c].corners, c, views, image_sizes) for c in found[i]} for i in candidates}
    span = np.ptp(np.asarray(grid, float), axis=0) if len(grid) else np.ones(3)
    span = np.where(span > 0, span, 1.0)

    chosen: List[int] = []
    count = {c: 0 for c in cameras}
    covisible = {c: 0 for c in cameras}
    axis_values = {c: [set(), set(), set()] for c in cameras}
    covered = {c: set() for c in cameras}
    groups = _Groups(cameras)
    remaining = list(candidates)
    while remaining and len(chosen) < max_points:
        def key(i):
            seen = found[i]
            merge = groups.count() - groups.count_after(seen)
            need = sum((count[c] < min_views) + (covisible[c] < min_covisible) for c in seen)
            axes = sum(len(axis_values[c][k]) < 2 and grid[i][k] not in axis_values[c][k]
                       for c in seen for k in range(3))
            new_cells = sum(len(cells[i][c] - covered[c]) for c in seen)
            spread = min((float(np.linalg.norm((np.subtract(grid[i], grid[j])) / span)) for j in chosen), default=0.0)
            return merge, need, axes, new_cells, round(spread, 9), -i

        best = max(remaining, key=key)
        remaining.remove(best)
        chosen.append(best)
        seen = found[best]
        groups.join(seen)
        for c in seen:
            count[c] += 1
            covisible[c] += 1  # every candidate is seen by two or more cameras
            for k in range(3):
                axis_values[c][k].add(grid[best][k])
            covered[c] |= cells[best][c]
    connected = groups.count() == 1 and all(count[c] > 0 for c in cameras)
    return Selection(points=[tuple(grid[i]) for i in chosen], per_camera=count, connected=connected,
                     covisible=dict(covisible), groups=groups.groups())


def selection_shortfalls(
    selection: Selection,
    cameras: Sequence[str],
    *,
    min_views: int = MIN_VIEWS,
    min_covisible: int = MIN_COVISIBLE,
) -> Dict[str, str]:
    """Plain-words advice for each camera the chosen move list itself leaves short of the rule.

    Fewer than `min_views` views, fewer than `min_covisible` of them shared with another
    camera, or not joined through shared views to the largest group of cameras (of equal
    groups, the one with the earliest camera in `cameras`). A camera's reasons are joined
    with "; ".
    """
    cameras = list(cameras)
    covisible = selection.per_camera if selection.covisible is None else selection.covisible
    groups = selection.groups
    if groups is None:
        groups = (tuple(cameras),) if selection.connected else tuple((c,) for c in cameras)
    rank = {c: k for k, c in enumerate(cameras)}
    main = max(groups, key=lambda g: (len(g), -min(rank.get(c, len(rank)) for c in g))) if groups else ()
    reasons: Dict[str, List[str]] = {}
    for camera in cameras:
        n = selection.per_camera.get(camera, 0)
        if n < min_views:
            reasons.setdefault(camera, []).append(SHORT_VIEWS.format(n=n, need=min_views))
        k = covisible.get(camera, 0)
        if k < min_covisible and k < n:
            reasons.setdefault(camera, []).append(SHORT_COVISIBLE.format(n=k, need=min_covisible))
        if camera not in main:
            others = ", ".join(c for c in cameras if c in main) or "the other cameras"
            reasons.setdefault(camera, []).append(SHORT_JOINED.format(others=others))
    return {camera: "; ".join(text) for camera, text in reasons.items()}


def merge_advice(*advice: Mapping[str, str]) -> Dict[str, str]:
    """Per camera, every distinct advice given for it, in order, joined with "; "."""
    merged: Dict[str, List[str]] = {}
    for one in advice:
        for camera, text in one.items():
            if text not in merged.setdefault(camera, []):
                merged[camera].append(text)
    return {camera: "; ".join(texts) for camera, texts in merged.items()}


def _cells(corners, camera, views, image_sizes) -> set:
    if image_sizes is not None and camera in image_sizes:
        width, height = image_sizes[camera]
        lo, size = np.zeros(2), np.array([width, height], float)
    else:
        every = np.concatenate([v[camera].corners for v in views.values()
                                if camera in v and v[camera].found])
        lo = every.min(axis=0)
        size = np.maximum(every.max(axis=0) - lo, 1e-9)
    cell = np.clip(((np.asarray(corners) - lo) / size * OCCUPANCY_CELLS).astype(int), 0, OCCUPANCY_CELLS - 1)
    return {(int(u), int(v)) for u, v in cell}


class _Groups:
    """Which cameras shared views join, as a union-find."""

    def __init__(self, cameras):
        self._parent = {c: c for c in cameras}

    def _root(self, c, parent=None):
        parent = self._parent if parent is None else parent
        while parent[c] != c:
            c = parent[c]
        return c

    def count(self, parent=None):
        parent = self._parent if parent is None else parent
        return len({self._root(c, parent) for c in parent})

    def count_after(self, cameras):
        parent = dict(self._parent)
        self._join(parent, cameras)
        return self.count(parent)

    def join(self, cameras):
        self._join(self._parent, cameras)

    def groups(self) -> Tuple[Tuple[str, ...], ...]:
        """The groups, each in camera order, in the order of their first camera."""
        by_root: Dict[str, List[str]] = {}
        for c in self._parent:
            by_root.setdefault(self._root(c), []).append(c)
        return tuple(tuple(g) for g in by_root.values())

    def _join(self, parent, cameras):
        cameras = list(cameras)
        for c in cameras[1:]:
            a, b = self._root(cameras[0], parent), self._root(c, parent)
            if a != b:
                parent[b] = a


def order_path(points: Sequence[Point]) -> List[Point]:
    """A short path through `points`, from the first: nearest neighbour by the distance the
    stage travels (the sum over axes), with a point inserted wherever two in a row differ in
    more than one axis, so each move is one axis, x first, then y, then z.

    The inserted points take each coordinate from one of the two points either side, so
    they stay in any box that holds both.
    """
    if not points:
        return []
    rest = [tuple(float(v) for v in p) for p in points]
    order = [rest.pop(0)]
    while rest:
        here = order[-1]
        k = min(range(len(rest)), key=lambda j: (sum(abs(a - b) for a, b in zip(here, rest[j])), j))
        order.append(rest.pop(k))
    path = [order[0]]
    for target in order[1:]:
        position = list(path[-1])
        for axis in range(3):
            if position[axis] != target[axis]:
                position[axis] = target[axis]
                path.append(tuple(position))
    return path


def choose_move_list(
    grid: Sequence[Point],
    views: Mapping[int, Mapping[str, ScoutView]],
    cameras: Sequence[str],
    *,
    image_sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
    max_points: int = MAX_POINTS,
    max_path: int = MAX_PATH,
) -> Tuple[Selection, List[Point]]:
    """(the selection, its ordered path) with the path at most `max_path` waypoints.

    The one-axis points order_path inserts stay dwell waypoints, as more views. When the
    path is longer than `max_path`, the selection is made again with one point fewer, until
    it fits.
    """
    limit = max_points
    while True:
        selection = select_waypoints(grid, views, cameras, max_points=limit, image_sizes=image_sizes)
        path = order_path(selection.points)
        if len(path) <= max_path or limit <= 1:
            if limit < max_points:
                logger.info("scout: %d points make a path of %d waypoints, within %d",
                            len(selection.points), len(path), max_path)
            return selection, path
        limit = min(limit, len(selection.points)) - 1


def scout_views_from_capture(
    videos: Mapping[str, Path],
    frame_times: Mapping[str, Sequence[float]],
    plan_targets: Sequence[Sequence[float]],
    dwells: Sequence[Tuple[float, float]],
    board: BoardSpec = BoardSpec(),
) -> Dict[int, Dict[str, ScoutView]]:
    """Plan index -> camera -> ScoutView, from a scout capture's videos.

    Each distinct waypoint's frames are the camera's frames whose time (`frame_times`,
    column 0 of its timestamps file) lies in the waypoint's logged dwell (dwell_segments);
    the board is looked for in the middle one (scout_view). A waypoint with no frame in its
    dwell, or whose frame the video does not have, is a view that found nothing.
    """
    targets = [np.asarray(t, float) for t in plan_targets]
    distinct = distinct_waypoints(targets)
    views: Dict[int, Dict[str, ScoutView]] = {i: {} for i in distinct}
    for camera, video in videos.items():
        segments = dwell_segments(frame_times[camera], targets, dwells)
        wanted = {}
        for i, (begin, stop) in zip(distinct, segments):
            if stop > begin:
                wanted[(begin + stop - 1) // 2] = i
        frames = _frames_at(Path(video), set(wanted))
        for i, (begin, stop) in zip(distinct, segments):
            k = (begin + stop - 1) // 2
            if stop > begin and k in frames:
                views[i][camera] = scout_view(camera, frames[k], board)
            else:
                logger.warning("scout: %s has no frame in waypoint %d's dwell", camera, i)
                views[i][camera] = ScoutView(camera, False, None, None, None, False)
    return views


def _frames_at(video: Path, indices: set) -> Dict[int, np.ndarray]:
    """The grey frames of `video` at `indices`, read through once."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"could not open the video {video}")
    frames, k, last = {}, 0, max(indices, default=-1)
    try:
        while k <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if k in indices:
                frames[k] = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            k += 1
    finally:
        capture.release()
    return frames


def video_sizes(videos: Mapping[str, Path]) -> Dict[str, Tuple[int, int]]:
    """camera -> (width, height) of its video."""
    sizes = {}
    for camera, video in videos.items():
        capture = cv2.VideoCapture(str(video))
        try:
            sizes[camera] = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            capture.release()
    return sizes


def configured_bounds(settings: Mapping) -> Optional[Bounds]:
    """The library settings' scoutBounds as Bounds; None when unset or malformed (logged)."""
    value = settings.get(SCOUT_BOUNDS_SETTING) if settings else None
    if value is None:
        return None
    try:
        if not isinstance(value, Mapping) or not all(isinstance(value.get(k), (list, tuple)) for k in ("lo", "hi")):
            raise TypeError("it must be {lo: [x, y, z], hi: [x, y, z]}")
        lo = tuple(float(v) for v in value["lo"])
        hi = tuple(float(v) for v in value["hi"])
        if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in lo + hi):
            raise ValueError("three finite values each")
    except (KeyError, TypeError, ValueError) as err:
        logger.warning("ignoring the %s setting %r: %s", SCOUT_BOUNDS_SETTING, value, err)
        return None
    return Bounds(lo, hi)


def diagnose(
    views: Mapping[int, Mapping[str, ScoutView]],
    cameras: Sequence[str],
    image_sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
    *,
    min_views: int = MIN_VIEWS,
) -> Dict[str, str]:
    """Plain-words advice for each camera that fails the rule, by camera name.

    A camera fails when it found the board at fewer than `min_views` points, or when its
    views' median foreshortening is below GRAZING_FORESHORTENING. Its advice is the first
    that fits:

    - no board pixels and no detection at any point: never in view;
    - found, but at a grazing angle: aim more face-on;
    - at half or more of the points with the board in view (board pixels, detected or
      not), it reaches one image edge (for a detection, its corners come within 10 px of
      it, given `image_sizes`): cut off at that edge, with how many of the scouted points;
    - board pixels without a detection: focus, then lighting and exposure;
    - otherwise, detected at too few points: seen at only k of m points.
    """
    advice = {}
    m = len(views)
    for camera in cameras:
        seen = [v[camera] for v in views.values() if camera in v]
        detected = [v for v in seen if v.found]
        aspects = [v.aspect for v in detected if v.aspect is not None]
        median_aspect = float(np.median(aspects)) if aspects else None
        grazing = median_aspect is not None and foreshortening(median_aspect) < GRAZING_FORESHORTENING
        if len(detected) >= min_views and not grazing:
            continue
        size = None if image_sizes is None else image_sizes.get(camera)
        # Like with like: the edge count and what it is compared with are both over the
        # points where the board was in view, detected or not.
        in_view = [v for v in seen if v.board_pixels or v.found]
        undetected = [v for v in in_view if not v.found]
        edges = [_view_edge(v, size) for v in in_view]
        tally = {e: edges.count(e) for e in _EDGES}
        edge = max(_EDGES, key=lambda e: tally[e])
        if not in_view:
            advice[camera] = ADVICE_NEVER
        elif grazing:
            advice[camera] = ADVICE_GRAZING.format(aspect=median_aspect)
        elif tally[edge] > 0 and tally[edge] * 2 >= len(in_view):
            advice[camera] = ADVICE_EDGE.format(edge=edge, n=tally[edge], m=m)
        elif undetected:
            advice[camera] = ADVICE_FOCUS
        else:
            advice[camera] = ADVICE_FEW.format(k=len(detected), m=m, need=min_views)
    return advice


def _view_edge(view: ScoutView, size: Optional[Tuple[int, int]]) -> Optional[str]:
    if not view.found:
        return view.touches_edge
    if size is None:
        return None
    width, height = size
    u, v = view.corners[:, 0], view.corners[:, 1]
    gaps = {"left": u.min(), "right": width - 1 - u.max(), "top": v.min(), "bottom": height - 1 - v.max()}
    edge = min(_EDGES, key=lambda e: gaps[e])
    return edge if gaps[edge] <= _CORNER_EDGE_MARGIN_PX else None


def scout_move_list_path(config_dir: Path) -> Path:
    """<config>/calibrations/scout_move_list.yaml."""
    return library_root(config_dir) / SCOUT_MOVE_LIST_FILE


def save_move_list(path: Path, points: Sequence[Point], cameras_serials: Sequence[str], created: datetime) -> None:
    """Write the move list with its date and the cameras it was scouted for, replacing the file whole."""
    data = {
        "created": created.isoformat(),
        "cameras": sorted(str(s) for s in cameras_serials),
        "points": [[float(v) for v in p] for p in points],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def load_move_list(path: Path, cameras_serials: Sequence[str]) -> Optional[List[Point]]:
    """The saved move list when it was scouted for exactly these cameras (by serial, in any
    order); None when it was not, or the file is missing or unreadable."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        saved = {str(s) for s in data["cameras"]}
        points = [tuple(float(v) for v in p) for p in data["points"]]
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as err:
        logger.info("no saved scout move list at %s: %s", path, err)
        return None
    if saved != {str(s) for s in cameras_serials}:
        logger.info("the saved scout move list is for cameras %s, not %s", sorted(saved), sorted(cameras_serials))
        return None
    if not points or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in points):
        return None
    return points
