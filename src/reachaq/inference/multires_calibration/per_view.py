"""The per-view fit: a free board rotation at every view, its position held to the motor.

The motor-frame fit (fit.py) models every view as the board translated by the commanded
motor position. The rig's stage does not do that exactly: on christielab10's session010
(2026-11-23, ChArUco) the board turned by up to 2.8 degrees between waypoints, the same in
every camera, and stood 3.3 mm short of the commanded z at z 35, so the motor-frame fit sat at
15-20 px RMS while each camera calibrated alone reached 0.8-1.3 px.

Here each camera has its intrinsics (fx, fy, cx, cy, k1, k2, p1, p2; k3 held at 0, as in
fit.py) and its pose, and each view (each waypoint seen by at least one camera) has its own
board pose, shared by every camera:

    X_cam = R_c @ (R_v @ P_j + t_v) + t_c

with (R_c, t_c) of the reference camera (the one with the most views) held at the identity,
which fixes the gauge; the board's squares give the scale. A view seen by one camera only
stays in: it constrains that camera's intrinsics.

The rotation R_v is free. The position t_v is held to a similarity of the commanded position,
t_v ~ s R m_v + t (s, R, t fitted), by a soft-L1 residual of scale MOTOR_PRIOR_MM per axis,
and a view whose slack is an outlier loses that hold (it is listed). The hold is what
determines the focal length: every view is the same plane orientation within the stage's
tilt, the degenerate case of planar calibration, so with t_v free (prior_mm=None) a fit
reaches the corner noise at any of a range of focal lengths. The motor-frame fit is this
model with no tilt and no slack, and is its first start.

The cameras are expressed in the motor frame through the similarity, so the result is a
MotorFrameFit like the motor-frame fit's: cameras in the motor frame (true mm), motor_scale =
s, board_rvec the mean board rotation there. Its `placement` holds the per-view slack and
tilts, which the report gives as stage-accuracy warnings.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.sparse import lil_matrix

from reachaq.core.logging import get_verbose_logger
from reachaq.inference.multires_calibration.fit import (
    CHARUCO,
    INTRINSIC_NAMES,
    BoardSpec,
    CameraModel,
    MotorFrameFit,
    MotorPlacement,
    WaypointObservation,
    _camera_data,
    _CameraData,
    _mean_rotation,
    _parameter_std,
    _scaled_rotation,
    fit_motor_frame,
)

logger = get_verbose_logger(__name__)

FIT_PER_VIEW = "per_view"
FIT_MOTOR_FRAME = "motor_frame"
FIT_MODES = (FIT_PER_VIEW, FIT_MOTOR_FRAME)
# The library setting that picks the fit (configured_fit).
FIT_SETTING = "fit"
# The library setting that holds the per-view fit's principal points (configured_principal_point):
# "free" (the default) or "centre", each camera's at its image centre. UNFINALIZED.
PRINCIPAL_POINT_SETTING = "fit_principal_point"
PRINCIPAL_POINT_MODES = ("free", "centre")

_N_INTRINSICS = 8
# The scale of the soft-L1 residual that holds each view's board centre to the motor
# similarity (fit_per_view's prior_mm), per axis. None frees the views' translations entirely,
# which leaves the focal length undetermined on a stage that never tilts the board.
# UNFINALIZED, and a relative weight, not a stage sigma: the prior residual is slack / prior_mm,
# beside reprojection residuals in px, so prior_mm of slack weighs as much as 1 px of
# reprojection error whatever the corner noise is. With real corner noise of 0.1-0.3 px the
# hold is therefore tighter than a 0.4 mm stage sigma would be, and the intrinsics' std
# (focal_rel_std) depends on this weight.
MOTOR_PRIOR_MM: Optional[float] = 0.4
# The fitted pivot is held to the board centre by a residual of this scale per axis (mm), so a
# capture whose board never turns, which says nothing about the pivot, still determines it.
_PIVOT_PRIOR_MM = 50.0
# The first pass's slack scale: the views held to the similarity almost exactly.
_TIGHT_PRIOR_MM = 0.01
# How many times the outlying views are re-chosen and the fit repeated.
_OUTLIER_ROUNDS = 3
# fit.py's planar start flags: k1 is the only distortion term of the start.
_INITIAL_CALIBRATION_FLAGS = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
_FOCAL_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
# Above this RMS after the planar start the focal grid is tried too (fit.py's rule). The rig's
# cameras calibrated alone reach 0.8-1.3 px, so the grid often runs; the lowest cost is kept.
_RETRY_RMS_PX = 1.0
_MAX_EVALUATIONS = 400
# The robust similarity: a view is an outlier when its residual is more than 3 MADs above the
# median; the MAD is floored so that a perfect stage does not flag noise.
_OUTLIER_MADS = 3.0
_MAD_FLOOR_MM = 0.05


def default_fit_mode(board: BoardSpec) -> str:
    """per_view for a ChArUco board, motor_frame for a checkerboard.

    A checkerboard's cross-camera corner order comes only from reconcile_corner_order, whose
    board-rotation test uses the motor-frame model, so it keeps the fit it was built for.
    """
    return FIT_PER_VIEW if board.kind == CHARUCO else FIT_MOTOR_FRAME


def configured_fit(settings, board: BoardSpec) -> str:
    """The library settings' `fit` (per_view or motor_frame); default_fit_mode(board) when it
    is unset or null. Raises ValueError naming the key for any other value."""
    value = settings.get(FIT_SETTING) if settings else None
    if value is None:
        return default_fit_mode(board)
    if value not in FIT_MODES:
        raise ValueError(f"{FIT_SETTING}: {value!r} is not one of {', '.join(FIT_MODES)}")
    return str(value)


# The library setting that picks the stage model the per-view fit holds the views to
# (configured_placement): "similarity" (the default), s R m + t, one scale for every axis; or
# "affine", A m + t with A any 3x3: a scale per motor axis and the angles between the axes.
# UNFINALIZED.
PLACEMENT_SETTING = "placement"
PLACEMENT_SIMILARITY = "similarity"
PLACEMENT_AFFINE = "affine"
PLACEMENT_MODES = (PLACEMENT_SIMILARITY, PLACEMENT_AFFINE)


def configured_placement(settings) -> str:
    """The library settings' `placement` (similarity or affine); similarity when unset or null.
    Raises ValueError naming the key for any other value."""
    value = settings.get(PLACEMENT_SETTING) if settings else None
    if value is None:
        return PLACEMENT_SIMILARITY
    if value not in PLACEMENT_MODES:
        raise ValueError(f"{PLACEMENT_SETTING}: {value!r} is not one of {', '.join(PLACEMENT_MODES)}")
    return str(value)


def configured_principal_point(settings) -> bool:
    """Whether the library settings hold the per-view fit's principal points at the image
    centre (`fit_principal_point: centre`); False when unset or "free". Raises ValueError
    naming the key for any other value."""
    value = settings.get(PRINCIPAL_POINT_SETTING) if settings else None
    if value is None:
        return False
    if value not in PRINCIPAL_POINT_MODES:
        raise ValueError(f"{PRINCIPAL_POINT_SETTING}: {value!r} is not one of {', '.join(PRINCIPAL_POINT_MODES)}")
    return value == "centre"


def fit_per_view(
    observations: Sequence[WaypointObservation],
    board: BoardSpec,
    image_size: Tuple[int, int],
    camera_names: Sequence[str],
    *,
    initial: Optional[MotorFrameFit] = None,
    prior_mm: Optional[float] = -1.0,
    tilt: bool = True,
    fix_principal_point: bool = False,
    placement: str = PLACEMENT_SIMILARITY,
) -> MotorFrameFit:
    """Fit every camera in `camera_names` with a free board pose per view (module docstring).

    `prior_mm`: when given, each view's board origin is also held to the motor similarity
    s R m_v + t (whose 7 parameters join the fit) by a soft-L1 residual of scale `prior_mm`
    per axis; the rotations stay free. Without it, nothing ties a view's depth to its motor
    position, and near-parallel views leave the focal length undetermined.

    Starts: first fit_motor_frame's result, kept when the fit converges with every camera at
    most _RETRY_RMS_PX; otherwise also each camera's planar calibration (k1 its only distortion
    term) and each focal length in _FOCAL_GRID times the image width, principal point at the
    centre; the converged start with the lowest cost wins. From each start, the cameras'
    relative poses are the chordal mean rotation and median translation over the views they
    share (chained from the reference camera through the cameras already placed), and each
    view's pose is the reference camera's, or another placed camera's mapped through its pose.

    `initial`, a fit in this image's pixels, is the only start when given: its intrinsics and
    relative camera poses, with each view's pose from solvePnP.

    `placement` is the stage model the views are held to: PLACEMENT_SIMILARITY, s R m + t, or
    PLACEMENT_AFFINE, A m + t (a scale per motor axis and the angles between the axes; the
    motor frame's orientation is then A's polar rotation, and motor_scale the cube root of
    det A).

    `fix_principal_point` holds every camera's (cx, cy): at `initial`'s when given (the
    cross-check's mapped sensor fit), else at the image centre ((w - 1) / 2, (h - 1) / 2).
    Near-parallel views barely separate the principal point from the camera's position; on
    christielab10's session002 the free principal points sat 0.9-1.4 ROI half-sides off centre.

    Raises ValueError as fit_motor_frame does for a camera whose views cannot place it, or
    when a camera shares no view with the cameras already placed; RuntimeError when no start
    converges.
    """
    if placement not in PLACEMENT_MODES:
        raise ValueError(f"placement must be one of {PLACEMENT_MODES}, got {placement!r}")
    obj = board.object_points()
    names = tuple(camera_names)
    data = {cam: _camera_data(observations, cam, len(obj), board) for cam in names}
    ref = max(names, key=lambda cam: len(data[cam].indices))
    if prior_mm is not None and prior_mm < 0:
        prior_mm = MOTOR_PRIOR_MM
    problem = _Problem(obj, data, names, ref, observations, prior_mm, tilt, affine=placement == PLACEMENT_AFFINE)
    if fix_principal_point:
        width, height = int(image_size[0]), int(image_size[1])
        problem.fixed_pp = {
            cam: (np.asarray(initial.cameras[cam].K, float)[:2, 2] if initial is not None
                  else np.array([(width - 1) / 2, (height - 1) / 2]))
            for cam in names
        }

    if initial is not None:
        x0 = problem.start_from(initial)
        result, outcome = problem.solve(x0)
        if result is None:
            raise RuntimeError(f"the per-view fit did not converge from the given start: {outcome}")
        return problem.to_fit(result, image_size)

    attempts, converged = [], []

    def attempt(label, make_start):
        try:
            x0 = make_start()
        except (cv2.error, np.linalg.LinAlgError, RuntimeError, ValueError) as error:
            attempts.append(f"{label}: no start ({' '.join(str(error).split())})")
            return None
        result, outcome = problem.solve(x0)
        attempts.append(f"{label}: {outcome}")
        if result is not None:
            converged.append(result)
        return result

    # The motor-frame fit is this model with no tilt and no slack, and its focal length is
    # determined, so it is the first start. The planar starts are degenerate in focal length
    # (every view is the same plane orientation): from one, a synthetic rig with no tilt
    # settled at fx 1802 against 2300, paying for it with 8.7 mm of slack.
    first = attempt("motor-frame start",
                    lambda: problem.start_from(fit_motor_frame(observations, board, image_size, names)))
    if first is not None and max(problem.rms(first.fun).values()) <= _RETRY_RMS_PX:
        return problem.to_fit(first, image_size)
    planar = attempt("planar start", lambda: problem.start_planar(image_size, None))
    if planar is not None and max(problem.rms(planar.fun).values()) <= _RETRY_RMS_PX and first is None:
        return problem.to_fit(planar, image_size)
    width, height = int(image_size[0]), int(image_size[1])
    for factor in _FOCAL_GRID:
        f0 = factor * width
        K0 = np.array([[f0, 0.0, (width - 1) / 2], [0.0, f0, (height - 1) / 2], [0.0, 0.0, 1.0]])
        attempt(f"fx0 = fy0 = {f0:g} px ({factor:g} x width)", lambda K0=K0: problem.start_planar(image_size, K0))
    if not converged:
        raise RuntimeError("the per-view fit did not converge from any start: " + "; ".join(attempts))
    best = min(converged, key=problem.common_cost(converged))
    logger.info("per-view fit retried from a focal grid; kept the lowest cost of: %s", "; ".join(attempts))
    return problem.to_fit(best, image_size)


class _Problem:
    """The parameter layout, residuals and start of one per-view fit.

    x = [8 intrinsics per camera] + [rvec, tvec per camera other than the reference]
        + [rvec, tvec per view], views in waypoint order.
    """

    def __init__(self, obj, data: Dict[str, _CameraData], names, ref, observations, prior_mm=None, tilt=True,
                 affine=False):
        self.obj, self.data, self.names, self.ref = obj, data, names, ref
        # The stage model: A m + t, A any 3x3 (12 parameters), or s R m + t (7)
        self.affine = affine
        self.prior_mm = prior_mm
        # The board point the prior holds to the motor similarity: the point the board turns
        # about, its mount, which is fitted (3 parameters shared by every view, in the board's
        # frame, started at the centre of the inner corners) when the views may tilt. Anchored
        # at a fixed point instead, every tilt about any other point turns into slack: at corner
        # 0, a 3 degree tilt about a mount 30 mm behind the board's centre moves the anchor 1.6
        # mm; at the centre, a tilt about corner 0 moved fx 1.7 % on the synthetic stage.
        self.anchor = np.asarray(obj, float).mean(axis=0)
        # The prior's loss and scale for the pass being solved (solve): soft-L1 at prior_mm
        # for the result.
        self.robust = True
        self.scale_mm = prior_mm
        # Views whose slack is an outlier (solve): their translation is left free, so a lost
        # step neither pulls the fit nor loses the view's corners.
        # Without tilt every view takes view 0's board rotation (the other views' rotation
        # parameters are unused), so only the translations have per-view freedom.
        self.tilt = tilt
        # camera -> (cx, cy) held there; None fits them
        self.fixed_pp: Optional[Dict[str, np.ndarray]] = None
        self.views = sorted(set().union(*[d.indices for d in data.values()]))
        self.view_pos = {w: k for k, w in enumerate(self.views)}
        self.prior_off = np.zeros(len(self.views), bool)
        motor = {o.index: np.asarray(o.motor_mm, float).reshape(3) for o in observations}
        self.motor = np.array([motor[w] for w in self.views])
        self.posed = [cam for cam in names if cam != ref]
        n = len(obj)
        # Per camera, per its view: (view position, corner indices used, observed px of those).
        self.blocks: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] = {}
        for cam in names:
            d = data[cam]
            corners = d.corners.reshape(len(d.indices), n, 2)
            masks = np.ones((len(d.indices), n), bool) if d.valid is None else d.valid.reshape(len(d.indices), n)
            self.blocks[cam] = [
                (self.view_pos[w], np.flatnonzero(m), c[m]) for w, c, m in zip(d.indices, corners, masks)
            ]
        self._groups = self._sparsity = None
        self.n_params = _N_INTRINSICS * len(names) + 6 * len(self.posed) + 6 * len(self.views)
        if prior_mm is not None:
            self.n_params += self._n_sim()  # the stage model: rvec, t, s; or A, t
            if tilt:
                self.n_params += 3  # the pivot, in the board's frame
        self.observed = np.concatenate([px.ravel() for cam in names for _, _, px in self.blocks[cam]])
        # Per camera, every corner row's view position and board point, in block order.
        self._rows = {
            cam: (np.concatenate([np.full(len(ids), p) for p, ids, _ in self.blocks[cam]]),
                  np.concatenate([obj[ids] for _, ids, _ in self.blocks[cam]]))
            for cam in names
        }

    # Parameter offsets.
    def _intr(self, c):
        return slice(_N_INTRINSICS * c, _N_INTRINSICS * (c + 1))

    def _pose(self, cam):
        k = self.posed.index(cam)
        start = _N_INTRINSICS * len(self.names) + 6 * k
        return slice(start, start + 6)

    def _view(self, p):
        start = _N_INTRINSICS * len(self.names) + 6 * len(self.posed) + 6 * p
        return slice(start, start + 6)

    def _n_sim(self) -> int:
        return 12 if self.affine else 7

    def _sim(self):
        start = _N_INTRINSICS * len(self.names) + 6 * len(self.posed) + 6 * len(self.views)
        return slice(start, start + self._n_sim())

    def stage_map(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """(A, t) of the stage model in x: A m + t is a view's held point for motor position m."""
        sim = x[self._sim()]
        if self.affine:
            return sim[:9].reshape(3, 3), sim[9:12]
        return sim[6] * _rodrigues(sim[None, :3])[0], sim[3:6]

    def _set_stage_map(self, x, A: np.ndarray, t: np.ndarray) -> None:
        if self.affine:
            x[self._sim()] = np.concatenate([np.asarray(A, float).ravel(), t])
        else:
            R, s = _polar(A)
            x[self._sim()] = np.concatenate([cv2.Rodrigues(R)[0].ravel(), t, [s]])

    def _pivot(self):
        """The pivot's 3 parameters (prior and tilt only)."""
        return slice(self._sim().stop, self._sim().stop + 3)

    def pivot(self, x) -> np.ndarray:
        """The board point held to the similarity: the fitted pivot, or the board centre."""
        return x[self._pivot()] if self.prior_mm is not None and self.tilt else self.anchor

    def _intrinsics(self, x, c, cam) -> np.ndarray:
        """Camera c's 8 intrinsics from x, with its principal point held where fixed_pp says."""
        values = x[self._intr(c)]
        if self.fixed_pp is not None:
            values = values.copy()
            values[2:4] = self.fixed_pp[cam]
        return values

    def _camera(self, x, c, cam):
        fx, fy, cx, cy, k1, k2, p1, p2 = self._intrinsics(x, c, cam)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        dist = np.array([k1, k2, p1, p2, 0.0])
        if cam == self.ref:
            return K, dist, np.zeros(3), np.zeros(3)
        pose = x[self._pose(cam)]
        return K, dist, pose[:3], pose[3:]

    def residuals(self, x) -> np.ndarray:
        """Projected minus observed px, every camera's detected corners in block order.

        Vectorised over all corners of a camera (cv2.projectPoints per view was 10 times
        slower); the projection is cv2's pinhole model with k1, k2, p1, p2 (k3 = 0), and
        test_the_projection_matches_cv2 holds it to cv2.projectPoints.
        """
        start = _N_INTRINSICS * len(self.names) + 6 * len(self.posed)
        views = x[start:start + 6 * len(self.views)].reshape(-1, 6)
        R_views = _rodrigues(views[:, :3]) if self.tilt else np.repeat(_rodrigues(views[:1, :3]), len(views), axis=0)
        out = []
        for c, cam in enumerate(self.names):
            view_of, points = self._rows[cam]
            X = np.einsum("nij,nj->ni", R_views[view_of], points) + views[view_of, 3:]
            if cam != self.ref:
                pose = x[self._pose(cam)]
                X = X @ _rodrigues(pose[None, :3])[0].T + pose[3:]
            out.append(_project(X, self._intrinsics(x, c, cam)).ravel())
        reprojection = np.concatenate(out) - self.observed
        if self.prior_mm is None:
            return reprojection
        A, t = self.stage_map(x)
        predicted = self.motor @ A.T + t
        e = (self._centres(views, R_views, self.pivot(x)) - predicted) / self.scale_mm
        e[self.prior_off] = 0.0
        if not self.robust:
            return np.concatenate([reprojection, e.ravel(), self._pivot_residual(x)])
        # soft-L1: the squared residual is 2 (sqrt(1 + e^2) - 1), quadratic near 0, linear far out.
        prior = np.sign(e) * np.sqrt(2.0 * (np.sqrt(1.0 + e * e) - 1.0))
        return np.concatenate([reprojection, prior.ravel(), self._pivot_residual(x)])

    def _pivot_residual(self, x) -> np.ndarray:
        """The pivot's pull to the board centre, 3 rows (none without a fitted pivot)."""
        if self.prior_mm is None or not self.tilt:
            return np.zeros(0)
        return (x[self._pivot()] - self.anchor) / _PIVOT_PRIOR_MM

    def sparsity(self):
        n_prior = 0 if self.prior_mm is None else 3 * len(self.views) + (3 if self.tilt else 0)
        S = lil_matrix((len(self.observed) + n_prior, self.n_params), dtype=int)
        row = 0
        for c, cam in enumerate(self.names):
            for p, ids, _ in self.blocks[cam]:
                rows = slice(row, row + 2 * len(ids))
                intrinsics = self._intr(c)
                S[rows, intrinsics] = 1
                if self.fixed_pp is not None:  # cx and cy are held: their columns stay empty
                    S[rows, slice(intrinsics.start + 2, intrinsics.start + 4)] = 0
                if cam != self.ref:
                    S[rows, self._pose(cam)] = 1
                view = self._view(p)
                S[rows, slice(view.start + 3, view.stop)] = 1
                rotation = view if self.tilt else self._view(0)
                S[rows, slice(rotation.start, rotation.start + 3)] = 1
                row += 2 * len(ids)
        if self.prior_mm is not None:
            for p in range(len(self.views)):
                view = self._view(p)
                S[row:row + 3, view] = 1  # the centre moves with the view's rotation too
                if not self.tilt:
                    rotation = self._view(0)
                    S[row:row + 3, slice(rotation.start, rotation.start + 3)] = 1
                S[row:row + 3, self._sim()] = 1
                if self.tilt:
                    S[row:row + 3, self._pivot()] = 1
                row += 3
            if self.tilt:
                S[row:row + 3, self._pivot()] = 1
        return S

    def _centres(self, views: np.ndarray, R_views: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        """(views, 3) each view's `pivot` (a point in the board's frame) in the reference frame."""
        return views[:, 3:] + R_views @ pivot

    def _views(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """(views (n, 6), their rotation matrices (n, 3, 3)) from x."""
        start = _N_INTRINSICS * len(self.names) + 6 * len(self.posed)
        views = x[start:start + 6 * len(self.views)].reshape(-1, 6)
        R_views = _rodrigues(views[:, :3]) if self.tilt else np.repeat(_rodrigues(views[:1, :3]), len(views), axis=0)
        return views, R_views

    def jacobian(self, x, *_):
        """The dense Jacobian by finite differences, perturbing the columns that share no
        residual together (the sparsity's column groups): about 20 evaluations rather than
        one per parameter. Dense, so least_squares can take exact trust-region steps; with
        its sparse LSMR steps it crawled and ran out of evaluations on every synthetic rig."""
        if self._groups is None:
            S = self.sparsity().tocsc()
            self._sparsity = S
            self._groups = group_columns(S)
        return approx_derivative(self.residuals, x, method="2-point", sparsity=(self._sparsity, self._groups)).toarray()

    def solve(self, x0):
        """least_squares from x0. With a prior, the slack is relaxed in passes: first held
        nearly exact (_TIGHT_PRIOR_MM, quadratic), which is the motor-frame model with free
        tilts and determines the focal length; then quadratic at prior_mm; then soft-L1 at
        prior_mm, the result. From the motor-frame start on a tilting synthetic stage (fx off
        by 5-13 % there), the soft-L1 pass alone, or after a quadratic pass at prior_mm, stopped
        at a cost of about 1250 with fx 13 % short, where the truth's is 79: once the tilts have
        absorbed the reprojection error, the slack takes up the focal error."""
        self.prior_off = np.zeros(len(self.views), bool)
        if self.prior_mm is not None:
            for scale in (_TIGHT_PRIOR_MM, self.prior_mm):
                self.robust, self.scale_mm = False, scale
                try:
                    step = least_squares(
                        self.residuals, x0, jac=self.jacobian, method="trf", x_scale="jac", loss="linear",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=_MAX_EVALUATIONS,
                    )
                finally:
                    self.robust, self.scale_mm = True, self.prior_mm
                if step.status > 0:
                    x0 = step.x
                else:
                    logger.info("per-view fit: the pass at slack scale %g mm did not converge (%s);"
                                " the next pass starts where it began", scale, step.message)
        result = least_squares(
            self.residuals, x0, jac=self.jacobian, method="trf", x_scale="jac", loss="linear",
            ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=_MAX_EVALUATIONS,
        )
        # A view whose slack is more than 3 MADs (at least 3 prior_mm) above the median loses
        # its prior and the fit is repeated, until the set holds. Soft-L1 bounds an outlier's
        # pull but does not remove it, and the focal length is the weak direction: three
        # lost steps of 2.5-3 mm moved fx by 0.8 % at prior_mm 0.05 and 3 % at 0.4 on the
        # synthetic stage, against 0.1 % without them.
        for _ in range(_OUTLIER_ROUNDS if self.prior_mm is not None and result.status > 0 else 0):
            slack = np.linalg.norm(self.slack_mm(result.x), axis=1)
            kept = slack[~self.prior_off]
            median = float(np.median(kept))
            mad = 1.4826 * float(np.median(np.abs(kept - median)))
            off = slack > median + _OUTLIER_MADS * max(mad, self.prior_mm)
            if np.array_equal(off, self.prior_off) or (~off).sum() < 3:
                break
            previous, previous_off = result, self.prior_off
            self.prior_off = off
            result = least_squares(
                self.residuals, result.x, jac=self.jacobian, method="trf", x_scale="jac", loss="linear",
                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=_MAX_EVALUATIONS,
            )
            if result.status <= 0:
                # Keep the converged fit this refit started from, with its outlier set.
                logger.warning("per-view fit: the refit without the outlying views' prior did not converge"
                               " (%s); keeping the fit before it", result.message)
                result, self.prior_off = previous, previous_off
                break
        if result.status <= 0:
            return None, f"least_squares status {result.status} after {result.nfev} evaluations ({result.message})"
        result.prior_off = self.prior_off.copy()
        rms = ", ".join(f"{cam} {v:.3g}" for cam, v in self.rms(result.fun).items())
        return result, f"converged in {result.nfev} evaluations, cost {result.cost:.6g}, rms px {rms}"

    def slack_mm(self, x) -> np.ndarray:
        """(views, 3) each view's board centre minus s R m + t, in the reference frame, mm."""
        views, R_views = self._views(x)
        A, t = self.stage_map(x)
        return self._centres(views, R_views, self.pivot(x)) - (self.motor @ A.T + t)

    def common_cost(self, results):
        """A key that scores each converged result over the same outlier set, the union of
        theirs: a start that freed more views would otherwise score lower for that alone."""
        off = np.zeros(len(self.views), bool)
        for result in results:
            off |= getattr(result, "prior_off", np.zeros(len(self.views), bool))

        def cost(result):
            saved, self.prior_off = self.prior_off, off
            try:
                r = self.residuals(result.x)
            finally:
                self.prior_off = saved
            return 0.5 * float(r @ r)

        return cost

    def _per_camera_squares(self, r) -> Dict[str, List[np.ndarray]]:
        sq = (r[:len(self.observed)].reshape(-1, 2) ** 2).sum(axis=1)
        out, k = {}, 0
        for cam in self.names:
            out[cam] = []
            for _, ids, _ in self.blocks[cam]:
                out[cam].append(sq[k:k + len(ids)])
                k += len(ids)
        return out

    def rms(self, r) -> Dict[str, float]:
        return {cam: float(np.sqrt(np.concatenate(v).mean())) for cam, v in self._per_camera_squares(r).items()}

    # Starts.
    def start_planar(self, image_size, K_fixed) -> np.ndarray:
        intrinsics, view_poses = {}, {}
        size = (int(image_size[0]), int(image_size[1]))
        for cam in self.names:
            objs = [self.obj[ids].astype(np.float32) for _, ids, _ in self.blocks[cam]]
            imgs = [px.reshape(-1, 1, 2).astype(np.float32) for _, _, px in self.blocks[cam]]
            if K_fixed is None:
                _, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None,
                                                               flags=_INITIAL_CALIBRATION_FLAGS)
            else:
                held = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
                try:
                    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                        objs, imgs, size, np.array(K_fixed, float), np.zeros(5), flags=_INITIAL_CALIBRATION_FLAGS | held)
                except cv2.error:
                    K, dist = np.array(K_fixed, float), np.zeros(5)
                    poses = [cv2.solvePnP(o, i, K, dist)[1:] for o, i in zip(objs, imgs)]
                    rvecs, tvecs = [r for r, _ in poses], [t for _, t in poses]
            intrinsics[cam] = (np.asarray(K, float), np.asarray(dist, float).ravel())
            view_poses[cam] = {p: (cv2.Rodrigues(np.asarray(r, float))[0], np.asarray(t, float).ravel())
                               for (p, _, _), r, t in zip(self.blocks[cam], rvecs, tvecs)}
        return self._pack(intrinsics, self._relative_poses(view_poses), view_poses)

    def start_from(self, initial: MotorFrameFit) -> np.ndarray:
        """The start a given fit gives (the motor-frame fit's, or the cross-check's mapped
        sensor fit): its intrinsics and relative camera poses, each view's pose from solvePnP
        with those intrinsics (the reference camera's, else the first camera that saw it,
        mapped through its pose), so the views keep their tilts; and the similarity the fit
        implies, its reference camera's pose and motor_scale."""
        intrinsics, cameras, view_poses = {}, {}, {}
        R_ref = cv2.Rodrigues(np.asarray(initial.cameras[self.ref].rvec, float))[0]
        t_ref = np.asarray(initial.cameras[self.ref].tvec, float).reshape(3)
        for cam in self.names:
            m = initial.cameras[cam]
            K, dist = np.asarray(m.K, float), np.asarray(m.dist, float).ravel()
            intrinsics[cam] = (K, dist)
            R = cv2.Rodrigues(np.asarray(m.rvec, float))[0] @ R_ref.T
            cameras[cam] = (R, np.asarray(m.tvec, float).reshape(3) - R @ t_ref)
            view_poses[cam] = {}
            for p, ids, px in self.blocks[cam]:
                _, rvec, tvec = cv2.solvePnP(self.obj[ids].astype(np.float32),
                                             px.reshape(-1, 1, 2).astype(np.float32), K, dist)
                view_poses[cam][p] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
        # The given fit puts the board centre at R_ref (R_b c + s m) + t_ref.
        R_board = cv2.Rodrigues(np.asarray(initial.board_rvec, float))[0]
        s = float(initial.motor_scale)
        return self._pack(intrinsics, cameras, view_poses,
                          similarity=(R_ref, s, t_ref + R_ref @ R_board @ self.anchor))

    def _relative_poses(self, view_poses) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Each camera's (R, t) from the reference camera's frame: BFS from the reference, each
        camera placed from the placed cameras it shares views with."""
        placed = {self.ref: (np.eye(3), np.zeros(3))}
        queue = deque([self.ref])
        while queue:
            known = queue.popleft()
            R_k, t_k = placed[known]
            for cam in self.names:
                if cam in placed:
                    continue
                shared = sorted(set(view_poses[cam]) & set(view_poses[known]))
                if not shared:
                    continue
                # T_cam<-known from each shared view: T_cam,v o T_known,v^-1.
                rotations = np.array([view_poses[cam][p][0] @ view_poses[known][p][0].T for p in shared])
                R_ck = _mean_rotation(rotations)
                t_ck = np.median([view_poses[cam][p][1] - R_ck @ view_poses[known][p][1] for p in shared], axis=0)
                placed[cam] = (R_ck @ R_k, R_ck @ t_k + t_ck)
                queue.append(cam)
        missing = [cam for cam in self.names if cam not in placed]
        if missing:
            raise ValueError(f"{', '.join(missing)}: no view shared with the other cameras; the camera cannot be placed")
        return placed

    def _pack(self, intrinsics, cameras, view_poses, similarity=None) -> np.ndarray:
        x = np.zeros(self.n_params)
        for c, cam in enumerate(self.names):
            K, dist = intrinsics[cam]
            x[self._intr(c)] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist[:4]]
            if cam != self.ref:
                R, t = cameras[cam]
                x[self._pose(cam)] = np.concatenate([cv2.Rodrigues(R)[0].ravel(), t])
        for p in range(len(self.views)):
            # The reference camera's view of it, else the first camera's that saw it, mapped
            # into the reference frame.
            for cam in (self.ref, *self.posed):
                if p in view_poses[cam]:
                    R_cv, t_cv = view_poses[cam][p]
                    R_c, t_c = cameras.get(cam, (np.eye(3), np.zeros(3)))
                    R_v, t_v = R_c.T @ R_cv, R_c.T @ (t_cv - t_c)
                    x[self._view(p)] = np.concatenate([cv2.Rodrigues(R_v)[0].ravel(), t_v])
                    break
        if self.prior_mm is not None:
            if self.tilt:
                x[self._pivot()] = self.anchor
            if similarity is None:
                similarity = _similarity(self.motor, self._centres(*self._views(x), self.anchor))
            R, s, t = similarity
            self._set_stage_map(x, s * R, t)
        return x

    # The result.
    def to_fit(self, result, image_size) -> MotorFrameFit:
        x = result.x
        J = result.jac.toarray() if hasattr(result.jac, "toarray") else np.asarray(result.jac)
        # The residual variance from the reprojections alone: the prior's residuals are in
        # its own units, and would otherwise inflate every sigma.
        # _parameter_std divides by len(r) - parameters; padding the prior rows with zeros would
        # count them as degrees of freedom, so the reprojection residuals are scaled to give
        # their own variance, r @ r / (n - parameters).
        n = len(self.observed)
        n_params = J.shape[1]
        scale = math.sqrt(max(len(result.fun) - n_params, 1) / max(n - n_params, 1))
        std = _parameter_std(J, np.concatenate([result.fun[:n] * scale, np.zeros(len(result.fun) - n)]))
        views, R_all = self._views(x)
        views_R = list(R_all)
        pivot = self.pivot(x)
        centres = self._centres(views, R_all, pivot)
        fixed = None
        if self.prior_mm is not None:
            A, t_fit = self.stage_map(x)
            fixed = (np.array(A, float), np.array(t_fit, float))
        R, s, t_centre, placement = _place(self.views, views_R, centres, self.motor, fixed)
        # The similarity holds the pivot; the motor frame's origin is corner 0 at motor
        # position 0 (fit.py), so X_ref = R @ X_motor + t with t = t_centre - R @ R_b @ pivot.
        board_motor = _mean_rotation(np.array([R.T @ Rv for Rv in views_R]))
        t = t_centre - R @ board_motor @ pivot
        placement.pivot_mm = tuple(float(v) for v in pivot)
        off = getattr(result, "prior_off", None)
        if fixed is not None and off is not None:
            # The views the fit freed are the outliers, and the RMS is over the others.
            placement.outliers = [w for w, o in zip(self.views, off) if o]
            kept = np.array([placement.residual_mm[w] for w, o in zip(self.views, off) if not o])
            placement.rms_mm = float(np.sqrt((kept ** 2).sum(axis=1).mean())) if len(kept) else math.nan

        cameras, rms_px, per_view, param_std, counts = {}, {}, {}, {}, {}
        squares = self._per_camera_squares(result.fun)
        for c, cam in enumerate(self.names):
            K, dist, rc, tc = self._camera(x, c, cam)
            R_c = cv2.Rodrigues(np.asarray(rc, float))[0]
            # X_ref = R @ X_motor + t, so X_cam = R_c R X_motor + (R_c t + t_c).
            cameras[cam] = CameraModel(K=K, dist=dist, rvec=cv2.Rodrigues(R_c @ R)[0].ravel(),
                                       tvec=R_c @ t + np.asarray(tc, float))
            rms_px[cam] = float(np.sqrt(np.concatenate(squares[cam]).mean()))
            per_view[cam] = {self.views[p]: float(np.sqrt(sq.mean())) for (p, _, _), sq in zip(self.blocks[cam], squares[cam])}
            counts[cam] = {self.views[p]: len(ids) for p, ids, _ in self.blocks[cam]}
            fitted = dict(zip(INTRINSIC_NAMES[:_N_INTRINSICS], std[self._intr(c)]))
            if self.fixed_pp is not None:  # held, not estimated, as k3 is
                fitted.pop("cx")
                fitted.pop("cy")
            param_std[cam] = {name: float(fitted.get(name, 0.0)) for name in INTRINSIC_NAMES}
        partial = any(d.valid is not None for d in self.data.values())
        return MotorFrameFit(
            cameras=cameras,
            board_rvec=cv2.Rodrigues(board_motor)[0].ravel(),
            board_tvec=np.zeros(3),
            motor_scale=float(s),
            image_size=(int(image_size[0]), int(image_size[1])),
            rms_px=rms_px,
            per_view_rms_px=per_view,
            param_std=param_std,
            corner_counts=counts if partial else None,
            placement=placement,
        )


def _rodrigues(rvecs: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation matrices of (n, 3) rotation vectors."""
    rvecs = np.asarray(rvecs, float).reshape(-1, 3)
    theta = np.linalg.norm(rvecs, axis=1)
    small = theta < 1e-12
    k = rvecs / np.where(small, 1.0, theta)[:, None]
    K = np.zeros((len(rvecs), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], (1.0 - np.cos(theta))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def _project(X: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """(n, 2) px of camera-frame points X (n, 3): cv2's model with k1, k2, p1, p2 and k3 = 0."""
    fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([fx * xd + cx, fy * yd + cy])


def _place(views, views_R, views_t, motor, fixed=None) -> Tuple[np.ndarray, float, np.ndarray, MotorPlacement]:
    """(R, s, t) of the stage model o_v ~ A m_v + t, with R the polar rotation of A and s the
    cube root of det A (A = s R for a similarity), robust to outlying views, and the
    MotorPlacement: per-view residuals in motor mm, tilts, scale, per-axis scales and axis
    angles, inlier RMS, outliers. `fixed`, an (A, t) the fit already solved, is used as it is;
    the outliers are then only those more than 3 MAD off it."""
    o = np.asarray(views_t, float)
    m = np.asarray(motor, float)
    inliers = np.ones(len(o), bool)

    def model(keep):
        if fixed is not None:
            return fixed
        R_, s_, t_ = _similarity(m[keep], o[keep])
        return s_ * R_, t_

    for _ in range(10):
        A, t = model(inliers)
        residual = (o - t) @ np.linalg.inv(A).T - m  # rows: A^-1 (o - t) - m
        distance = np.linalg.norm(residual, axis=1)
        median = float(np.median(distance[inliers]))
        mad = max(1.4826 * float(np.median(np.abs(distance[inliers] - median))), _MAD_FLOOR_MM)
        keep = distance <= median + _OUTLIER_MADS * mad
        if keep.sum() < 3 or np.linalg.matrix_rank(m[keep] - m[keep].mean(axis=0), tol=1e-6) < 2:
            break
        if np.array_equal(keep, inliers):
            break
        inliers = keep
    A, t = model(inliers)
    residual = (o - t) @ np.linalg.inv(A).T - m
    R, s = _polar(A)
    board_motor = [R.T @ Rv for Rv in views_R]
    mean = _mean_rotation(np.array(board_motor))
    # Each view's turn from the mean board rotation, B = mean @ dR, as an angle and its axis in
    # the motor frame (the axis mean @ rotvec(dR)).
    turns = {w: cv2.Rodrigues(mean.T @ B)[0].ravel() for w, B in zip(views, board_motor)}
    tilt = {w: math.degrees(float(np.linalg.norm(v))) for w, v in turns.items()}
    axis = {w: tuple(float(a) for a in (mean @ v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.zeros(3)))
            for w, v in turns.items()}
    placement = MotorPlacement(
        scale=float(s),
        rms_mm=float(np.sqrt((residual[inliers] ** 2).sum(axis=1).mean())),
        residual_mm={w: tuple(float(v) for v in r) for w, r in zip(views, residual)},
        tilt_deg=tilt,
        tilt_axis=axis,
        outliers=[w for w, keep in zip(views, inliers) if not keep],
        axis_scale=tuple(float(v) for v in np.linalg.norm(A, axis=0)),
        axis_angle_deg=tuple(float(_angle(A[:, i], A[:, j])) for i, j in ((0, 1), (0, 2), (1, 2))),
    )
    return R, float(s), t, placement


def _polar(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """(R, s): the rotation of A's polar decomposition A = R S, and the cube root of det A."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, float(np.cbrt(np.linalg.det(A)))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two vectors, degrees."""
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b))))


def _similarity(m: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """(R, s, t) minimising sum |o_i - (s R m_i + t)|^2 (Umeyama)."""
    m_mean, o_mean = m.mean(axis=0), o.mean(axis=0)
    R, s = _scaled_rotation(m - m_mean, o - o_mean)
    return R, s, o_mean - s * R @ m_mean
