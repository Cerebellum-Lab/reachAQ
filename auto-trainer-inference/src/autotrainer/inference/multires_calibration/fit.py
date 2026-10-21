"""Fit both cameras in the motor frame from the board at paused waypoints.

The stage slides the board between waypoints and never tilts it, so every view a
camera has of the board lies in a parallel plane. A planar calibration of those
views cannot separate focal length from distance. The commanded motor positions
can: the corners of every waypoint together form one rigid, non-planar target in
the motor frame,

    X_wj = R(rvec_b) @ P_j + tvec_b + sigma * m_w

where P_j is board corner j, m_w the commanded motor position of waypoint w in mm,
and sigma the motor scale (true mm per commanded mm), which the fit measures. The
board's own squares carry the metric scale, so sigma and the focal length are
both determined. Camera c maps the motor frame into its own frame as
X_cam = R(rvec_c) @ X + tvec_c, in mm.

tvec_b is held at zero rather than fitted: it is a gauge choice. Moving the board
origin by d and every camera by -R_c @ d changes no projection, so the motor frame's
origin is a choice, not a measurement. Here it is the board's corner 0 at motor
position 0; the DCS calibration sets the live 3D origin later. Fitting tvec_b too
would leave J^T J singular and the parameter uncertainties undefined.

k3 is held at 0 (dist keeps all five coefficients, with k3 = 0.0). The board covers
only the middle of the sensor, about 520 px from the principal point on the
synthetic rig, and with k3 free the fit put it at -8 +/- 6. That moved the
distortion at the corners of the 1024 preset's ROI by 2.7 px while fitting the
covered area no better.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)

# The order of the intrinsic parameters in param_std, matching cv2's distortion order
# (k1, k2, p1, p2, k3).
INTRINSIC_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
# The intrinsics the fit solves: all but k3, which is held at 0 (see above).
_FITTED_INTRINSICS = INTRINSIC_NAMES[:-1]
# Per camera: the fitted intrinsics in that order, then rvec (3) and tvec (3).
_CAMERA_PARAMS = len(_FITTED_INTRINSICS) + 6
# The planar calibration that starts the fit sees only parallel planes, so it is
# degenerate in focal length, and free distortion terms let it wander further. With
# all five free, its fx on the synthetic rig ranged from 0.6x to 5.5x the truth
# (k2 down to -2800), and from such a start the joint fit diverged on 1 seed in 30.
# Fitting k1 alone kept all 30 in the basin. The joint fit then frees k2, p1 and p2
# too; k3 stays at 0 in both.
_INITIAL_CALIBRATION_FLAGS = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
# Converging fits on the synthetic rig took 7-44 evaluations. A fit that wanders along
# an undetermined direction can take thousands (one took 105 s for 1586), so stop it
# early and raise instead.
_MAX_EVALUATIONS = 400
# The planar start can still land outside the joint fit's basin: on the pipeline's
# synthetic capture it put fx at 3 to 13 times the truth and the board rotation up to 77
# degrees off, and the fit of the 512 preset's detections ran out of evaluations from it.
# When the planar start does not converge, or leaves a camera above _RETRY_RMS_PX, the fit
# is retried with fx0 = fy0 = image width times each factor here, which brackets lenses
# from wide to long (the synthetic rig's fx is 1.6 times its width).
_FOCAL_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
# Detected corners sit within a few tenths of a pixel of a good fit; a camera above this
# RMS after the planar start is taken as a sign of a wrong basin, not of bad data.
_RETRY_RMS_PX = 1.0
# reconcile_corner_order's board rotations differ by about 0 or about 180 degrees; between
# these the decision is ambiguous and is logged.
_AMBIGUOUS_FLIP_DEG = (60.0, 120.0)
# Directions of the normalised J^T J whose eigenvalue is below 1/_MAX_CONDITION of the
# largest are treated as undetermined by the capture. Good captures on the synthetic
# rig sit at a condition number of 4.5e6 in every one of 30 seeds; a camera that sees
# the board in only one plane sits at 3.8e10 to 2.6e11, with its fx off by 40-60 %.
_MAX_CONDITION = 1e9


@dataclass(frozen=True)
class BoardSpec:
    """A chessboard, counted by its inner corners."""

    square_mm: float = 4.0
    rows: int = 6
    cols: int = 8

    def object_points(self) -> np.ndarray:
        """(rows*cols, 3) inner-corner positions in mm, row-major with the column fastest, z = 0.

        x runs along a row (the columns) and y down the rows, which is the order
        cv2.findChessboardCorners reports for a pattern size of (cols, rows).
        """
        row, col = np.mgrid[0:self.rows, 0:self.cols]
        return np.column_stack([col.ravel(), row.ravel(), np.zeros(row.size)]).astype(float) * self.square_mm


@dataclass
class WaypointObservation:
    """The board at one paused waypoint. A camera is absent when it did not detect the board."""

    index: int
    motor_mm: np.ndarray
    corners: Dict[str, np.ndarray]  # camera name -> (rows*cols, 2) sensor px, in object_points order


@dataclass
class CameraModel:
    """One camera: X_cam = R(rvec) @ X_motor + tvec, in mm; dist is cv2's 5 coefficients."""

    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass
class MotorFrameFit:
    cameras: Dict[str, CameraModel]
    board_rvec: np.ndarray
    board_tvec: np.ndarray
    motor_scale: float
    image_size: Tuple[int, int]
    # RMS of the per-corner reprojection distance, sqrt(mean(du^2 + dv^2)), as cv2.calibrateCamera reports it.
    rms_px: Dict[str, float]
    per_view_rms_px: Dict[str, Dict[int, float]]  # camera -> waypoint index -> RMS px
    param_std: Dict[str, Dict[str, float]]  # camera -> INTRINSIC_NAMES -> 1-sigma

    def stereo(self, first: str = "left", second: str = "right") -> Tuple[np.ndarray, np.ndarray]:
        """(R, T) with X_second = R @ X_first + T, T in mm."""
        a, b = self.cameras[first], self.cameras[second]
        R_a = cv2.Rodrigues(np.asarray(a.rvec, float))[0]
        R_b = cv2.Rodrigues(np.asarray(b.rvec, float))[0]
        R = R_b @ R_a.T
        return R, np.asarray(b.tvec, float).reshape(3) - R @ np.asarray(a.tvec, float).reshape(3)


@dataclass
class _CameraData:
    indices: List[int]
    motor_mm: np.ndarray  # (n_views, 3)
    corners: np.ndarray  # (n_views * n_corners, 2)


def fit_motor_frame(
    observations: Sequence[WaypointObservation],
    board: BoardSpec,
    image_size: Tuple[int, int],
    camera_names: Sequence[str] = ("left", "right"),
    *,
    initial: Optional[MotorFrameFit] = None,
) -> MotorFrameFit:
    """Fit every camera's intrinsics and motor-frame pose, the board rotation and the motor scale.

    A waypoint seen by only one camera still constrains that camera: each camera's
    residuals come from exactly the waypoints it observed.

    Starts. The planar start (a planar calibration per camera, k1 its only distortion
    term, then the motor axes from how the board moved in each camera) is tried first and
    kept when it converges with every camera's RMS at most _RETRY_RMS_PX. It is degenerate
    in focal length, because every view is a parallel plane, so otherwise the fit is
    retried from each focal length in _FOCAL_GRID times the image width, with the principal
    point at the image centre: the same planar calibration with that K held fixed gives k1
    and every view's pose (zero distortion and per-view solvePnP if it cannot). Of all the
    starts that converged, planar included, the one with the lowest total cost is kept,
    even if its RMS is still high: that is what the data supports, and the report grades
    it. RuntimeError, listing every start and its outcome, only when none converged.

    `initial`, a fit whose cameras are in this image's pixels, is the only start when
    given; RuntimeError if the fit does not converge from it. The pipeline's per-preset
    cross-check starts from the sensor fit mapped to the preset this way. The result is the
    least-squares optimum of these observations whichever start reaches it.
    """
    obj = board.object_points()
    data = {cam: _camera_data(observations, cam, len(obj)) for cam in camera_names}

    if initial is not None:
        fit, _, outcome = _solve(_parameters_of(initial, camera_names), obj, data, camera_names, image_size)
        if fit is None:
            raise RuntimeError(f"the motor-frame fit did not converge from the given start: {outcome}")
        return fit

    attempts: List[str] = []
    converged: List[Tuple[float, MotorFrameFit]] = []

    def attempt(label: str, K_fixed: Optional[np.ndarray]) -> Optional[MotorFrameFit]:
        try:
            x0 = _initial_parameters(data, obj, image_size, camera_names, K_fixed)
        except (cv2.error, np.linalg.LinAlgError) as error:
            attempts.append(f"{label}: no start ({' '.join(str(error).split())})")
            return None
        fit, cost, outcome = _solve(x0, obj, data, camera_names, image_size)
        attempts.append(f"{label}: {outcome}")
        if fit is not None:
            converged.append((cost, fit))
        return fit

    planar = attempt("planar start", None)
    if planar is not None and max(planar.rms_px.values()) <= _RETRY_RMS_PX:
        return planar
    width, height = int(image_size[0]), int(image_size[1])
    for factor in _FOCAL_GRID:
        f0 = factor * width
        K0 = np.array([[f0, 0.0, (width - 1) / 2], [0.0, f0, (height - 1) / 2], [0.0, 0.0, 1.0]])
        attempt(f"fx0 = fy0 = {f0:g} px ({factor:g} x width)", K0)
    if not converged:
        raise RuntimeError("the motor-frame fit did not converge from any start: " + "; ".join(attempts))
    cost, best = min(converged, key=lambda item: item[0])
    logger.info("motor-frame fit retried from a focal grid; kept the lowest cost %.6g of: %s", cost, "; ".join(attempts))
    return best


def _initial_parameters(
    data: Dict[str, _CameraData],
    obj: np.ndarray,
    image_size: Tuple[int, int],
    camera_names: Sequence[str],
    K_fixed: Optional[np.ndarray],
) -> np.ndarray:
    """The start vector: each camera's planar start (with K held at K_fixed when given), the
    board rotation of the camera with the most views, and the mean scale over the cameras."""
    x0_cameras, scales, board_rotations = [], [], []
    for cam in camera_names:
        params, scale, R_board = _initial_camera(data[cam], obj, image_size, K_fixed=K_fixed)
        x0_cameras.append(params)
        scales.append(scale)
        board_rotations.append((len(data[cam].indices), R_board))
    # The camera with the most views gives the board rotation; s estimates sigma in each camera.
    R_board0 = max(board_rotations, key=lambda item: item[0])[1]
    return np.concatenate(x0_cameras + [cv2.Rodrigues(R_board0)[0].ravel(), [float(np.mean(scales))]])


def _solve(
    x0: np.ndarray,
    obj: np.ndarray,
    data: Dict[str, _CameraData],
    camera_names: Sequence[str],
    image_size: Tuple[int, int],
) -> Tuple[Optional[MotorFrameFit], float, str]:
    """(fit, total cost, outcome) of the joint fit from x0; the fit is None and the cost inf
    when it did not converge, and the outcome says how it ended."""

    def residuals(x: np.ndarray) -> np.ndarray:
        return np.concatenate(_camera_residuals(x, obj, data, camera_names))

    # The Jacobian is by finite differences. With 14 parameters per camera plus 4 shared
    # (32 for a pair) and one projectPoints call per camera per evaluation, a fit takes
    # 1-2 s on christielab10, so an analytic or sparse Jacobian would buy nothing.
    result = least_squares(
        residuals,
        x0,
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=_MAX_EVALUATIONS,
    )
    # Status 0 is the evaluation limit, and below that bad input: no usable fit either way.
    if result.status <= 0:
        return None, math.inf, (
            f"least_squares status {result.status} after {result.nfev} evaluations ({result.message})"
        )
    x = result.x

    cameras, rms_px, per_view, param_std = {}, {}, {}, {}
    std = _parameter_std(result.jac, result.fun)
    for c, (cam, r) in enumerate(zip(camera_names, _camera_residuals(x, obj, data, camera_names))):
        params = x[c * _CAMERA_PARAMS:(c + 1) * _CAMERA_PARAMS]
        K, dist, rvec, tvec = _unpack_camera(params)
        cameras[cam] = CameraModel(K=K, dist=dist, rvec=rvec, tvec=tvec)
        sq = (r.reshape(-1, 2) ** 2).sum(axis=1).reshape(len(data[cam].indices), len(obj))
        rms_px[cam] = float(np.sqrt(sq.mean()))
        per_view[cam] = {i: float(np.sqrt(v)) for i, v in zip(data[cam].indices, sq.mean(axis=1))}
        block = std[c * _CAMERA_PARAMS:c * _CAMERA_PARAMS + len(_FITTED_INTRINSICS)]
        fitted = dict(zip(_FITTED_INTRINSICS, block))
        # k3 is held, not estimated, so it has no spread of its own.
        param_std[cam] = {name: float(fitted.get(name, 0.0)) for name in INTRINSIC_NAMES}

    fit = MotorFrameFit(
        cameras=cameras,
        board_rvec=x[-4:-1].copy(),
        board_tvec=np.zeros(3),
        motor_scale=float(x[-1]),
        image_size=(int(image_size[0]), int(image_size[1])),
        rms_px=rms_px,
        per_view_rms_px=per_view,
        param_std=param_std,
    )
    rms = ", ".join(f"{cam} {rms_px[cam]:.3g}" for cam in camera_names)
    return fit, float(result.cost), f"converged in {result.nfev} evaluations, cost {result.cost:.6g}, rms px {rms}"


def reconcile_corner_order(
    observations: Sequence[WaypointObservation],
    board: BoardSpec,
    image_size: Tuple[int, int],
    camera_names: Sequence[str] = ("left", "right"),
) -> Tuple[List[WaypointObservation], Dict[str, bool]]:
    """Put every camera's corners in the board order of the camera with the most views.

    The 6x8 board looks the same turned 180 degrees about its normal, so two cameras can
    each be consistent across their own waypoints yet label the board from opposite ends.
    A camera labelling it from the other end sees the board turned 180 degrees about the
    board normal, so the board's rotation in the motor frame from that camera is 180
    degrees from the reference camera's. Every corner array of a camera more than 90
    degrees away is reversed.

    Each camera's board rotation comes from a converged motor-frame fit of that camera
    alone (fit_motor_frame with its starts), not from the planar start: that start was up
    to 77 degrees off on the pipeline's synthetic capture, too close to the 90-degree
    threshold to decide by. An angle between 60 and 120 degrees is logged as ambiguous.

    Returns new observations (the inputs are not changed) and, per camera, whether its
    corners were reversed. Raises ValueError as fit_motor_frame does for a camera whose
    waypoints cannot place it, and RuntimeError when a camera's fit converges from no start.
    """
    obj = board.object_points()
    data = {cam: _camera_data(observations, cam, len(obj)) for cam in camera_names}
    board_rotation = {
        cam: cv2.Rodrigues(np.asarray(fit_motor_frame(observations, board, image_size, (cam,)).board_rvec, float))[0]
        for cam in camera_names
    }
    reference = max(camera_names, key=lambda cam: len(data[cam].indices))
    flipped = {}
    for cam in camera_names:
        relative = board_rotation[reference].T @ board_rotation[cam]
        angle = math.degrees(float(np.linalg.norm(cv2.Rodrigues(relative)[0])))
        low, high = _AMBIGUOUS_FLIP_DEG
        if low < angle < high:
            logger.warning(
                "%s: its board rotation is %.1f degrees from %s's; between %g and %g degrees,"
                " whether its corners are reversed is ambiguous", cam, angle, reference, low, high,
            )
        flipped[cam] = angle > 90.0
    reconciled = [
        WaypointObservation(
            index=o.index,
            motor_mm=np.array(o.motor_mm, dtype=float),
            corners={
                cam: np.array(np.asarray(c)[::-1] if flipped.get(cam, False) else c, dtype=float)
                for cam, c in o.corners.items()
            },
        )
        for o in observations
    ]
    return reconciled, flipped


def _camera_data(observations: Sequence[WaypointObservation], cam: str, n_corners: int) -> _CameraData:
    seen = [o for o in observations if cam in o.corners]
    for o in seen:
        if np.shape(o.corners[cam]) != (n_corners, 2):
            raise ValueError(
                f"{cam}: waypoint {o.index} has corners of shape {np.shape(o.corners[cam])}, expected ({n_corners}, 2)"
            )
    motor = np.array([np.asarray(o.motor_mm, float).reshape(3) for o in seen]).reshape(-1, 3)
    # The motor axes come from how the board moved in this camera; motion along a single
    # line leaves the rotation about that line undetermined.
    if len(seen) < 3 or np.linalg.matrix_rank(motor - motor.mean(axis=0), tol=1e-6) < 2:
        raise ValueError(
            f"{cam}: the board was seen at {len(seen)} waypoint(s), which do not span two motor axes;"
            " the camera's pose in the motor frame is undetermined"
        )
    corners = np.concatenate([np.asarray(o.corners[cam], float) for o in seen])
    return _CameraData(indices=[o.index for o in seen], motor_mm=motor, corners=corners)


def _initial_camera(
    d: _CameraData, obj: np.ndarray, image_size, K_fixed: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """(camera parameter block, scale s, board rotation in the motor frame) for one camera.

    With K_fixed the planar calibration holds K there and fits only k1 and the views' poses;
    if it cannot, the distortion is zero and each view's pose comes from solvePnP.
    """
    n = len(d.indices)
    image_points = [c.reshape(-1, 1, 2).astype(np.float32) for c in d.corners.reshape(n, -1, 2)]
    object_points = [obj.astype(np.float32)] * n
    size = (int(image_size[0]), int(image_size[1]))
    if K_fixed is None:
        _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, size, None, None, flags=_INITIAL_CALIBRATION_FLAGS
        )
    else:
        held = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
        try:
            _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, size, np.array(K_fixed, float), np.zeros(5),
                flags=_INITIAL_CALIBRATION_FLAGS | held,
            )
        except cv2.error:
            K, dist = np.array(K_fixed, float), np.zeros(5)
            poses = [cv2.solvePnP(object_points[0], p, K, dist)[1:] for p in image_points]
            rvecs, tvecs = [r for r, _ in poses], [t for _, t in poses]
    R_views = np.array([cv2.Rodrigues(r)[0] for r in rvecs])  # board -> camera, per view
    t_views = np.array([t.ravel() for t in tvecs])  # board origin in the camera, per view

    m_mean, t_mean = d.motor_mm.mean(axis=0), t_views.mean(axis=0)
    R_cm, s = _scaled_rotation(d.motor_mm - m_mean, t_views - t_mean)
    tvec = t_mean - s * R_cm @ m_mean
    R_board = R_cm.T @ _mean_rotation(R_views)

    k1, k2, p1, p2 = np.asarray(dist, float).ravel()[:4]
    params = np.concatenate([
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2, p1, p2], cv2.Rodrigues(R_cm)[0].ravel(), tvec,
    ])
    return params, float(s), R_board


def _scaled_rotation(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """R, s minimising sum |b_i - s R a_i|^2 over centred rows a_i, b_i (Umeyama 1991)."""
    U, D, Vt = np.linalg.svd(b.T @ a)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ S @ Vt, float(np.trace(np.diag(D) @ S) / np.sum(a * a))


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """The rotation closest, in the Frobenius norm, to the mean of `rotations`."""
    U, _, Vt = np.linalg.svd(rotations.sum(axis=0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _parameters_of(fit: MotorFrameFit, camera_names: Sequence[str]) -> np.ndarray:
    """`fit` as the parameter vector fit_motor_frame solves, the inverse of _unpack_camera
    per camera; k3 is not a parameter, since the fit holds it at 0."""
    blocks = []
    for cam in camera_names:
        m = fit.cameras[cam]
        K = np.asarray(m.K, float)
        k1, k2, p1, p2 = np.asarray(m.dist, float).ravel()[:4]
        blocks.append(np.concatenate([
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2, p1, p2],
            np.asarray(m.rvec, float).ravel(),
            np.asarray(m.tvec, float).ravel(),
        ]))
    return np.concatenate(blocks + [np.asarray(fit.board_rvec, float).ravel(), [float(fit.motor_scale)]])


def _unpack_camera(p: np.ndarray):
    """(K, dist, rvec, tvec) from one camera's block; dist is cv2's five, with k3 = 0."""
    fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, np.array([k1, k2, p1, p2, 0.0]), p[8:11].copy(), p[11:14].copy()


def _camera_residuals(x: np.ndarray, obj: np.ndarray, data: Dict[str, _CameraData], camera_names) -> List[np.ndarray]:
    R_board = cv2.Rodrigues(x[-4:-1])[0]
    sigma = x[-1]
    board_points = obj @ R_board.T
    out = []
    for c, cam in enumerate(camera_names):
        d = data[cam]
        K, dist, rvec, tvec = _unpack_camera(x[c * _CAMERA_PARAMS:(c + 1) * _CAMERA_PARAMS])
        X = (board_points[None, :, :] + sigma * d.motor_mm[:, None, :]).reshape(-1, 3)
        projected = cv2.projectPoints(X, rvec, tvec, K, dist)[0].reshape(-1, 2)
        out.append((projected - d.corners).ravel())
    return out


def _parameter_std(J: np.ndarray, r: np.ndarray) -> np.ndarray:
    """sqrt(diag(inv(J^T J)) * residual variance) per parameter, or inf where the
    capture does not determine the parameter.

    The columns are normalised first because fx (~2000) and p1 (~1e-4) differ by orders
    of magnitude. The inverse is taken through the eigenvectors of the normalised J^T J,
    over the determined directions only, so no variance can come out negative or be
    clipped to a 0 that would read as "perfectly determined". A parameter is reported
    as inf when the undetermined directions outweigh the rest of its variance even at
    the largest eigenvalue still counted as undetermined; the true weight is larger.
    """
    dof = max(len(r) - J.shape[1], 1)
    variance = float(r @ r) / dof
    std = np.full(J.shape[1], np.inf)
    norms = np.linalg.norm(J, axis=0)
    used = norms > 0  # a parameter that moves no residual is undetermined outright
    w, V = np.linalg.eigh((J[:, used] / norms[used]).T @ (J[:, used] / norms[used]))
    if w[-1] <= 0:
        return std
    floor = w[-1] / _MAX_CONDITION
    weak = w <= floor
    determined_var = (V[:, ~weak] ** 2 / w[~weak]).sum(axis=1)
    undetermined = (V[:, weak] ** 2).sum(axis=1) / floor > determined_var
    s = np.sqrt(determined_var * variance) / norms[used]
    s[undetermined] = np.inf
    std[used] = s
    return std
