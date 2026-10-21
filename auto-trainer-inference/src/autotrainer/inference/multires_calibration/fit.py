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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

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

    `initial`, a fit whose cameras are in this image's pixels, starts the fit in place of
    the planar initialisation. That start is degenerate in focal length (see
    _INITIAL_CALIBRATION_FLAGS): on the pipeline's synthetic capture it put fx at 3 to 13
    times the truth and the board rotation up to 77 degrees off, and from one such start
    the fit of the 512 preset's own detections ran out of evaluations. The pipeline's
    per-preset cross-check therefore starts from the sensor fit mapped to the preset. The
    result is the least-squares optimum of these observations either way; only the
    starting point differs.
    """
    obj = board.object_points()
    data = {cam: _camera_data(observations, cam, len(obj)) for cam in camera_names}

    if initial is not None:
        x0 = _parameters_of(initial, camera_names)
    else:
        # Initialisation: planar calibration per camera, then the motor axes from how the
        # board origin moved in each camera (scaled Kabsch/Umeyama).
        x0_cameras, scales, board_rotations = [], [], []
        for cam in camera_names:
            params, scale, R_board = _initial_camera(data[cam], obj, image_size)
            x0_cameras.append(params)
            scales.append(scale)
            board_rotations.append((len(data[cam].indices), R_board))
        # The camera with the most views gives the board rotation; s estimates sigma in each camera.
        R_board0 = max(board_rotations, key=lambda item: item[0])[1]
        x0 = np.concatenate(x0_cameras + [cv2.Rodrigues(R_board0)[0].ravel(), [float(np.mean(scales))]])

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
        raise RuntimeError(
            f"the motor-frame fit did not converge: least_squares status {result.status}"
            f" after {result.nfev} evaluations ({result.message})"
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

    return MotorFrameFit(
        cameras=cameras,
        board_rvec=x[-4:-1].copy(),
        board_tvec=np.zeros(3),
        motor_scale=float(x[-1]),
        image_size=(int(image_size[0]), int(image_size[1])),
        rms_px=rms_px,
        per_view_rms_px=per_view,
        param_std=param_std,
    )


def reconcile_corner_order(
    observations: Sequence[WaypointObservation],
    board: BoardSpec,
    image_size: Tuple[int, int],
    camera_names: Sequence[str] = ("left", "right"),
) -> Tuple[List[WaypointObservation], Dict[str, bool]]:
    """Put every camera's corners in the board order of the camera with the most views.

    The 6x8 board looks the same turned 180 degrees about its normal, so two cameras can
    each be consistent across their own waypoints yet label the board from opposite ends.
    A camera labelling it from the other end sees, in its planar calibration, the board
    turned 180 degrees about the board normal, so the board's rotation in the motor frame
    from that camera (the initialisation fit_motor_frame starts from) is 180 degrees from
    the reference camera's. Every corner array of a camera more than 90 degrees away is
    reversed.

    Returns new observations (the inputs are not changed) and, per camera, whether its
    corners were reversed. Raises ValueError as fit_motor_frame does for a camera whose
    waypoints cannot place it.
    """
    obj = board.object_points()
    data = {cam: _camera_data(observations, cam, len(obj)) for cam in camera_names}
    board_rotation = {cam: _initial_camera(data[cam], obj, image_size)[2] for cam in camera_names}
    reference = max(camera_names, key=lambda cam: len(data[cam].indices))
    # The angle between two rotations exceeds 90 degrees when trace(R_a^T R_b) < 1.
    flipped = {
        cam: bool(np.trace(board_rotation[reference].T @ board_rotation[cam]) < 1.0) for cam in camera_names
    }
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


def _initial_camera(d: _CameraData, obj: np.ndarray, image_size) -> Tuple[np.ndarray, float, np.ndarray]:
    """(camera parameter block, scale s, board rotation in the motor frame) for one camera."""
    n = len(d.indices)
    image_points = [c.reshape(-1, 1, 2).astype(np.float32) for c in d.corners.reshape(n, -1, 2)]
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj.astype(np.float32)] * n,
        image_points,
        (int(image_size[0]), int(image_size[1])),
        None,
        None,
        flags=_INITIAL_CALIBRATION_FLAGS,
    )
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
