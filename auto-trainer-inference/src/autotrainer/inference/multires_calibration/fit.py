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

tvec_b is held at zero rather than fitted. Moving the board origin by d and every
camera by -R_c @ d changes no projection, so the motor frame's origin is a choice,
not a measurement: it is the board's corner 0 at motor position 0. Fitting it too
would leave J^T J singular and the parameter uncertainties undefined.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

# The order of the intrinsic parameters in param_std, and in each camera's block of
# the parameter vector, matching cv2's distortion order (k1, k2, p1, p2, k3).
INTRINSIC_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
# Per camera: the nine intrinsics, then rvec (3) and tvec (3).
_CAMERA_PARAMS = len(INTRINSIC_NAMES) + 6
# The planar calibration that starts the fit sees only parallel planes, so it is
# degenerate in focal length, and free distortion terms let it wander further. With
# all five free, its fx on the synthetic rig ranged from 0.6x to 5.5x the truth
# (k2 down to -2800), and from such a start the joint fit diverged on 1 seed in 30.
# Fitting k1 alone kept all 30 in the basin. The joint fit then frees all nine.
_INITIAL_CALIBRATION_FLAGS = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST


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
) -> MotorFrameFit:
    """Fit every camera's intrinsics and motor-frame pose, the board rotation and the motor scale.

    A waypoint seen by only one camera still constrains that camera: each camera's
    residuals come from exactly the waypoints it observed.
    """
    obj = board.object_points()
    data = {cam: _camera_data(observations, cam, len(obj)) for cam in camera_names}

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

    # The Jacobian is by finite differences. With 15 parameters per camera plus 4 shared
    # (34 for a pair) and one projectPoints call per camera per evaluation, a fit takes
    # 1-2 s on christielab10, so an analytic or sparse Jacobian would buy nothing.
    result = least_squares(
        residuals, x0, method="trf", x_scale="jac", loss="linear", ftol=1e-10, xtol=1e-10, gtol=1e-10
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
        block = std[c * _CAMERA_PARAMS:c * _CAMERA_PARAMS + len(INTRINSIC_NAMES)]
        param_std[cam] = {name: float(v) for name, v in zip(INTRINSIC_NAMES, block)}

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

    dist = np.asarray(dist, float).ravel()[:5]
    params = np.concatenate([
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dist, cv2.Rodrigues(R_cm)[0].ravel(), tvec,
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


def _unpack_camera(p: np.ndarray):
    fx, fy, cx, cy = p[:4]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, p[4:9].copy(), p[9:12].copy(), p[12:15].copy()


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
    """sqrt(diag(inv(J^T J)) * residual variance), per parameter.

    The columns are normalised before the inverse because fx (~2000) and k3 differ by
    orders of magnitude, which would otherwise cost the inverse most of its precision.
    """
    dof = max(len(r) - J.shape[1], 1)
    variance = float(r @ r) / dof
    norms = np.linalg.norm(J, axis=0)
    norms[norms == 0] = 1.0
    Js = J / norms
    try:
        cov = np.linalg.inv(Js.T @ Js) / np.outer(norms, norms)
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.inf)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None) * variance)
