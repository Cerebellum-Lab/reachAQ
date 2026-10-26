"""What each preset gets from a motor-frame fit: stereo parameters, the 3D frame and an anipose export.

The fit (fit.py) is in full-sensor pixels with X_cam = R(rvec) @ X_motor + tvec in mm.
Each preset gets:

- stereo parameters with today's keys, so the existing loaders, the live path and
  DeepLabCut read them unchanged. The intrinsics are mapped to the preset's pixels;
  R and T are the fitted extrinsics; rectification is recomputed at the preset's
  (width, height). rot_cor is 0.0 because new calibrations rotate no pixels.
- the 3D frame: the matrix M in the live path's `oriented = (raw - offsets) @ M`.
  It replaces the reorientation from hand-entered camera angles and the square-size
  scaling. T is in mm, so the triangulated points already are.
- an anipose calibration.toml with the cameras' motor-frame poses.
"""

import json
import math
from typing import Tuple

import cv2
import numpy as np

from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.inference.config import StereoParams
from autotrainer.inference.multires_calibration.fit import MotorFrameFit
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, map_intrinsics

# DeepLabCut keys the pair "{cam1}-{cam2}" by the project's camera names.
STEREO_KEY = "left-right"
_CAMERAS = ("left", "right")


def preset_stereo_params(fit: MotorFrameFit, g: PresetGeometry, *, alpha: float = 1.0) -> StereoParams:
    """Stereo parameters of preset `g`, with the keys and array shapes of today's stereo_params.pickle."""
    K1, K2 = (map_intrinsics(fit.cameras[cam].K, g) for cam in _CAMERAS)
    # (1, 5) and (3, 1), the shapes stereoCalibrate returns and today's pickles hold.
    d1, d2 = (np.asarray(fit.cameras[cam].dist, float).reshape(1, -1) for cam in _CAMERAS)
    R, T = fit.stereo(*_CAMERAS)
    T = T.reshape(3, 1)
    # cv2 wants the image size as (width, height).
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, d1, K2, d2, (g.width, g.height), R, T, alpha=alpha)
    # What stereoCalibrate would return for these K, R and T: x2^T F x1 = 0 on undistorted
    # pixels, with F scaled to F[2, 2] = 1 as OpenCV does.
    E = _cross_matrix(T.ravel()) @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    if abs(F[2, 2]) > 0:
        F = F / F[2, 2]
    matrix = {
        "cameraMatrix1": K1,
        "cameraMatrix2": K2,
        "distCoeffs1": d1,
        "distCoeffs2": d2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "roi1": roi1,
        "roi2": roi2,
        "Q": Q,
        "image_shape": [(g.width, g.height), (g.width, g.height)],
        "rot_cor": 0.0,
    }
    return StereoParams(key_name=STEREO_KEY, matrix=matrix)


def motor_to_inference_signs() -> Tuple[int, int, int]:
    """The sign of each oriented (inference) axis relative to the same motor axis, read from the DCS chain.

    The oriented axes are the DCS axes (flips_inference_diamond is (1, 1, 1)), so a part
    the motor carries must move in oriented coordinates as motor_to_diamond says it does.
    The production consumer depends on exactly that: drift monitoring computes
    diamond_to_motor(offset) - motor_position, with offset = triangle - diamond in
    oriented coordinates, and the drift is zero only when the triangle's oriented
    displacement is motor_to_diamond's response to the motor's move.

    The signs are measured from motor_to_diamond, not copied from its flips, so they
    follow any change there. A map that is not a per-axis sign flip raises ValueError: no
    diagonal of signs could represent it. DiamondTriangleOffsetConfig's motor_to_inference
    and inference_to_motor are not used: they are test-only helpers that relate the motor
    to a diamond - triangle offset, and they give the opposite signs.
    """
    zero = Offset3DTuple(0.0, 0.0, 0.0)
    dcs = DiamondTriangleOffsetConfig(used_position=zero, measured_offset=zero)
    origin = np.array(dcs.motor_to_diamond(zero), float)
    # Column k is the DCS displacement of a unit motor step along axis k.
    J = np.column_stack([
        np.array(dcs.motor_to_diamond(Offset3DTuple(*np.eye(3)[k])), float) - origin for k in range(3)
    ])
    signs = np.diag(J)
    if not (np.allclose(J, np.diag(signs)) and np.allclose(np.abs(signs), 1.0)):
        raise ValueError(f"the DCS motor-to-diamond map is not a per-axis sign flip:\n{J}")
    return tuple(int(round(s)) for s in signs)


def frame_transform(fit: MotorFrameFit, stereo: StereoParams, signs: Tuple[int, int, int]) -> np.ndarray:
    """3x3 M such that (raw - offsets) @ M is the motor-axis displacement in mm, times `signs`.

    raw is what the live path triangulates with P1 and P2: rectified left-camera
    coordinates, X_rect = R1 @ X_cam, with X_cam = R_L @ X_motor + t_L. A displacement
    therefore maps back as dX_motor = R_L^T @ R1^T @ dX_rect, and the row-vector form
    of diag(signs) @ dX_motor is dX_rect @ M with M = (diag(signs) @ R_L^T @ R1^T)^T.
    The units stay mm: T is in mm, so no square-size factor applies.
    """
    R_left = cv2.Rodrigues(np.asarray(fit.cameras["left"].rvec, float))[0]
    R1 = np.asarray(stereo.matrix["R1"], float)
    return (np.diag(np.asarray(signs, float)) @ R_left.T @ R1.T).T


def anipose_toml(fit: MotorFrameFit, g: PresetGeometry) -> str:
    """An anipose calibration.toml for preset `g`, in the layout aniposelib's CameraGroup.dump writes.

    One table per camera, cam_0 (left) and cam_1 (right), holding its name, size [w, h],
    the preset's intrinsic matrix, the 5 distortion coefficients and the motor-frame
    pose: rotation is the Rodrigues vector of the motor-frame -> camera rotation and
    translation is in mm, as aniposelib's Camera projects with cv2.projectPoints.
    metadata.error is the mean of the fit's per-camera RMS in the preset's px (sensor px /
    binning), like every other pixel value in the file.

    Raises ValueError, naming the camera, for a camera parameter that is not finite: TOML
    can spell nan and inf, and anipose would read them back as a calibration.
    """
    lines = []
    for i, cam in enumerate(_CAMERAS):
        m = fit.cameras[cam]
        try:
            lines += [
                f"[cam_{i}]",
                f"name = {_toml_value(cam)}",
                f"size = {_toml_value([g.width, g.height])}",
                f"matrix = {_toml_value(map_intrinsics(m.K, g))}",
                f"distortions = {_toml_value(np.asarray(m.dist, float).ravel())}",
                f"rotation = {_toml_value(np.asarray(m.rvec, float).ravel())}",
                f"translation = {_toml_value(np.asarray(m.tvec, float).ravel())}",
                "",
            ]
        except ValueError as error:
            raise ValueError(f"anipose export of preset {g.name}, camera {cam}: {error}") from error
    error = float(np.mean([fit.rms_px[cam] for cam in _CAMERAS])) / g.binning
    lines += ["[metadata]", f"adjusted = {_toml_value(False)}", f"error = {_toml_value(error)}"]
    return "\n".join(lines) + "\n"


def _cross_matrix(t: np.ndarray) -> np.ndarray:
    """[t]x, the matrix with [t]x @ v = t x v."""
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def _toml_value(value) -> str:
    """A TOML literal for the few types the export writes; floats at repr precision so they read back exactly.

    Raises ValueError for a float that is not finite.
    """
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError(f"{float(value)!r} is not a finite calibration value")
        # repr gives the shortest exact form.
        return repr(float(value))
    if isinstance(value, str):
        # A JSON string is a valid TOML basic string for the camera names written here.
        return json.dumps(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"no TOML form for {type(value).__name__}")
