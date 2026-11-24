"""A stand-in base preset and motor-frame fit, shared by the multi-resolution calibration tests.

A module of its own so that test files share them without importing each other.
"""

import cv2
import numpy as np

from reachaq.inference.multires_calibration.fit import INTRINSIC_NAMES, CameraModel, MotorFrameFit
from reachaq.inference.multires_calibration.geometry import PresetGeometry


BASE = PresetGeometry(4, 256, 256, 52, 6)


def _fit() -> MotorFrameFit:
    """A plausible motor-frame fit in 1440x1080 sensor px: two cameras 40 degrees apart,
    150 mm from the motor origin and looking at it."""
    cameras = {}
    for cam, side in (("left", -1.0), ("right", 1.0)):
        a = np.radians(20.0)
        eye = 150.0 * np.array([side * np.sin(a), 0.0, -np.cos(a)])
        z = -eye / np.linalg.norm(eye)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])  # rows are the camera axes in the motor frame
        cameras[cam] = CameraModel(
            K=np.array([[2300.0, 0.0, 719.5], [0.0, 2300.0, 539.5], [0.0, 0.0, 1.0]]),
            dist=np.zeros(5),
            rvec=cv2.Rodrigues(R)[0].ravel(),
            tvec=-R @ eye,
        )
    return MotorFrameFit(
        cameras=cameras,
        board_rvec=np.zeros(3),
        board_tvec=np.zeros(3),
        motor_scale=1.0,
        image_size=(1440, 1080),
        rms_px={cam: 0.1 for cam in cameras},
        per_view_rms_px={cam: {0: 0.1} for cam in cameras},
        param_std={cam: {name: 0.0 for name in INTRINSIC_NAMES} for cam in cameras},
    )


FIT = _fit()
