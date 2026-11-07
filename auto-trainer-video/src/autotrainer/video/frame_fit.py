"""Fit camera frames to a smaller consumer queue by integer area averaging.

A camera can capture at less binning than its consumers (live inference, the
display) were sized for. The consumers then receive each frame averaged down
by a whole factor k: one k x k block of camera pixels per queue pixel. Over the
same sensor area that lands on the same pixel grid as a capture at k times the
binning, so models and calibrations made at that binning still apply.
"""
from typing import Optional, Sequence, Tuple

import cv2
import numpy


def fit_factor(frame_shape: Sequence[int], target_shape: Optional[Sequence[int]]) -> int:
    """How many times smaller ``target_shape`` is than ``frame_shape`` in each direction.

    1 when there is no target or it is the same size. Anything that is not the
    frame divided by one integer in both directions raises: it would resample
    rather than average whole blocks, and silently change what consumers see.
    """
    rows, cols = int(frame_shape[0]), int(frame_shape[1])
    if target_shape is None:
        return 1
    t_rows, t_cols = int(target_shape[0]), int(target_shape[1])
    if (t_rows, t_cols) == (rows, cols):
        return 1
    if (
        t_rows <= 0 or t_cols <= 0
        or rows % t_rows or cols % t_cols
        or rows // t_rows != cols // t_cols
    ):
        raise ValueError(
            f"queue shape {t_rows}x{t_cols} is not the camera frame {rows}x{cols} "
            "divided by one integer factor")
    return rows // t_rows


class QueueFit:
    """Area-averages each camera frame down to one queue's shape."""

    def __init__(self, frame_shape: Sequence[int], target_shape: Optional[Sequence[int]]):
        self.factor = fit_factor(frame_shape, target_shape)
        source = frame_shape if self.factor == 1 else target_shape
        self.shape: Tuple[int, int] = (int(source[0]), int(source[1]))

    def __call__(self, frame: numpy.ndarray) -> numpy.ndarray:
        if self.factor == 1:
            return frame
        # INTER_AREA at an integer ratio is a plain block mean, and it is ~170x
        # faster than the numpy equivalent (0.014 ms vs 2.4 ms at 512x512 on
        # christielab10). It releases the GIL while it runs.
        return cv2.resize(frame, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
