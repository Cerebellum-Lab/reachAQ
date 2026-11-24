"""Geometry of the camera presets and the mapping from the sensor to each preset.

The calibration is fitted once in full-sensor pixels and the intrinsics of every
preset are derived from it. A preset is a binning factor plus a rectangular ROI
cut from the binned sensor, so the mapping is a scale and a shift.

Pixel-centre convention: sensor pixel i is centred on coordinate i. A binned
pixel j (counted from the ROI corner) averages sensor pixels
b*(j + offset) .. b*(j + offset) + b - 1, so its centre is at
b*(j + offset) + (b - 1)/2 and

    u_preset = (u_sensor + 0.5) / b - 0.5 - offset

Treating the binned pixel as a point at b*(j + offset) instead would leave an
error of (b - 1)/2 sensor px, which is 1.5 sensor px at bin 4.

The distortion coefficients act on normalised image coordinates, before the
intrinsic matrix, so binning and cropping do not change them.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PresetGeometry:
    """One camera preset. The offsets are the ROI corner in binned pixels."""

    binning: int
    width: int
    height: int
    offset_x: int
    offset_y: int

    @property
    def name(self) -> str:
        return f"{self.width}x{self.height}_bin{self.binning}"

    def sensor_roi(self) -> Tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the ROI in sensor pixels, as pixel edges.

        Edge coordinates have pixel i spanning [i, i+1), half a pixel away from the
        centre coordinates the point mapping uses. Subtract 0.5 to compare the two.
        """
        b = self.binning
        return (
            float(self.offset_x * b),
            float(self.offset_y * b),
            float((self.offset_x + self.width) * b),
            float((self.offset_y + self.height) * b),
        )


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {pts.shape}")
    return pts


def _centre_shift(g: PresetGeometry) -> np.ndarray:
    """The [x, y] shift in u_preset = u_sensor / b + shift."""
    return np.array([0.5 / g.binning - 0.5 - g.offset_x, 0.5 / g.binning - 0.5 - g.offset_y])


def sensor_to_preset_points(points: np.ndarray, g: PresetGeometry) -> np.ndarray:
    """Map (N, 2) sensor pixel positions to pixel positions in preset `g`."""
    return _as_points(points) / g.binning + _centre_shift(g)


def preset_to_sensor_points(points: np.ndarray, g: PresetGeometry) -> np.ndarray:
    """Map (N, 2) pixel positions in preset `g` back to sensor pixels."""
    return (_as_points(points) - _centre_shift(g)) * g.binning


def map_intrinsics(K_sensor: np.ndarray, g: PresetGeometry) -> np.ndarray:
    """The 3x3 intrinsic matrix of preset `g` from the sensor's.

    Left-multiplying by the affine map of the pixel coordinates scales fx, fy and
    any skew by 1/b and moves the principal point exactly as sensor_to_preset_points
    moves a pixel. The distortion coefficients stay as they are.
    """
    shift_x, shift_y = _centre_shift(g)
    to_preset = np.array([
        [1.0 / g.binning, 0.0, shift_x],
        [0.0, 1.0 / g.binning, shift_y],
        [0.0, 0.0, 1.0],
    ])
    return to_preset @ np.asarray(K_sensor, dtype=float)


def presets_from_base(base: PresetGeometry, binnings: Sequence[int] = (4, 2, 1)) -> List[PresetGeometry]:
    """The presets at each binning in `binnings`, covering the same sensor window as `base`.

    A preset at binning b has k = base.binning // b times as many pixels across each
    side and k times the offsets. Each b must divide base.binning so the ROI edges
    stay on whole pixels of both presets.
    """
    presets = []
    for b in binnings:
        if b <= 0 or base.binning % b != 0:
            raise ValueError(f"binning {b} does not divide the base binning {base.binning}")
        k = base.binning // b
        presets.append(PresetGeometry(b, base.width * k, base.height * k, base.offset_x * k, base.offset_y * k))
    return presets



@dataclass(frozen=True)
class PresetSet:
    """One geometry per camera, for the cameras recorded together: `name` is "live" for the
    cameras' live geometries and f"bin{b}" for the set derived at binning b from each camera's
    live window. It names the set's folder in a calibration (presets/<name>/)."""

    name: str
    cameras: Mapping[str, PresetGeometry]


def rebinned(g: PresetGeometry, binning: int) -> Optional[PresetGeometry]:
    """`g`'s sensor window at `binning`, or None when the window's edges do not fall on whole
    pixels at that binning."""
    if binning <= 0:
        raise ValueError(f"binning must be positive, got {binning}")
    x0, y0, x1, y1 = (int(v) for v in g.sensor_roi())
    if any(v % binning for v in (x0, y0, x1, y1)):
        return None
    return PresetGeometry(binning, (x1 - x0) // binning, (y1 - y0) // binning, x0 // binning, y0 // binning)


def presets_for_cameras(live: Mapping[str, PresetGeometry], binnings: Sequence[int] = (1, 2, 4)) -> List[PresetSet]:
    """The "live" set, then a f"bin{b}" set for each b in `binnings`: every camera's own live
    window at binning b.

    A binning is left out when some camera's window does not fall on whole pixels at it, and
    when its set is the live set again (every camera already records at b). Any geometry left
    out can still be derived on demand (store.load_preset_set).
    """
    if not live:
        raise ValueError("no cameras to make presets for")
    live = dict(live)
    sets = [PresetSet("live", live)]
    for b in binnings:
        cameras: Dict[str, PresetGeometry] = {}
        for cam, g in live.items():
            at_b = rebinned(g, b)
            if at_b is None:
                break
            cameras[cam] = at_b
        else:
            if cameras != live:
                sets.append(PresetSet(f"bin{b}", cameras))
    return sets
