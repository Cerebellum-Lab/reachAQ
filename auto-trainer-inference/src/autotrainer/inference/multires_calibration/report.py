"""The quality report of a multi-resolution 3D calibration.

Every calibration session grades its fit against fixed thresholds and records the
thresholds with the grades, so a bad month is caught and months can be compared even
after the thresholds are tuned. The report is written as JSON for machines and as text
for the operator.

Thresholds are (warn, fail) pairs; None never triggers. For most metrics a larger value
is worse: the status is "fail" when value > fail, else "warn" when value > warn. For the
metrics in LOWER_IS_WORSE a smaller value is worse: "fail" when value < fail, else "warn"
when value < warn. Every Metric records which way it grades (lower_is_worse), so the
JSON reads on its own and `thresholds` keeps the plain (warn, fail) shape.

A value that cannot be determined (a parameter the capture does not fix, a stereo check
with no waypoint seen by both cameras) is +inf with a note, never NaN, and takes the worst
status its thresholds allow: an undetermined quantity cannot pass a check. JSON has no
infinity, so non-finite floats are written as the strings "inf", "-inf" and "nan", which
keeps the file strict JSON, and from_json turns them back into floats.

Pixel conventions are geometry.py's: corners and principal points are pixel-centre
coordinates (pixel i is centred on i), and roi_sensor is in pixel edges, as
PresetGeometry.sensor_roi() returns it, so a pixel-centre coordinate u sits at edge
coordinate u + 0.5.

The change since the previous calibration has to respect the fit's gauge. The motor
frame's origin is the board's corner 0 at motor position 0 (fit.py), a choice rather than
a measurement, and the board is remounted on the stage every month. Between calibrations
the origin therefore moves by an unknown vector, and every camera centre (-R^T t) shifts
with it, so comparing each camera's centre on its own would report "moved" every month.
The report compares the baseline instead, the vector from the left camera's centre to
the right's, which a common shift leaves unchanged: baseline_change_mm catches one camera
moving relative to the other. A shift of both cameras together by the same vector cannot
be told from an origin shift and is not visible here; the DCS calibration, redone after
every 3D calibration, measures the origin. Camera rotations are relative to the motor
axes, which are physical, so camera_rotation_change_deg compares each camera on its own.
"""

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autotrainer.inference.multires_calibration.fit import (
    BoardSpec,
    CameraModel,
    MotorFrameFit,
    WaypointObservation,
)
from autotrainer.inference.multires_calibration.geometry import PresetGeometry

Thresholds = Dict[str, Tuple[Optional[float], Optional[float]]]

# The initial (warn, fail) limits, recorded verbatim in every report. Tune them on real
# captures; a report always carries the limits it was graded with.
DEFAULT_THRESHOLDS: Thresholds = {
    "views_kept": (20.0, 10.0),
    "reprojection_rms_px": (0.5, 1.0),
    "coverage_roi_fraction": (0.5, 0.25),
    "fx_fy_ratio_dev": (0.01, 0.03),
    "focal_rel_std": (0.005, 0.02),
    "focal_left_right_rel": (0.10, None),
    "principal_point_offset_frac": (0.2, 0.4),
    "max_distortion_shift_px": (60.0, 150.0),
    "epipolar_error_px": (0.5, 1.0),
    "square_size_rel_error": (0.01, 0.03),
    "motor_scale_dev": (0.01, None),
    "waypoints_unverified": (0.0, None),
    # Informational: a preset's direct fit has a noise floor near 0.5 preset px in K even on
    # perfect synthetic frames (0.33 and 0.87 px measured), so it cannot fail a calibration.
    "cross_check_delta_px": (1.5, None),
    # The pixel-centre mapping itself, which the spec holds to 0.1 px: on the synthetic
    # capture the mean offset is about 0.001 preset px, and the pixel-edge convention, off by
    # (b - 1) / 2b on each axis (0.53 px in all at bin 4), fails it.
    "cross_check_corner_offset_px": (0.1, 0.25),
    "camera_rotation_change_deg": (0.5, None),
    "baseline_change_mm": (1.0, None),
    "focal_change_rel": (0.02, None),
}
# The metrics where a smaller value is worse; see the module docstring.
LOWER_IS_WORSE = frozenset({"views_kept", "coverage_roi_fraction"})
NO_PREVIOUS_NOTE = "no comparable previous calibration"
MOVED_NOTE = "camera moved since the previous calibration"

_STATUS_ORDER = ("fail", "warn", "pass")
# The coverage mask is rasterised at 1/4 of sensor resolution: one mask pixel is 16 sensor
# px^2, and the rasteriser's boundary error of up to one mask pixel per edge is under 1 %
# of a 1024 px ROI.
_COVERAGE_SCALE = 4
# cv2.fillPoly takes fixed-point vertices with this many fractional bits.
_FILL_SHIFT = 4
_DISTORTION_SAMPLES_PER_EDGE = 40
# Undistortion is iterative and cv2's default stops after 5 iterations; iterate to
# convergence so the stereo checks measure the calibration, not the undistortion.
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
_NON_FINITE = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


@dataclass
class Metric:
    name: str
    camera: Optional[str]
    value: float
    unit: str
    warn: Optional[float]
    fail: Optional[float]
    status: str  # "pass", "warn" or "fail"
    note: str = ""
    lower_is_worse: bool = False


@dataclass
class QualityReport:
    created: str  # ISO 8601, local time with its UTC offset
    verdict: str  # the worst metric status
    metrics: List[Metric]
    thresholds: Thresholds  # name -> (warn, fail), the limits the metrics were graded with
    notes: List[str]
    override: Optional[dict] = None  # the operator's confirmation to install despite a fail

    def to_json(self) -> str:
        data = {
            "created": self.created,
            "verdict": self.verdict,
            "metrics": [asdict(m) for m in self.metrics],
            "thresholds": {name: list(limits) for name, limits in self.thresholds.items()},
            "notes": list(self.notes),
            "override": self.override,
        }
        return json.dumps(_encode_non_finite(data), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "QualityReport":
        data = json.loads(text)
        metrics = [
            Metric(
                name=m["name"],
                camera=m["camera"],
                value=_decode_float(m["value"]),
                unit=m["unit"],
                warn=_decode_limit(m["warn"]),
                fail=_decode_limit(m["fail"]),
                status=m["status"],
                note=m.get("note", ""),
                lower_is_worse=bool(m.get("lower_is_worse", False)),
            )
            for m in data["metrics"]
        ]
        thresholds = {
            name: (_decode_limit(warn), _decode_limit(fail)) for name, (warn, fail) in data["thresholds"].items()
        }
        return cls(
            created=data["created"],
            verdict=data["verdict"],
            metrics=metrics,
            thresholds=thresholds,
            notes=list(data["notes"]),
            override=data.get("override"),
        )

    def to_text(self) -> str:
        """A summary for the operator: the verdict, then every metric, failures first."""
        counts = {s: sum(m.status == s for m in self.metrics) for s in _STATUS_ORDER}
        lines = [
            f"3D calibration quality: {self.verdict.upper()}"
            f" ({counts['fail']} fail, {counts['warn']} warn, {counts['pass']} pass; created {self.created})"
        ]
        if self.override:
            details = ", ".join(f"{key} {value}" for key, value in self.override.items())
            lines.append(f"Override recorded: {details}")
        lines.append("")
        # sorted is stable, so metrics of one status keep the order they were graded in.
        for m in sorted(self.metrics, key=lambda m: _STATUS_ORDER.index(m.status)):
            line = (
                f"{m.status.upper():<6}{m.name:<30}{m.camera or '-':<7}"
                f"{_format_value(m.value)} {m.unit}  [{_format_limits(m)}]"
            )
            lines.append(f"{line}  {m.note}" if m.note else line)
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class CaptureSummary:
    waypoints_total: int
    waypoints_verified: int  # waypoints whose arrival the motor's last_position confirmed
    views_kept: Dict[str, int]  # camera -> waypoints with the board found
    rejections: Dict[str, Dict[str, int]]  # camera -> reason -> views rejected


@dataclass
class PresetCrossCheck:
    """A preset fitted directly from its own pooled frames, against the mapped sensor fit."""

    geometry: PresetGeometry
    max_intrinsic_delta_px: float  # max |dfx, dfy, dcx, dcy| over both cameras, preset px
    # |mean over every corner of both cameras of (corner detected in the preset's frames -
    # the sensor detection mapped into the preset)|, preset px: the mapping, free of fit noise
    corner_offset_px: float


@dataclass
class PreviousCalibration:
    """The calibration this one replaces. A legacy one has no motor-frame fit to compare with."""

    legacy: bool
    fit: Optional[MotorFrameFit]


def build_report(
    fit: MotorFrameFit,
    observations: Sequence[WaypointObservation],
    board: BoardSpec,
    roi_sensor: Tuple[float, float, float, float],
    capture: CaptureSummary,
    cross_checks: Sequence[PresetCrossCheck],
    previous: Optional[PreviousCalibration],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> QualityReport:
    """Grade `fit` and its capture against `thresholds`.

    roi_sensor is the base preset's sensor_roi(), in pixel edges. The stereo checks use
    the first two cameras of the fit. The verdict is "fail" if any metric fails, else
    "warn" if any warns, else "pass".
    """
    missing = sorted(set(DEFAULT_THRESHOLDS) - set(thresholds))
    if missing:
        raise ValueError(f"the thresholds have no limits for {missing}")
    roi = tuple(float(v) for v in roi_sensor)
    if not (roi[2] > roi[0] and roi[3] > roi[1]):
        raise ValueError(f"roi_sensor {roi_sensor} is empty")

    metrics: List[Metric] = []
    notes: List[str] = []

    def add(name: str, camera: Optional[str], value: float, unit: str, note: str = "") -> Metric:
        graded = _grade(name, camera, value, unit, thresholds[name], note)
        metrics.append(graded)
        return graded

    cameras = list(fit.cameras)

    # The capture.
    for cam in cameras:
        counted = sum(cam in o.corners for o in observations)
        add("views_kept", cam, capture.views_kept.get(cam, counted), "views")
    add("waypoints_unverified", None, capture.waypoints_total - capture.waypoints_verified, "waypoints")
    for cam, reasons in capture.rejections.items():
        rejected = sum(reasons.values())
        if rejected:
            detail = ", ".join(f"{reason} {count}" for reason, count in reasons.items())
            notes.append(f"{cam}: {rejected} views rejected ({detail})")

    # Each camera: its residual, its coverage and the sanity of its intrinsics.
    for cam in cameras:
        model = fit.cameras[cam]
        fx, fy = model.K[0, 0], model.K[1, 1]
        add("reprojection_rms_px", cam, fit.rms_px.get(cam, math.inf), "sensor px")
        add("coverage_roi_fraction", cam, coverage_fraction(observations, cam, roi), "of ROI")
        add("fx_fy_ratio_dev", cam, abs(fx / fy - 1.0), "ratio")
        fx_std = fit.param_std.get(cam, {}).get("fx", math.inf)
        undetermined = "" if math.isfinite(fx_std) else "the capture does not determine fx"
        add("focal_rel_std", cam, fx_std / fx, "ratio", undetermined)
        add("principal_point_offset_frac", cam, _principal_point_offset(model, roi), "of ROI half-side")
        add("max_distortion_shift_px", cam, _max_distortion_shift(model, roi), "sensor px")

    # The pair.
    if len(cameras) >= 2:
        first, second = cameras[:2]
        fx_a, fx_b = fit.cameras[first].K[0, 0], fit.cameras[second].K[0, 0]
        add("focal_left_right_rel", None, abs(fx_a - fx_b) / ((fx_a + fx_b) / 2), "ratio")
        epipolar, square, stereo_note = _stereo_checks(fit, observations, board, first, second)
    else:
        stereo_note = f"the fit has {len(cameras)} camera(s); a pair is needed"
        add("focal_left_right_rel", None, math.inf, "ratio", stereo_note)
        epipolar = square = math.inf
    add("epipolar_error_px", None, epipolar, "sensor px", stereo_note)
    add("square_size_rel_error", None, square, "ratio", stereo_note)
    add("motor_scale_dev", None, abs(fit.motor_scale - 1.0), "ratio")
    for check in cross_checks:
        add("cross_check_delta_px", None, check.max_intrinsic_delta_px, "preset px", f"preset {check.geometry.name}")
        add("cross_check_corner_offset_px", None, check.corner_offset_px, "preset px", f"preset {check.geometry.name}")

    # The change since the previous calibration.
    if previous is None or previous.legacy or previous.fit is None:
        notes.append(NO_PREVIOUS_NOTE)
    else:
        notes.extend(_change_metrics(fit, previous.fit, add))

    statuses = {m.status for m in metrics}
    verdict = next((s for s in _STATUS_ORDER if s in statuses), "pass")
    return QualityReport(
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        verdict=verdict,
        metrics=metrics,
        thresholds={name: (_as_limit(warn), _as_limit(fail)) for name, (warn, fail) in thresholds.items()},
        notes=notes,
    )


def coverage_fraction(
    observations: Sequence[WaypointObservation], camera: str, roi_sensor: Tuple[float, float, float, float]
) -> float:
    """The fraction of roi_sensor (pixel edges) inside the union of the per-waypoint convex
    hulls of `camera`'s detected corners, rasterised at 1/4 resolution."""
    x0, y0, x1, y1 = (float(v) for v in roi_sensor)
    shape = (int(math.ceil((y1 - y0) / _COVERAGE_SCALE)), int(math.ceil((x1 - x0) / _COVERAGE_SCALE)))
    mask = _hull_mask(observations, camera, (x0, y0), shape, _COVERAGE_SCALE)
    return float(np.count_nonzero(mask)) / mask.size


def write_coverage_images(
    observations: Sequence[WaypointObservation],
    image_size: Tuple[int, int],
    roi_sensor: Tuple[float, float, float, float],
    out_dir: Path,
    camera_names: Sequence[str] = ("left", "right"),
) -> List[Path]:
    """Write coverage_<cam>.png per camera: the full sensor with the covered area (the union
    of the per-waypoint corner hulls), the ROI rectangle and every detected corner."""
    width, height = int(image_size[0]), int(image_size[1])
    x0, y0, x1, y1 = (float(v) for v in roi_sensor)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cam in camera_names:
        image = np.full((height, width, 3), 40, np.uint8)
        image[_hull_mask(observations, cam, (0.0, 0.0), (height, width), 1) > 0] = (70, 130, 70)
        # The ROI's edges are x0 .. x1; its outermost pixels are x0 and x1 - 1.
        corner_a, corner_b = (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1)
        cv2.rectangle(image, corner_a, corner_b, (0, 210, 255), 2)
        seen = [o for o in observations if cam in o.corners]
        for o in seen:
            for u, v in np.asarray(o.corners[cam], float):
                cv2.circle(image, (int(round(u)), int(round(v))), 2, (80, 80, 255), -1)
        label = f"{cam}: {coverage_fraction(observations, cam, roi_sensor):.2f} of ROI covered, {len(seen)} views"
        cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        path = out_dir / f"coverage_{cam}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write {path}")
        paths.append(path)
    return paths


def _grade(name, camera, value, unit, limits, note="") -> Metric:
    value = float(value)
    if math.isnan(value):
        value = math.inf
        note = note or "could not be computed"
    elif math.isinf(value):
        note = note or "could not be determined"
    warn, fail = (_as_limit(limit) for limit in limits)
    lower = name in LOWER_IS_WORSE
    return Metric(name, camera, value, unit, warn, fail, _status(value, warn, fail, lower), note, lower)


def _status(value: float, warn: Optional[float], fail: Optional[float], lower_is_worse: bool) -> str:
    if not math.isfinite(value):
        # An undetermined value cannot pass: it takes the worst status the limits allow.
        return "fail" if fail is not None else "warn" if warn is not None else "pass"

    def beyond(limit):
        return limit is not None and (value < limit if lower_is_worse else value > limit)

    return "fail" if beyond(fail) else "warn" if beyond(warn) else "pass"


def _as_limit(limit) -> Optional[float]:
    return None if limit is None else float(limit)


def _hull_mask(observations, camera, origin, shape, scale) -> np.ndarray:
    """A uint8 mask of `shape` whose pixel (0, 0) covers sensor edges origin .. origin + scale,
    with the union of the per-waypoint convex hulls of `camera`'s corners set to 255.

    Pixel-centre sensor coordinate u maps to mask pixel-centre coordinate
    (u + 0.5 - origin) / scale - 0.5. Each hull is filled on its own, so overlaps add up
    to the union whatever rule fillPoly uses for overlapping contours.
    """
    mask = np.zeros(shape, np.uint8)
    offset = np.asarray(origin, float)
    for o in observations:
        if camera not in o.corners:
            continue
        hull = cv2.convexHull(np.asarray(o.corners[camera], np.float32)).reshape(-1, 2).astype(float)
        vertices = ((hull + 0.5 - offset) / scale - 0.5) * (1 << _FILL_SHIFT)
        cv2.fillPoly(mask, [np.round(vertices).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=_FILL_SHIFT)
    return mask


def _principal_point_offset(model: CameraModel, roi) -> float:
    """|(cx, cy) - ROI centre|, each axis divided by that axis's ROI half-side.

    For the square presets this is |(cx, cy) - centre| / (side / 2); dividing per axis keeps
    a rectangular ROI's limit at the same fraction of the way to its edge in both directions.
    """
    x0, y0, x1, y1 = roi
    # The centre in pixel-centre coordinates, half a pixel inside the edge coordinates.
    centre_x, centre_y = (x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5
    return math.hypot(
        (model.K[0, 2] - centre_x) / ((x1 - x0) / 2),
        (model.K[1, 2] - centre_y) / ((y1 - y0) / 2),
    )


def _max_distortion_shift(model: CameraModel, roi) -> float:
    """The largest |distorted - undistorted| projection, in sensor px, over points along the
    ROI's border: each point is taken as an undistorted pixel, turned into a ray with K^-1,
    and projected again through K and dist."""
    x0, y0, x1, y1 = roi
    # The border in pixel-centre coordinates, corners included.
    xs = np.linspace(x0, x1, _DISTORTION_SAMPLES_PER_EDGE) - 0.5
    ys = np.linspace(y0, y1, _DISTORTION_SAMPLES_PER_EDGE) - 0.5
    border = np.concatenate([
        np.column_stack([xs, np.full_like(xs, y0 - 0.5)]),
        np.column_stack([xs, np.full_like(xs, y1 - 0.5)]),
        np.column_stack([np.full_like(ys, x0 - 0.5), ys]),
        np.column_stack([np.full_like(ys, x1 - 0.5), ys]),
    ])
    rays = np.linalg.solve(model.K, np.column_stack([border, np.ones(len(border))]).T).T
    rays /= rays[:, 2:3]
    distorted = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), model.K, model.dist)[0].reshape(-1, 2)
    return float(np.max(np.linalg.norm(distorted - border, axis=1)))


def _stereo_checks(fit: MotorFrameFit, observations, board: BoardSpec, first: str, second: str):
    """(mean symmetric epipolar distance in sensor px, |mean triangulated square - square_mm| /
    square_mm, note) over the waypoints where both cameras found the board."""
    both = [o for o in observations if first in o.corners and second in o.corners]
    if not both:
        return math.inf, math.inf, f"no waypoint has the board in both {first} and {second}"
    a, b = fit.cameras[first], fit.cameras[second]
    na = _normalised(np.concatenate([o.corners[first] for o in both]), a)
    nb = _normalised(np.concatenate([o.corners[second] for o in both]), b)
    ha = np.column_stack([na, np.ones(len(na))])
    hb = np.column_stack([nb, np.ones(len(nb))])

    # X_second = R X_first + T gives the essential matrix E = [T]x R, with xb^T E xa = 0.
    R, T = fit.stereo(first, second)
    E = _skew(T) @ R
    line_b = ha @ E.T  # E xa: the epipolar line of xa in the second image
    line_a = hb @ E  # E^T xb: the epipolar line of xb in the first image
    residual = np.abs(np.sum(hb * line_b, axis=1))
    # Distances are in normalised coordinates; each camera's mean focal length makes them px.
    d_a = residual / np.hypot(line_a[:, 0], line_a[:, 1]) * _mean_focal(a)
    d_b = residual / np.hypot(line_b[:, 0], line_b[:, 1]) * _mean_focal(b)
    epipolar = float(np.mean((d_a + d_b) / 2))

    X = cv2.triangulatePoints(_projection(a), _projection(b), na.T, nb.T)
    grid = (X[:3] / X[3]).T.reshape(len(both), board.rows, board.cols, 3)
    steps = np.concatenate([
        np.linalg.norm(grid[:, :, 1:] - grid[:, :, :-1], axis=-1).ravel(),  # along each row
        np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=-1).ravel(),  # down each column
    ])
    square = abs(float(steps.mean()) - board.square_mm) / board.square_mm
    return epipolar, square, f"{len(both)} waypoints with both cameras"


def _change_metrics(fit: MotorFrameFit, previous: MotorFrameFit, add: Callable[..., Metric]) -> List[str]:
    """Grade the change since `previous`: each camera's rotation and focal length, and the
    baseline between the first two cameras (see the module docstring). Return the report notes."""
    notes = []
    for cam, now in fit.cameras.items():
        before = previous.cameras.get(cam)
        if before is None:
            notes.append(f"{cam}: not in the previous calibration, so its change is not reported")
            continue
        R_now = cv2.Rodrigues(np.asarray(now.rvec, float))[0]
        R_before = cv2.Rodrigues(np.asarray(before.rvec, float))[0]
        # The norm of the rotation vector is the angle, without arccos's loss of precision near 0.
        angle = math.degrees(float(np.linalg.norm(cv2.Rodrigues(R_before.T @ R_now)[0])))
        rotation = add("camera_rotation_change_deg", cam, angle, "deg")
        focal = add("focal_change_rel", cam, abs(now.K[0, 0] / before.K[0, 0] - 1.0), "ratio")
        if rotation.status != "pass":
            rotation.note = MOVED_NOTE
            notes.append(f"{cam} {MOVED_NOTE}: turned {angle:.2f} deg")
        if focal.status != "pass":
            focal.note = "focal length changed since the previous calibration"

    pair = list(fit.cameras)[:2]
    if len(pair) == 2 and all(cam in previous.cameras for cam in pair):
        first, second = pair
        baseline_now = _centre(fit.cameras[second]) - _centre(fit.cameras[first])
        baseline_before = _centre(previous.cameras[second]) - _centre(previous.cameras[first])
        change = float(np.linalg.norm(baseline_now - baseline_before))
        baseline = add("baseline_change_mm", None, change, "mm")
        if baseline.status != "pass":
            baseline.note = f"a camera moved since the previous calibration: the {first}-{second} baseline changed"
            notes.append(f"{baseline.note} by {change:.2f} mm")
    else:
        notes.append("the previous calibration lacks one of the pair, so the baseline change is not reported")
    return notes


def _normalised(pixels: np.ndarray, model: CameraModel) -> np.ndarray:
    points = np.asarray(pixels, float).reshape(-1, 1, 2)
    return cv2.undistortPointsIter(points, model.K, model.dist, None, None, _UNDISTORT_CRITERIA).reshape(-1, 2)


def _projection(model: CameraModel) -> np.ndarray:
    """[R | t] of a camera, for normalised image coordinates."""
    return np.hstack([cv2.Rodrigues(np.asarray(model.rvec, float))[0], np.reshape(model.tvec, (3, 1))])


def _centre(model: CameraModel) -> np.ndarray:
    """The camera centre in the motor frame, -R^T t."""
    return -cv2.Rodrigues(np.asarray(model.rvec, float))[0].T @ np.asarray(model.tvec, float).reshape(3)


def _mean_focal(model: CameraModel) -> float:
    return float((model.K[0, 0] + model.K[1, 1]) / 2)


def _skew(v) -> np.ndarray:
    x, y, z = np.asarray(v, float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _encode_non_finite(obj):
    """`obj` with every non-finite float replaced by "inf", "-inf" or "nan", for strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return "nan" if math.isnan(obj) else "inf" if obj > 0 else "-inf"
    if isinstance(obj, dict):
        return {key: _encode_non_finite(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_non_finite(value) for value in obj]
    return obj


def _decode_float(value) -> float:
    if isinstance(value, str):
        if value not in _NON_FINITE:
            raise ValueError(f"expected a number or one of {sorted(_NON_FINITE)}, got {value!r}")
        return _NON_FINITE[value]
    return float(value)


def _decode_limit(value) -> Optional[float]:
    return None if value is None else _decode_float(value)


def _format_value(value: float) -> str:
    return f"{value:.4g}"


def _format_limits(m: Metric) -> str:
    sign = "<" if m.lower_is_worse else ">"
    limits = (("warn", m.warn), ("fail", m.fail))
    parts = [f"{label} {sign} {_format_value(limit)}" for label, limit in limits if limit is not None]
    return ", ".join(parts) if parts else "no limits"
