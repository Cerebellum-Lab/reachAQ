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
with no waypoint seen by both cameras of a pair) is +inf with a note, never NaN, and takes the worst
status its thresholds allow: an undetermined quantity cannot pass a check. JSON has no
infinity, so non-finite floats are written as the strings "inf", "-inf" and "nan", which
keeps the file strict JSON, and from_json turns them back into floats.

The one exception is the cross-camera triangulation check (loo_triangulation_px). A camera
with no waypoint that two other fitted cameras also saw cannot be checked that way, which
is a property of where the cameras look, not a fault of the calibration. Its metric has
status UNDETERMINED and value None: it is neither a pass nor a fail and does not enter the
verdict. When no camera can be checked (always so with 2 fitted cameras), the report's
verdict_note says so, so the verdict does not read as a fully checked pass.

The fit has 2 to 6 cameras. Per-camera metrics are graded for every fitted camera, and the
pair metrics for every co-visible pair (two cameras that saw the board at a common
waypoint), with the pair named "a-b" in the fit's camera order. A camera the pipeline's
per-camera rule left out of the fit is listed in `cameras` as not calibrated (2D only),
first in the text report, and makes the verdict at least "warn".

Pixel conventions are geometry.py's: corners and principal points are pixel-centre
coordinates (pixel i is centred on i), and roi_sensor is in pixel edges, as
PresetGeometry.sensor_roi() returns it, so a pixel-centre coordinate u sits at edge
coordinate u + 0.5.

The change since the previous calibration has to respect the fit's gauge. The motor
frame's origin is the board's corner 0 at motor position 0 (fit.py), a choice rather than
a measurement, and the board is remounted on the stage every month. Between calibrations
the origin therefore moves by an unknown vector, and every camera centre (-R^T t) shifts
with it, so comparing each camera's centre on its own would report "moved" every month.
The report compares the baselines instead, the vector from one camera's centre to
another's for every pair, which a common shift leaves unchanged: baseline_change_mm catches
one camera moving relative to another. A shift of every camera together by the same vector
cannot be told from an origin shift and is not visible here; the DCS calibration, redone
after every 3D calibration, measures the origin. Camera rotations are relative to the motor
axes, which are physical, so camera_rotation_change_deg compares each camera on its own.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from reachaq.inference.multires_calibration.fit import (
    CHARUCO,
    BoardSpec,
    CameraModel,
    MotorFrameFit,
    WaypointObservation,
    usable_view,
    valid_corners,
)
from reachaq.inference.multires_calibration.geometry import PresetGeometry

Thresholds = Dict[str, Tuple[Optional[float], Optional[float]]]

# The initial (warn, fail) limits, recorded verbatim in every report. Tune them on real
# captures; a report always carries the limits it was graded with.
DEFAULT_THRESHOLDS: Thresholds = {
    "views_kept": (20.0, 10.0),
    # The views a camera shares with another fitted camera; the fit does not need them, but
    # they expose stage geometry the model leaves out (axis skew).
    "covisible_views": (8.0, 4.0),
    "reprojection_rms_px": (0.5, 1.0),
    "coverage_roi_fraction": (0.5, 0.25),
    "fx_fy_ratio_dev": (0.01, 0.03),
    "focal_rel_std": (0.005, 0.02),
    "focal_pair_rel": (0.10, None),
    "principal_point_offset_frac": (0.2, 0.4),
    "max_distortion_shift_px": (60.0, 150.0),
    "epipolar_error_px": (0.5, 1.0),
    "square_size_rel_error": (0.01, 0.03),
    # With 3 or more cameras: the median distance between a camera's corners and the points
    # the other cameras triangulate, projected into it.
    "loo_triangulation_px": (0.5, 1.0),
    "motor_scale_dev": (0.01, None),
    "waypoints_unverified": (0.0, None),
    # A waypoint the motor reported elsewhere than commanded puts its view where the board
    # was not: the fit takes the commanded position, so any such waypoint fails.
    "waypoints_unreached": (None, 0.0),
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
    # Informational (ruling R50): reported for the record and for months to be compared,
    # never graded. The joint fit's RMS over every corner of every camera, in sensor px;
    # each camera's RMS in each preset's own px (sensor RMS / binning); the length of each
    # pair's baseline; and each camera's board coverage of the full sensor frame.
    "stereo_reprojection_rms_px": (None, None),
    "preset_reprojection_rms_px": (None, None),
    "baseline_mm": (None, None),
    "coverage_frame_fraction": (None, None),
}
# The metrics where a smaller value is worse; see the module docstring.
LOWER_IS_WORSE = frozenset({"views_kept", "covisible_views", "coverage_roi_fraction", "coverage_frame_fraction"})
NO_PREVIOUS_NOTE = "no comparable previous calibration"
MOVED_NOTE = "camera moved since the previous calibration"
# The motor's position report echoes its target (christielab10 motor check, 2026-10-26), so a
# waypoint counts as verified on the motor's word alone.
VERIFIED_NOTE = "verified: the motor reported the commanded position, not a traced arrival"
# The first lines of the text report, one per camera the calibration left out.
NOT_CALIBRATED = "NOT CALIBRATED (2D only)"

# The status of a cross-camera triangulation check that is not possible for a camera.
UNDETERMINED = "undetermined"
LOO_UNDETERMINED_NOTE = (
    "triangulation check not possible: no other two cameras saw the board at the same positions"
)

_STATUS_ORDER = ("fail", "warn", "pass")
# The order of the text report's metric lines: an undetermined check comes before the passes.
_DISPLAY_ORDER = ("fail", "warn", UNDETERMINED, "pass")
# The verdict as the verdict note words it.
_VERDICT_WORDS = {"pass": "passed", "warn": "warn", "fail": "fail"}
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
    value: Optional[float]  # None only when status is UNDETERMINED
    unit: str
    warn: Optional[float]
    fail: Optional[float]
    status: str  # "pass", "warn", "fail", or UNDETERMINED (not graded)
    note: str = ""
    lower_is_worse: bool = False


@dataclass
class CameraVerdict:
    """Whether a capture camera is calibrated. `reason` is "" when it is, and otherwise says
    in plain words why not (pipeline.per_camera_rule): the camera is then 2D only."""

    name: str
    calibrated: bool
    reason: str


@dataclass
class QualityReport:
    created: str  # ISO 8601, local time with its UTC offset
    # the worst metric status, and at least "warn" when a camera is not calibrated
    verdict: str
    metrics: List[Metric]
    thresholds: Thresholds  # name -> (warn, fail), the limits the metrics were graded with
    notes: List[str]
    override: Optional[dict] = None  # the operator's confirmation to install despite a fail
    # Every capture camera in capture order; empty in a report written before 2026-11-20.
    cameras: Dict[str, CameraVerdict] = field(default_factory=dict)
    # Said with the verdict when no camera could have the cross-camera triangulation check,
    # e.g. "passed; cross-camera triangulation check not possible with 2 cameras"; else "".
    verdict_note: str = ""

    def to_json(self) -> str:
        data = {
            "created": self.created,
            "verdict": self.verdict,
            "verdict_note": self.verdict_note,
            "metrics": [asdict(m) for m in self.metrics],
            "thresholds": {name: list(limits) for name, limits in self.thresholds.items()},
            "notes": list(self.notes),
            "override": self.override,
            "cameras": {name: asdict(v) for name, v in self.cameras.items()},
        }
        return json.dumps(_encode_non_finite(data), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "QualityReport":
        data = json.loads(text)
        metrics = [
            Metric(
                name=m["name"],
                camera=m["camera"],
                value=None if m["value"] is None else _decode_float(m["value"]),
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
            verdict_note=data.get("verdict_note", ""),
            cameras={
                name: CameraVerdict(name=v["name"], calibrated=bool(v["calibrated"]), reason=v["reason"])
                for name, v in (data.get("cameras") or {}).items()
            },
        )

    def to_text(self) -> str:
        """A summary for the operator: every camera that is not calibrated, then the verdict,
        then every metric, failures first."""
        counts = {s: sum(m.status == s for m in self.metrics) for s in _DISPLAY_ORDER}
        undetermined = f", {counts[UNDETERMINED]} {UNDETERMINED}" if counts[UNDETERMINED] else ""
        lines = [f"{NOT_CALIBRATED}: {v.name}: {v.reason}" for v in self.cameras.values() if not v.calibrated]
        verdict_line = (
            f"3D calibration quality: {self.verdict.upper()}"
            f" ({counts['fail']} fail, {counts['warn']} warn, {counts['pass']} pass{undetermined};"
            f" created {self.created})"
        )
        lines.append(f"{verdict_line} - {self.verdict_note}" if self.verdict_note else verdict_line)
        if self.override:
            details = ", ".join(f"{key} {value}" for key, value in self.override.items())
            lines.append(f"Override recorded: {details}")
        lines.append("")
        # sorted is stable, so metrics of one status keep the order they were graded in.
        for m in sorted(self.metrics, key=lambda m: _DISPLAY_ORDER.index(m.status)):
            value = "-" if m.value is None else f"{_format_value(m.value)} {m.unit}"
            line = f"{m.status.upper():<5} {m.name:<30}{m.camera or '-':<7}{value}  [{_format_limits(m)}]"
            lines.append(f"{line}  {m.note}" if m.note else line)
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class CaptureSummary:
    waypoints_total: int
    # waypoints at which the motor reported the commanded position: its word, not a traced
    # arrival (pipeline.WaypointLog.verified)
    waypoints_verified: int
    # waypoints whose finite reported position is more than 0.1 mm from the commanded one
    # on some axis (pipeline.unreached_waypoints)
    waypoints_unreached: int
    views_kept: Dict[str, int]  # camera -> waypoints with the board found
    rejections: Dict[str, Dict[str, int]]  # camera -> reason -> views rejected


@dataclass
class PresetCrossCheck:
    """A preset fitted directly from its own pooled frames, against the mapped sensor fit."""

    geometry: PresetGeometry
    max_intrinsic_delta_px: float  # max |dfx, dfy, dcx, dcy| over every fitted camera, preset px
    # |mean over every corner of every camera of (corner detected in the preset's frames -
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
    presets: Sequence[PresetGeometry] = (),
    cameras: Optional[Mapping[str, CameraVerdict]] = None,
) -> QualityReport:
    """Grade `fit` and its capture against `thresholds`.

    roi_sensor is the base preset's sensor_roi(), in pixel edges. Only the fit's cameras are
    graded, and only their corners in `observations` are used. The pair checks cover every
    pair of them that saw the board at a common waypoint. Each camera's RMS is also reported
    in the pixels of every preset in `presets`.

    `cameras` is every capture camera's verdict, in capture order; a fitted camera it leaves
    out is calibrated, and by default every fitted camera is. The verdict is "fail" if any
    metric fails, else "warn" if any warns or any camera is not calibrated, else "pass". An
    UNDETERMINED triangulation check is neither: it does not enter the verdict, and when
    every camera's is undetermined (always so with 2 fitted cameras) verdict_note says so.
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

    fitted = list(fit.cameras)
    # The fitted cameras that found the board at each waypoint: for a ChArUco board, those
    # whose corners there are a view (fit.usable_view), so a co-visible view is one where
    # both cameras pass that minimum.
    seen = [{cam for cam, c in o.corners.items() if cam in fit.cameras and usable_view(c, board)}
            for o in observations]

    # The capture.
    for cam in fitted:
        counted = sum(cam in s for s in seen)
        add("views_kept", cam, capture.views_kept.get(cam, counted), "views")
        add("covisible_views", cam, sum(cam in s and len(s) > 1 for s in seen), "views")
    add("waypoints_unverified", None, capture.waypoints_total - capture.waypoints_verified, "waypoints",
        VERIFIED_NOTE)
    add("waypoints_unreached", None, capture.waypoints_unreached, "waypoints")
    for cam, reasons in capture.rejections.items():
        rejected = sum(reasons.values())
        if rejected:
            detail = ", ".join(f"{reason} {count}" for reason, count in reasons.items())
            notes.append(f"{cam}: {rejected} views rejected ({detail})")
    if board.kind == CHARUCO:
        # A ChArUco view may be partial, so how much of the board each view held is shown.
        for cam in fitted:
            notes.append(_views_used_note(cam, fit, observations))

    # Each camera: its residual, its coverage and the sanity of its intrinsics.
    frame = (0.0, 0.0, float(fit.image_size[0]), float(fit.image_size[1]))
    for cam in fitted:
        model = fit.cameras[cam]
        fx, fy = model.K[0, 0], model.K[1, 1]
        rms = fit.rms_px.get(cam, math.inf)
        add("reprojection_rms_px", cam, rms, "sensor px")
        for g in presets:
            # A binned pixel is b sensor pixels wide, so a distance in sensor px is 1/b of it in preset px.
            add("preset_reprojection_rms_px", cam, rms / g.binning, "preset px", f"preset {g.name}")
        add("coverage_roi_fraction", cam, coverage_fraction(observations, cam, roi), "of ROI")
        add("coverage_frame_fraction", cam, coverage_fraction(observations, cam, frame), "of frame")
        add("fx_fy_ratio_dev", cam, abs(fx / fy - 1.0), "ratio")
        fx_std = fit.param_std.get(cam, {}).get("fx", math.inf)
        undetermined = "" if math.isfinite(fx_std) else "the capture does not determine fx"
        add("focal_rel_std", cam, fx_std / fx, "ratio", undetermined)
        add("principal_point_offset_frac", cam, _principal_point_offset(model, roi), "of ROI half-side")
        add("max_distortion_shift_px", cam, _max_distortion_shift(model, roi), "sensor px")

    # Every co-visible pair, and each camera against the rest.
    add("stereo_reprojection_rms_px", None, _joint_rms(fit), "sensor px")
    pairs = [(a, b) for a, b in combinations(fitted, 2) if any(a in s and b in s for s in seen)]
    for first, second in pairs:
        pair = f"{first}-{second}"
        fx_a, fx_b = fit.cameras[first].K[0, 0], fit.cameras[second].K[0, 0]
        add("focal_pair_rel", pair, abs(fx_a - fx_b) / ((fx_a + fx_b) / 2), "ratio")
        add("baseline_mm", pair, float(np.linalg.norm(_centre(fit.cameras[second]) - _centre(fit.cameras[first]))),
            "mm")
        epipolar, square, stereo_note = _stereo_checks(fit, observations, board, first, second)
        add("epipolar_error_px", pair, epipolar, "sensor px", stereo_note)
        add("square_size_rel_error", pair, square, "ratio", stereo_note)
    if not pairs:
        # Undetermined, so the pair checks cannot pass.
        stereo_note = (
            f"the fit has {len(fitted)} camera(s); a pair is needed" if len(fitted) < 2
            else "no two cameras found the board at the same waypoint"
        )
        add("focal_pair_rel", None, math.inf, "ratio", stereo_note)
        add("baseline_mm", None, math.inf, "mm", stereo_note)
        add("epipolar_error_px", None, math.inf, "sensor px", stereo_note)
        add("square_size_rel_error", None, math.inf, "ratio", stereo_note)
    loo_checked = False
    if len(fitted) >= 3:
        for cam in fitted:
            loo, loo_note = _leave_one_out(fit, observations, cam)
            if loo is None:
                warn, fail = (_as_limit(limit) for limit in thresholds["loo_triangulation_px"])
                metrics.append(Metric("loo_triangulation_px", cam, None, "sensor px", warn, fail, UNDETERMINED,
                                      LOO_UNDETERMINED_NOTE))
            else:
                loo_checked = True
                add("loo_triangulation_px", cam, loo, "sensor px", loo_note)
    add("motor_scale_dev", None, abs(fit.motor_scale - 1.0), "ratio")
    for check in cross_checks:
        add("cross_check_delta_px", None, check.max_intrinsic_delta_px, "preset px", f"preset {check.geometry.name}")
        add("cross_check_corner_offset_px", None, check.corner_offset_px, "preset px", f"preset {check.geometry.name}")

    # The change since the previous calibration.
    if previous is None or previous.legacy or previous.fit is None:
        notes.append(NO_PREVIOUS_NOTE)
    else:
        notes.extend(_change_metrics(fit, previous.fit, add))

    verdicts = dict(cameras or {})
    for cam in fit.cameras:
        verdicts.setdefault(cam, CameraVerdict(cam, True, ""))
    statuses = {m.status for m in metrics}
    if any(not v.calibrated for v in verdicts.values()):
        statuses.add("warn")  # a 2D-only camera never leaves the calibration a plain pass
    verdict = next((s for s in _STATUS_ORDER if s in statuses), "pass")
    verdict_note = ""
    if not loo_checked and len(fitted) >= 2:
        where = "with 2 cameras" if len(fitted) == 2 else "for any camera"
        verdict_note = f"{_VERDICT_WORDS[verdict]}; cross-camera triangulation check not possible {where}"
    return QualityReport(
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        verdict=verdict,
        metrics=metrics,
        thresholds={name: (_as_limit(warn), _as_limit(fail)) for name, (warn, fail) in thresholds.items()},
        notes=notes,
        cameras=verdicts,
        verdict_note=verdict_note,
    )


def _views_used_note(cam: str, fit: MotorFrameFit, observations: Sequence[WaypointObservation]) -> str:
    """"{cam}: views used N of M waypoints, median K corners per view": N the views the fit
    used, M the distinct waypoints, K the median of the corners fitted per view."""
    used = fit.per_view_rms_px.get(cam, {})
    counts = (fit.corner_counts or {}).get(cam)
    if counts is None:
        by_index = {o.index: o.corners[cam] for o in observations if cam in o.corners}
        counts = {i: int(valid_corners(by_index[i]).sum()) for i in used if i in by_index}
    median = float(np.median(list(counts.values()))) if counts else 0.0
    return f"{cam}: views used {len(used)} of {len(observations)} waypoints, median {median:g} corners per view"


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
            corners = np.asarray(o.corners[cam], float)
            for u, v in corners[valid_corners(corners)]:
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
        corners = np.asarray(o.corners[camera], np.float32)
        corners = corners[valid_corners(corners)]  # a ChArUco view's detected corners
        if len(corners) < 3:
            continue
        hull = cv2.convexHull(corners).reshape(-1, 2).astype(float)
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
    both = [o for o in observations if first in o.corners and second in o.corners
            and usable_view(o.corners[first], board) and usable_view(o.corners[second], board)]
    if not both:
        return math.inf, math.inf, f"no waypoint has the board in both {first} and {second}"
    a, b = fit.cameras[first], fit.cameras[second]
    # The corners both cameras detected: every corner of a checkerboard.
    common = np.concatenate([valid_corners(o.corners[first]) & valid_corners(o.corners[second]) for o in both])
    na = _normalised(np.concatenate([o.corners[first] for o in both])[common], a)
    nb = _normalised(np.concatenate([o.corners[second] for o in both])[common], b)
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
    points = np.full((len(common), 3), np.nan)
    points[common] = (X[:3] / X[3]).T
    grid = points.reshape(len(both), board.rows, board.cols, 3)
    steps = np.concatenate([
        np.linalg.norm(grid[:, :, 1:] - grid[:, :, :-1], axis=-1).ravel(),  # along each row
        np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=-1).ravel(),  # down each column
    ])
    steps = steps[np.isfinite(steps)]  # the neighbours both ends of which both cameras found
    square = abs(float(steps.mean()) - board.square_mm) / board.square_mm
    return epipolar, square, f"{len(both)} waypoints with {first} and {second}"


def _change_metrics(fit: MotorFrameFit, previous: MotorFrameFit, add: Callable[..., Metric]) -> List[str]:
    """Grade the change since `previous`: each camera's rotation and focal length, and the
    baseline of every pair of cameras both calibrations have (see the module docstring).
    Return the report notes."""
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

    both = [cam for cam in fit.cameras if cam in previous.cameras]
    for first, second in combinations(both, 2):
        baseline_now = _centre(fit.cameras[second]) - _centre(fit.cameras[first])
        baseline_before = _centre(previous.cameras[second]) - _centre(previous.cameras[first])
        change = float(np.linalg.norm(baseline_now - baseline_before))
        baseline = add("baseline_change_mm", f"{first}-{second}", change, "mm")
        if baseline.status != "pass":
            baseline.note = f"a camera moved since the previous calibration: the {first}-{second} baseline changed"
            notes.append(f"{baseline.note} by {change:.2f} mm")
    if len(both) < 2:
        notes.append("the previous calibration has fewer than two of these cameras, so no baseline change is reported")
    return notes


def _leave_one_out(fit: MotorFrameFit, observations, cam: str) -> Tuple[Optional[float], str]:
    """(median px, note): over every corner of every waypoint where `cam` and at least two
    other fitted cameras found the board, the distance between `cam`'s corner and the point
    the others triangulate (each undistorted, then a linear DLT), projected through `cam`'s
    model. (None, "") when no waypoint has two others: the check is undetermined."""
    others = [c for c in fit.cameras if c != cam]
    model = fit.cameras[cam]
    distances, waypoints = [], 0
    for o in observations:
        seen = [c for c in others if c in o.corners]
        if cam not in o.corners or len(seen) < 2:
            continue
        # The corners every one of them detected: every corner of a checkerboard.
        common = valid_corners(o.corners[cam])
        for c in seen:
            common = common & valid_corners(o.corners[c])
        if not common.any():
            continue
        X = _triangulate([fit.cameras[c] for c in seen], [np.asarray(o.corners[c], float)[common] for c in seen])
        projected = cv2.projectPoints(X, np.asarray(model.rvec, float), np.asarray(model.tvec, float),
                                      model.K, model.dist)[0].reshape(-1, 2)
        distances.append(np.linalg.norm(projected - np.asarray(o.corners[cam], float)[common], axis=1))
        waypoints += 1
    if not distances:
        return None, ""
    return float(np.median(np.concatenate(distances))), f"{waypoints} waypoints with {cam} and 2 or more others"


def _triangulate(models: Sequence[CameraModel], pixels: Sequence[np.ndarray]) -> np.ndarray:
    """(n, 3) motor-frame points from n corners seen in every one of `models` (2 or more):
    each camera's pixels are undistorted to normalised coordinates, and each point is the
    least-squares null vector of its DLT system, 2 rows per camera."""
    rows = []
    for model, p in zip(models, pixels):
        n = _normalised(p, model)
        P = _projection(model)
        rows.append(n[:, :1] * P[2] - P[0])
        rows.append(n[:, 1:] * P[2] - P[1])
    A = np.stack(rows, axis=1)  # (n, 2 * cameras, 4)
    X = np.linalg.svd(A)[2][:, -1, :]
    return X[:, :3] / X[:, 3:]


def _joint_rms(fit: MotorFrameFit) -> float:
    """The RMS reprojection distance over every corner of every camera together, in sensor
    px: the residual of the joint fit itself. Every view has the same corners, so the mean
    of the per-view mean squares is the mean over the corners. Infinite with no views.
    Partial views (fit.corner_counts) are weighted by the corners each held."""
    if fit.corner_counts:
        weighted = [(v * v * fit.corner_counts.get(cam, {}).get(i, 1), fit.corner_counts.get(cam, {}).get(i, 1))
                    for cam, views in fit.per_view_rms_px.items() for i, v in views.items()]
        total = sum(n for _, n in weighted)
        return math.sqrt(sum(s for s, _ in weighted) / total) if total else math.inf
    squares = [v * v for views in fit.per_view_rms_px.values() for v in views.values()]
    return math.sqrt(sum(squares) / len(squares)) if squares else math.inf


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
