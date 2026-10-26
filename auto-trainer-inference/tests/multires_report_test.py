"""The calibration quality report: graded metrics, recorded thresholds and the change since the last calibration.

The fit comes from the synthetic rig, fitted once for the module (about 1 s on
christielab10). Tests that need exact values use the rig's noise-free truth cameras as
the fit, so a metric's definition can be checked against a known answer.

"Full coverage" uses the rectangle the board's corners sweep as the ROI. The synthetic
move list sweeps only 0.41 (left) and 0.43 (right) of the 1024 base ROI, measured with
the same rasteriser, which is under the 0.5 warn, so on the base ROI a clean fit would
warn on coverage alone.
"""

import dataclasses
import json
import math

import cv2
import numpy as np
import pytest

from autotrainer.inference.multires_calibration.fit import WaypointObservation, fit_motor_frame
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.report import (
    DEFAULT_THRESHOLDS,
    CaptureSummary,
    PresetCrossCheck,
    PreviousCalibration,
    QualityReport,
    build_report,
    coverage_fraction,
    write_coverage_images,
)
from multires_synthetic import synthetic_rig


BASE = PresetGeometry(4, 256, 256, 52, 6)
BASE_ROI = BASE.sensor_roi()  # (208, 24, 1232, 1048), pixel edges
IMAGE_SIZE = (1440, 1080)
CAMERAS = ("left", "right")
CHANGE_METRICS = {"camera_rotation_change_deg", "baseline_change_mm", "focal_change_rel"}
STATUS_WORDS = ("FAIL", "WARN", "PASS")


@pytest.fixture(scope="module")
def rig():
    return synthetic_rig(noise_px=0.1)


@pytest.fixture(scope="module")
def fitted(rig):
    return fit_motor_frame(rig.observations, rig.board, IMAGE_SIZE)


@pytest.fixture(scope="module")
def exact_rig():
    return synthetic_rig(noise_px=0.0)


def swept_roi(observations):
    """The sensor rectangle, in pixel edges, that the board's corners sweep in either camera."""
    points = np.concatenate([c for o in observations for c in o.corners.values()])
    # Corners are pixel-centre coordinates; pixel i spans edges [i - 0.5, i + 0.5) + 0.5.
    lo = np.floor(points.min(axis=0) + 0.5)
    hi = np.ceil(points.max(axis=0) + 0.5)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def capture_for(observations, unverified=0, views_kept=None, rejections=None, unreached=0):
    counted = {cam: sum(cam in o.corners for o in observations) for cam in CAMERAS}
    return CaptureSummary(
        waypoints_total=len(observations),
        waypoints_verified=len(observations) - unverified,
        waypoints_unreached=unreached,
        views_kept=counted if views_kept is None else views_kept,
        rejections={cam: {} for cam in CAMERAS} if rejections is None else rejections,
    )


def cross_checks(deltas=(0.1, 0.1), offsets=(0.01, 0.01)):
    """One cross-check per preset the pipeline fits directly (binning > 1), with these K
    deltas and corner offsets."""
    presets = [g for g in presets_from_base(BASE) if g.binning > 1]
    return [
        PresetCrossCheck(geometry=g, max_intrinsic_delta_px=d, corner_offset_px=o)
        for g, d, o in zip(presets, deltas, offsets)
    ]


def report_for(fit, observations, board, roi, *, previous=None, capture=None, checks=None):
    return build_report(
        fit,
        observations,
        board,
        roi,
        capture_for(observations) if capture is None else capture,
        cross_checks() if checks is None else checks,
        previous,
    )


def metric(report, name, camera=None):
    found = [m for m in report.metrics if m.name == name and m.camera == camera]
    assert len(found) == 1, f"expected one {name} for {camera}, got {found}"
    return found[0]


def not_passing(report):
    return [(m.name, m.camera, m.value, m.status) for m in report.metrics if m.status != "pass"]


def turned(model, angle_deg, axis):
    """A copy of camera `model` turned by `angle_deg` about `axis` (camera frame) about its own centre."""
    R = cv2.Rodrigues(np.asarray(model.rvec, float))[0]
    centre = -R.T @ np.asarray(model.tvec, float)
    axis = np.asarray(axis, float)
    R_new = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.radians(angle_deg))[0] @ R
    return dataclasses.replace(model, rvec=cv2.Rodrigues(R_new)[0].ravel(), tvec=-R_new @ centre)


def shifted(model, shift_mm):
    """A copy of camera `model` whose centre is moved by `shift_mm` (motor frame), not turned."""
    R = cv2.Rodrigues(np.asarray(model.rvec, float))[0]
    centre = -R.T @ np.asarray(model.tvec, float)
    return dataclasses.replace(model, tvec=-R @ (centre + np.asarray(shift_mm, float)))


def with_camera(fit, cam, model):
    return dataclasses.replace(fit, cameras={**fit.cameras, cam: model})


def test_clean_synthetic_fit_passes(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))

    assert report.verdict == "pass", not_passing(report)
    assert set(DEFAULT_THRESHOLDS) <= set(report.thresholds)
    assert report.thresholds == DEFAULT_THRESHOLDS
    # Every metric that applies without a previous calibration is graded, per camera where it is per camera.
    assert {m.name for m in report.metrics} == set(DEFAULT_THRESHOLDS) - CHANGE_METRICS
    for name in ("views_kept", "reprojection_rms_px", "coverage_roi_fraction", "fx_fy_ratio_dev",
                 "focal_rel_std", "principal_point_offset_frac", "max_distortion_shift_px"):
        for cam in CAMERAS:
            metric(report, name, cam)
    assert all(not math.isnan(m.value) for m in report.metrics)
    assert "no comparable previous calibration" in report.notes


def test_low_coverage_fails(rig, fitted):
    x0, y0, x1, y1 = BASE_ROI
    centre = np.array([(x0 + x1) / 2 - 0.5, (y0 + y1) / 2 - 0.5])
    distance = [
        np.linalg.norm(np.concatenate(list(o.corners.values())).mean(axis=0) - centre) for o in rig.observations
    ]
    nearest = [rig.observations[i] for i in np.argsort(distance)[:4]]

    report = report_for(fitted, nearest, rig.board, BASE_ROI)

    for cam in CAMERAS:
        coverage = metric(report, "coverage_roi_fraction", cam)
        assert coverage.status == "fail"
        assert coverage.value < 0.25
        # Four views is under both views_kept limits, which grade lower-is-worse.
        assert metric(report, "views_kept", cam).value == 4
        assert metric(report, "views_kept", cam).status == "fail"
    assert report.verdict == "fail"


def test_coverage_fraction_is_the_area_of_the_union_of_hulls_clipped_to_the_roi():
    roi = (100.0, 100.0, 900.0, 900.0)  # pixel edges, 800 px square

    def waypoint(index, x_edges, y_edges):
        # A 6x8 grid of corners whose hull spans the given pixel edges exactly.
        xs = np.linspace(*x_edges, 8) - 0.5
        ys = np.linspace(*y_edges, 6) - 0.5
        corners = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        return WaypointObservation(index=index, motor_mm=np.zeros(3), corners={"left": corners})

    left_half = waypoint(0, (100, 500), (100, 900))
    overlapping = waypoint(1, (300, 700), (100, 900))
    half_outside = waypoint(2, (-300, 300), (100, 900))

    # One mask pixel is 4 sensor px, 1/200 of the side; the rasteriser may add one column per edge.
    assert coverage_fraction([left_half], "left", roi) == pytest.approx(0.5, abs=0.011)
    assert coverage_fraction([left_half, overlapping], "left", roi) == pytest.approx(0.75, abs=0.011)
    assert coverage_fraction([half_outside], "left", roi) == pytest.approx(0.25, abs=0.011)
    assert coverage_fraction([left_half], "right", roi) == 0.0


def test_motor_scale_disagreement_only_warns(rig, fitted):
    fit = dataclasses.replace(fitted, motor_scale=1.02)

    report = report_for(fit, rig.observations, rig.board, swept_roi(rig.observations))

    scale = metric(report, "motor_scale_dev")
    assert scale.value == pytest.approx(0.02)
    assert scale.status == "warn"
    # The rest of the clean fit passes, so the scale alone makes the verdict warn, never fail.
    assert report.verdict == "warn", not_passing(report)


def test_previous_calibration_change_warns_camera_moved(rig, fitted):
    previous = with_camera(fitted, "left", turned(fitted.cameras["left"], 1.0, (0.2, 1.0, 0.1)))

    report = report_for(
        fitted, rig.observations, rig.board, swept_roi(rig.observations),
        previous=PreviousCalibration(legacy=False, fit=previous),
    )

    left = metric(report, "camera_rotation_change_deg", "left")
    assert left.value == pytest.approx(1.0, abs=1e-9)
    assert left.status == "warn"
    assert "moved" in left.note
    # Turned about its own centre, the left camera did not translate; the right one did not change.
    assert metric(report, "baseline_change_mm").value == pytest.approx(0.0, abs=1e-9)
    assert metric(report, "camera_rotation_change_deg", "right").status == "pass"
    assert metric(report, "camera_rotation_change_deg", "right").value == pytest.approx(0.0, abs=1e-9)
    assert any("left" in note and "moved" in note for note in report.notes)
    assert "no comparable previous calibration" not in report.notes


def test_a_common_shift_of_both_cameras_does_not_read_as_moved(rig, fitted):
    # Remounting the board moves the motor frame's origin, which shifts every camera centre
    # by the same vector: the cameras did not move relative to each other.
    shift = (3.0, -2.0, 5.0)
    previous = with_camera(
        with_camera(fitted, "left", shifted(fitted.cameras["left"], shift)),
        "right", shifted(fitted.cameras["right"], shift),
    )

    report = report_for(
        fitted, rig.observations, rig.board, swept_roi(rig.observations),
        previous=PreviousCalibration(legacy=False, fit=previous),
    )

    baseline = metric(report, "baseline_change_mm")
    assert baseline.value == pytest.approx(0.0, abs=1e-9)
    assert baseline.status == "pass"
    assert "moved" not in baseline.note
    assert not any("moved" in note for note in report.notes)
    assert "camera_translation_change_mm" not in {m.name for m in report.metrics}
    assert report.verdict == "pass", not_passing(report)


def test_moving_one_camera_changes_the_baseline(rig, fitted):
    previous = with_camera(fitted, "right", shifted(fitted.cameras["right"], (1.2, -1.6, 0.0)))  # 2 mm

    report = report_for(
        fitted, rig.observations, rig.board, swept_roi(rig.observations),
        previous=PreviousCalibration(legacy=False, fit=previous),
    )

    baseline = metric(report, "baseline_change_mm")
    assert baseline.value == pytest.approx(2.0, abs=1e-9)
    assert baseline.status == "warn"
    assert "moved" in baseline.note
    assert any("moved" in note for note in report.notes)
    assert metric(report, "camera_rotation_change_deg", "right").value == pytest.approx(0.0, abs=1e-9)


def test_previous_calibration_measures_focal_change(rig, fitted):
    left = fitted.cameras["left"]
    left_K = left.K.copy()
    left_K[0, 0] *= 1.03
    previous = with_camera(fitted, "left", dataclasses.replace(left, K=left_K))

    report = report_for(
        fitted, rig.observations, rig.board, swept_roi(rig.observations),
        previous=PreviousCalibration(legacy=False, fit=previous),
    )

    focal = metric(report, "focal_change_rel", "left")
    assert focal.value == pytest.approx(abs(1 / 1.03 - 1))
    assert focal.status == "warn"
    assert metric(report, "focal_change_rel", "right").status == "pass"


@pytest.mark.parametrize(
    "previous",
    [None, "legacy", "legacy_with_fit", "no_fit"],
)
def test_legacy_or_missing_previous_calibration_is_noted_without_change_metrics(rig, fitted, previous):
    previous = {
        None: None,
        "legacy": PreviousCalibration(legacy=True, fit=None),
        "legacy_with_fit": PreviousCalibration(legacy=True, fit=fitted),
        "no_fit": PreviousCalibration(legacy=False, fit=None),
    }[previous]

    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations), previous=previous)

    assert "no comparable previous calibration" in report.notes
    assert not CHANGE_METRICS & {m.name for m in report.metrics}


def test_json_round_trip(rig, fitted):
    previous = with_camera(fitted, "left", turned(fitted.cameras["left"], 1.0, (0.2, 1.0, 0.1)))
    report = report_for(
        fitted, rig.observations, rig.board, swept_roi(rig.observations),
        previous=PreviousCalibration(legacy=False, fit=previous),
    )
    report.override = {"confirmed": True, "by": "tester", "at": "2026-10-20T12:00:00"}

    assert QualityReport.from_json(report.to_json()) == report


def test_json_round_trip_keeps_undetermined_parameters_as_strict_json(rig, fitted):
    param_std = {**fitted.param_std, "left": {**fitted.param_std["left"], "fx": math.inf}}
    fit = dataclasses.replace(fitted, param_std=param_std)

    report = report_for(fit, rig.observations, rig.board, swept_roi(rig.observations))

    focal = metric(report, "focal_rel_std", "left")
    assert focal.value == math.inf
    assert focal.status == "fail"
    assert focal.note
    assert report.verdict == "fail"
    text = report.to_json()

    def reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    json.loads(text, parse_constant=reject)
    assert QualityReport.from_json(text) == report


def test_text_lists_failures_first(rig, fitted):
    fit = dataclasses.replace(fitted, motor_scale=1.02)  # a warn among the fails
    report = report_for(fit, rig.observations[:4], rig.board, BASE_ROI)
    assert {m.status for m in report.metrics} == {"fail", "warn", "pass"}

    lines = report.to_text().splitlines()

    assert "FAIL" in lines[0]
    metric_lines = [line for line in lines if line.split() and line.split()[0] in STATUS_WORDS]
    statuses = [line.split()[0] for line in metric_lines]
    assert statuses[0] == "FAIL"
    assert statuses == sorted(statuses, key=STATUS_WORDS.index)
    assert len(metric_lines) == len(report.metrics)
    # Each line names the metric and its camera.
    for m in report.metrics:
        assert any(m.name in line and (m.camera or "-") in line for line in metric_lines)


def test_noise_free_truth_has_no_stereo_error(exact_rig):
    report = report_for(exact_rig.truth, exact_rig.observations, exact_rig.board, swept_roi(exact_rig.observations))

    assert metric(report, "epipolar_error_px").value < 1e-6
    assert metric(report, "square_size_rel_error").value < 1e-8
    assert report.verdict == "pass", not_passing(report)


def test_epipolar_error_detects_a_stereo_inconsistency(exact_rig):
    # Pitching the right camera by 0.1 degrees moves its image about 4 px across the
    # roughly horizontal epipolar lines.
    fit = with_camera(exact_rig.truth, "right", turned(exact_rig.truth.cameras["right"], 0.1, (1.0, 0.0, 0.0)))

    report = report_for(fit, exact_rig.observations, exact_rig.board, swept_roi(exact_rig.observations))

    epipolar = metric(report, "epipolar_error_px")
    assert epipolar.value > 1.0
    assert epipolar.status == "fail"


def test_distortion_shift_is_the_largest_radial_shift_on_the_roi_border(exact_rig):
    left = exact_rig.truth.cameras["left"]
    k1 = -0.25
    fit = with_camera(exact_rig.truth, "left", dataclasses.replace(left, dist=np.array([k1, 0.0, 0.0, 0.0, 0.0])))

    report = report_for(fit, exact_rig.observations, exact_rig.board, BASE_ROI)

    # With k1 alone the shift is |k1| r^2 times the distance from the principal point, so it
    # is largest at the ROI corner farthest from it. Corners in pixel-centre coordinates:
    x0, y0, x1, y1 = BASE_ROI
    fx, fy, cx, cy = left.K[0, 0], left.K[1, 1], left.K[0, 2], left.K[1, 2]
    expected = 0.0
    for u, v in [(x0 - 0.5, y0 - 0.5), (x1 - 0.5, y0 - 0.5), (x0 - 0.5, y1 - 0.5), (x1 - 0.5, y1 - 0.5)]:
        x, y = (u - cx) / fx, (v - cy) / fy
        r2 = x * x + y * y
        expected = max(expected, math.hypot(fx * x * k1 * r2, fy * y * k1 * r2))
    assert metric(report, "max_distortion_shift_px", "left").value == pytest.approx(expected, rel=1e-9)


def test_principal_point_offset_is_measured_from_the_roi_centre(exact_rig):
    x0, y0, x1, y1 = BASE_ROI
    left = exact_rig.truth.cameras["left"]
    K = left.K.copy()
    # The ROI centre in pixel-centre coordinates, then 0.3 of the half-side to the right.
    K[0, 2] = (x0 + x1) / 2 - 0.5 + 0.3 * (x1 - x0) / 2
    K[1, 2] = (y0 + y1) / 2 - 0.5
    fit = with_camera(exact_rig.truth, "left", dataclasses.replace(left, K=K))

    report = report_for(fit, exact_rig.observations, exact_rig.board, BASE_ROI)

    offset = metric(report, "principal_point_offset_frac", "left")
    assert offset.value == pytest.approx(0.3)
    assert offset.status == "warn"


def test_capture_problems_and_cross_checks_are_graded(rig, fitted):
    capture = capture_for(
        rig.observations,
        unverified=1,
        views_kept={"left": 15, "right": 37},
        rejections={"left": {"blurred": 2, "no board": 1}, "right": {}},
    )

    report = report_for(
        fitted, rig.observations, rig.board, swept_roi(rig.observations),
        capture=capture, checks=cross_checks(deltas=(0.1, 2.0), offsets=(0.15, 0.3)),
    )

    assert metric(report, "waypoints_unverified").value == 1
    assert metric(report, "waypoints_unverified").status == "warn"
    assert metric(report, "waypoints_unreached").value == 0
    assert metric(report, "waypoints_unreached").status == "pass"
    # The capture's own count wins over the observations'.
    assert metric(report, "views_kept", "left").value == 15
    assert metric(report, "views_kept", "left").status == "warn"
    assert any("left" in note and "blurred 2" in note for note in report.notes)
    # The K delta is limited by fit noise, so it only ever warns, however large.
    assert DEFAULT_THRESHOLDS["cross_check_delta_px"] == (1.5, None)
    deltas = {m.note: m.status for m in report.metrics if m.name == "cross_check_delta_px"}
    assert deltas == {"preset 256x256_bin4": "pass", "preset 512x512_bin2": "warn"}
    # The corner offset checks the mapping itself, to the spec's 0.1 px.
    assert DEFAULT_THRESHOLDS["cross_check_corner_offset_px"] == (0.1, 0.25)
    offsets = {m.note: m.status for m in report.metrics if m.name == "cross_check_corner_offset_px"}
    assert offsets == {"preset 256x256_bin4": "warn", "preset 512x512_bin2": "fail"}
    assert report.verdict == "fail"


def test_a_waypoint_the_stage_did_not_reach_fails_the_calibration(rig, fitted):
    # The fit tags each view with the commanded position, so one waypoint the motor reported
    # elsewhere is a view placed in the wrong spot: the calibration fails, not just warns.
    assert DEFAULT_THRESHOLDS["waypoints_unreached"] == (None, 0.0)
    clean = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    assert metric(clean, "waypoints_unreached").value == 0 and clean.verdict == "pass"

    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations),
                        capture=capture_for(rig.observations, unreached=1))

    assert metric(report, "waypoints_unreached").value == 1
    assert metric(report, "waypoints_unreached").status == "fail"
    assert report.verdict == "fail"
    assert not_passing(report) == [("waypoints_unreached", None, 1.0, "fail")]


def test_write_coverage_images(rig, tmp_path):
    paths = write_coverage_images(rig.observations, IMAGE_SIZE, BASE_ROI, tmp_path)

    assert [p.name for p in paths] == ["coverage_left.png", "coverage_right.png"]
    for path in paths:
        assert path.parent == tmp_path
        assert path.stat().st_size > 0
        image = cv2.imread(str(path))
        assert image is not None
        assert image.shape[:2] == (IMAGE_SIZE[1], IMAGE_SIZE[0])
        assert image.std() > 0
