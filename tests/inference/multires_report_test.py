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

from reachaq.inference.multires_calibration.fit import MotorPlacement, WaypointObservation, fit_motor_frame
from reachaq.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from reachaq.inference.multires_calibration.pipeline import per_camera_rule
from reachaq.inference.multires_calibration.report import (
    DEFAULT_THRESHOLDS,
    LOO_UNDETERMINED_NOTE,
    STAGE_HEADING,
    STAGE_METRICS,
    UNDETERMINED,
    CameraVerdict,
    CaptureSummary,
    PresetCrossCheck,
    PreviousCalibration,
    QualityReport,
    build_report,
    coverage_fraction,
    planar_rms_px,
    write_coverage_images,
)
from multires_synthetic import synthetic_rig


BASE = PresetGeometry(4, 256, 256, 52, 6)
PRESETS = presets_from_base(BASE)
BASE_ROI = BASE.sensor_roi()  # (208, 24, 1232, 1048), pixel edges
IMAGE_SIZE = (1440, 1080)
CAMERAS = ("left", "right")
PAIR = "left-right"  # a pair metric's camera
CHANGE_METRICS = {"camera_rotation_change_deg", "baseline_change_mm", "focal_change_rel"}
STATUS_WORDS = ("FAIL", "WARN", "PASS")


@pytest.fixture(scope="module")
def rig():
    return synthetic_rig(noise_px=0.1)


@pytest.fixture(scope="module")
def fitted(rig):
    return fit_motor_frame(rig.observations, rig.board, IMAGE_SIZE, CAMERAS)


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
    cameras = sorted({cam for o in observations for cam in o.corners})
    counted = {cam: sum(cam in o.corners for o in observations) for cam in cameras}
    return CaptureSummary(
        waypoints_total=len(observations),
        waypoints_verified=len(observations) - unverified,
        waypoints_unreached=unreached,
        views_kept=counted if views_kept is None else views_kept,
        rejections={cam: {} for cam in cameras} if rejections is None else rejections,
    )


def cross_checks(deltas=(0.1, 0.1), offsets=(0.01, 0.01)):
    """One cross-check per preset the pipeline fits directly (binning > 1), with these K
    deltas and corner offsets."""
    presets = [g for g in presets_from_base(BASE) if g.binning > 1]
    return [
        PresetCrossCheck(geometry=g, max_intrinsic_delta_px=d, corner_offset_px=o)
        for g, d, o in zip(presets, deltas, offsets)
    ]


def report_for(fit, observations, board, roi, *, previous=None, capture=None, checks=None, cameras=None):
    return build_report(
        fit,
        observations,
        board,
        roi,
        capture_for(observations) if capture is None else capture,
        cross_checks() if checks is None else checks,
        previous,
        presets=PRESETS,
        cameras=cameras,
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
    # Every metric that applies without a previous calibration is graded, per camera where it
    # is per camera; the leave-one-camera-out check needs a third camera.
    # The stage block's slack and repeat visits need a per-view fit's placement.
    assert {m.name for m in report.metrics} == (
        set(DEFAULT_THRESHOLDS) - CHANGE_METRICS
        - {"loo_triangulation_px", "focal_vs_nominal_rel", "stage_slack_rms_mm", "stage_repeat_mm"})
    for name in ("views_kept", "covisible_views", "reprojection_rms_px", "joint_reprojection_rms_px",
                 "coverage_roi_fraction", "fx_fy_ratio_dev",
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
    assert metric(report, "baseline_change_mm", PAIR).value == pytest.approx(0.0, abs=1e-9)
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

    baseline = metric(report, "baseline_change_mm", PAIR)
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

    baseline = metric(report, "baseline_change_mm", PAIR)
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
    # Failures first within the calibration's metrics and within the stage block after them.
    stage_at = lines.index(STAGE_HEADING)
    for block in ([line for line in metric_lines if lines.index(line) < stage_at],
                  [line for line in metric_lines if lines.index(line) > stage_at]):
        block_statuses = [line.split()[0] for line in block]
        assert block and block_statuses == sorted(block_statuses, key=STATUS_WORDS.index)
    assert all(line.split()[1] in STAGE_METRICS for line in metric_lines if lines.index(line) > stage_at)
    assert len(metric_lines) == len(report.metrics)
    # Each line names the metric and its camera.
    for m in report.metrics:
        assert any(m.name in line and (m.camera or "-") in line for line in metric_lines)


def test_noise_free_truth_has_no_stereo_error(exact_rig):
    report = report_for(exact_rig.truth, exact_rig.observations, exact_rig.board, swept_roi(exact_rig.observations))

    assert metric(report, "epipolar_error_px", PAIR).value < 1e-6
    assert metric(report, "square_size_rel_error", PAIR).value < 1e-8
    assert report.verdict == "pass", not_passing(report)


def test_epipolar_error_detects_a_stereo_inconsistency(exact_rig):
    # Pitching the right camera by 0.1 degrees moves its image about 4 px across the
    # roughly horizontal epipolar lines.
    fit = with_camera(exact_rig.truth, "right", turned(exact_rig.truth.cameras["right"], 0.1, (1.0, 0.0, 0.0)))

    report = report_for(fit, exact_rig.observations, exact_rig.board, swept_roi(exact_rig.observations))

    epipolar = metric(report, "epipolar_error_px", PAIR)
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


# Informational metrics (ruling R50): the spec's report contents that grade nothing.
INFORMATIONAL = ("stereo_reprojection_rms_px", "preset_reprojection_rms_px", "baseline_mm", "coverage_frame_fraction")


def test_informational_metrics_have_no_limits_and_never_grade(rig, fitted):
    assert {name: DEFAULT_THRESHOLDS[name] for name in INFORMATIONAL} == {name: (None, None) for name in INFORMATIONAL}
    # Few views and a small ROI fail other metrics; these stay pass whatever their value.
    report = report_for(fitted, rig.observations[:4], rig.board, BASE_ROI)
    informational = [m for m in report.metrics if m.name in INFORMATIONAL]
    assert {m.name for m in informational} == set(INFORMATIONAL)
    assert all(m.status == "pass" and m.warn is None and m.fail is None for m in informational)


def test_the_stereo_fit_rms_pools_every_corner_of_both_cameras(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    # Every view has the same corners, so pooling the per-view RMS pools the corners.
    per_view = [v for cam in CAMERAS for v in fitted.per_view_rms_px[cam].values()]
    joint = metric(report, "stereo_reprojection_rms_px")
    assert joint.value == pytest.approx(math.sqrt(np.mean(np.square(per_view))), rel=1e-12)
    assert joint.unit == "sensor px"
    assert min(fitted.rms_px.values()) <= joint.value <= max(fitted.rms_px.values())


def test_reprojection_rms_is_reported_in_each_presets_px(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    for cam in CAMERAS:
        found = {m.note: m for m in report.metrics if m.name == "preset_reprojection_rms_px" and m.camera == cam}
        assert set(found) == {f"preset {g.name}" for g in PRESETS}, cam
        for g in PRESETS:
            m = found[f"preset {g.name}"]
            assert m.value == pytest.approx(fitted.rms_px[cam] / g.binning, rel=1e-12) and m.unit == "preset px"


def test_the_baseline_length_is_the_distance_between_the_camera_centres(exact_rig):
    report = report_for(exact_rig.truth, exact_rig.observations, exact_rig.board, swept_roi(exact_rig.observations))
    centres = {}
    for cam in CAMERAS:
        model = exact_rig.truth.cameras[cam]
        centres[cam] = -cv2.Rodrigues(np.asarray(model.rvec, float))[0].T @ np.asarray(model.tvec, float).reshape(3)
    baseline = metric(report, "baseline_mm", PAIR)
    assert baseline.value == pytest.approx(float(np.linalg.norm(centres["right"] - centres["left"])), rel=1e-12)
    assert baseline.unit == "mm" and baseline.value > 0


def test_coverage_of_the_full_frame_is_reported_beside_the_roi_coverage(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, BASE_ROI)
    for cam in CAMERAS:
        full = metric(report, "coverage_frame_fraction", cam)
        assert full.value == pytest.approx(coverage_fraction(rig.observations, cam, (0, 0, *IMAGE_SIZE)))
        assert 0 < full.value <= 1 and full.unit == "of frame"
        metric(report, "coverage_roi_fraction", cam)


def test_verified_means_the_motor_reported_the_commanded_position(rig, fitted):
    # The motor's position report echoes its target about 0.1 s after the command
    # (christielab10 motor check, 2026-10-26), so "verified" is the motor's word, not a
    # traced arrival (ruling R53), and the report says so.
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    note = metric(report, "waypoints_unverified").note
    assert "the motor reported the commanded position" in note and "not a traced arrival" in note


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


THREE = ("left", "right", "top")


@pytest.fixture(scope="module")
def rig3():
    return synthetic_rig(cameras=THREE, noise_px=0.1)


@pytest.fixture(scope="module")
def exact_rig3():
    return synthetic_rig(cameras=THREE, noise_px=0.0)


def test_pair_metrics_cover_every_covisible_pair(exact_rig3):
    report = report_for(exact_rig3.truth, exact_rig3.observations, exact_rig3.board,
                        swept_roi(exact_rig3.observations))
    pairs = ["left-right", "left-top", "right-top"]
    for name in ("epipolar_error_px", "square_size_rel_error", "focal_pair_rel", "baseline_mm"):
        assert [m.camera for m in report.metrics if m.name == name] == pairs, name
    for pair in pairs:
        assert metric(report, "epipolar_error_px", pair).value < 1e-6
        assert metric(report, "square_size_rel_error", pair).value < 1e-8
    assert report.verdict == "pass", not_passing(report)


def test_a_pair_that_shares_no_waypoint_has_no_pair_metrics(exact_rig3):
    # top sees only the first half of the move list and right only the second, so the two
    # never see the board together; left sees all of it.
    observations = [
        WaypointObservation(o.index, o.motor_mm, {
            cam: c for cam, c in o.corners.items()
            if not (cam == "top" and o.index >= 18) and not (cam == "right" and o.index < 18)
        })
        for o in exact_rig3.observations
    ]
    report = report_for(exact_rig3.truth, observations, exact_rig3.board, swept_roi(observations))
    assert [m.camera for m in report.metrics if m.name == "epipolar_error_px"] == ["left-right", "left-top"]


def test_covisible_views_count_the_views_shared_with_another_camera(rig, fitted):
    observations = [
        WaypointObservation(o.index, o.motor_mm, dict(o.corners)) for o in rig.observations
    ]
    for o in observations[:30]:
        del o.corners["right"]
    report = report_for(fitted, observations, rig.board, swept_roi(observations))
    # Both cameras saw the last 7 waypoints; the left camera saw 30 more on its own.
    for cam in CAMERAS:
        shared = metric(report, "covisible_views", cam)
        assert shared.value == 7 and shared.status == "warn" and shared.lower_is_worse
    assert DEFAULT_THRESHOLDS["covisible_views"] == (8.0, 4.0)


def test_loo_triangulation_is_small_on_the_truth(rig3):
    report = report_for(rig3.truth, rig3.observations, rig3.board, swept_roi(rig3.observations))
    assert DEFAULT_THRESHOLDS["loo_triangulation_px"] == (0.5, 1.0)
    for cam in THREE:
        loo = metric(report, "loo_triangulation_px", cam)
        assert loo.value < 0.3 and loo.status == "pass", loo
        assert loo.unit == "sensor px"


def test_loo_triangulation_catches_a_camera_the_others_disagree_with(exact_rig3):
    # The top camera turned 0.1 degrees: the other two still agree with each other, so
    # it is top's own detections that land away from what they triangulate.
    fit = with_camera(exact_rig3.truth, "top", turned(exact_rig3.truth.cameras["top"], 0.1, (1.0, 0.0, 0.0)))
    report = report_for(fit, exact_rig3.observations, exact_rig3.board, swept_roi(exact_rig3.observations))
    assert metric(report, "loo_triangulation_px", "top").value > 1.0
    assert metric(report, "loo_triangulation_px", "top").status == "fail"


def test_uncalibrated_cameras_come_first_and_make_the_verdict_warn(rig, fitted):
    verdicts = {
        "left": CameraVerdict("left", True, ""),
        "right": CameraVerdict("right", True, ""),
        "camera4": CameraVerdict("camera4", False, "too few board views: 4 of 12 needed"),
    }
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations), cameras=verdicts)
    assert report.cameras == verdicts
    assert list(report.cameras) == ["left", "right", "camera4"]
    # Every metric passes; the 2D-only camera alone makes the verdict warn.
    assert not not_passing(report)
    assert report.verdict == "warn"
    lines = report.to_text().splitlines()
    assert lines[0] == "NOT CALIBRATED (2D only): camera4: too few board views: 4 of 12 needed"
    assert "WARN" in lines[1]
    assert QualityReport.from_json(report.to_json()) == report
    assert json.loads(report.to_json())["cameras"]["camera4"] == {
        "name": "camera4", "calibrated": False, "reason": "too few board views: 4 of 12 needed"}


def test_every_fitted_camera_is_calibrated_by_default(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    assert report.cameras == {cam: CameraVerdict(cam, True, "") for cam in CAMERAS}
    assert not report.to_text().startswith("NOT CALIBRATED")


def test_a_report_written_before_the_camera_verdicts_still_loads(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    data = json.loads(report.to_json())
    del data["cameras"]
    assert QualityReport.from_json(json.dumps(data)).cameras == {}


def verdict_line(report):
    (line,) = [line for line in report.to_text().splitlines() if line.startswith("3D calibration quality:")]
    return line


def test_two_camera_good_capture_passes_with_note(rig, fitted):
    # With 2 cameras no camera can be checked against points the other two triangulate: the
    # calibration passes, and says it was not checked that way.
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    note = "passed; cross-camera triangulation check not possible with 2 cameras"
    assert report.verdict == "pass", not_passing(report)
    assert report.verdict_note == note
    assert verdict_line(report).startswith("3D calibration quality: PASS (")
    assert verdict_line(report).endswith(f" - {note}")
    assert json.loads(report.to_json())["verdict_note"] == note
    assert QualityReport.from_json(report.to_json()) == report
    # The note follows the verdict: a fail says fail.
    failing = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations),
                         capture=capture_for(rig.observations, unreached=1))
    assert failing.verdict_note == "fail; cross-camera triangulation check not possible with 2 cameras"
    # A report written before the note still loads, without one.
    data = json.loads(report.to_json())
    del data["verdict_note"]
    assert QualityReport.from_json(json.dumps(data)).verdict_note == ""


def test_chain_topology_is_not_failed_by_loo():
    # C sees 0-15, B all 37, A 16-36: every waypoint has exactly two cameras, so no camera
    # can be checked against two others. Each camera still meets the per-camera rule.
    seen = {"cam_c": range(16), "cam_b": range(37), "cam_a": range(16, 37)}
    cameras = ("cam_a", "cam_b", "cam_c")
    chain = synthetic_rig(cameras=cameras, seen=seen, noise_px=0.1)
    verdicts = per_camera_rule(chain.observations, cameras)
    assert all(v.calibrated for v in verdicts.values()), verdicts
    fit = fit_motor_frame(chain.observations, chain.board, IMAGE_SIZE, cameras)

    report = report_for(fit, chain.observations, chain.board, swept_roi(chain.observations), cameras=verdicts)

    assert report.verdict != "fail", not_passing(report)
    loo = [m for m in report.metrics if m.name == "loo_triangulation_px"]
    assert [m.camera for m in loo] == list(cameras)
    assert all(m.status == UNDETERMINED and m.value is None and m.note == LOO_UNDETERMINED_NOTE for m in loo)
    word = {"pass": "passed", "warn": "warn"}[report.verdict]
    assert report.verdict_note == f"{word}; cross-camera triangulation check not possible for any camera"
    assert verdict_line(report).endswith(f" - {report.verdict_note}")


def test_undetermined_loo_is_shown_and_not_a_pass():
    # left sees every waypoint, right and top 0-18 with it, camera4 19-36 with left alone:
    # camera4 cannot be checked, the other three can.
    cameras = ("left", "right", "top", "camera4")
    seen = {"right": range(19), "top": range(19), "camera4": range(19, 37)}
    four = synthetic_rig(cameras=cameras, seen=seen, noise_px=0.0)

    report = report_for(four.truth, four.observations, four.board, swept_roi(four.observations))

    undetermined = metric(report, "loo_triangulation_px", "camera4")
    assert undetermined.status == UNDETERMINED and undetermined.status != "pass"
    assert undetermined.value is None
    assert undetermined.note == LOO_UNDETERMINED_NOTE
    assert (undetermined.warn, undetermined.fail) == DEFAULT_THRESHOLDS["loo_triangulation_px"]
    for cam in ("left", "right", "top"):
        assert metric(report, "loo_triangulation_px", cam).status == "pass"
    # Some cameras were checked, so the verdict carries no note, and the undetermined check
    # does not enter it: the verdict is the warn of the 18 and 19 views kept alone.
    assert report.verdict_note == ""
    assert report.verdict == "warn"
    assert {(name, status) for name, _, _, status in not_passing(report)} == {
        ("views_kept", "warn"), ("loo_triangulation_px", UNDETERMINED)}
    # The text shows it, as undetermined with no value, and counts it apart from the passes.
    lines = report.to_text().splitlines()
    (line,) = [line for line in lines if "loo_triangulation_px" in line and "camera4" in line]
    assert line.startswith("UNDETERMINED loo_triangulation_px") and LOO_UNDETERMINED_NOTE in line
    assert ", 1 undetermined;" in verdict_line(report)
    # It survives the JSON as null, not as a number.
    data = json.loads(report.to_json())
    (saved,) = [m for m in data["metrics"] if m["name"] == "loo_triangulation_px" and m["camera"] == "camera4"]
    assert saved["value"] is None and saved["status"] == UNDETERMINED
    assert QualityReport.from_json(report.to_json()) == report


# The graded RMS (Ben, 2026-11-24): each camera's own planar fit, so stage error cannot fail
# a calibration; the stage-tied joint RMS only warns, and the stage has its own block.


def test_the_graded_rms_is_each_cameras_planar_fit(rig, fitted):
    report = report_for(fitted, rig.observations, rig.board, swept_roi(rig.observations))
    obj = rig.board.object_points().astype(np.float32)
    for cam in CAMERAS:
        model = fitted.cameras[cam]
        image = [o.corners[cam].astype(np.float32).reshape(-1, 1, 2) for o in rig.observations if cam in o.corners]
        dist = np.zeros(5)
        dist[:4] = np.ravel(model.dist)[:4]
        rms = cv2.calibrateCamera([obj] * len(image), image, IMAGE_SIZE, np.array(model.K, float), dist,
                                  flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)[0]
        graded = metric(report, "reprojection_rms_px", cam)
        assert graded.value == pytest.approx(rms, rel=1e-3)
        assert graded.value == pytest.approx(planar_rms_px(rig.observations, rig.board, IMAGE_SIZE, cam, model)[0])
        # Free poses fit at least as well as poses tied to the stage.
        assert graded.value <= fitted.rms_px[cam] + 1e-6
        joint = metric(report, "joint_reprojection_rms_px", cam)
        assert joint.value == pytest.approx(fitted.rms_px[cam], rel=1e-12)
        assert (joint.warn, joint.fail) == (1.0, None)


def test_holding_the_principal_point_holds_it_in_the_planar_fit(rig, fitted):
    cam = CAMERAS[0]
    model = fitted.cameras[cam]
    off = dataclasses.replace(model, K=np.array(model.K, float) + np.array([[0, 0, 40.0], [0, 0, -30.0], [0, 0, 0]]))
    free, free_note = planar_rms_px(rig.observations, rig.board, IMAGE_SIZE, cam, off)
    held, held_note = planar_rms_px(rig.observations, rig.board, IMAGE_SIZE, cam, off, fix_principal_point=True)
    assert held > free
    assert "principal point held" in held_note and "principal point held" not in free_note


@pytest.fixture(scope="module")
def tilting_rig():
    # The board turns up to 3 degrees between waypoints: a stage error the motor-frame model,
    # a pure translation, cannot take up.
    return synthetic_rig(noise_px=0.1, tilt_deg=3.0, stage_error_mm=0.05)


def test_stage_error_passes_on_the_planar_rms_and_only_warns_on_the_joint_rms(tilting_rig):
    fit = fit_motor_frame(tilting_rig.observations, tilting_rig.board, IMAGE_SIZE, CAMERAS)
    report = report_for(fit, tilting_rig.observations, tilting_rig.board, swept_roi(tilting_rig.observations))
    for cam in CAMERAS:
        graded = metric(report, "reprojection_rms_px", cam)
        assert graded.status == "pass" and graded.value < 0.2, graded
        joint = metric(report, "joint_reprojection_rms_px", cam)
        assert joint.status == "warn" and joint.value > 1.0, joint


def test_the_stage_block_reports_slack_and_repeat_visits_and_never_fails(rig, fitted):
    # Waypoint 10 is commanded back to waypoint 3's position, where the board stood 2 mm higher in z.
    first, again = rig.observations[3], rig.observations[10]
    observations = [dataclasses.replace(o, motor_mm=first.motor_mm) if o is again else o for o in rig.observations]
    residual = {o.index: (0.0, 0.0, 2.0 if o.index == again.index else 0.0) for o in observations}
    placement = MotorPlacement(scale=1.0, rms_mm=0.0, residual_mm=residual, tilt_deg={}, outliers=[])
    report = report_for(dataclasses.replace(fitted, placement=placement), observations, rig.board,
                        swept_roi(observations))

    repeat = metric(report, "stage_repeat_mm", f"{first.index}-{again.index}")
    assert repeat.value == pytest.approx(2.0) and repeat.status == "warn"
    assert "z +2.00" in repeat.note
    slack = {axis: metric(report, "stage_slack_rms_mm", axis) for axis in "xyz"}
    assert slack["z"].value == pytest.approx(2.0 / math.sqrt(len(observations)))
    assert slack["x"].value == 0.0 and slack["y"].value == 0.0
    assert all(DEFAULT_THRESHOLDS[name][1] is None for name in STAGE_METRICS)
    assert report.verdict == "warn", not_passing(report)
    lines = report.to_text().splitlines()
    stage_at = lines.index(STAGE_HEADING)
    stage_names = {line.split()[1] for line in lines[stage_at + 1:] if line.split() and line.split()[0] in STATUS_WORDS}
    assert stage_names >= set(STAGE_METRICS)
