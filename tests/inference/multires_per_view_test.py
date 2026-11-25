"""The per-view fit: a free board rotation at every view, its translation held to the motor
similarity by a robust slack.

The synthetic stage here turns the board by up to 3 degrees at each waypoint, places it within
0.2 mm of where it was commanded, and loses 2.5-3 mm at three waypoints, as christielab10's
stage seemed to on session010 (2026-11-23). The per-view fit must recover the cameras at the
noise level; the motor-frame fit, whose model is a pure translation, and the same fit with
the rotation held, must not, which shows the data tells them apart. With the translations
entirely free (prior_mm=None) the focal length is undetermined: every view is the same plane
orientation.
"""

import math

import cv2
import numpy as np
import pytest

from reachaq.inference.multires_calibration import pipeline, store
from reachaq.inference.multires_calibration.fit import (
    RIG_CHARUCO_BOARD,
    BoardSpec,
    WaypointObservation,
    fit_motor_frame,
)
from reachaq.inference.multires_calibration.geometry import (
    PresetGeometry,
    map_intrinsics,
    presets_from_base,
    sensor_to_preset_points,
)
from reachaq.inference.multires_calibration.per_view import (
    FIT_MOTOR_FRAME,
    FIT_PER_VIEW,
    configured_fit,
    default_fit_mode,
    fit_per_view,
)
from reachaq.inference.multires_calibration.products import (
    frame_transform,
    motor_to_inference_signs,
    preset_stereo_params,
)
from reachaq.inference.multires_calibration.report import (
    DEFAULT_THRESHOLDS,
    CaptureSummary,
    build_report,
    placement_notes,
)
from multires_synthetic import synthetic_rig

SIZE = (1440, 1080)
CAMERAS = ("left", "right")
BASE = PresetGeometry(4, 256, 256, 52, 6)
BOARD = RIG_CHARUCO_BOARD


def angle_deg(R):
    return math.degrees(float(np.linalg.norm(cv2.Rodrigues(np.asarray(R, float))[0])))


def rotation(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float))[0]


@pytest.fixture(scope="module")
def stage_rig():
    # Placed within 0.05 mm elsewhere: the christielab10 motor check measured arrivals within
    # 0.007 mm. Random placement error is what limits the focal length here: at 0.2 mm, fx
    # came out 1.3 % short.
    return synthetic_rig(cameras=("left", "right", "top"), noise_px=0.1, tilt_deg=3.0, stage_error_mm=0.05,
                         stage_offsets=LOST_STEPS)


# Waypoint -> the stage's error there, mm: the views the slack must absorb.
LOST_STEPS = {5: (0.0, 0.0, -3.0), 17: (0.0, 0.0, -3.0), 29: (2.5, 0.0, 0.0)}


def assert_cameras_match(fit, truth, cams, *, f_rel=5e-3, c_px=10.0, deg=0.2, mm=0.5):
    for cam in cams:
        t, f = truth.cameras[cam], fit.cameras[cam]
        assert f.K[0, 0] == pytest.approx(t.K[0, 0], rel=f_rel), cam
        assert f.K[1, 1] == pytest.approx(t.K[1, 1], rel=f_rel), cam
        assert abs(f.K[0, 2] - t.K[0, 2]) < c_px and abs(f.K[1, 2] - t.K[1, 2]) < c_px, cam
    # The cameras' poses relative to one another, which no gauge or placement changes.
    first = cams[0]
    for cam in cams[1:]:
        R_fit, T_fit = fit.stereo(first, cam)
        R_true, T_true = truth.stereo(first, cam)
        assert angle_deg(R_fit.T @ R_true) < deg, cam
        assert np.linalg.norm(T_fit - T_true) < mm, cam


def test_per_view_recovers_the_cameras_of_a_tilting_inexact_stage(stage_rig):
    cams = ("left", "right", "top")
    fit = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    assert_cameras_match(fit, stage_rig.truth, cams)
    for cam in cams:
        # 0.1 px noise per coordinate is 0.141 px per corner.
        assert fit.rms_px[cam] < 0.17, cam
        assert math.isfinite(fit.param_std[cam]["fx"]), cam


def test_the_motor_frame_fit_cannot_fit_the_same_capture(stage_rig):
    cams = ("left", "right", "top")
    fit = fit_motor_frame(stage_rig.observations, BOARD, SIZE, cams)
    assert max(fit.rms_px.values()) > 2.0


@pytest.mark.parametrize("pivot", [
    (0.0, 0.0, 0.0),  # corner 0
    (14.0, 10.0, 0.0),  # the board centre
    (14.0, 10.0, 30.0),  # 30 mm behind the centre
    (34.0, 10.0, 0.0),  # 20 mm to one side
], ids=["corner0", "centre", "30mm_behind", "20mm_aside"])
def test_a_board_turning_about_its_mount_is_recovered(pivot):
    # A real board turns about its mount, wherever that is: the fitted pivot takes the tilt,
    # so it neither shows up as slack nor biases the focal length. Anchored at a fixed point,
    # the reviewer measured fx 2.4-2.8 % short with the pivot at the centre, with no sign of
    # it in the RMS or the outliers.
    rig = synthetic_rig(cameras=("left", "right", "top"), noise_px=0.1, tilt_deg=3.0, stage_error_mm=0.05,
                        stage_offsets=LOST_STEPS, tilt_pivot_mm=pivot)
    cams = ("left", "right", "top")
    fit = fit_per_view(rig.observations, BOARD, SIZE, cams)
    assert_cameras_match(fit, rig.truth, cams)
    assert fit.placement.outliers == sorted(LOST_STEPS)
    others = [np.linalg.norm(r) for w, r in fit.placement.residual_mm.items() if w not in LOST_STEPS]
    assert max(others) < 0.3
    np.testing.assert_allclose(fit.placement.pivot_mm, pivot, atol=3.0)


def test_holding_the_rotation_cannot_fit_a_tilting_stage(stage_rig):
    cams = ("left", "right", "top")
    fit = fit_per_view(stage_rig.observations, BOARD, SIZE, cams, tilt=False)
    assert max(fit.rms_px.values()) > 1.0
    assert max(fit.placement.tilt_deg.values()) < 1e-6


def test_the_slack_shows_the_lost_steps(stage_rig):
    cams = ("left", "right", "top")
    fit = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    for w, offset in LOST_STEPS.items():
        np.testing.assert_allclose(fit.placement.residual_mm[w], offset, atol=0.15, err_msg=str(w))
    others = [np.linalg.norm(r) for w, r in fit.placement.residual_mm.items() if w not in LOST_STEPS]
    assert max(others) < 0.3
    assert fit.placement.outliers == sorted(LOST_STEPS)
    tilts = [fit.placement.tilt_deg[w] for w in stage_rig.view_poses]
    assert 1.0 < max(tilts) < 7.0


def test_free_translations_leave_the_focal_length_undetermined():
    # The pure free-pose bundle adjustment: every view is the same plane orientation, so
    # focal length and principal point trade off against distance (Sturm and Maybank).
    rig = synthetic_rig(noise_px=0.1)
    fit = fit_per_view(rig.observations, BOARD, SIZE, CAMERAS, prior_mm=None)
    assert max(fit.rms_px.values()) < 0.17  # the corners fit
    for cam in CAMERAS:
        rel = abs(fit.cameras[cam].K[0, 0] / rig.truth.cameras[cam].K[0, 0] - 1.0)
        assert rel > 0.01 or not math.isfinite(fit.param_std[cam]["fx"]), cam


def test_a_stage_that_only_translates_gives_no_tilt_and_the_motor_frame_result():
    rig = synthetic_rig(noise_px=0.1)
    per_view = fit_per_view(rig.observations, BOARD, SIZE, CAMERAS)
    motor = fit_motor_frame(rig.observations, BOARD, SIZE, CAMERAS)
    # Noise alone turns each view's fitted board by a few hundredths of a degree.
    assert max(per_view.placement.tilt_deg.values()) < 0.1
    assert per_view.placement.rms_mm < 0.05
    for cam in CAMERAS:
        a, b = per_view.cameras[cam], motor.cameras[cam]
        np.testing.assert_allclose(np.diag(a.K)[:2], np.diag(b.K)[:2], rtol=1e-3, err_msg=cam)
        np.testing.assert_allclose(a.K[:2, 2], b.K[:2, 2], atol=1.5, err_msg=cam)
        assert angle_deg(rotation(a.rvec).T @ rotation(b.rvec)) < 0.05, cam
        assert np.linalg.norm(np.asarray(a.tvec) - np.asarray(b.tvec)) < 0.2, cam
        assert per_view.rms_px[cam] == pytest.approx(motor.rms_px[cam], abs=0.01), cam
    assert per_view.motor_scale == pytest.approx(motor.motor_scale, abs=1e-3)


def test_views_seen_by_one_camera_are_used():
    n = 37
    seen = {"left": range(0, 30), "right": range(10, n)}  # 0-9 left only, 30-36 right only
    # Random placement error weakens the focal length (1 mm put fx 10 % short with only two
    # cameras); 0.05 mm is the stage's own, as in stage_rig.
    rig = synthetic_rig(noise_px=0.1, tilt_deg=2.0, stage_error_mm=0.05, seen=seen)
    fit = fit_per_view(rig.observations, BOARD, SIZE, CAMERAS)
    assert sorted(fit.per_view_rms_px["left"]) == list(range(0, 30))
    assert sorted(fit.per_view_rms_px["right"]) == list(range(10, n))
    assert sorted(fit.placement.residual_mm) == list(range(n))
    # Two cameras sharing only waypoints 10-29: the principal point is the weak direction.
    assert_cameras_match(fit, rig.truth, CAMERAS, c_px=15.0, deg=0.3)  # 10 px of cx is 0.25 deg


def test_the_gauge_is_fixed_so_two_runs_agree(stage_rig):
    cams = ("left", "right", "top")
    a = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    b = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    for cam in cams:
        np.testing.assert_array_equal(a.cameras[cam].K, b.cameras[cam].K)
        np.testing.assert_array_equal(a.cameras[cam].rvec, b.cameras[cam].rvec)
        np.testing.assert_array_equal(a.cameras[cam].tvec, b.cameras[cam].tvec)
    assert a.motor_scale == b.motor_scale


def test_the_similarity_places_the_cameras_in_the_motor_frame():
    # Tilted but exactly where commanded, at a motor scale of 0.95: the placement is exact.
    rig = synthetic_rig(noise_px=0.05, tilt_deg=3.0, stage_error_mm=0.0, motor_scale=0.95, frame_rotation_deg=35.0)
    fit = fit_per_view(rig.observations, BOARD, SIZE, CAMERAS)
    assert fit.motor_scale == pytest.approx(0.95, abs=5e-4)
    for cam in CAMERAS:
        t, f = rig.truth.cameras[cam], fit.cameras[cam]
        assert angle_deg(rotation(f.rvec).T @ rotation(t.rvec)) < 0.05, cam
        assert np.linalg.norm(np.asarray(f.tvec) - np.asarray(t.tvec)) < 0.3, cam
    # The mean board rotation in the motor frame is the untilted board's, to the tilts' mean.
    assert angle_deg(rotation(fit.board_rvec).T @ rotation(rig.truth.board_rvec)) < 1.0
    assert fit.placement.rms_mm < 0.05 and fit.placement.outliers == []
    assert max(fit.placement.tilt_deg.values()) < 6.0


def test_a_view_far_from_its_commanded_position_is_named_and_left_out_of_the_placement():
    rig = synthetic_rig(noise_px=0.05, tilt_deg=1.0, stage_error_mm=0.0)
    truth = rig.truth
    # Waypoint 13 stood 3.3 mm short of its commanded z.
    R_v, t_v = rig.view_poses[13]
    shifted = t_v + np.array([0.0, 0.0, -3.3])
    X = BOARD.object_points() @ R_v.T + shifted
    observations = list(rig.observations)
    corners = {cam: cv2.projectPoints(X, m.rvec, m.tvec, m.K, m.dist)[0].reshape(-1, 2) for cam, m in truth.cameras.items()}
    observations[13] = WaypointObservation(index=13, motor_mm=observations[13].motor_mm, corners=corners)
    fit = fit_per_view(observations, BOARD, SIZE, CAMERAS)
    assert fit.placement.outliers == [13]
    np.testing.assert_allclose(fit.placement.residual_mm[13], (0.0, 0.0, -3.3), atol=0.05)
    assert max(fit.rms_px.values()) < 0.1  # the calibration itself is unaffected
    notes = placement_notes(fit.placement)
    assert any(n.startswith("waypoint 13: board 3.3 mm off commanded z (-3.3 mm)") for n in notes), notes
    assert notes[0].startswith("motor placement (per-view fit): similarity scale")
    assert "left out of it (more than 3 MAD off): waypoints 13" in notes[0]


def test_the_per_view_fit_starts_from_a_given_fit(stage_rig):
    cams = ("left", "right", "top")
    first = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    again = fit_per_view(stage_rig.observations, BOARD, SIZE, cams, initial=first)
    for cam in cams:
        np.testing.assert_allclose(again.cameras[cam].K, first.cameras[cam].K, atol=0.05)


def test_downstream_products_load(tmp_path, stage_rig):
    cams = ("left", "right", "top")
    fit = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    n = len(stage_rig.observations)
    capture = CaptureSummary(waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
                             views_kept={cam: n for cam in cams}, rejections={cam: {} for cam in cams})
    report = build_report(fit, stage_rig.observations, BOARD, BASE.sensor_roi(), capture, [], None,
                          presets=presets_from_base(BASE))
    folder = tmp_path / "calibration"
    store.write_calibration(folder, fit=fit, presets=presets_from_base(BASE), board=BOARD, report=report,
                            capture={"fit": FIT_PER_VIEW})
    loaded = store.load_sensor_fit(folder)
    for cam in cams:
        np.testing.assert_allclose(loaded.cameras[cam].K, fit.cameras[cam].K)
        R_a, T_a = loaded.stereo("left", cam)
        R_b, T_b = fit.stereo("left", cam)
        np.testing.assert_allclose(R_a, R_b, atol=1e-12)
        np.testing.assert_allclose(T_a, T_b, atol=1e-9)
    assert loaded.motor_scale == pytest.approx(fit.motor_scale)
    assert list(folder.rglob("*.toml")) and list(folder.rglob("stereo_params.pickle"))
    M = frame_transform(loaded, preset_stereo_params(loaded, BASE), motor_to_inference_signs())
    np.testing.assert_allclose(np.abs(np.linalg.det(M)), 1.0, atol=1e-9)
    for name, limit in (("epipolar_error_px", 0.3), ("loo_triangulation_px", 0.3), ("square_size_rel_error", 0.005)):
        values = [m.value for m in report.metrics if m.name == name]
        assert values and all(v is not None and v < limit for v in values), (name, values)
    assert any(note.startswith("motor placement (per-view fit)") for note in report.notes)
    # The graded RMS is each camera's own planar fit; the synthetic stage's three offsets show in the stage block.
    assert all(m.status == "pass" for m in report.metrics if m.name in ("reprojection_rms_px", "joint_reprojection_rms_px"))
    slack = {m.camera: m for m in report.metrics if m.name == "stage_slack_rms_mm"}
    assert set(slack) == {"x", "y", "z"} and all(m.fail is None for m in slack.values())
    assert slack["z"].value > slack["y"].value


def test_the_projection_matches_cv2():
    from reachaq.inference.multires_calibration.per_view import _project, _rodrigues

    rng = np.random.default_rng(3)
    rvecs = rng.normal(0, 0.8, (20, 3))
    rvecs[0] = 0.0
    np.testing.assert_allclose(_rodrigues(rvecs), [cv2.Rodrigues(r)[0] for r in rvecs], atol=1e-12)
    X = np.column_stack([rng.uniform(-30, 30, (200, 2)), rng.uniform(120, 180, 200)])
    intrinsics = np.array([2300.0, 2305.0, 722.0, 537.0, -0.25, 0.1, 0.0005, -0.0003])
    K = np.array([[2300.0, 0, 722.0], [0, 2305.0, 537.0], [0, 0, 1]])
    dist = np.array([-0.25, 0.1, 0.0005, -0.0003, 0.0])
    expected = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    np.testing.assert_allclose(_project(X, intrinsics), expected, atol=1e-8)


def test_an_affine_placement_recovers_a_stage_with_a_scale_per_axis():
    # The stage moves 3 % long in x, 4 % short in y and 6 % long in z: one scale for all three
    # cannot place the views, a scale per axis can. With the principal point held (variant f):
    # with it free, the z scale trades against the focal length and principal point, and the
    # affine fit left fx 2 % short with its std undetermined.
    from reachaq.inference.multires_calibration.per_view import PLACEMENT_AFFINE

    cams = ("left", "right", "top")
    scales = (1.03, 0.96, 1.06)
    rig = synthetic_rig(cameras=cams, noise_px=0.1, stage_error_mm=0.05, axis_scale=scales)
    affine = fit_per_view(rig.observations, BOARD, SIZE, cams, tilt=False, placement=PLACEMENT_AFFINE,
                          fix_principal_point=True)
    assert_cameras_match(affine, rig.truth, cams)
    np.testing.assert_allclose(affine.placement.axis_scale, scales, atol=3e-3)
    np.testing.assert_allclose(affine.placement.axis_angle_deg, (90.0, 90.0, 90.0), atol=0.2)
    assert affine.placement.rms_mm < 0.15
    similarity = fit_per_view(rig.observations, BOARD, SIZE, cams, tilt=False, fix_principal_point=True)
    assert similarity.placement.rms_mm > 3 * affine.placement.rms_mm
    assert any(note.startswith("stage axes: scale x 1.02") or note.startswith("stage axes: scale x 1.03") for note in placement_notes(affine.placement))


def test_the_placement_setting():
    from reachaq.inference.multires_calibration.per_view import configured_placement

    assert configured_placement({}) == "similarity"
    assert configured_placement({"placement": "affine"}) == "affine"
    with pytest.raises(ValueError, match="placement"):
        configured_placement({"placement": "projective"})
    with pytest.raises(ValueError, match="placement"):
        fit_per_view([], BOARD, SIZE, CAMERAS, placement="projective")


def test_the_principal_point_can_be_held_at_the_image_centre(stage_rig):
    cams = ("left", "right", "top")
    fit = fit_per_view(stage_rig.observations, BOARD, SIZE, cams, fix_principal_point=True)
    for cam in cams:
        np.testing.assert_array_equal(fit.cameras[cam].K[:2, 2], [(SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2])
        assert fit.param_std[cam]["cx"] == 0.0 and fit.param_std[cam]["cy"] == 0.0
        # The rig's principal points are 1.5-4 px off centre, so the focal length moves a little.
        assert fit.cameras[cam].K[0, 0] == pytest.approx(stage_rig.truth.cameras[cam].K[0, 0], rel=0.01), cam
        assert fit.rms_px[cam] < 0.25, cam


def test_a_binned_fit_with_its_pixel_scale_solves_the_sensor_problem(stage_rig):
    # The slack prior is in mm beside reprojection residuals in the image's own px, so in a
    # preset binned b times it weighs b^2 as much. On christielab10's session004 that alone
    # moved the 256 preset's direct fx 1-3 % from the same corners fitted in sensor px.
    # pixel_scale (sensor px per image px) restores the sensor fit's weighting.
    import dataclasses

    from reachaq.inference.multires_calibration.geometry import map_intrinsics, sensor_to_preset_points

    cams = ("left", "right", "top")
    g = PresetGeometry(2, 720, 540, 0, 0)  # the whole sensor at binning 2: the same corners, half the px
    sensor = fit_per_view(stage_rig.observations, BOARD, SIZE, cams)
    binned = [WaypointObservation(o.index, o.motor_mm, {c: sensor_to_preset_points(v, g) for c, v in o.corners.items()})
              for o in stage_rig.observations]
    mapped = dataclasses.replace(sensor, cameras={c: dataclasses.replace(m, K=map_intrinsics(m.K, g))
                                                  for c, m in sensor.cameras.items()}, image_size=(g.width, g.height))
    scaled = fit_per_view(binned, BOARD, (g.width, g.height), cams, initial=mapped, pixel_scale=g.binning)
    for cam in cams:
        np.testing.assert_allclose(scaled.cameras[cam].K, mapped.cameras[cam].K, atol=0.02, err_msg=cam)
        # The RMS stays in the image's own px.
        assert scaled.rms_px[cam] == pytest.approx(sensor.rms_px[cam] / g.binning, rel=1e-3), cam
    assert scaled.placement.outliers == sensor.placement.outliers
    with pytest.raises(ValueError, match="pixel_scale"):
        fit_per_view(binned, BOARD, (g.width, g.height), cams, initial=mapped, pixel_scale=0.0)


@pytest.fixture(scope="module")
def held_sensor_fit(stage_rig):
    """The recommended per-view fit, principal points held at the image centre."""
    return fit_per_view(stage_rig.observations, BOARD, SIZE, CAMERAS, fix_principal_point=True)


def _cx_plus_5(K, g):
    out = map_intrinsics(K, g)
    out[0, 2] += 5.0
    return out


def _fx_plus_1_percent(K, g):
    out = map_intrinsics(K, g)
    out[0, 0] *= 1.01
    return out


@pytest.mark.parametrize("mapping, wrong", [(map_intrinsics, False), (_cx_plus_5, True), (_fx_plus_1_percent, True)],
                         ids=["correct", "cx+5px", "fx+1%"])
def test_the_preset_cross_check_catches_a_wrong_intrinsics_mapping(stage_rig, held_sensor_fit, monkeypatch, mapping,
                                                                   wrong):
    # Review 4b, I-3: with the principal point held, the direct fit kept it at the mapped value
    # under test, and a 5 px cx error read 0.9 preset px, under the 1.5 warn.
    g = PresetGeometry(4, 360, 270, 0, 0)  # the whole sensor at binning 4
    binned = [WaypointObservation(o.index, o.motor_mm, {c: sensor_to_preset_points(v, g) for c, v in o.corners.items()
                                                        if c in CAMERAS})
              for o in stage_rig.observations]
    sensor = [WaypointObservation(o.index, o.motor_mm, {c: v for c, v in o.corners.items() if c in CAMERAS})
              for o in stage_rig.observations]
    monkeypatch.setattr(pipeline, "map_intrinsics", mapping)
    delta = pipeline._preset_delta(sensor, binned, held_sensor_fit, g, BOARD, CAMERAS, FIT_PER_VIEW,
                                   {"fix_principal_point": True})
    warn = DEFAULT_THRESHOLDS["cross_check_delta_px"][0]
    assert (delta > warn) == wrong, delta


def test_the_principal_point_setting():
    from reachaq.inference.multires_calibration.per_view import configured_principal_point

    assert configured_principal_point({}) is False
    assert configured_principal_point({"fit_principal_point": "free"}) is False
    assert configured_principal_point({"fit_principal_point": "centre"}) is True
    with pytest.raises(ValueError, match="fit_principal_point"):
        configured_principal_point({"fit_principal_point": "middle"})


def test_the_fit_mode_defaults_by_board_and_reads_the_setting():
    assert default_fit_mode(RIG_CHARUCO_BOARD) == FIT_PER_VIEW
    assert default_fit_mode(BoardSpec()) == FIT_MOTOR_FRAME
    assert configured_fit({}, RIG_CHARUCO_BOARD) == FIT_PER_VIEW
    assert configured_fit({}, BoardSpec()) == FIT_MOTOR_FRAME
    assert configured_fit({"fit": "motor_frame"}, RIG_CHARUCO_BOARD) == FIT_MOTOR_FRAME
    assert configured_fit({"fit": "per_view"}, BoardSpec()) == FIT_PER_VIEW
    with pytest.raises(ValueError, match="fit"):
        configured_fit({"fit": "bundle"}, RIG_CHARUCO_BOARD)
    with pytest.raises(ValueError, match="fit mode"):
        pipeline._fit("bundle", [], BOARD, SIZE, CAMERAS)


def test_the_motor_frame_fit_is_unchanged_by_the_dispatch():
    rig = synthetic_rig(noise_px=0.1)
    a = fit_motor_frame(rig.observations, rig.board, SIZE, CAMERAS)
    b = pipeline._fit(FIT_MOTOR_FRAME, rig.observations, rig.board, SIZE, CAMERAS)
    for cam in CAMERAS:
        np.testing.assert_array_equal(a.cameras[cam].K, b.cameras[cam].K)
    assert a.placement is None and b.placement is None
