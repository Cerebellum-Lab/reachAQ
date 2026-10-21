"""The motor-frame joint fit, checked on a synthetic rig with a known answer.

The board only slides between waypoints, so each camera sees parallel planes and a
planar calibration cannot separate focal length from distance. The fit treats all
waypoints as one non-planar target in the motor frame (board corners plus the
commanded motor translation, times an unknown motor scale). These tests check that
it recovers the synthetic rig's cameras and motor scale, keeps a waypoint seen by
only one camera for that camera, and reports a stereo pair consistent with its poses.
"""

import cv2
import numpy as np
import pytest

from autotrainer.inference.multires_calibration import fit as fit_module
from autotrainer.inference.multires_calibration.fit import (
    INTRINSIC_NAMES,
    BoardSpec,
    WaypointObservation,
    fit_motor_frame,
    reconcile_corner_order,
)
from multires_synthetic import render_board_frame, synthetic_rig


def rotation_angle_deg(a, b):
    """The angle of R(a)^T R(b), in degrees."""
    R = cv2.Rodrigues(np.asarray(a, float))[0].T @ cv2.Rodrigues(np.asarray(b, float))[0]
    # The norm of the rotation vector is the angle, without arccos's loss near zero.
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))


# Undistortion is iterative; cv2's default stops after 5 iterations, which would let the
# triangulation test measure the undistortion instead of the fit.
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)


def triangulate(cameras, pixels):
    """Motor-frame points from the left and right pixels: undistort each camera's pixels to
    normalised coordinates with its K and dist, then triangulate with its [R|t]."""
    projections, normalised = [], []
    for cam in ("left", "right"):
        m = cameras[cam]
        n = cv2.undistortPointsIter(
            np.asarray(pixels[cam], float).reshape(-1, 1, 2), m.K, m.dist, None, None, _UNDISTORT_CRITERIA
        )
        normalised.append(n.reshape(-1, 2).T)
        projections.append(np.hstack([cv2.Rodrigues(m.rvec)[0], np.reshape(m.tvec, (3, 1))]))
    X = cv2.triangulatePoints(projections[0], projections[1], normalised[0], normalised[1])
    return (X[:3] / X[3]).T


def distortion_displacement(model, pixels):
    """Distorted minus undistorted projection, in px, of the rays whose undistorted
    projection through `model` lands on `pixels`."""
    pixels = np.asarray(pixels, float)
    rays = np.column_stack([(pixels - model.K[:2, 2]) / np.diag(model.K)[:2], np.ones(len(pixels))])
    distorted = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), model.K, model.dist)[0].reshape(-1, 2)
    return distorted - pixels


def test_fit_recovers_the_synthetic_rig():
    rig = synthetic_rig(noise_px=0.1)
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    for cam in ("left", "right"):
        t, f = rig.truth.cameras[cam], fit.cameras[cam]
        assert f.K[0, 0] == pytest.approx(t.K[0, 0], rel=0.005)
        assert f.K[1, 1] == pytest.approx(t.K[1, 1], rel=0.005)
        # The principal point and the camera pose are bounded at about 3 sigma of what this
        # target can measure, not at the plan's 1 px, 0.05 deg and 0.1 mm. The board's depth
        # range (13 mm at 150 mm) barely separates a principal-point shift from a sideways
        # camera move: they correlate at -1.0 in the fit. With 0.1 px of corner noise the
        # fit's sigma is 1.07-1.26 px for cx and cy, matching the scatter over 30 seeds
        # (max 3.2 px, 0.077 deg, 0.30 mm), and the plan's bounds failed 27 of those 30.
        # What the cameras are for is unaffected: see the triangulation test.
        # Noise-free, the fit recovers them exactly (see the stereo test).
        assert abs(f.K[0, 2] - t.K[0, 2]) < 4.0 and abs(f.K[1, 2] - t.K[1, 2]) < 4.0
        assert abs(f.dist[0] - t.dist[0]) < 0.02
        assert rotation_angle_deg(f.rvec, t.rvec) < 0.15
        assert np.linalg.norm(f.tvec - t.tvec) < 0.5
        assert fit.rms_px[cam] < 0.2
    assert fit.motor_scale == pytest.approx(1.0, abs=0.002)


def test_fit_measures_a_motor_scale_error():
    rig = synthetic_rig(motor_scale=1.01)
    assert fit_motor_frame(rig.observations, rig.board, (1440, 1080)).motor_scale == pytest.approx(1.01, abs=0.002)


def test_waypoints_seen_by_one_camera_still_constrain_that_camera():
    rig = synthetic_rig()
    for obs in rig.observations[::3]:
        obs.corners.pop("right")
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    assert fit.cameras["right"].K[0, 0] == pytest.approx(rig.truth.cameras["right"].K[0, 0], rel=0.01)
    # Each camera keeps exactly the waypoints it saw: the left camera all of them,
    # the right camera all but the ones it missed.
    assert sorted(fit.per_view_rms_px["left"]) == [o.index for o in rig.observations]
    assert sorted(fit.per_view_rms_px["right"]) == [o.index for o in rig.observations if "right" in o.corners]


def test_stereo_is_consistent_with_the_camera_poses():
    rig = synthetic_rig(noise_px=0.0)
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    R, T = fit.stereo()
    # A motor-frame point mapped to the left camera, then through (R, T), equals its right-camera mapping.
    X = np.random.default_rng(1).uniform(-60.0, 60.0, (20, 3))
    left, right = fit.cameras["left"], fit.cameras["right"]
    X_left = X @ cv2.Rodrigues(left.rvec)[0].T + left.tvec
    X_right = X @ cv2.Rodrigues(right.rvec)[0].T + right.tvec
    np.testing.assert_allclose(X_left @ R.T + T, X_right, rtol=0, atol=1e-6)
    # Without noise the fit lands on the truth, so the pair is the true one too.
    R_true, T_true = rig.truth.stereo()
    assert rotation_angle_deg(cv2.Rodrigues(R)[0], cv2.Rodrigues(R_true)[0]) < 1e-4
    np.testing.assert_allclose(T, T_true, rtol=0, atol=1e-4)


def test_fit_converges_when_the_planar_start_is_far_off():
    # Parallel views leave the planar calibration that starts the fit degenerate in focal
    # length. With this seed it starts the left camera at well over twice the true fx;
    # with all five distortion terms free there it started at 5.5x, k2 = -2800, and the
    # joint fit diverged to rms 4.4 px.
    rig = synthetic_rig(seed=12)
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    for cam in ("left", "right"):
        assert fit.rms_px[cam] < 0.2
        assert fit.cameras[cam].K[0, 0] == pytest.approx(rig.truth.cameras[cam].K[0, 0], rel=0.005)
    assert fit.motor_scale == pytest.approx(1.0, abs=0.002)


@pytest.mark.parametrize("frame_rotation_deg", [0.0, 35.0])
def test_fit_triangulates_the_working_volume_like_truth(frame_rotation_deg):
    # At 35 degrees the motor axes are turned against the board and cameras about an
    # oblique axis, so nothing in the fit can lean on them being aligned.
    rig = synthetic_rig(noise_px=0.1, frame_rotation_deg=frame_rotation_deg)
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    # A 10 x 10 x 5 grid over the bounding box of every corner at every waypoint. Motor
    # z, the axis the stage moves least, gets the 5 steps.
    corners = np.concatenate([rig.world_corners(o.motor_mm) for o in rig.observations])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    axes = [np.linspace(lo[i], hi[i], n) for i, n in enumerate((10, 10, 5))]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    pixels = {
        cam: cv2.projectPoints(X, t.rvec, t.tvec, t.K, t.dist)[0].reshape(-1, 2)
        for cam, t in rig.truth.cameras.items()
    }
    # The measurement adds no error of its own: the truth models give the grid back.
    assert np.linalg.norm(triangulate(rig.truth.cameras, pixels) - X, axis=1).max() < 1e-6
    assert np.linalg.norm(triangulate(fit.cameras, pixels) - X, axis=1).max() < 0.05


def working_volume_error_mm(rig, cameras) -> float:
    """The largest triangulation error of `cameras` over the working-volume grid of
    test_fit_triangulates_the_working_volume_like_truth, in mm."""
    corners = np.concatenate([rig.world_corners(o.motor_mm) for o in rig.observations])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    axes = [np.linspace(lo[i], hi[i], n) for i, n in enumerate((10, 10, 5))]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pixels = {
        cam: cv2.projectPoints(X, t.rvec, t.tvec, t.K, t.dist)[0].reshape(-1, 2)
        for cam, t in rig.truth.cameras.items()
    }
    return float(np.linalg.norm(triangulate(cameras, pixels) - X, axis=1).max())


def with_reversed_corners(observations, cam):
    """Copies of `observations` with every one of `cam`'s corner arrays in the other 180-degree order."""
    return [
        WaypointObservation(
            index=o.index,
            motor_mm=o.motor_mm.copy(),
            corners={c: (v[::-1].copy() if c == cam else v.copy()) for c, v in o.corners.items()},
        )
        for o in observations
    ]


def test_reconcile_reverses_a_camera_that_labels_the_board_from_the_other_end():
    # The 6x8 board looks the same turned 180 degrees, so the right camera here is
    # consistent across its own waypoints but starts from the left camera's last corner.
    rig = synthetic_rig(noise_px=0.1)
    flipped = with_reversed_corners(rig.observations, "right")
    inputs = [{c: v.copy() for c, v in o.corners.items()} for o in flipped]
    reconciled, reversed_cameras = reconcile_corner_order(flipped, rig.board, (1440, 1080))
    assert reversed_cameras == {"left": False, "right": True}
    for o, before in zip(flipped, inputs):
        for c in before:
            np.testing.assert_array_equal(o.corners[c], before[c])  # the inputs are not changed
    for r, o in zip(reconciled, rig.observations):
        assert r.index == o.index
        for c in ("left", "right"):
            np.testing.assert_array_equal(r.corners[c], o.corners[c])
    fit = fit_motor_frame(reconciled, rig.board, (1440, 1080))
    assert working_volume_error_mm(rig, fit.cameras) < 0.05


def test_reconcile_leaves_consistent_cameras_alone():
    rig = synthetic_rig(noise_px=0.1)
    reconciled, reversed_cameras = reconcile_corner_order(rig.observations, rig.board, (1440, 1080))
    assert reversed_cameras == {"left": False, "right": False}
    for r, o in zip(reconciled, rig.observations):
        assert r is not o and r.index == o.index
        np.testing.assert_array_equal(r.motor_mm, o.motor_mm)
        for c in ("left", "right"):
            assert r.corners[c] is not o.corners[c]
            np.testing.assert_array_equal(r.corners[c], o.corners[c])


def test_reconcile_takes_the_camera_with_the_most_views_as_reference():
    # The left camera misses five waypoints, so the right camera is the reference even
    # though it is the one reversed here; the left camera is turned to match it.
    rig = synthetic_rig(noise_px=0.1)
    flipped = with_reversed_corners(rig.observations, "right")
    for o in flipped[:5]:
        del o.corners["left"]
    reconciled, reversed_cameras = reconcile_corner_order(flipped, rig.board, (1440, 1080))
    assert reversed_cameras == {"left": True, "right": False}
    for r, f in zip(reconciled, flipped):
        np.testing.assert_array_equal(r.corners["right"], f.corners["right"])
        if "left" in f.corners:
            np.testing.assert_array_equal(r.corners["left"], f.corners["left"][::-1])


def test_fitted_distortion_matches_truth_at_the_1024_roi_corners():
    rig = synthetic_rig(noise_px=0.1)
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    # The sensor ROI of the 1024x1024 bin-1 preset, PresetGeometry(1, 1024, 1024, 208, 24).
    # The board does not reach these corners, so this is where the fitted distortion
    # extrapolates furthest.
    roi_corners = [(208.0, 24.0), (1232.0, 24.0), (208.0, 1048.0), (1232.0, 1048.0)]
    for cam in ("left", "right"):
        fitted = distortion_displacement(fit.cameras[cam], roi_corners)
        truth = distortion_displacement(rig.truth.cameras[cam], roi_corners)
        assert np.linalg.norm(fitted - truth, axis=1).max() < 1.0


def test_param_std_covers_every_intrinsic():
    rig = synthetic_rig(noise_px=0.1)
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    for cam in ("left", "right"):
        std = fit.param_std[cam]
        assert tuple(std) == INTRINSIC_NAMES == ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
        # k3 is held at 0, so it is reported as exactly 0 with no spread.
        dist = fit.cameras[cam].dist
        assert dist.shape == (5,) and dist[4] == 0.0 and std["k3"] == 0.0
        assert all(np.isfinite(v) and v > 0 for name, v in std.items() if name != "k3")
        # 0.1 px of noise on ~1800 corners pins the focal length far inside the 0.5 % the
        # recovery test allows; a std near that size would mean the Jacobian is wrong.
        assert std["fx"] / fit.cameras[cam].K[0, 0] < 0.001


def test_a_camera_the_capture_cannot_determine_reports_infinite_std():
    rig = synthetic_rig()
    # The right camera sees the board only at the 8 waypoints at z = -29.4, so its whole
    # target is one plane and its focal length trades against its distance. Fitted, its
    # fx lands near 1400 (truth 2310) with an ordinary rms.
    at_one_height = [o for o in rig.observations if o.motor_mm[2] == -29.4]
    assert len(at_one_height) == 8
    for obs in rig.observations:
        if obs.motor_mm[2] != -29.4:
            obs.corners.pop("right")
    fit = fit_motor_frame(rig.observations, rig.board, (1440, 1080))
    right, left = fit.param_std["right"], fit.param_std["left"]
    assert right["fx"] == np.inf
    # Never 0.0, which would read as perfectly determined; k3 is held at 0 by design.
    assert all(v != 0.0 for name, v in right.items() if name != "k3")
    # The left camera saw every waypoint, and its uncertainty is still reported.
    assert all(np.isfinite(v) and v > 0 for name, v in left.items() if name != "k3")


def test_a_fit_that_runs_out_of_evaluations_raises(monkeypatch):
    real_least_squares = fit_module.least_squares

    def capped(*args, **kwargs):
        # The production limit is explicit; cut it to 2 here so the solver stops early.
        assert kwargs["max_nfev"] == 400
        kwargs["max_nfev"] = 2
        return real_least_squares(*args, **kwargs)

    monkeypatch.setattr(fit_module, "least_squares", capped)
    rig = synthetic_rig()
    with pytest.raises(RuntimeError, match=r"status 0 after \d+ evaluations"):
        fit_motor_frame(rig.observations, rig.board, (1440, 1080))


def test_frame_rotation_turns_the_scene_against_the_motor_axes():
    plain, turned = synthetic_rig(noise_px=0.0), synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0)
    # The same move list, but the board (and the cameras with it) turned 35 degrees about
    # the default oblique axis, as seen from the motor frame.
    np.testing.assert_array_equal(
        [o.motor_mm for o in plain.observations], [o.motor_mm for o in turned.observations]
    )
    R_plain = cv2.Rodrigues(plain.truth.board_rvec)[0]
    R_turned = cv2.Rodrigues(turned.truth.board_rvec)[0]
    turn = cv2.Rodrigues(R_turned @ R_plain.T)[0].ravel()
    assert np.degrees(np.linalg.norm(turn)) == pytest.approx(35.0, abs=1e-9)
    axis = np.array([0.3, -0.5, 0.81])
    np.testing.assert_allclose(turn / np.linalg.norm(turn), axis / np.linalg.norm(axis), atol=1e-12)


def test_a_camera_with_too_few_waypoints_is_refused():
    rig = synthetic_rig()
    for obs in rig.observations[2:]:
        obs.corners.pop("right")
    with pytest.raises(ValueError, match="right"):
        fit_motor_frame(rig.observations, rig.board, (1440, 1080))


def test_board_object_points_are_row_major_with_columns_fastest():
    pts = BoardSpec().object_points()
    assert pts.shape == (48, 3)
    np.testing.assert_array_equal(pts[:2], [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    np.testing.assert_array_equal(pts[8], [0.0, 4.0, 0.0])
    np.testing.assert_array_equal(pts[-1], [28.0, 20.0, 0.0])


def test_synthetic_moves_follow_the_rig_move_list_shape():
    motor = np.array([o.motor_mm for o in synthetic_rig().observations])
    assert motor.shape == (37, 3)
    np.testing.assert_array_equal(motor[0], [-9.4, 10.0, -35.0])
    for axis, (lo, hi) in enumerate([(-10.0, 10.0), (10.0, 35.0), (-35.0, -22.0)]):
        assert lo <= motor[:, axis].min() and motor[:, axis].max() <= hi
    # Every step moves exactly one axis, so no waypoint repeats the one before it.
    assert np.all(np.count_nonzero(np.diff(motor, axis=0), axis=1) == 1)


def test_rendered_board_is_detected_at_the_projected_corners():
    rig = synthetic_rig(noise_px=0.0)
    board, cam = rig.board, rig.truth.cameras["left"]
    X = rig.world_corners(rig.observations[5].motor_mm)
    gray = render_board_frame(cam, X, (1440, 1080), board)
    assert gray.dtype == np.uint8 and gray.shape == (1080, 1440)

    found, corners = cv2.findChessboardCorners(gray, (board.cols, board.rows))
    assert found
    corners = cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    ).reshape(-1, 2)
    truth = cv2.projectPoints(X, cam.rvec, cam.tvec, cam.K, np.zeros(5))[0].reshape(-1, 2)
    # The 6x8 board is symmetric under a 180 degree turn, so the detector may report either order.
    err = min(np.linalg.norm(corners - truth, axis=1).max(), np.linalg.norm(corners[::-1] - truth, axis=1).max())
    assert err < 0.25
