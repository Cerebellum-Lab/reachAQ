"""What a preset gets from the motor-frame fit: stereo parameters, the 3D frame and the exports.

The fit is in full-sensor pixels; every preset gets its own stereo_params with the
intrinsics mapped to its pixels, so today's loaders and live path read it unchanged.
The frame transform replaces the old reorientation in `oriented = (raw - offsets) @ M`
and must give motor-axis millimetres with the signs the DCS chain expects, so the live
drift monitor sees no drift when the motor carries the triangle. The DeepLabCut and
anipose exports must reproduce the same 3D points as reachAQ's triangulation.

The tests use the synthetic rig's truth cameras, noise-free, as the fit, and points
in the working volume the board sweeps: board corners at a grid of motor positions
spanning the waypoints, kept where both cameras see them inside the presets' ROI.

The default synthetic cameras stand symmetric about motor z, which makes their
rectified frame the motor frame (R1 @ R_left is the identity), so M is just
diag(signs) and a transposed or misordered M would pass. The frame tests therefore
also run on a rig whose scene is turned 35 degrees about a generic axis, as on the
real rig, where the cameras sit at unequal azimuths and elevations.
"""

import dataclasses
import importlib
import pickle

import cv2
import numpy as np
import pandas as pd
import pytest

from reachaq.core import Offset3DTuple
from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from reachaq.inference.analysis.prepare_jetson_data import _undistort_points
from reachaq.inference.multires_calibration.fit import CaptureCamera
from reachaq.inference.multires_calibration.geometry import (
    PresetGeometry,
    PresetSet,
    map_intrinsics,
    presets_from_base,
    sensor_to_preset_points,
)
from reachaq.inference.multires_calibration.products import (
    STEREO_KEY,
    anipose_toml,
    frame_transform,
    motor_to_inference_signs,
    preset_stereo_params,
)
from multires_synthetic import synthetic_rig


RIG = synthetic_rig(noise_px=0.0)
FIT = RIG.truth  # noise-free, the truth cameras stand in for a fit
PRESETS = presets_from_base(PresetGeometry(4, 256, 256, 52, 6))
CAMERAS = ("left", "right")
# The scene, board and cameras together, turned 35 degrees about a generic axis in the motor frame.
RIGS = {"symmetric": RIG, "turned": synthetic_rig(noise_px=0.0, frame_rotation_deg=35.0)}


def preset_id(g):
    return g.name


def toml_loads():
    """tomllib where Python has it (3.11+), else its backport tomli, which the rig's 3.10 has."""
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads


def pair_toml(fit, g):
    """The anipose export of left and right, both at preset `g`."""
    return anipose_toml(
        fit, PresetSet(g.name, {cam: g for cam in CAMERAS}), calibration_id="0" * 32,
        cameras=[CaptureCamera(cam, None, i) for i, cam in enumerate(CAMERAS)],
    )


def working_volume_points(rig=RIG):
    """(N, 3) motor-frame points: board corners at a 3x3x3 grid of motor positions spanning
    the waypoints, kept where both truth cameras see them inside the presets' sensor ROI."""
    motor = np.array([o.motor_mm for o in rig.observations])
    axes = [np.linspace(lo, hi, 3) for lo, hi in zip(motor.min(axis=0), motor.max(axis=0))]
    X = np.concatenate([rig.world_corners(m) for m in np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 3)])
    # sensor_roi is in pixel edges; in the pixel-centre convention the ROI spans [x0 - 0.5, x1 - 0.5].
    # Every preset covers the same sensor window.
    x0, y0, x1, y1 = PRESETS[0].sensor_roi()
    inside = np.ones(len(X), bool)
    for cam in CAMERAS:
        u = sensor_pixels(X, cam, rig.truth)
        inside &= (u[:, 0] >= x0 - 0.5) & (u[:, 0] <= x1 - 0.5) & (u[:, 1] >= y0 - 0.5) & (u[:, 1] <= y1 - 0.5)
    assert inside.sum() > 500, "the working-volume grid barely overlaps the ROI; the tests would prove little"
    return X[inside]


def sensor_pixels(X, cam, fit=FIT):
    m = fit.cameras[cam]
    return cv2.projectPoints(np.asarray(X, float), m.rvec, m.tvec, m.K, m.dist)[0].reshape(-1, 2)


def preset_pixels(X, g, fit=FIT):
    """{camera: (N, 2) pixel positions in preset g} of motor-frame points X, through fit's cameras."""
    return {cam: sensor_to_preset_points(sensor_pixels(X, cam, fit), g) for cam in CAMERAS}


def detections(pixels):
    """One frame of (x, y, likelihood) per point, flattened, as pose detections arrive."""
    return np.column_stack([pixels, np.ones(len(pixels))]).reshape(1, -1)


def reachaq_raw(stereo, pixels):
    """Rectified-left 3D points, computed as PoseAlgorithm._triangulate_3d_live does."""
    m = stereo.matrix
    undistorted = [
        _undistort_points(
            detections(pixels[cam]), m[f"cameraMatrix{i}"], m[f"distCoeffs{i}"], m[f"R{i}"], m[f"P{i}"], m["rot_cor"]
        ).reshape(-1, 3)[:, :2].T.astype(np.float64)
        for i, cam in enumerate(CAMERAS, start=1)
    ]
    homogeneous = cv2.triangulatePoints(m["P1"][:3], m["P2"][:3], undistorted[0], undistorted[1])
    return (homogeneous / homogeneous[3])[:3].T


# Converged undistortion: cv2's default stops after 5 iterations.
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-15)


def exact_raw(stereo, pixels):
    """reachaq_raw's chain (undistort with R_i and P_i, triangulate with P1 and P2) in float64 throughout.

    _undistort_points casts the pixels to float32, and cv2 then returns float32. One
    float32 ulp is 2e-6 mm in every preset here (7.6e-6 px at 128 px x 0.26 mm/px at
    bin 4; 3.1e-5 px at 512 px x 0.065 mm/px at bin 1), and the live chain measured
    up to 5.7e-6 mm against 1e-13 mm in float64. A check at 1e-6 mm therefore needs this.
    """
    m = stereo.matrix
    undistorted = [
        cv2.undistortPointsIter(
            np.asarray(pixels[cam], np.float64).reshape(-1, 1, 2), m[f"cameraMatrix{i}"], m[f"distCoeffs{i}"],
            m[f"R{i}"], m[f"P{i}"], _UNDISTORT_CRITERIA,
        ).reshape(-1, 2).T
        for i, cam in enumerate(CAMERAS, start=1)
    ]
    homogeneous = cv2.triangulatePoints(m["P1"][:3], m["P2"][:3], undistorted[0], undistorted[1])
    return (homogeneous / homogeneous[3])[:3].T


def test_preset_stereo_params_keys_and_shapes():
    g = PresetGeometry(4, 256, 256, 52, 6)
    sp = preset_stereo_params(FIT, g)
    assert sp.key_name == STEREO_KEY == "left-right"
    assert set(sp.matrix) >= {"cameraMatrix1", "cameraMatrix2", "distCoeffs1", "distCoeffs2", "R", "T",
                              "E", "F", "R1", "R2", "P1", "P2", "roi1", "roi2", "Q", "image_shape", "rot_cor"}
    assert sp.matrix["image_shape"] == [(256, 256), (256, 256)] and sp.matrix["rot_cor"] == 0.0
    assert isinstance(sp.matrix["rot_cor"], float)
    # The shapes today's stereoCalibrate pickles have, so code that indexes them keeps working.
    shapes = {"cameraMatrix1": (3, 3), "cameraMatrix2": (3, 3), "distCoeffs1": (1, 5), "distCoeffs2": (1, 5),
              "R": (3, 3), "T": (3, 1), "E": (3, 3), "F": (3, 3), "R1": (3, 3), "R2": (3, 3),
              "P1": (3, 4), "P2": (3, 4), "Q": (4, 4)}
    assert {k: np.shape(sp.matrix[k]) for k in shapes} == shapes
    assert len(sp.matrix["roi1"]) == 4 and len(sp.matrix["roi2"]) == 4
    for i, cam in enumerate(CAMERAS, start=1):
        np.testing.assert_allclose(sp.matrix[f"cameraMatrix{i}"], map_intrinsics(FIT.cameras[cam].K, g), rtol=1e-12)
        np.testing.assert_array_equal(sp.matrix[f"distCoeffs{i}"].ravel(), FIT.cameras[cam].dist)
    R, T = FIT.stereo("left", "right")
    np.testing.assert_allclose(sp.matrix["R"], R, atol=1e-12)
    np.testing.assert_allclose(sp.matrix["T"].ravel(), T, atol=1e-9)


@pytest.mark.parametrize("alpha", [1.0, 0.0])
def test_rectification_uses_width_then_height(alpha):
    # The presets are square, so only a non-square image tells (width, height) from the
    # (height, width) the legacy calibration passed. The full sensor is 1440x1080.
    g = PresetGeometry(1, 1440, 1080, 0, 0)
    m = preset_stereo_params(FIT, g, alpha=alpha).matrix
    args = (m["cameraMatrix1"], m["distCoeffs1"], m["cameraMatrix2"], m["distCoeffs2"])
    R, T = FIT.stereo("left", "right")
    right_order = cv2.stereoRectify(*args, (1440, 1080), R, T.reshape(3, 1), alpha=alpha)
    swapped = cv2.stereoRectify(*args, (1080, 1440), R, T.reshape(3, 1), alpha=alpha)
    names = ("R1", "R2", "P1", "P2", "Q", "roi1", "roi2")
    for name, value in zip(names, right_order):
        np.testing.assert_allclose(m[name], value, rtol=0, atol=1e-9)
    # The comparison can tell the two orders apart: the rectified principal point moves.
    assert np.abs(swapped[2] - right_order[2]).max() > 10.0
    assert m["image_shape"] == [(1440, 1080), (1440, 1080)]


def test_essential_and_fundamental_matrices_satisfy_the_epipolar_constraint():
    g = PresetGeometry(2, 512, 512, 104, 12)
    m = preset_stereo_params(FIT, g).matrix
    X = working_volume_points()
    # Undistorted projections: normalised camera coordinates, and preset pixels through the preset K.
    normalised = {}
    for cam in CAMERAS:
        c = FIT.cameras[cam]
        X_cam = X @ cv2.Rodrigues(c.rvec)[0].T + c.tvec
        normalised[cam] = np.column_stack([X_cam[:, :2] / X_cam[:, 2:], np.ones(len(X))])
    algebraic = np.einsum("ni,ij,nj->n", normalised["right"], m["E"], normalised["left"])
    assert np.abs(algebraic).max() < 1e-9 * np.linalg.norm(m["T"])
    x1 = normalised["left"] @ m["cameraMatrix1"].T
    x2 = normalised["right"] @ m["cameraMatrix2"].T
    lines = x1 @ m["F"].T  # epipolar line in the right image of each left point
    distance_px = np.abs(np.sum(x2 * lines, axis=1)) / np.linalg.norm(lines[:, :2], axis=1)
    assert distance_px.max() < 1e-6
    assert m["F"][2, 2] == pytest.approx(1.0)


def test_signs_are_read_from_the_dcs_flips(monkeypatch):
    # Today's DCS convention (flips_motor_diamond): DCS x runs with motor x, y and z against it.
    # The oriented axes are the DCS axes, so the oriented 3D frame takes the same signs.
    assert motor_to_inference_signs() == (1, -1, -1)
    monkeypatch.setattr(DiamondTriangleOffsetConfig, "flips_motor_diamond", Offset3DTuple(-1, 1, -1))
    assert motor_to_inference_signs() == (-1, 1, -1)
    # A DCS map that scales an axis is not a sign flip, and must not be read as one.
    monkeypatch.setattr(DiamondTriangleOffsetConfig, "flips_motor_diamond", Offset3DTuple(2, -1, -1))
    with pytest.raises(ValueError):
        motor_to_inference_signs()


@pytest.mark.parametrize("frame", RIGS)
@pytest.mark.parametrize("g", PRESETS, ids=preset_id)
def test_live_pipeline_on_preset_points_recovers_motor_displacements(g, frame):
    fit = RIGS[frame].truth
    X = working_volume_points(RIGS[frame])
    stereo = preset_stereo_params(fit, g)
    signs = motor_to_inference_signs()
    raw = reachaq_raw(stereo, preset_pixels(X, g, fit))
    oriented = (raw - raw[0]) @ frame_transform(fit, stereo, signs)
    expected = (X - X[0]) * np.array(signs)
    assert np.abs(oriented - expected).max() < 0.05


# A DCS calibration away from zero: the motor position at which it recorded the offset.
_USED_POSITION = Offset3DTuple(3.5, 22.0, -28.0)


@pytest.mark.parametrize("frame", RIGS)
@pytest.mark.parametrize("g", PRESETS, ids=preset_id)
def test_live_drift_is_zero_when_the_motor_carries_the_triangle(g, frame):
    # The production consumer of the oriented frame: drift = diamond_to_motor(offset) - motor_position
    # (BehaviorAlgorithm.handle_diamond_triangle_offset), with offset = triangle - diamond in oriented
    # coordinates (PoseResponse.get_parts_3d_offset(Diamond, Triangle)). The diamond stays put, the
    # motor carries the triangle, and the DCS calibration recorded the offset at _USED_POSITION.
    rig = RIGS[frame]
    fit = rig.truth
    stereo = preset_stereo_params(fit, g)
    M = frame_transform(fit, stereo, motor_to_inference_signs())
    camera_offsets = np.array([4.0, -6.0, 150.0])  # cancels in the offset; non-zero as live

    def live_offset(triangle, diamond):
        raw = reachaq_raw(stereo, preset_pixels(np.array([triangle, diamond]), g, fit))
        oriented = (raw - camera_offsets) @ M
        return oriented[0] - oriented[1]

    volume = working_volume_points(rig)
    diamond, triangle = volume[0], volume.mean(axis=0)
    measured = live_offset(triangle, diamond)
    assert np.linalg.norm(measured) > 1.0, "the calibrated offset should be away from zero"
    cfg = DiamondTriangleOffsetConfig(used_position=_USED_POSITION, measured_offset=Offset3DTuple(*measured))
    for k in range(3):
        move = 2.0 * np.eye(3)[k]
        offset = live_offset(triangle + move, diamond)
        drift = np.array(cfg.diamond_to_motor(Offset3DTuple(*offset))) - (np.array(_USED_POSITION) + move)
        assert np.abs(drift).max() < 1e-3, (k, drift)


@pytest.mark.parametrize("frame", RIGS)
@pytest.mark.parametrize("g", PRESETS, ids=preset_id)
def test_a_motor_step_maps_back_through_diamond_to_motor(g, frame):
    # The test above runs the live float32 chain; this one runs the same chain in float64
    # (see exact_raw) so the signs and axes are pinned to 1e-6 mm.
    rig = RIGS[frame]
    fit = rig.truth
    stereo = preset_stereo_params(fit, g)
    M = frame_transform(fit, stereo, motor_to_inference_signs())
    base = Offset3DTuple(-2.0, 5.0, 12.0)  # an oriented triangle - diamond offset
    cfg = DiamondTriangleOffsetConfig(used_position=_USED_POSITION, measured_offset=base)
    start = working_volume_points(rig).mean(axis=0)
    for k in range(3):
        e_k = np.eye(3)[k]
        raw = exact_raw(stereo, preset_pixels(np.array([start, start + e_k]), g, fit))
        d = (raw[1] - raw[0]) @ M
        step = np.array(cfg.diamond_to_motor(base + Offset3DTuple(*d))) - np.array(cfg.diamond_to_motor(base))
        np.testing.assert_allclose(step, e_k, rtol=0, atol=1e-6)


def test_anipose_toml_reads_back():
    loads = toml_loads()
    # A noisy rig's truth carries the injected noise as its rms_px, so the error is not 0.
    fit = synthetic_rig(noise_px=0.1).truth
    g = PresetGeometry(1, 1024, 1024, 208, 24)
    data = loads(pair_toml(fit, g))
    assert sorted(data) == ["cam_0", "cam_1", "metadata"]
    for key, cam in (("cam_0", "left"), ("cam_1", "right")):
        table, m = data[key], fit.cameras[cam]
        assert table["name"] == cam and table["size"] == [1024, 1024]
        # Written at repr precision, so every value reads back exactly.
        np.testing.assert_array_equal(table["matrix"], map_intrinsics(m.K, g))
        np.testing.assert_array_equal(table["distortions"], m.dist)
        np.testing.assert_array_equal(table["rotation"], m.rvec)
        np.testing.assert_array_equal(table["translation"], m.tvec)
        assert len(table["distortions"]) == 5
    assert data["metadata"]["adjusted"] is False
    # The error is in the preset's px, like the rest of the file: sensor px / binning.
    sensor_rms = np.mean([fit.rms_px[c] for c in CAMERAS])
    assert data["metadata"]["error"] == sensor_rms > 0
    binned = PresetGeometry(4, 256, 256, 52, 6)
    assert loads(pair_toml(fit, binned))["metadata"]["error"] == pytest.approx(sensor_rms / 4, rel=1e-15)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", ["K", "dist", "rvec", "tvec"])
def test_anipose_toml_refuses_non_finite_camera_parameters(field, value):
    # TOML can spell nan and inf, but a calibration holding them would be read back by
    # anipose as if it were one; the export refuses it instead.
    model = FIT.cameras["right"]
    broken = np.array(getattr(model, field), float)
    broken.flat[-1] = value
    fit = dataclasses.replace(FIT, cameras={**FIT.cameras, "right": dataclasses.replace(model, **{field: broken})})
    with pytest.raises(ValueError, match="right") as refused:
        pair_toml(fit, PRESETS[0])
    assert "finite" in str(refused.value)
    # The cameras as fitted still export.
    pair_toml(FIT, PRESETS[0])


@pytest.mark.parametrize("g", PRESETS, ids=preset_id)
def test_anipose_toml_cameras_triangulate_preset_pixels_to_motor_points(g):
    # aniposelib is not installed, so do what CameraGroup.triangulate does with the read-back
    # cameras: cv2.undistortPoints to normalised coordinates, then a DLT with each [R|t].
    data = toml_loads()(pair_toml(FIT, g))
    X = working_volume_points()
    pixels = preset_pixels(X, g)
    normalised, projections = [], []
    for key, cam in (("cam_0", "left"), ("cam_1", "right")):
        table = data[key]
        n = cv2.undistortPoints(pixels[cam].reshape(-1, 1, 2), np.array(table["matrix"]), np.array(table["distortions"]))
        normalised.append(n.reshape(-1, 2).T)
        R = cv2.Rodrigues(np.array(table["rotation"]))[0]
        projections.append(np.hstack([R, np.reshape(table["translation"], (3, 1))]))
    homogeneous = cv2.triangulatePoints(projections[0], projections[1], normalised[0], normalised[1])
    np.testing.assert_allclose((homogeneous[:3] / homogeneous[3]).T, X, rtol=0, atol=1e-4)


def _import_deeplabcut(name):
    """Import a deeplabcut module, or skip the test when it cannot be imported for any reason.

    importorskip only catches ImportError; DLC 2.3.11 under Keras 3 fails with an AttributeError.
    """
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.skip(f"deeplabcut not importable: {exc!r}")


@pytest.mark.parametrize("g", PRESETS, ids=preset_id)
def test_deeplabcut_triangulates_the_export_like_reachaq(g, tmp_path):
    dlc_tri = _import_deeplabcut("deeplabcut.pose_estimation_3d.triangulation")
    auxiliaryfunctions = _import_deeplabcut("deeplabcut.utils.auxiliaryfunctions")
    auxiliaryfunctions_3d = _import_deeplabcut("deeplabcut.utils.auxiliaryfunctions_3d")

    X = working_volume_points()
    stereo = preset_stereo_params(FIT, g)
    pixels = preset_pixels(X, g)
    reachaq = reachaq_raw(stereo, pixels)

    # Through a pickle and DLC's own reader, as deeplabcut.triangulate loads stereo_params.pickle.
    path = tmp_path / "stereo_params.pickle"
    with path.open("wb") as fh:
        pickle.dump(stereo.as_pickle_dict(), fh)
    stereo_file = auxiliaryfunctions.read_pickle(path)
    # deeplabcut.triangulate keys the pair by the project's camera names, cam_names[0] + "-" + cam_names[1].
    cam_names = list(CAMERAS)
    stereo_matrix = stereo_file[cam_names[0] + "-" + cam_names[1]]

    # A single-animal DLC frame: (scorer, bodyparts, coords) columns of x, y, likelihood.
    columns = pd.MultiIndex.from_product(
        [["synthetic"], [f"point{j}" for j in range(len(X))], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    views = tuple(pd.DataFrame(detections(pixels[cam]), columns=columns) for cam in CAMERAS)
    left, right = dlc_tri._undistort_views([views], stereo_file)[0]
    # As deeplabcut.triangulate reshapes them for one individual.
    x1 = left.to_numpy().reshape((1, 1, -1, 3))[..., :2][:, 0].reshape((-1, 2)).T
    x2 = right.to_numpy().reshape((1, 1, -1, 3))[..., :2][:, 0].reshape((-1, 2)).T
    dlc = auxiliaryfunctions_3d.triangulatePoints(stereo_matrix["P1"], stereo_matrix["P2"], x1, x2)[:3].T

    np.testing.assert_allclose(dlc, reachaq, rtol=0, atol=1e-6)


# Three cameras in capture order with an uncalibrated one among them: the export lists the
# calibrated ones in that order, each at its own geometry, and leaves stimCam out.
RIG3 = synthetic_rig(noise_px=0.1, frame_rotation_deg=35.0, cameras=("left", "right", "camera4"))
CAPTURE3 = [
    CaptureCamera("camera4", "24152535", 4),
    CaptureCamera("left", "24152533", 0),
    CaptureCamera("stimCam", "20142630", 2),
    CaptureCamera("right", None, 1),
]
MIXED = PresetSet("live", {
    "left": PresetGeometry(4, 256, 256, 52, 6),
    "right": PresetGeometry(2, 512, 512, 104, 12),
    "camera4": PresetGeometry(1, 1024, 1024, 208, 24),
    "stimCam": PresetGeometry(1, 720, 540, 0, 0),
})
CALIBRATION_ID = "5d1f0e5a9c2b4c39a7e6f1b2c3d4e5f6"


def test_toml_has_one_table_per_calibrated_camera_in_capture_order():
    data = toml_loads()(anipose_toml(RIG3.truth, MIXED, calibration_id=CALIBRATION_ID, cameras=CAPTURE3))
    assert sorted(data) == ["cam_0", "cam_1", "cam_2", "metadata"]
    names = [data[f"cam_{i}"]["name"] for i in range(3)]
    assert names == ["camera4", "left", "right"]
    for i, cam in enumerate(names):
        g = MIXED.cameras[cam]
        assert data[f"cam_{i}"]["size"] == [g.width, g.height]
        assert set(data[f"cam_{i}"]) == {"name", "size", "matrix", "distortions", "rotation", "translation"}
    metadata = data["metadata"]
    assert set(metadata) == {"adjusted", "error", "calibration_id", "cameras", "units", "frame"}
    assert metadata["calibration_id"] == CALIBRATION_ID
    # TOML has no null: a camera without a serial has no serial key.
    assert metadata["cameras"] == [
        {"name": "camera4", "serial": "24152535"}, {"name": "left", "serial": "24152533"}, {"name": "right"},
    ]
    assert metadata["units"] == "mm" and metadata["frame"] == "motor" and metadata["adjusted"] is False
    # The error is in each camera's own preset px: sensor px / its binning, averaged.
    expected = np.mean([RIG3.truth.rms_px[cam] / MIXED.cameras[cam].binning for cam in names])
    assert metadata["error"] == pytest.approx(expected, rel=1e-15)


def test_toml_projects_like_the_fit():
    # What aniposelib's Camera.project does with a table: cv2.projectPoints with its matrix,
    # distortions, rotation and translation. It must land where the fit's camera puts the
    # point in the sensor, mapped into that camera's preset.
    data = toml_loads()(anipose_toml(RIG3.truth, MIXED, calibration_id=CALIBRATION_ID, cameras=CAPTURE3))
    X = np.concatenate([RIG3.world_corners(o.motor_mm) for o in RIG3.observations])
    for i in range(3):
        table = data[f"cam_{i}"]
        cam = table["name"]
        got = cv2.projectPoints(X, np.array(table["rotation"]), np.array(table["translation"]),
                                np.array(table["matrix"]), np.array(table["distortions"]))[0].reshape(-1, 2)
        expected = sensor_to_preset_points(sensor_pixels(X, cam, RIG3.truth), MIXED.cameras[cam])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9, err_msg=cam)


def test_toml_refuses_more_than_six_cameras_and_a_set_without_a_fitted_camera():
    fit = RIG3.truth
    with pytest.raises(ValueError, match="no calibrated camera"):
        anipose_toml(fit, PresetSet("x", {"stimCam": MIXED.cameras["stimCam"]}), calibration_id=CALIBRATION_ID,
                     cameras=CAPTURE3)
    many = dataclasses.replace(fit, cameras={f"c{i}": fit.cameras["left"] for i in range(7)},
                               rms_px={f"c{i}": 0.1 for i in range(7)})
    seven = PresetSet("x", {f"c{i}": MIXED.cameras["left"] for i in range(7)})
    with pytest.raises(ValueError, match="6"):
        anipose_toml(many, seven, calibration_id=CALIBRATION_ID, cameras=[CaptureCamera(f"c{i}", None, i) for i in range(7)])
