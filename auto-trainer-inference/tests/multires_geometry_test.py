"""Preset geometry and the pixel-centre mapping between the sensor and a preset.

The rig runs the same 1440x1080 sensor in three presets (bin 4 at 256x256, bin 2
at 512x512, bin 1 at 1024x1024). One calibration is fitted in sensor pixels and
every preset's intrinsics are derived from it, so the mapping has to be exact:
a half-pixel error here is a systematic 3D offset in every later session.

The convention is pixel centres. Sensor pixel i is centred on coordinate i, so
binned pixel j (j >= 0 inside the ROI) is centred on b*(j + offset) + (b-1)/2.
`sensor_roi` returns pixel *edges* (pixel i spans [i, i+1)), which sit half a
pixel away from that convention; the edge test below pins the relation.
"""

import cv2
import numpy as np
import pytest

from autotrainer.inference.multires_calibration.geometry import (
    PresetGeometry,
    PresetSet,
    map_intrinsics,
    preset_to_sensor_points,
    presets_for_cameras,
    presets_from_base,
    sensor_to_preset_points,
)


BASE = PresetGeometry(binning=4, width=256, height=256, offset_x=52, offset_y=6)

K_SENSOR = np.array([[2300.0, 0.0, 720.0], [0.0, 2290.0, 540.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.3, 0.12, 0.001, -0.002, 0.05])


def test_presets_from_base_match_the_rig_presets():
    assert presets_from_base(BASE) == [
        BASE,
        PresetGeometry(2, 512, 512, 104, 12),
        PresetGeometry(1, 1024, 1024, 208, 24),
    ]
    assert [g.name for g in presets_from_base(BASE)] == ["256x256_bin4", "512x512_bin2", "1024x1024_bin1"]


def test_presets_from_base_rejects_a_binning_that_does_not_divide_the_base():
    # bin 3 cannot be reached from bin 4 by whole sensor pixels, so the ROI would not align.
    with pytest.raises(ValueError):
        presets_from_base(BASE, binnings=(4, 3))
    # Going coarser than the base is also not a whole-pixel relation.
    with pytest.raises(ValueError):
        presets_from_base(PresetGeometry(2, 512, 512, 104, 12), binnings=(4, 2, 1))


def test_sensor_roi_is_in_pixel_edges_of_the_sensor():
    assert BASE.sensor_roi() == (208.0, 24.0, 1232.0, 1048.0)
    # All three presets cover the same sensor window, which is the whole point of the offsets.
    assert {g.sensor_roi() for g in presets_from_base(BASE)} == {(208.0, 24.0, 1232.0, 1048.0)}


def test_pixel_centre_mapping_of_the_first_roi_pixel():
    # sensor pixels 208..211 x 24..27 bin into preset pixel (0, 0); their centre is (209.5, 25.5)
    np.testing.assert_allclose(sensor_to_preset_points(np.array([[209.5, 25.5]]), BASE), [[0.0, 0.0]])
    np.testing.assert_allclose(preset_to_sensor_points(np.array([[0.0, 0.0]]), BASE), [[209.5, 25.5]])


@pytest.mark.parametrize("g", presets_from_base(BASE), ids=lambda g: g.name)
def test_roi_edges_map_onto_the_preset_image_edges(g):
    x0, y0, x1, y1 = g.sensor_roi()
    # Edge coordinates are half a pixel above centre coordinates (pixel i spans [i, i+1) as an edge
    # coordinate and [i-0.5, i+0.5] as a centre coordinate), so subtract 0.5 first. The first
    # preset pixel then starts at -0.5 and the last ends at size-0.5.
    corners = np.array([[x0, y0], [x1, y1]]) - 0.5
    np.testing.assert_allclose(sensor_to_preset_points(corners, g), [[-0.5, -0.5], [g.width - 0.5, g.height - 0.5]])


@pytest.mark.parametrize("g", presets_from_base(BASE), ids=lambda g: g.name)
def test_sensor_to_preset_and_back_is_the_identity(g):
    rng = np.random.default_rng(1)
    sensor = rng.uniform([0.0, 0.0], [1440.0, 1080.0], size=(50, 2))
    np.testing.assert_allclose(preset_to_sensor_points(sensor_to_preset_points(sensor, g), g), sensor, atol=1e-9)


def test_point_mapping_rejects_arrays_that_are_not_n_by_2():
    with pytest.raises(ValueError):
        sensor_to_preset_points(np.zeros((5, 3)), BASE)
    with pytest.raises(ValueError):
        preset_to_sensor_points(np.zeros(2), BASE)


def test_point_mapping_returns_n_by_2_and_handles_no_points():
    assert sensor_to_preset_points(np.zeros((0, 2)), BASE).shape == (0, 2)
    assert preset_to_sensor_points(np.zeros((7, 2)), BASE).shape == (7, 2)


def test_map_intrinsics_values():
    K = np.array([[2000.0, 0, 720.0], [0, 2010.0, 540.0], [0, 0, 1]])
    np.testing.assert_allclose(map_intrinsics(K, BASE),
                               [[500.0, 0, 127.625], [0, 502.5, 128.625], [0, 0, 1]])


def test_map_intrinsics_does_not_modify_its_input():
    before = K_SENSOR.copy()
    map_intrinsics(K_SENSOR, BASE)
    np.testing.assert_array_equal(K_SENSOR, before)


def test_projection_then_mapping_equals_projection_with_mapped_intrinsics():
    # The distortion coefficients act on normalised coordinates, so they are untouched by binning
    # and cropping. Projecting with the mapped K must therefore land exactly where projecting with
    # the sensor K and then mapping the pixel does, including through the distortion polynomial.
    rng = np.random.default_rng(0)
    points = np.column_stack([
        rng.uniform(-50.0, 50.0, 200),
        rng.uniform(-50.0, 50.0, 200),
        rng.uniform(100.0, 300.0, 200),
    ])
    zero = np.zeros(3)
    sensor_px = cv2.projectPoints(points, zero, zero, K_SENSOR, DIST)[0].reshape(-1, 2)

    for g in presets_from_base(BASE):
        expected = sensor_to_preset_points(sensor_px, g)
        via_mapped_k = cv2.projectPoints(points, zero, zero, map_intrinsics(K_SENSOR, g), DIST)[0].reshape(-1, 2)
        np.testing.assert_allclose(via_mapped_k, expected, rtol=0.0, atol=1e-9, err_msg=g.name)


def test_presets_for_cameras_derives_each_cameras_own_window_at_every_binning():
    # Each camera keeps its own window: left is the rig's bin-4 preset, camera4 a bin-2 one
    # elsewhere on the sensor. bin2 and bin4 sets exist only where every window is whole there.
    live = {"left": BASE, "camera4": PresetGeometry(2, 300, 200, 10, 40)}
    sets = presets_for_cameras(live)
    assert [s.name for s in sets] == ["live", "bin1", "bin2", "bin4"]
    by_name = {s.name: s for s in sets}
    assert by_name["live"] == PresetSet("live", live)
    assert by_name["bin1"].cameras == {
        "left": PresetGeometry(1, 1024, 1024, 208, 24),
        "camera4": PresetGeometry(1, 600, 400, 20, 80),
    }
    assert by_name["bin2"].cameras == {"left": PresetGeometry(2, 512, 512, 104, 12), "camera4": live["camera4"]}
    assert by_name["bin4"].cameras == {"left": BASE, "camera4": PresetGeometry(4, 150, 100, 5, 20)}
    for s in sets:
        for cam, g in s.cameras.items():
            assert g.sensor_roi() == live[cam].sensor_roi(), (s.name, cam)


def test_presets_for_cameras_leaves_out_a_binning_a_window_does_not_divide_and_a_copy_of_live():
    # 302 sensor px at bin 1 do not bin by 4; and with every camera at bin 4 the bin4 set is the live one.
    odd = {"left": BASE, "camera4": PresetGeometry(1, 302, 200, 10, 40)}
    assert [s.name for s in presets_for_cameras(odd)] == ["live", "bin1", "bin2"]
    assert [s.name for s in presets_for_cameras({"left": BASE, "right": BASE})] == ["live", "bin1", "bin2"]
    with pytest.raises(ValueError):
        presets_for_cameras({})
