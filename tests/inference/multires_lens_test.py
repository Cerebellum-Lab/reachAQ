"""Nominal lenses in the calibration library's settings, and the report's checks against them."""

import dataclasses
import math

import numpy as np
import pytest

from reachaq.inference.multires_calibration.fit import CameraModel, CaptureCamera
from reachaq.inference.multires_calibration.geometry import PresetGeometry
from reachaq.inference.multires_calibration.report import (
    CaptureSummary,
    LensSettings,
    build_report,
    configured_lenses,
)
from multires_synthetic import synthetic_rig

BASE = PresetGeometry(4, 256, 256, 52, 6)
RIG = synthetic_rig(noise_px=0.0)
PIXEL_UM = 3.45


def report_of(fit, **lens_options):
    n = len(RIG.observations)
    capture = CaptureSummary(waypoints_total=n, waypoints_verified=n, waypoints_unreached=0,
                             views_kept={cam: n for cam in fit.cameras}, rejections={cam: {} for cam in fit.cameras})
    return build_report(fit, RIG.observations, RIG.board, BASE.sensor_roi(), capture, [], None, **lens_options)


def metrics(report, name):
    return {m.camera: m for m in report.metrics if m.name == name}


def with_fx(fit, cam, fx):
    m = fit.cameras[cam]
    K = np.array(m.K, float)
    K[0, 0], K[1, 1] = fx, fx * K[1, 1] / K[0, 0]
    return dataclasses.replace(fit, cameras={**fit.cameras, cam: CameraModel(K=K, dist=m.dist, rvec=m.rvec, tvec=m.tvec)})


def test_lenses_parse_by_serial_or_name():
    lenses = configured_lenses({"lenses": {"24152533": {"lens_mm": 8.5}, "right": {"lens_mm": 12}}, "pixel_um": 3.45})
    assert lenses == LensSettings(lens_mm={"24152533": 8.5, "right": 12.0}, pixel_um=3.45)
    cameras = [CaptureCamera("left", "24152533", 0), CaptureCamera("right", "24152513", 1),
               CaptureCamera("camera4", None, 2)]
    # left by its serial; right's serial is not listed, so by its name; camera4 has none.
    assert lenses.by_camera(cameras) == {"left": 8.5, "right": 12.0}
    assert configured_lenses({}) == LensSettings(lens_mm={}, pixel_um=None)
    assert configured_lenses({"pixel_um": 3.45}).pixel_um == 3.45


@pytest.mark.parametrize("settings, key", [
    ({"lenses": {"left": {"lens_mm": -8.5}}, "pixel_um": 3.45}, r"lenses\.left\.lens_mm"),
    ({"lenses": {"left": {"lens_mm": math.inf}}, "pixel_um": 3.45}, r"lenses\.left\.lens_mm"),
    ({"lenses": {"left": {"lens_mm": "8.5"}}, "pixel_um": 3.45}, r"lenses\.left\.lens_mm"),
    ({"lenses": {"left": {}}, "pixel_um": 3.45}, r"lenses\.left\.lens_mm"),
    ({"lenses": {"left": {"lens_mm": 8.5, "fov": 30}}, "pixel_um": 3.45}, r"lenses\.left\.fov"),
    ({"lenses": {"left": 8.5}, "pixel_um": 3.45}, r"lenses\.left"),
    ({"lenses": [8.5], "pixel_um": 3.45}, "lenses"),
    ({"lenses": {"left": {"lens_mm": 8.5}}}, "pixel_um"),
    ({"pixel_um": 0}, "pixel_um"),
])
def test_bad_lens_settings_are_refused_naming_the_key(settings, key):
    with pytest.raises(ValueError, match=key):
        configured_lenses(settings)


def test_the_report_gives_each_cameras_focal_length_in_mm_and_its_field_of_view():
    report = report_of(RIG.truth, lenses={"left": 8.0}, pixel_um=PIXEL_UM)
    fx = RIG.truth.cameras["left"].K[0, 0]  # 2300 px: 7.935 mm, HFOV 2 atan(1440 / 4600) = 34.8 deg
    assert f"left: fitted {fx * PIXEL_UM / 1000:.1f} mm, HFOV 34.8\N{DEGREE SIGN} (nominal 8 mm)" in report.notes
    assert any(n.startswith("right: fitted 8.0 mm, HFOV ") and n.endswith("(no nominal lens)") for n in report.notes)
    (check,) = metrics(report, "focal_vs_nominal_rel").values()
    assert check.camera == "left" and check.status == "pass"


def test_a_focal_length_more_than_15_percent_off_nominal_warns():
    report = report_of(RIG.truth, lenses={"left": 6.0, "right": 8.0}, pixel_um=PIXEL_UM)
    checks = metrics(report, "focal_vs_nominal_rel")
    assert checks["left"].status == "warn" and checks["left"].value == pytest.approx(7.935 / 6.0 - 1.0, abs=1e-3)
    assert checks["right"].status == "pass"
    assert report.verdict != "fail"


def test_focal_pair_rel_compares_against_the_nominal_lenses():
    # Right with a 12 mm lens and left with 8.5 mm: fx 12/8.5 times left's.
    fit = with_fx(RIG.truth, "right", RIG.truth.cameras["left"].K[0, 0] * 12 / 8.5)
    matched = metrics(report_of(fit, lenses={"left": 8.5, "right": 12.0}, pixel_um=PIXEL_UM), "focal_pair_rel")
    assert matched["left-right"].value == pytest.approx(0.0, abs=1e-9) and matched["left-right"].status == "pass"
    off = metrics(report_of(fit, lenses={"left": 8.5, "right": 8.5}, pixel_um=PIXEL_UM), "focal_pair_rel")
    # (fx left / fx right) / (8.5 / 8.5) - 1 = 8.5 / 12 - 1
    assert off["left-right"].value == pytest.approx(1.0 - 8.5 / 12) and off["left-right"].status == "warn"


def test_without_lenses_focal_pair_rel_is_reported_ungraded():
    fit = with_fx(RIG.truth, "right", RIG.truth.cameras["left"].K[0, 0] * 12 / 8.5)
    report = report_of(fit)
    pair = metrics(report, "focal_pair_rel")["left-right"]
    assert pair.status == "pass" and pair.warn is None and pair.fail is None
    assert pair.value == pytest.approx(abs(1 - 12 / 8.5) / ((1 + 12 / 8.5) / 2))
    assert not metrics(report, "focal_vs_nominal_rel")
    assert not any("fitted" in n and "HFOV" in n for n in report.notes)
