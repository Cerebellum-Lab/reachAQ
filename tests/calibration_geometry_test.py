"""The pixel space live 3D triangulates in, read from the left camera's configuration.

Live frames reach inference pooled to the base capture, so the geometry is the
configured base: hbin/vbin, width, height and ROI offset. A key the configuration
leaves out is whatever SpinCam runs at by default, not 0, because that is the ROI
the camera actually captures.
"""

import dataclasses

import pytest

from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from types import SimpleNamespace

from tools.acquisition.model.calibration_geometry import (
    NOT_IN_CALIBRATION,
    SPINCAM_DEFAULT_GEOMETRY,
    camera_serial,
    live_inference_geometry,
    match_cameras,
)

BASE_SIZE = {"hbin": 4, "vbin": 4, "width": 256, "height": 256}


def test_defaults_fill_missing_offsets():
    assert live_inference_geometry(BASE_SIZE) == PresetGeometry(4, 256, 256, 52, 6)


def test_an_empty_configuration_is_spincams_default_capture():
    assert live_inference_geometry({}) == PresetGeometry(4, 256, 256, 52, 6)


def test_configured_values_are_used():
    params = {"hbin": 2, "vbin": 2, "width": 512, "height": 400, "offsetx": 104, "offsety": 12}
    assert live_inference_geometry(params) == PresetGeometry(2, 512, 400, 104, 12)


def test_aliases():
    # offset_x/offset_y win over offsetx/offsety, as in capture_binning._base_offset:
    # SpinCam applies them after its defaults.
    both = {**BASE_SIZE, "offsetx": 52, "offsety": 6, "offset_x": 40, "offset_y": 10}
    assert live_inference_geometry(both) == PresetGeometry(4, 256, 256, 40, 10)
    alias_only = {**BASE_SIZE, "offset_x": 0, "offset_y": 0}
    assert live_inference_geometry(alias_only) == PresetGeometry(4, 256, 256, 0, 0)


def test_values_parsed_from_a_camera_url_are_read():
    # Parameters read back from a capture URL are strings; blank and "none" mean unset.
    params = {"hbin": "4", "vbin": "4.0", "width": "256", "height": "256", "offsetx": "", "offsety": "None"}
    assert live_inference_geometry(params) == PresetGeometry(4, 256, 256, 52, 6)


def test_unequal_binning_raises():
    with pytest.raises(ValueError, match="hbin=4 vbin=2"):
        live_inference_geometry({**BASE_SIZE, "vbin": 2})
    # A missing vbin is SpinCam's default, so it does not take hbin's configured value.
    with pytest.raises(ValueError, match="hbin=2 vbin=4"):
        live_inference_geometry({"hbin": 2, "width": 512, "height": 512})


@pytest.mark.parametrize("value", ["four", 52.5])
def test_a_value_that_is_not_a_whole_number_raises(value):
    with pytest.raises(ValueError, match="offsetx"):
        live_inference_geometry({**BASE_SIZE, "offsetx": value})


def test_defaults_mirror_spincam():
    # spinnaker_cam imports PySpin at load, so only a machine with the camera SDK checks this.
    pytest.importorskip("PySpin")
    from autotrainer.video.camera.spinnaker_cam import SpinCamDefaultParams

    defaults = dataclasses.asdict(SpinCamDefaultParams())
    assert {key: defaults[key] for key in SPINCAM_DEFAULT_GEOMETRY} == SPINCAM_DEFAULT_GEOMETRY
    # live_inference_geometry takes a missing vbin as the hbin default.
    assert defaults["vbin"] == SPINCAM_DEFAULT_GEOMETRY["hbin"]


def _cal(name, serial, calibrated=True, reason=""):
    return {"name": name, "serial": serial, "camera_id": None, "calibrated": calibrated, "reason": reason}


def test_the_serial_is_a_spinnaker_camera_s_host():
    assert camera_serial(SimpleNamespace(scheme="spinnaker", host="21000001")) == "21000001"
    assert camera_serial(SimpleNamespace(scheme="Spinnaker", host=" 21000001 ")) == "21000001"
    assert camera_serial(SimpleNamespace(scheme="random", host="localhost")) is None
    assert camera_serial(SimpleNamespace(scheme="spinnaker", host="")) is None
    assert camera_serial(None) is None


def test_cameras_match_by_serial_then_by_name_without_one():
    calibration = [_cal("left", "1"), _cal("right", "2"), _cal("stimCam", None),
                   _cal("camera4", "4", False, "too few board views: 4 of 12 needed")]
    match = match_cameras(calibration, {"left": "2", "right": "1", "stimCam": None, "camera4": "4", "camera5": "5"})
    # Renamed by serial; matched by name without a serial; the calibration's own 2D-only reason.
    assert match.calibrated == {"left": "right", "right": "left", "stimCam": "stimCam"}
    assert match.reasons == {"camera4": "too few board views: 4 of 12 needed", "camera5": NOT_IN_CALIBRATION}
    assert match.swapped == {}
    assert set(match.matched) == {"left", "right", "stimCam", "camera4"}


def test_a_calibration_without_serials_matches_every_camera_by_name():
    # Format 2 and legacy calibrations record no serials.
    match = match_cameras([_cal("left", None), _cal("right", None)], {"left": "1", "right": "2"})
    assert match.calibrated == {"left": "left", "right": "right"} and match.reasons == {}


def test_a_new_body_under_a_calibrated_name_is_swapped_unless_the_old_body_is_elsewhere():
    calibration = [_cal("left", "1"), _cal("right", "2")]
    swapped = match_cameras(calibration, {"left": "9", "right": "2"})
    assert swapped.reasons == {"left": "swapped: calibrated serial 1, now 9"}
    assert swapped.swapped == {"left": ("1", "9")}
    # left's calibrated body is now configured as camera4: left is a new camera, not a swap.
    moved = match_cameras(calibration, {"left": "9", "right": "2", "camera4": "1"})
    assert moved.reasons == {"left": NOT_IN_CALIBRATION} and moved.calibrated == {"right": "right", "camera4": "left"}
