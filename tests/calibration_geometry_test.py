"""The pixel space live 3D triangulates in, read from the left camera's configuration.

Live frames reach inference pooled to the base capture, so the geometry is the
configured base: hbin/vbin, width, height and ROI offset. A key the configuration
leaves out is whatever SpinCam runs at by default, not 0, because that is the ROI
the camera actually captures.
"""

import dataclasses

import pytest

from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from tools.acquisition.model.calibration_geometry import (
    SPINCAM_DEFAULT_GEOMETRY,
    live_inference_geometry,
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
