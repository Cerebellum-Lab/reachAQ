"""The pixel space live 3D triangulates in, read from a camera's configuration.

Live frames reach inference pooled to the base capture (capture_binning.py) whatever the
capture-binning preset, so live 3D always works in the base geometry: the saved
hbin/vbin, width, height and ROI offset. The configuration may leave any of them out,
and the camera then runs at SpinCam's default. Those defaults are mirrored here rather
than read from SpinCamDefaultParams, because spinnaker_cam imports PySpin when it loads.
"""

from typing import Any, Mapping, Sequence

from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from tools.acquisition.model.capture_binning import _number

# SpinCamDefaultParams' geometry (auto-trainer-video camera/spinnaker_cam.py). Its vbin
# default equals its hbin default; tests/calibration_geometry_test.py checks both on a
# machine with PySpin.
SPINCAM_DEFAULT_GEOMETRY = dict(hbin=4, width=256, height=256, offsetx=52, offsety=6)


def live_inference_geometry(params: Mapping[str, Any]) -> PresetGeometry:
    """The base PresetGeometry of a camera configured with `params`.

    The offsets accept SpinCam's aliases, offset_x and offset_y, which win over offsetx and
    offsety as in capture_binning._base_offset. Unlike there, a missing key is SpinCam's
    default rather than 0, since that is the ROI the camera actually captures.

    Raises ValueError when hbin and vbin differ, as no preset bins the two directions
    differently, or when a value is not a whole number.
    """
    defaults = SPINCAM_DEFAULT_GEOMETRY
    hbin = _whole(params, ("hbin",), defaults["hbin"])
    vbin = _whole(params, ("vbin",), defaults["hbin"])
    if hbin != vbin:
        raise ValueError(f"live 3D needs the same binning both ways; the camera has hbin={hbin} vbin={vbin}")
    return PresetGeometry(
        binning=hbin,
        width=_whole(params, ("width",), defaults["width"]),
        height=_whole(params, ("height",), defaults["height"]),
        offset_x=_whole(params, ("offset_x", "offsetx"), defaults["offsetx"]),
        offset_y=_whole(params, ("offset_y", "offsety"), defaults["offsety"]),
    )


def _whole(params: Mapping[str, Any], keys: Sequence[str], default: int) -> int:
    """The first of `keys` set in `params`, as a whole number, else `default`."""
    for key in keys:
        # Parsed as capture_binning parses the same params: blank and "none" are unset.
        value = _number(params, key)
        if value is None:
            continue
        if not value.is_integer():
            raise ValueError(f"{key}={params[key]!r} is not a whole number")
        return int(value)
    return default
