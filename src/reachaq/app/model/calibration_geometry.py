"""The pixel space live 3D triangulates in, read from a camera's configuration.

Live frames reach inference pooled to the base capture (capture_binning.py) whatever the
capture-binning preset, so live 3D always works in the base geometry: the saved
hbin/vbin, width, height and ROI offset. The configuration may leave any of them out,
and the camera then runs at SpinCam's default. Those defaults are mirrored here rather
than read from SpinCamDefaultParams, because spinnaker_cam imports PySpin when it loads.

It also says which configured camera is which camera of a calibration (match_cameras): a
camera is its serial, CameraConfiguration.host for a Spinnaker camera, so a renamed camera
keeps its calibration and a swapped body loses it. A camera without a serial on either
side (the random and playback schemes, or a calibration from before serials were recorded)
is matched by name.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from reachaq.inference.multires_calibration.geometry import PresetGeometry
from reachaq.app.model.capture_binning import _number

# SpinCamDefaultParams' geometry (reachaq camera/spinnaker_cam.py). Its vbin
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


NOT_IN_CALIBRATION = "not in the active calibration"


def camera_serial(config) -> Optional[str]:
    """The serial of the camera `config` (a CameraConfiguration) opens: its host for a
    Spinnaker camera, None for any other scheme or a blank host."""
    if config is None or str(getattr(config, "scheme", "") or "").lower() != "spinnaker":
        return None
    host = str(getattr(config, "host", "") or "").strip()
    return host or None


@dataclass(frozen=True)
class CameraMatch:
    """Configured cameras against a calibration's cameras.

    `calibrated`: configured name -> the calibration's name for it, for each configured camera
    the calibration calibrated. `reasons`: configured name -> why it is 2D only, for every
    other configured camera. `swapped`: configured name -> (the calibrated serial, the serial
    now) for a camera whose name the calibration has with another body. `matched`: the
    calibration's names that some configured camera is.
    """

    calibrated: Dict[str, str] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)
    swapped: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    matched: Tuple[str, ...] = ()


def match_cameras(calibration: Sequence[Mapping[str, Any]], configured: Mapping[str, Optional[str]]) -> CameraMatch:
    """Which calibration camera each configured camera is.

    `calibration` is a CalibrationIndex.cameras list ({name, serial, calibrated, reason}).
    `configured` maps each configured camera's name to its serial (camera_serial), None when
    it has none. A camera matches the calibration camera with its serial; failing that, the
    calibration camera of its name when either has no serial. A camera named like a
    calibration camera whose serial is another, and no configured camera has that serial,
    is swapped: "swapped: calibrated serial X, now Y". Otherwise it is NOT_IN_CALIBRATION.
    A matched camera the calibration left 2D only gets the calibration's own reason.
    """
    by_serial = {c["serial"]: c for c in calibration if c.get("serial")}
    by_name = {c["name"]: c for c in calibration}
    configured_serials = {serial for serial in configured.values() if serial}
    calibrated: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    swapped: Dict[str, Tuple[str, str]] = {}
    matched = []
    for name, serial in configured.items():
        camera = by_serial.get(serial) if serial else None
        same = by_name.get(name)
        if camera is None and same is not None:
            same_serial = same.get("serial")
            if not same_serial or (not serial and same_serial not in configured_serials):
                camera = same
            elif serial and same_serial not in configured_serials:
                swapped[name] = (same_serial, serial)
                reasons[name] = f"swapped: calibrated serial {same_serial}, now {serial}"
                continue
        if camera is None:
            reasons[name] = NOT_IN_CALIBRATION
            continue
        matched.append(camera["name"])
        if camera.get("calibrated", True):
            calibrated[name] = camera["name"]
        else:
            reasons[name] = camera.get("reason") or "not calibrated"
    return CameraMatch(calibrated=calibrated, reasons=reasons, swapped=swapped, matched=tuple(matched))
