"""Capture-binning presets: the same field of view at less binning.

A camera's saved params describe its base capture (hbin/vbin B, width, height,
offsets, gain). Live inference, the display and the stereo calibration are
sized for that base. A ``capture_binning`` param b asks for the same sensor
area at binning b instead. With k = B / b the camera then delivers frames k
times larger in each direction. The capture process averages k x k blocks
back to the base size for the consumers (autotrainer.video.frame_fit), and
recording keeps the larger frames.

The Blackfly S BFS-U3-16S2M offers only Sum binning, so less binning is
darker by k^2 at the same exposure (measured: bin 2 is 0.38x as bright as
bin 4 through gamma 0.7). The gain makes that up, adding 20 log10(k^2) dB,
rather than the exposure, which would blur fast reaches.

Only the runtime capture URL carries the result. The saved params, and so
everything sized from them, stay at the base.
"""
import math
from typing import Any, Dict, Iterable, Mapping, Optional

CAPTURE_BINNING_PARAM = "capture_binning"

# SpinCam accepts these aliases for offsetx/offsety. A preset drops them from
# the runtime URL, so they cannot be applied after the scaled offsets.
OFFSET_ALIASES = ("offset_x", "offset_y")


def _number(params: Mapping[str, Any], key: str) -> Optional[float]:
    value = params.get(key)
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none"}):
        return None
    return float(value)


def _base_offset(params: Mapping[str, Any], axis: str) -> int:
    # The alias wins when present: SpinCam applies URL params after its
    # defaults, so offset_x is set after the offsetx default.
    value = _number(params, f"offset_{axis}")
    if value is None:
        value = _number(params, f"offset{axis}")
    return 0 if value is None else int(value)


def capture_binning(params: Mapping[str, Any]) -> Optional[int]:
    """The requested capture binning, or None when the base capture is used as is."""
    value = _number(params, CAPTURE_BINNING_PARAM)
    return None if value is None else int(value)


def base_binning(params: Mapping[str, Any]) -> Optional[int]:
    """The base binning when it is the same in both directions, else None."""
    hbin, vbin = _number(params, "hbin"), _number(params, "vbin")
    if hbin is None or vbin is None or hbin != vbin:
        return None
    return int(hbin)


def capture_factor(params: Mapping[str, Any]) -> int:
    """k: how many times larger the captured frame is than the base frame in each direction."""
    binning = capture_binning(params)
    if binning is None:
        return 1
    hbin, vbin = _number(params, "hbin"), _number(params, "vbin")
    if hbin is None or vbin is None:
        raise ValueError(f"{CAPTURE_BINNING_PARAM}={binning} needs the base hbin and vbin params")
    if hbin != vbin:
        raise ValueError(
            f"{CAPTURE_BINNING_PARAM} needs equal base binning; got hbin={hbin:g} vbin={vbin:g}")
    if binning < 1 or int(hbin) % binning:
        raise ValueError(
            f"{CAPTURE_BINNING_PARAM}={binning} does not divide the base binning {hbin:g} into a whole factor")
    return int(hbin) // binning


def runtime_capture_params(params: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """The params a preset sets on the camera, or None when there is no preset."""
    binning = capture_binning(params)
    if binning is None:
        return None
    k = capture_factor(params)
    width, height = _number(params, "width"), _number(params, "height")
    if width is None or height is None:
        raise ValueError(f"{CAPTURE_BINNING_PARAM} needs the base width and height params")
    gain = _number(params, "gain")
    base_gain = 0.0 if gain is None else gain  # SpinCam.init leaves the camera at 0 dB
    return {
        "hbin": str(binning),
        "vbin": str(binning),
        "width": str(int(width) * k),
        "height": str(int(height) * k),
        "offsetx": str(_base_offset(params, "x") * k),
        "offsety": str(_base_offset(params, "y") * k),
        "gain": f"{base_gain + 20 * math.log10(k * k):.4f}",
    }


def common_capture_binning(cameras: Iterable[Any]) -> Optional[int]:
    """The binning all these cameras capture at, or None if they differ or there are none."""
    values = {camera.effective_capture_binning for camera in cameras}
    return values.pop() if len(values) == 1 else None


def capture_binning_available(cameras: Iterable[Any], binning: int) -> bool:
    """Whether ``binning`` divides every camera's base binning into a whole factor."""
    bases = [camera.base_binning for camera in cameras]
    return (
        len(bases) > 0
        and binning >= 1
        and all(base is not None and base % binning == 0 for base in bases)
    )


def apply_capture_binning(cameras: Iterable[Any], binning: int) -> None:
    """Set the preset on every camera; at the base binning the preset is removed."""
    cameras = tuple(cameras)
    if not capture_binning_available(cameras, binning):
        raise ValueError(f"capture binning {binning} does not divide every camera's base binning")
    for camera in cameras:
        camera.set_capture_binning(None if binning == camera.base_binning else int(binning))
