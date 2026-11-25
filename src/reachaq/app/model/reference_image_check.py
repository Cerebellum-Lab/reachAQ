"""The reference-image check: has a camera moved since its calibration?

A calibration can keep one still frame per camera, at the camera's live geometry and with
nothing in view, as reference/<camera>.png in its library entry. At recording start each
camera's current frame is registered against its reference (register), and a shift,
rotation or scale beyond the limits marks the camera "moved" (check_cameras). A moved camera
joins the "calibration looks needed, record anyway?" confirmation with its measured values.

The registration is OpenCV's ECC (findTransformECC, MOTION_AFFINE) on the Sobel gradient
magnitude of both images, started from a phase correlation of the same gradients so a
shift of tens of pixels converges. The fixed rig structure dominates the gradients; an
animal or a hand in view lowers the ECC correlation (`quality`) rather than faking a move,
and a quality below `min_quality` is "can't tell", which never warns.

The affine is reduced to a similarity: the shift is how far the image centre moves (the
translation taken at the centre, so a rotation about it is not counted as a shift), the
rotation is that of the affine's best-fitting rotation, and the scale is
(sqrt(det) - 1) * 100 %.

Settings, in <library>/settings.yaml under checks.reference_image, global with an optional
per_camera override of any limit (see DEFAULTS and docs/acquisition/stereo-calibration.md):

    checks:
      reference_image:
        enabled: false
        max_shift_px: 4.0
        max_rotation_deg: 1.5
        max_scale_pct: 2.0
        min_quality: 0.80
        per_camera: {left: {max_shift_px: 6.0}}
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from reachaq.core.logging import get_verbose_logger
from reachaq.inference.multires_calibration.pipeline import REFERENCE_DIR

logger = get_verbose_logger(__name__)

CHECKS_KEY = "checks"
REFERENCE_KEY = "reference_image"
PER_CAMERA_KEY = "per_camera"
ENABLED_KEY = "enabled"
DEFAULTS = {
    "max_shift_px": 4.0,
    "max_rotation_deg": 1.5,
    "max_scale_pct": 2.0,
    "min_quality": 0.80,
}

OK = "ok"
MOVED = "moved"
CANT_TELL = "can't tell"
NO_REFERENCE = "no reference"

# ECC: at most this many iterations, or until the warp changes by less than the epsilon.
_ECC_ITERATIONS = 100
_ECC_EPSILON = 1e-5
_ECC_GAUSS_SIZE = 5
# The images are smoothed before the gradient so sensor noise does not dominate it.
_BLUR_SIGMA = 1.0


@dataclass(frozen=True)
class Registration:
    shift_px: float  # how far the image centre moved, in pixels
    rotation_deg: float  # signed, about the optical axis
    scale_pct: float  # signed, (sqrt(det) - 1) * 100
    quality: float  # the ECC correlation, 0 when ECC did not converge


@dataclass(frozen=True)
class Limits:
    max_shift_px: float = DEFAULTS["max_shift_px"]
    max_rotation_deg: float = DEFAULTS["max_rotation_deg"]
    max_scale_pct: float = DEFAULTS["max_scale_pct"]
    min_quality: float = DEFAULTS["min_quality"]


@dataclass(frozen=True)
class ReferenceCheckSettings:
    enabled: bool = False
    limits: Limits = Limits()
    per_camera: Dict[str, Limits] = field(default_factory=dict)

    def limits_for(self, camera: str) -> Limits:
        return self.per_camera.get(camera, self.limits)


@dataclass(frozen=True)
class ReferenceVerdict:
    camera: str
    status: str  # OK, MOVED, CANT_TELL or NO_REFERENCE
    registration: Optional[Registration]
    limits: Limits = Limits()
    reason: str = ""  # why it is CANT_TELL or NO_REFERENCE without a registration


def _limits(values: Any, base: Limits, where: str) -> Limits:
    """`base` with each limit `values` gives replaced; a value that is not a positive number is warned about and ignored."""
    if not isinstance(values, Mapping):
        if values is not None:
            logger.warning("ignoring %s: it is not a mapping", where)
        return base
    changes = {}
    for name in DEFAULTS:
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            logger.warning("ignoring %s.%s = %r: it is not a number of 0 or more", where, name, value)
            continue
        changes[name] = float(value)
    return Limits(**{**base.__dict__, **changes})


def reference_check_settings(library_settings: Optional[Mapping[str, Any]]) -> ReferenceCheckSettings:
    """The check's settings from the library's settings.yaml (library.read_settings); the defaults where unset."""
    checks = (library_settings or {}).get(CHECKS_KEY)
    section = checks.get(REFERENCE_KEY) if isinstance(checks, Mapping) else None
    if not isinstance(section, Mapping):
        return ReferenceCheckSettings()
    enabled = section.get(ENABLED_KEY, False)
    if not isinstance(enabled, bool):
        logger.warning("ignoring checks.reference_image.enabled = %r: it is not true or false", enabled)
        enabled = False
    limits = _limits(section, Limits(), "checks.reference_image")
    per_camera = {}
    overrides = section.get(PER_CAMERA_KEY)
    if isinstance(overrides, Mapping):
        for camera, values in overrides.items():
            per_camera[str(camera)] = _limits(values, limits, f"checks.reference_image.per_camera.{camera}")
    elif overrides is not None:
        logger.warning("ignoring checks.reference_image.per_camera: it is not a mapping")
    return ReferenceCheckSettings(enabled=enabled, limits=limits, per_camera=per_camera)


def with_enabled(library_settings: Optional[Mapping[str, Any]], enabled: bool) -> Dict[str, Any]:
    """The `checks` value to store (library.update_settings(checks=...)) with the check switched
    on or off, keeping every other key under it."""
    checks = (library_settings or {}).get(CHECKS_KEY)
    checks = dict(checks) if isinstance(checks, Mapping) else {}
    section = checks.get(REFERENCE_KEY)
    section = dict(section) if isinstance(section, Mapping) else {}
    section[ENABLED_KEY] = bool(enabled)
    checks[REFERENCE_KEY] = section
    return checks


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    return image


def _gradient(image: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(_gray(image).astype(np.float32), (0, 0), _BLUR_SIGMA)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def register(reference: np.ndarray, current: np.ndarray) -> Registration:
    """How `current` is shifted, rotated and scaled against `reference`, and how well they match.

    Both are 2D (or BGR) images of the same size. A non-converged ECC gives quality 0 and
    NaN for the rest. Raises ValueError when the sizes differ.
    """
    ref, cur = _gradient(reference), _gradient(current)
    if ref.shape != cur.shape:
        raise ValueError(f"the current frame is {cur.shape[1]}x{cur.shape[0]}, "
                         f"its reference {ref.shape[1]}x{ref.shape[0]}")
    height, width = ref.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(ref, cur, window)
    warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, _ECC_ITERATIONS, _ECC_EPSILON)
    try:
        quality, warp = cv2.findTransformECC(ref, cur, warp, cv2.MOTION_AFFINE, criteria, None, _ECC_GAUSS_SIZE)
    except cv2.error as err:
        logger.info("the reference-image registration did not converge: %s", str(err).strip().splitlines()[-1])
        return Registration(math.nan, math.nan, math.nan, 0.0)
    a = warp[:, :2].astype(float)
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    shift = a @ centre + warp[:, 2].astype(float) - centre
    rotation = math.degrees(math.atan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1]))
    det = float(np.linalg.det(a))
    scale = (math.sqrt(abs(det)) - 1.0) * 100.0
    quality = float(quality) if math.isfinite(quality) else 0.0
    return Registration(float(np.hypot(*shift)), rotation, scale, quality)


def judge(camera: str, registration: Registration, limits: Limits) -> ReferenceVerdict:
    """MOVED over any limit, CANT_TELL below the minimum quality, else OK.

    A registration with a value that is not finite (ECC did not converge) is always
    CANT_TELL, whatever min_quality is: NaN compares false against every limit, so it would
    otherwise pass as OK with a min_quality of 0.
    """
    values = (registration.shift_px, registration.rotation_deg, registration.scale_pct, registration.quality)
    if not all(math.isfinite(v) for v in values):
        return ReferenceVerdict(camera, CANT_TELL, registration, limits, "the registration did not converge")
    if not registration.quality >= limits.min_quality:
        return ReferenceVerdict(camera, CANT_TELL, registration, limits,
                                f"match {registration.quality:.2f} below {limits.min_quality:.2f}")
    over = (registration.shift_px > limits.max_shift_px
            or abs(registration.rotation_deg) > limits.max_rotation_deg
            or abs(registration.scale_pct) > limits.max_scale_pct)
    return ReferenceVerdict(camera, MOVED if over else OK, registration, limits)


def reference_path(entry, camera: str) -> Path:
    """reference/<camera>.png in the calibration folder `entry` (a path or a library.LibraryEntry)."""
    return Path(getattr(entry, "path", entry)) / REFERENCE_DIR / f"{camera}.png"


def check_cameras(
    entry,
    frames: Mapping[str, Optional[np.ndarray]],
    settings,
    *,
    names: Optional[Mapping[str, str]] = None,
) -> Dict[str, ReferenceVerdict]:
    """Each camera of `frames` (camera -> its current frame, None when there is none) against
    its reference in the calibration folder `entry`.

    `settings` are the library's settings (library.read_settings) or a ReferenceCheckSettings;
    {} when the check is off. `names` maps a camera to its name in the calibration, where the
    two differ; the reference is looked up by that name, the limits and the result by the
    camera's own. A camera without a reference is NO_REFERENCE; one without a frame, with a
    frame of another size, or below the minimum quality is CANT_TELL. Never raises on a
    missing or unreadable reference.
    """
    if not isinstance(settings, ReferenceCheckSettings):
        settings = reference_check_settings(settings)
    if not settings.enabled:
        return {}
    verdicts = {}
    for camera, frame in frames.items():
        limits = settings.limits_for(camera)
        path = reference_path(entry, (names or {}).get(camera, camera))
        reference = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if reference is None:
            verdicts[camera] = ReferenceVerdict(camera, NO_REFERENCE, None, limits,
                                                "the calibration has no reference image for it")
            continue
        if frame is None:
            verdicts[camera] = ReferenceVerdict(camera, CANT_TELL, None, limits, "no frame from the camera")
            continue
        try:
            registration = register(reference, frame)
        except ValueError as err:
            verdicts[camera] = ReferenceVerdict(camera, CANT_TELL, None, limits, str(err))
            continue
        verdicts[camera] = judge(camera, registration, limits)
        logger.info("reference image %s: %s, %s", camera, verdicts[camera].status, registration)
    return verdicts


def moved_detail(verdict: ReferenceVerdict) -> str:
    """The confirmation's line for a camera: its measured values against their limits."""
    r, limits = verdict.registration, verdict.limits
    return (f"{verdict.camera}: shifted {r.shift_px:.1f} px (limit {limits.max_shift_px:.1f}), "
            f"rotated {abs(r.rotation_deg):.1f}\N{DEGREE SIGN} (limit {limits.max_rotation_deg:.1f}), "
            f"scale {abs(r.scale_pct):.1f}% (limit {limits.max_scale_pct:.1f}), match {r.quality:.2f}")


def _number(value: float) -> Optional[float]:
    return round(float(value), 4) if math.isfinite(value) else None


def verdict_metadata(verdict: ReferenceVerdict) -> Dict[str, Any]:
    """A camera's result for calibration3d.movedCameraChecks.referenceImage."""
    r, limits = verdict.registration, verdict.limits
    return {
        "status": verdict.status,
        "shiftPx": None if r is None else _number(r.shift_px),
        "rotationDeg": None if r is None else _number(r.rotation_deg),
        "scalePct": None if r is None else _number(r.scale_pct),
        "quality": None if r is None else _number(r.quality),
        "reason": verdict.reason or None,
        "limits": {
            "maxShiftPx": limits.max_shift_px,
            "maxRotationDeg": limits.max_rotation_deg,
            "maxScalePct": limits.max_scale_pct,
            "minQuality": limits.min_quality,
        },
    }
