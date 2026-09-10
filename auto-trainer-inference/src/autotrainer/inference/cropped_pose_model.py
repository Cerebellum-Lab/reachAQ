"""Region-of-interest cropping wrapper for a pose model.

Cropping is the safe way to cut live inference cost. Rescaling a 256x256 frame to
128x128 halves spatial resolution, so keypoint error roughly doubles in
original-image terms and that error feeds stereo triangulation, where 2D error is
geometrically amplified. Cropping to the same pixel count keeps full resolution
for everything inside the region.

It also needs no retraining. The DeepLabCut backbone is fully convolutional and a
crop preserves scale, so keypoints inside the region are detected as before; only
the coordinate frame shifts, and this wrapper shifts it back. The one real cost is
context: a keypoint close to the crop edge has less surrounding image than it did
in training, so keep a margin around the reach volume rather than cropping tight.
DeepLabCut-Live reported RMSE moving from 4.4 px to 5.5 px with a 50 px margin for
roughly 75% faster inference.

Coordinates leaving this wrapper are in full-frame space, which is what stereo
calibration and triangulation expect.
"""

from __future__ import annotations

import dataclasses
import os
import typing

import numpy

from autotrainer.core.logging import get_verbose_logger

from .pose_model import PoseModel


logger = get_verbose_logger(__name__)

INFERENCE_ROI_ENV_VAR = "REACHAQ_INFERENCE_ROI"
"""Optional "x,y,width,height" override, in full-frame pixels.

Deliberately an environment variable rather than a `SystemConfiguration` field:
`SystemConfiguration` rejects any version but its own, so adding a key would
invalidate every saved configuration file. Promoting this to the schema is a
separate decision that needs a version bump.
"""


@dataclasses.dataclass(frozen=True)
class InferenceRoi:
    """A crop region in full-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Inference ROI must have positive size: {self}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Inference ROI origin cannot be negative: {self}")

    @classmethod
    def parse(cls, text: typing.Optional[str]) -> typing.Optional["InferenceRoi"]:
        """Parse "x,y,width,height". Returns None for empty or unset input."""
        if text is None:
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        parts = [item.strip() for item in cleaned.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Inference ROI must be 'x,y,width,height', got {text!r}"
            )
        try:
            values = [int(item) for item in parts]
        except ValueError as err:
            raise ValueError(f"Inference ROI values must be integers: {text!r}") from err
        return cls(*values)

    @classmethod
    def from_environment(cls, environ=None) -> typing.Optional["InferenceRoi"]:
        source = os.environ if environ is None else environ
        return cls.parse(source.get(INFERENCE_ROI_ENV_VAR))

    def validate_for_shape(self, shape: typing.Tuple[int, int]) -> None:
        """Check the region fits inside a (rows, columns) frame shape."""
        rows, columns = shape
        if self.y + self.height > rows or self.x + self.width > columns:
            raise ValueError(
                f"Inference ROI {self} does not fit inside a frame of shape "
                f"{shape} (rows, columns)"
            )

    def covers_shape(self, shape: typing.Tuple[int, int]) -> bool:
        """True when the region is the whole frame, so cropping is a no-op."""
        rows, columns = shape
        return self.x == 0 and self.y == 0 and self.height == rows and self.width == columns


class CroppedPoseModel(PoseModel):
    """Runs an inner pose model on a crop and returns full-frame coordinates."""

    def __init__(self, inner: PoseModel, roi: InferenceRoi):
        super().__init__()
        self._inner = inner
        self._roi = roi
        self._buffer: typing.Optional[numpy.ndarray] = None

    @property
    def roi(self) -> InferenceRoi:
        return self._roi

    @property
    def inner(self) -> PoseModel:
        return self._inner

    @classmethod
    def pre_validate(cls, location: str):
        raise RuntimeError("Validate the wrapped model instead")

    def is_valid(self) -> bool:
        return self._inner.is_valid()

    def load(self) -> None:
        self._inner.load()
        # Mirror the inner model's part metadata; callers read these off the
        # outer object.
        self._body_parts = self._inner.body_parts
        self._body_part_categories = self._inner.body_part_categories
        self._body_parts_by_category = getattr(
            self._inner, "_body_parts_by_category", {}
        )
        logger.notice(
            "live inference cropping to ROI x=%d y=%d w=%d h=%d",
            self._roi.x, self._roi.y, self._roi.width, self._roi.height,
        )

    def predict(self, frames) -> typing.List[numpy.ndarray]:
        roi = self._roi
        frames = numpy.asarray(frames)
        rows, columns = frames.shape[1], frames.shape[2]
        if roi.y + roi.height > rows or roi.x + roi.width > columns:
            raise ValueError(
                f"Inference ROI {roi} does not fit frames of shape {frames.shape}"
            )

        view = frames[:, roi.y:roi.y + roi.height, roi.x:roi.x + roi.width, :]
        # Copy into a reused contiguous buffer: the slice is a non-contiguous
        # view, and the inference backend would copy it anyway. Reusing the
        # buffer keeps the hot path free of per-call allocation.
        if self._buffer is None or self._buffer.shape != view.shape:
            self._buffer = numpy.empty(view.shape, dtype=frames.dtype)
        numpy.copyto(self._buffer, view)

        poses = self._inner.predict(self._buffer)

        # Shift back into full-frame coordinates. Column offset applies to x,
        # row offset to y; likelihood is untouched.
        for pose in poses:
            pose[:, 0] += roi.x
            pose[:, 1] += roi.y
        return poses


def maybe_crop(
    model: PoseModel,
    shape: typing.Tuple[int, int],
    roi: typing.Optional[InferenceRoi] = None,
) -> PoseModel:
    """Wrap `model` for ROI cropping when a useful region is configured.

    Returns the model unchanged when no region is set or the region is the whole
    frame, so the default path is exactly what it was before.
    """
    roi = InferenceRoi.from_environment() if roi is None else roi
    if roi is None:
        return model
    roi.validate_for_shape(shape)
    if roi.covers_shape(shape):
        logger.info("configured inference ROI covers the full frame; not cropping")
        return model
    return CroppedPoseModel(model, roi)
