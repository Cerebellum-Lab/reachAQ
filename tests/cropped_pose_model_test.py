"""Tests for ROI-cropped live inference.

The behaviour that matters is that coordinates come back in full-frame space.
Stereo calibration and triangulation are defined in full-frame pixels, so an
un-shifted crop coordinate would silently corrupt every 3D result rather than
fail loudly.
"""

import numpy
import pytest

from autotrainer.inference.cropped_pose_model import (
    INFERENCE_ROI_ENV_VAR,
    CroppedPoseModel,
    InferenceRoi,
    maybe_crop,
)
from autotrainer.inference.pose_model import PoseModel


PARTS = ["Pellet", "RH_grab", "Triangle"]


class _RecordingModel(PoseModel):
    """Reports the centre of whatever frame it is given, and records the input."""

    def __init__(self):
        super().__init__()
        self.received = None
        self.call_count = 0

    @classmethod
    def pre_validate(cls, location: str):
        return None

    def load(self):
        self._body_parts = list(PARTS)
        self._body_part_categories = ["default"]
        self._body_parts_by_category = {"default": list(PARTS)}

    def predict(self, frames):
        frames = numpy.asarray(frames)
        self.received = frames.copy()
        self.call_count += 1
        rows, columns = frames.shape[1], frames.shape[2]
        centre_x, centre_y = columns / 2.0, rows / 2.0
        return [
            numpy.array(
                [[centre_x, centre_y, 0.99]] * len(PARTS), dtype=float
            )
            for _ in range(frames.shape[0])
        ]


def _frames(batch=2, rows=256, columns=256):
    # Distinct value per pixel so a wrong crop is detectable.
    base = numpy.arange(rows * columns, dtype=float).reshape(rows, columns)
    stack = numpy.empty((batch, rows, columns, 3), dtype=float)
    for index in range(batch):
        for plane in range(3):
            stack[index, :, :, plane] = base + index * 1000000 + plane
    return stack


def test_roi_parsing_and_validation():
    roi = InferenceRoi.parse(" 10, 20 ,64,48 ")
    assert (roi.x, roi.y, roi.width, roi.height) == (10, 20, 64, 48)
    assert InferenceRoi.parse("") is None
    assert InferenceRoi.parse(None) is None
    with pytest.raises(ValueError):
        InferenceRoi.parse("1,2,3")
    with pytest.raises(ValueError):
        InferenceRoi.parse("a,b,c,d")
    with pytest.raises(ValueError):
        InferenceRoi(0, 0, 0, 10)
    with pytest.raises(ValueError):
        InferenceRoi(-1, 0, 10, 10)


def test_roi_must_fit_the_frame():
    roi = InferenceRoi(200, 200, 128, 128)
    with pytest.raises(ValueError):
        roi.validate_for_shape((256, 256))
    InferenceRoi(64, 64, 128, 128).validate_for_shape((256, 256))


def test_crop_selects_the_requested_region():
    inner = _RecordingModel()
    inner.load()
    roi = InferenceRoi(x=32, y=16, width=64, height=48)
    model = CroppedPoseModel(inner, roi)
    model.load()

    frames = _frames()
    model.predict(frames)

    assert inner.received.shape == (2, 48, 64, 3)
    # Rows are y, columns are x.
    expected = frames[:, 16:16 + 48, 32:32 + 64, :]
    numpy.testing.assert_array_equal(inner.received, expected)


def test_coordinates_are_returned_in_full_frame_space():
    inner = _RecordingModel()
    inner.load()
    roi = InferenceRoi(x=32, y=16, width=64, height=48)
    model = CroppedPoseModel(inner, roi)
    model.load()

    poses = model.predict(_frames())

    # The stub reports the crop centre: (64/2, 48/2) = (32, 24) in crop space,
    # which is (32+32, 16+24) = (64, 40) in full-frame space.
    for pose in poses:
        assert pose[:, 0] == pytest.approx(64.0)
        assert pose[:, 1] == pytest.approx(40.0)
        # Likelihood must pass through untouched: the 3D path gates on it.
        assert pose[:, 2] == pytest.approx(0.99)


def test_zero_offset_roi_still_returns_crop_relative_centre():
    inner = _RecordingModel()
    inner.load()
    model = CroppedPoseModel(inner, InferenceRoi(0, 0, 128, 128))
    model.load()

    poses = model.predict(_frames())
    for pose in poses:
        assert pose[:, 0] == pytest.approx(64.0)
        assert pose[:, 1] == pytest.approx(64.0)


def test_body_part_metadata_is_mirrored_from_the_inner_model():
    inner = _RecordingModel()
    model = CroppedPoseModel(inner, InferenceRoi(0, 0, 64, 64))
    model.load()
    assert model.body_parts == PARTS
    assert model.body_part_categories == ["default"]


def test_buffer_is_reused_across_calls():
    inner = _RecordingModel()
    inner.load()
    model = CroppedPoseModel(inner, InferenceRoi(8, 8, 32, 32))
    model.load()

    frames = _frames()
    model.predict(frames)
    first = model._buffer
    model.predict(frames)
    # Same object: no per-call allocation in the hot path.
    assert model._buffer is first
    assert inner.call_count == 2


def test_predict_rejects_a_roi_larger_than_the_frames():
    inner = _RecordingModel()
    inner.load()
    model = CroppedPoseModel(inner, InferenceRoi(200, 200, 128, 128))
    model.load()
    with pytest.raises(ValueError):
        model.predict(_frames())


def test_input_frames_are_not_modified():
    inner = _RecordingModel()
    inner.load()
    model = CroppedPoseModel(inner, InferenceRoi(4, 4, 16, 16))
    model.load()
    frames = _frames()
    original = frames.copy()
    model.predict(frames)
    numpy.testing.assert_array_equal(frames, original)


def test_maybe_crop_passes_through_when_unset():
    inner = _RecordingModel()
    assert maybe_crop(inner, (256, 256), roi=None) is inner


def test_maybe_crop_passes_through_for_a_full_frame_roi():
    inner = _RecordingModel()
    result = maybe_crop(inner, (256, 256), roi=InferenceRoi(0, 0, 256, 256))
    assert result is inner


def test_maybe_crop_wraps_for_a_real_region():
    inner = _RecordingModel()
    result = maybe_crop(inner, (256, 256), roi=InferenceRoi(16, 16, 128, 128))
    assert isinstance(result, CroppedPoseModel)
    assert result.inner is inner


def test_maybe_crop_rejects_a_region_outside_the_frame():
    with pytest.raises(ValueError):
        maybe_crop(_RecordingModel(), (256, 256), roi=InferenceRoi(200, 0, 128, 64))


def test_roi_from_environment(monkeypatch):
    monkeypatch.setenv(INFERENCE_ROI_ENV_VAR, "5,6,32,16")
    roi = InferenceRoi.from_environment()
    assert (roi.x, roi.y, roi.width, roi.height) == (5, 6, 32, 16)
    monkeypatch.delenv(INFERENCE_ROI_ENV_VAR)
    assert InferenceRoi.from_environment() is None


def test_cropping_reduces_the_pixels_the_model_sees():
    """The whole point: fewer pixels through the network."""
    inner = _RecordingModel()
    inner.load()
    model = CroppedPoseModel(inner, InferenceRoi(64, 64, 128, 128))
    model.load()
    frames = _frames()
    model.predict(frames)
    full = frames.shape[1] * frames.shape[2]
    cropped = inner.received.shape[1] * inner.received.shape[2]
    assert cropped * 4 == full
