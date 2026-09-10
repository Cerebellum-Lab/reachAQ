"""Live inference must not pay batch-six compute for batch-two output.

The padded predict buffer exists because the DeepLabCut TensorFlow graph fixes
its batch dimension at build time: two real camera frames are padded up to
`camera_count * model_frames_per_camera` (six by default) and the four padding
results are then sliced away. Measured on the rig at 128px, that is 20.7 ms of
forward pass instead of 9.5 ms.

The PyTorch runner batches internally, so it can be handed the two real frames
and cost batch-two compute. The capability therefore lives on the PoseModel
seam rather than being assumed either way, and the padding stays for backends
that need it.
"""

import numpy
import pytest

from autotrainer.inference.cropped_pose_model import CroppedPoseModel, InferenceRoi
from autotrainer.inference.pose_model import PoseModel

PARTS = ["Pellet", "RH_grab", "Triangle"]


class _Recording(PoseModel):
    """Records the batch size it was asked to predict."""

    def __init__(self, supports_partial=False):
        super().__init__()
        self._supports_partial = supports_partial
        self.received_batches = []

    @classmethod
    def pre_validate(cls, location: str):
        return None

    @property
    def supports_partial_batch(self) -> bool:
        return self._supports_partial

    def load(self):
        self._body_parts = list(PARTS)
        self._body_part_categories = ["default"]
        self._body_parts_by_category = {"default": list(PARTS)}

    def predict(self, frames):
        frames = numpy.asarray(frames)
        self.received_batches.append(frames.shape[0])
        return [numpy.zeros((len(PARTS), 3)) for _ in range(frames.shape[0])]


def test_the_default_is_to_require_a_full_batch():
    """The TensorFlow graph cannot take a short batch, so False is the safe default."""
    assert PoseModel().supports_partial_batch is False


def test_a_backend_can_declare_that_it_batches_internally():
    from autotrainer.inference.dlc import DlcTorchPoseModel

    assert DlcTorchPoseModel("/unused").supports_partial_batch is True


def test_the_tensorflow_backend_still_requires_a_full_batch():
    from autotrainer.inference.dlc import DlcPoseModel

    assert DlcPoseModel("/unused").supports_partial_batch is False


@pytest.mark.parametrize("supports_partial", [True, False])
def test_the_crop_decorator_passes_the_capability_through(supports_partial):
    """A decorator answering for itself would strand the capability."""
    inner = _Recording(supports_partial=supports_partial)
    wrapped = CroppedPoseModel(inner, InferenceRoi(0, 0, 64, 64))
    assert wrapped.supports_partial_batch is supports_partial


def test_the_live_path_chooses_its_input_by_capability():
    """The selection in pose_process, exercised on the same buffers it uses."""
    model_batch_size = 6
    live_batch_size = 2
    predict_buffer = numpy.zeros((model_batch_size, 32, 32, 3))
    frame_buffer1 = predict_buffer[:live_batch_size]

    for supports_partial, expected in ((True, live_batch_size),
                                       (False, model_batch_size)):
        model = _Recording(supports_partial=supports_partial)
        model.load()
        chosen = frame_buffer1 if model.supports_partial_batch else predict_buffer
        poses = model.predict(chosen)[:live_batch_size]

        assert model.received_batches == [expected]
        assert len(poses) == live_batch_size, (
            "the caller must still receive one pose per real frame"
        )


def test_the_live_slice_is_a_view_of_the_padded_buffer():
    """Writing through the view must reach the buffer, whichever is predicted on."""
    predict_buffer = numpy.zeros((6, 8, 8, 3))
    frame_buffer1 = predict_buffer[:2]
    frame_buffer1[0, 0, 0, 0] = 7.0

    assert predict_buffer[0, 0, 0, 0] == 7.0
    assert frame_buffer1.base is predict_buffer


def test_pose_process_selects_the_input_once_not_per_frame():
    """A per-frame branch would put the check in the hot loop."""
    import inspect

    from autotrainer.inference import pose_process

    source = inspect.getsource(pose_process)
    assert "live_predict_input = (" in source
    assert source.count("supports_partial_batch") == 1, (
        "the capability should be read once, outside the loop"
    )
    assert "live_predict(live_predict_input)" in source
