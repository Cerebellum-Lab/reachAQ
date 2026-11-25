"""Which DeepLabCut engine can be handed a short batch.

The padded predict buffer in core exists because the DeepLabCut TensorFlow
graph fixes its batch dimension at build time. The PyTorch runner batches
internally, so it declares the capability on the PoseModel seam; the pipeline
side of this is tested in core (tests/live_partial_batch_test.py).
"""


def test_a_backend_can_declare_that_it_batches_internally():
    from reachaq.pose_plugins.deeplabcut import DlcTorchPoseModel

    assert DlcTorchPoseModel("/unused").supports_partial_batch is True


def test_the_tensorflow_backend_still_requires_a_full_batch():
    from reachaq.pose_plugins.deeplabcut import DlcPoseModel

    assert DlcPoseModel("/unused").supports_partial_batch is False
