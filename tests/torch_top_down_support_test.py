"""Top-down PyTorch models must run in the live path, and not as their detector.

DeepLabCut's `rtmpose_*` nets are top-down: a detector proposes a box, and the
pose model estimates keypoints inside a crop of it. Two things follow that the
bottom-up backbones never exercised.

A top-down train folder holds `snapshot-detector-<epoch>.pt` beside the pose
`snapshot-<epoch>.pt`. `_find_snapshot` excluded only the `snapshot-best-*`
files, so a detector snapshot matched the pattern, and `epoch_of` parsed its
trailing number - a detector trained for 250 epochs sorted above a pose model
trained for 200, and the default `snapshotindex` of -1 would have loaded the
object detector as the pose model. It would not have failed cleanly: the load
raises, but only after the process spawn, and the name in the message is a
snapshot the operator did ask for.

The pose model also needs a bounding box. The live frames are already the ROI
the camera is cropped to and hold one animal, so a full-frame box is the
identity crop and the detector is not run: it would add its own latency and its
own failure modes to a closed loop that already has the animal framed.
"""

import numpy
import pytest

from autotrainer.inference.dlc import DlcTorchPoseModel


def _train_folder(tmp_path, names):
    train = (tmp_path / "dlc-models-pytorch" / "iteration-2"
             / "projectSep18-trainset95shuffle1" / "train")
    train.mkdir(parents=True)
    for name in names:
        (train / name).write_text("weights")
    return train


def _model(tmp_path, snapshot_index=-1):
    model = DlcTorchPoseModel(str(tmp_path), shuffle_index=1)
    model._sys_configuration = {DlcTorchPoseModel.SNAPSHOT_INDEX_KEY: snapshot_index}
    return model


def test_a_detector_snapshot_is_not_mistaken_for_the_pose_model(tmp_path):
    """snapshot-detector-250 outranks snapshot-200 on the trailing number."""
    train = _train_folder(tmp_path, [
        "snapshot-100.pt", "snapshot-200.pt",
        "snapshot-detector-225.pt", "snapshot-detector-250.pt",
    ])
    chosen = _model(tmp_path)._find_snapshot(str(train))
    assert chosen.endswith("snapshot-200.pt")


def test_a_best_detector_snapshot_is_also_excluded(tmp_path):
    train = _train_folder(tmp_path, [
        "snapshot-200.pt", "snapshot-detector-best-220.pt",
    ])
    assert _model(tmp_path)._find_snapshot(str(train)).endswith("snapshot-200.pt")


def test_a_folder_holding_only_a_detector_is_refused(tmp_path):
    """Better to refuse than to load a detector and report keypoints."""
    train = _train_folder(tmp_path, ["snapshot-detector-250.pt"])
    with pytest.raises(FileNotFoundError):
        _model(tmp_path)._find_snapshot(str(train))


def test_the_bottom_up_ordering_is_unchanged(tmp_path):
    train = _train_folder(tmp_path, [
        "snapshot-100.pt", "snapshot-150.pt", "snapshot-200.pt",
        "snapshot-best-090.pt",
    ])
    model = _model(tmp_path)
    assert model._find_snapshot(str(train)).endswith("snapshot-200.pt")
    model._sys_configuration = {DlcTorchPoseModel.SNAPSHOT_INDEX_KEY: 0}
    assert model._find_snapshot(str(train)).endswith("snapshot-100.pt")


class _RecordingRunner:
    """Captures what predict() hands the DeepLabCut runner."""

    def __init__(self, parts):
        self.parts = parts
        self.received = None

    def inference(self, images):
        self.received = list(images)
        return [{"bodyparts": numpy.zeros((1, self.parts, 3))}
                for _ in self.received]


def _loaded(tmp_path, method):
    model = DlcTorchPoseModel(str(tmp_path), shuffle_index=1)
    model._body_parts_count = 3
    model._fast_path_enabled = False
    model._model_configuration = {"method": method}
    model._top_down = method == "td"
    model._runner = _RecordingRunner(3)
    return model


def test_a_top_down_model_is_given_a_full_frame_box(tmp_path):
    model = _loaded(tmp_path, "td")
    frames = numpy.zeros((2, 256, 192, 3), dtype=numpy.uint8)
    model.predict(frames)

    assert len(model._runner.received) == 2
    for entry in model._runner.received:
        assert isinstance(entry, tuple), "top-down needs (frame, context)"
        _frame, context = entry
        # x, y, width, height in the frame's own pixels.
        assert context["bboxes"].tolist() == [[0, 0, 192, 256]]


def test_a_bottom_up_model_is_given_plain_frames(tmp_path):
    model = _loaded(tmp_path, "bu")
    model.predict(numpy.zeros((2, 256, 256, 3), dtype=numpy.uint8))
    for entry in model._runner.received:
        assert not isinstance(entry, tuple)


def test_both_shapes_decode_to_one_pose_per_frame(tmp_path):
    for method in ("bu", "td"):
        model = _loaded(tmp_path, method)
        poses = model.predict(numpy.zeros((2, 256, 256, 3), dtype=numpy.uint8))
        assert len(poses) == 2
        assert all(pose.shape == (3, 3) for pose in poses)


def test_the_lean_path_refuses_a_top_down_model():
    """top_down_crop changes geometry, which the lean path does not reproduce."""
    model = DlcTorchPoseModel("/unused")
    model._runner = object()
    model._device = "cuda:0"
    model._model_configuration = {
        "method": "td",
        "data": {"inference": {"normalize_images": True,
                               "top_down_crop": {"width": 256, "height": 256}}},
        "model": {"heads": {"bodypart": {}}},
    }
    assert model._fast_path_reason() == "top_down_crop is configured"
