"""Tests for the DeepLabCut PyTorch backend.

The whole point of this class is that it is drop-in: PoseAlgorithm and
CroppedPoseModel index `pose[:, 0]`, `pose[:, 1]` and `pose[:, 2]` on whatever
predict() returns, so the return shape and column order have to match
DlcPoseModel exactly.  A backend that returned the runner's native
(individuals, bodyparts, 3) block would not fail loudly - it would shift every
coordinate and quietly corrupt triangulation.

The runner shape asserted here is measured, not assumed.  Running DeepLabCut
3.0.1 on the rig against a trained resnet_50 shuffle gave:

    [{'bodyparts': ndarray(1, 14, 3) float32}, ...]   # one dict per image

DeepLabCut itself is not importable in this test environment (and pulls in
torch when it is), so `load()` is exercised against stub modules.  That still
covers the two things most likely to be wrong: resolving the PyTorch model tree
rather than the TensorFlow one, and picking the right snapshot.
"""

import os
import sys
import types

import numpy
import pytest

from autotrainer.inference.dlc import DlcTorchPoseModel


PARTS = ["Pellet", "RH_grab", "Triangle"]


def _project(tmp_path, body_parts=None, snapshot_index=-1, snapshots=("snapshot-200.pt",),
             write_pytorch_config=True, engine_folder="dlc-models-pytorch"):
    """Build a minimal DeepLabCut project tree and return its path."""
    project = tmp_path / "project"
    train = project / engine_folder / "iteration-0" / "shuffle1" / "train"
    train.mkdir(parents=True)
    if write_pytorch_config:
        (train / "pytorch_config.yaml").write_text("net_type: resnet_50\n")
    for name in snapshots:
        (train / name).write_text("weights")
    (project / "config.yaml").write_text("bodyparts: []\n")

    configuration = {
        "bodyparts": list(PARTS) if body_parts is None else body_parts,
        "project_path": str(project),
        "TrainingFraction": [0.95],
        "snapshotindex": snapshot_index,
    }
    return project, configuration, train


def _install_stub_deeplabcut(monkeypatch, configuration, engine_folder="dlc-models-pytorch"):
    """Stub the two DeepLabCut modules load() imports, recording the engine."""
    seen = {}

    engine_module = types.ModuleType("deeplabcut.core.engine")

    class _Engine:
        PYTORCH = "pytorch"
        TENSORFLOW = "tensorflow"

    engine_module.Engine = _Engine

    aux = types.ModuleType("deeplabcut.utils.auxiliaryfunctions")

    def read_config(path):
        seen["config_path"] = path
        return dict(configuration)

    def get_model_folder(train_fraction, shuffle, cfg, engine=None):
        seen["train_fraction"] = train_fraction
        seen["shuffle"] = shuffle
        seen["engine"] = engine
        return os.path.join(engine_folder, "iteration-0", "shuffle1")

    aux.read_config = read_config
    aux.get_model_folder = get_model_folder

    for name, module in (
        ("deeplabcut", types.ModuleType("deeplabcut")),
        ("deeplabcut.core", types.ModuleType("deeplabcut.core")),
        ("deeplabcut.core.engine", engine_module),
        ("deeplabcut.utils", types.ModuleType("deeplabcut.utils")),
        ("deeplabcut.utils.auxiliaryfunctions", aux),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return seen


class _FakeRunner:
    """Returns the runner's real output shape: one dict per image, (1, parts, 3)."""

    def __init__(self, part_count=len(PARTS)):
        self.part_count = part_count
        self.received = None

    def inference(self, images):
        self.received = images
        results = []
        for index, _ in enumerate(images):
            pose = numpy.zeros((1, self.part_count, 3), dtype=numpy.float32)
            for part in range(self.part_count):
                pose[0, part] = (index * 100 + part, index * 100 + part + 0.5, 0.9)
            results.append({"bodyparts": pose})
        return results


def _loaded_model(tmp_path, monkeypatch, runner=None, **kwargs):
    project, configuration, train = _project(tmp_path, **kwargs)
    seen = _install_stub_deeplabcut(monkeypatch, configuration)
    model = DlcTorchPoseModel(str(project), batch_size=2)
    runner = runner if runner is not None else _FakeRunner()
    monkeypatch.setattr(
        DlcTorchPoseModel, "_build_runner",
        lambda self, config_path, snapshot_path: runner,
    )
    monkeypatch.setattr(
        "autotrainer.inference.dlc.dlc_torch_pose_model.detect_gpu_capability",
        lambda: None,
    )
    model.load()
    return model, runner, seen, train


def test_pre_validate_requires_a_project_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        DlcTorchPoseModel.pre_validate(str(tmp_path))


def test_is_valid_matches_the_tensorflow_backend(tmp_path):
    (tmp_path / "config.yaml").write_text("bodyparts: []\n")
    assert DlcTorchPoseModel(str(tmp_path)).is_valid() is True
    assert DlcTorchPoseModel(str(tmp_path / "missing")).is_valid() is False


def test_body_parts_from_a_flat_list_keep_their_order():
    model = DlcTorchPoseModel("unused")
    model._load_body_parts(list(PARTS))
    assert model.body_parts == PARTS
    assert model.body_part_categories == []


def test_body_parts_from_categories_are_grouped_in_declaration_order():
    """PoseAlgorithm gates per category, so the grouping is part of the contract."""
    model = DlcTorchPoseModel("unused")
    model._load_body_parts({"mouse": ["RH_grab"], "rig": ["Pellet", "Triangle"]})
    assert model.body_part_categories == ["mouse", "rig"]
    assert model.body_parts == ["RH_grab", "Pellet", "Triangle"]


def test_unhandled_body_parts_type_is_rejected():
    model = DlcTorchPoseModel("unused")
    with pytest.raises(TypeError):
        model._load_body_parts("RH_grab")


def test_load_resolves_the_pytorch_model_tree(tmp_path, monkeypatch):
    """dlc-models/ and dlc-models-pytorch/ are separate trees.

    Omitting the engine argument silently resolves to the TensorFlow folder,
    where there is no pytorch_config.yaml to find.
    """
    _, _, seen, _ = _loaded_model(tmp_path, monkeypatch)
    assert seen["engine"] == "pytorch"
    assert seen["train_fraction"] == 0.95
    assert seen["shuffle"] == 1


def test_load_reports_the_snapshot_and_defaults_to_cpu_without_a_gpu(tmp_path, monkeypatch):
    model, _, _, train = _loaded_model(tmp_path, monkeypatch)
    assert model.snapshot_path == str(train / "snapshot-200.pt")
    assert model.device == "cpu"
    assert model.precision == "fp32"
    assert model.body_parts == PARTS


def test_load_requires_a_pytorch_config(tmp_path, monkeypatch):
    """A shuffle created for the TensorFlow engine must fail loudly."""
    with pytest.raises(FileNotFoundError, match="PyTorch engine"):
        _loaded_model(tmp_path, monkeypatch, write_pytorch_config=False)


def test_snapshots_are_ordered_numerically_not_lexically(tmp_path, monkeypatch):
    """snapshotindex -1 means latest; "snapshot-9" sorts after "snapshot-100" as text."""
    model, _, _, train = _loaded_model(
        tmp_path, monkeypatch,
        snapshots=("snapshot-9.pt", "snapshot-100.pt", "snapshot-20.pt"),
    )
    assert model.snapshot_path == str(train / "snapshot-100.pt")


def test_best_snapshots_are_excluded_from_the_ordering(tmp_path, monkeypatch):
    """snapshot-best-*.pt has no place in an increasing-progress ordering."""
    model, _, _, train = _loaded_model(
        tmp_path, monkeypatch,
        snapshots=("snapshot-100.pt", "snapshot-best-40.pt"),
    )
    assert model.snapshot_path == str(train / "snapshot-100.pt")


def test_snapshot_index_selects_an_earlier_snapshot(tmp_path, monkeypatch):
    model, _, _, train = _loaded_model(
        tmp_path, monkeypatch,
        snapshots=("snapshot-50.pt", "snapshot-100.pt"),
        snapshot_index=0,
    )
    assert model.snapshot_path == str(train / "snapshot-50.pt")


def test_an_untrained_shuffle_fails_loudly(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="snapshot"):
        _loaded_model(tmp_path, monkeypatch, snapshots=())


def test_predict_before_load_is_an_error(tmp_path):
    (tmp_path / "config.yaml").write_text("bodyparts: []\n")
    with pytest.raises(RuntimeError, match="load"):
        DlcTorchPoseModel(str(tmp_path)).predict(numpy.zeros((1, 8, 8, 3), dtype=numpy.uint8))


def test_predict_returns_one_parts_by_three_array_per_frame(tmp_path, monkeypatch):
    """The individual axis must be dropped; this is the drop-in contract."""
    model, _, _, _ = _loaded_model(tmp_path, monkeypatch)
    poses = model.predict(numpy.zeros((2, 16, 16, 3), dtype=numpy.uint8))
    assert len(poses) == 2
    for pose in poses:
        assert pose.shape == (len(PARTS), 3)


def test_predict_preserves_the_x_y_confidence_column_order(tmp_path, monkeypatch):
    model, _, _, _ = _loaded_model(tmp_path, monkeypatch)
    poses = model.predict(numpy.zeros((2, 16, 16, 3), dtype=numpy.uint8))
    # Frame 1, part 2, as built by the fake runner.
    assert poses[1][2][0] == pytest.approx(102.0)
    assert poses[1][2][1] == pytest.approx(102.5)
    assert poses[1][2][2] == pytest.approx(0.9)


def test_predict_hands_the_runner_individual_frames(tmp_path, monkeypatch):
    """inference() takes a sequence of frames, not a stacked array."""
    model, runner, _, _ = _loaded_model(tmp_path, monkeypatch)
    frames = numpy.zeros((2, 16, 16, 3), dtype=numpy.uint8)
    model.predict(frames)
    assert isinstance(runner.received, list)
    assert len(runner.received) == 2
    assert runner.received[0].shape == (16, 16, 3)


def test_predict_survives_a_single_body_part_project(tmp_path, monkeypatch):
    """squeeze() would collapse the body part axis here; reshape must not."""
    model, _, _, _ = _loaded_model(
        tmp_path, monkeypatch,
        runner=_FakeRunner(part_count=1),
        body_parts=["Pellet"],
    )
    poses = model.predict(numpy.zeros((1, 16, 16, 3), dtype=numpy.uint8))
    assert poses[0].shape == (1, 3)


def test_predict_returns_float_arrays(tmp_path, monkeypatch):
    """The runner emits float32; downstream pandas assembly expects float."""
    model, _, _, _ = _loaded_model(tmp_path, monkeypatch)
    poses = model.predict(numpy.zeros((1, 16, 16, 3), dtype=numpy.uint8))
    assert poses[0].dtype == numpy.dtype("float64")


def test_requested_device_and_precision_win_over_detection(tmp_path, monkeypatch):
    model, _, _, _ = _loaded_model(tmp_path, monkeypatch)
    assert model.device == "cpu"

    project, configuration, _ = _project(tmp_path / "explicit")
    _install_stub_deeplabcut(monkeypatch, configuration)
    monkeypatch.setattr(
        DlcTorchPoseModel, "_build_runner",
        lambda self, config_path, snapshot_path: _FakeRunner(),
    )
    explicit = DlcTorchPoseModel(str(project), device="cuda:1", precision="fp16")
    explicit.load()
    assert explicit.device == "cuda:1"
    assert explicit.precision == "fp16"
