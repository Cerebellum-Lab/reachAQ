"""Selecting the PyTorch backend on an untrained project must be refused early.

Until a PyTorch shuffle is trained, `REACHAQ_POSE_BACKEND=torch` on a project
that has only ever been trained under TensorFlow cannot work. It already failed
loudly - `_find_snapshot` raises rather than returning poses from an untrained
network - but it failed inside the spawned pose process, after
`can_start_live_inference` had already passed. That reads as a crash rather
than as a refusal, and it costs a process spawn and a model load to find out.

`is_valid()` is what the preflight calls, so the check belongs there. It is
filesystem-only on purpose: the parent process must not import DeepLabCut or
torch merely to discover that it cannot run.
"""

import pytest

from autotrainer.inference.dlc import DlcTorchPoseModel


def _project(tmp_path, *, pytorch_snapshot=False, tensorflow_only=False):
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yaml").write_text("bodyparts: []\n")

    if tensorflow_only:
        train = project / "dlc-models" / "iteration-0" / "shuffle1" / "train"
        train.mkdir(parents=True)
        (train / "snapshot-200.index").write_text("tf")

    if pytorch_snapshot:
        train = (project / "dlc-models-pytorch" / "iteration-2"
                 / "mouseGYMSep18-trainset95shuffle2" / "train")
        train.mkdir(parents=True)
        (train / "pytorch_config.yaml").write_text("net_type: resnet_50\n")
        (train / "snapshot-200.pt").write_text("weights")

    return project


def test_a_project_with_a_trained_pytorch_snapshot_is_valid(tmp_path):
    project = _project(tmp_path, pytorch_snapshot=True)
    assert DlcTorchPoseModel(str(project)).is_valid() is True


def test_a_tensorflow_only_project_is_refused(tmp_path):
    """The exact case that would otherwise crash the pose process."""
    project = _project(tmp_path, tensorflow_only=True)
    assert DlcTorchPoseModel(str(project)).is_valid() is False


def test_a_project_with_no_models_at_all_is_refused(tmp_path):
    project = _project(tmp_path)
    assert DlcTorchPoseModel(str(project)).is_valid() is False


def test_an_untrained_pytorch_shuffle_is_refused(tmp_path):
    """A created-but-never-trained shuffle has a config and no weights."""
    project = _project(tmp_path)
    train = (project / "dlc-models-pytorch" / "iteration-2"
             / "mouseGYMSep18-trainset95shuffle2" / "train")
    train.mkdir(parents=True)
    (train / "pytorch_config.yaml").write_text("net_type: resnet_50\n")

    assert DlcTorchPoseModel(str(project)).is_valid() is False


def test_a_best_only_snapshot_still_counts_as_trained(tmp_path):
    """snapshot-best-*.pt is excluded from ordering, but it is still weights.

    Refusing to start when the only artefact is a best snapshot would be a
    false negative; load() decides which snapshot to use.
    """
    project = _project(tmp_path)
    train = (project / "dlc-models-pytorch" / "iteration-2"
             / "mouseGYMSep18-trainset95shuffle2" / "train")
    train.mkdir(parents=True)
    (train / "snapshot-best-050.pt").write_text("weights")

    assert DlcTorchPoseModel.has_trained_snapshot(str(project)) is True


def test_a_missing_project_is_still_refused(tmp_path):
    assert DlcTorchPoseModel(str(tmp_path / "nope")).is_valid() is False


def test_the_check_does_not_import_deeplabcut_or_torch(tmp_path):
    """The parent process must not pay a framework import to be told no."""
    import subprocess
    import sys
    import pathlib

    project = _project(tmp_path, pytorch_snapshot=True)
    repo = pathlib.Path(__file__).resolve().parents[1]
    script = (
        "import sys\n"
        f"sys.path[:0] = [r'{repo / 'auto-trainer-core' / 'src'}', "
        f"r'{repo / 'auto-trainer-inference' / 'src'}']\n"
        "from autotrainer.inference.dlc import DlcTorchPoseModel\n"
        f"assert DlcTorchPoseModel.has_trained_snapshot(r'{project}')\n"
        "leaked = [n for n in ('deeplabcut', 'torch') if n in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"imported: {completed.stdout.strip()}"


def test_the_tensorflow_backend_is_unaffected(tmp_path):
    """A TensorFlow project must keep validating as it always did."""
    from autotrainer.inference.dlc import DlcPoseModel

    project = _project(tmp_path, tensorflow_only=True)
    assert DlcPoseModel(str(project)).is_valid() is True
