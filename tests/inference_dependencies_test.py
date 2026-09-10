"""TensorFlow must be optional, never a hard requirement.

The two frameworks cannot share one environment: nvidia-cudnn-cu11 and
nvidia-cudnn-cu12 both install libcudnn.so.8, so whichever lands last wins and
the other framework's GPU path breaks. On the reachAQ rig that is exactly what
happened - the cu11 build won, TensorFlow works, and every torch convolution
aborts. Extras let a deployment install exactly one.

DeepLabCut 3.x still ships both engines, verified against the installed 3.0.1,
so raising the floor does not strand the TensorFlow path.
"""

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "auto-trainer-inference" / "pyproject.toml"


def _text():
    return PYPROJECT.read_text(encoding="utf-8")


def _dependencies_block(text):
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert match, "no dependencies block found"
    return match.group(1)


def _extras_block(text, name):
    match = re.search(
        r"^\[project\.optional-dependencies\](.*?)(?=^\[|\Z)", text, re.S | re.M
    )
    assert match, "no [project.optional-dependencies] table found"
    table = match.group(1)
    entry = re.search(rf"^{re.escape(name)}\s*=\s*\[(.*?)^\]", table, re.S | re.M)
    assert entry, f"no {name!r} extra found"
    return entry.group(1)


def test_neither_framework_is_a_hard_dependency():
    block = _dependencies_block(_text())
    for package in ("tensorflow", "tensorpack", "tf-slim", "torch"):
        assert package not in block, (
            f"{package} is still a hard dependency; it must move to an extra so "
            f"the cu11/cu12 cuDNN collision cannot be installed by accident"
        )


def test_deeplabcut_is_still_required_at_3_x():
    block = _dependencies_block(_text())
    match = re.search(r"deeplabcut\s*>=\s*([0-9.]+)", block)
    assert match, "deeplabcut is not a hard dependency"
    major = int(match.group(1).split(".")[0])
    assert major >= 3, f"deeplabcut floor is {match.group(1)}; the PyTorch engine needs 3.x"


def test_core_pandas_and_numpy_stay_required():
    block = _dependencies_block(_text())
    for package in ("auto-trainer-core", "pandas", "numpy"):
        assert package in block, f"{package} should remain a hard dependency"


def test_both_extras_are_declared():
    text = _text()
    assert _extras_block(text, "tensorflow")
    assert _extras_block(text, "torch")


def test_tensorflow_extra_excludes_2_13():
    """2.13 pairs with a keras that moved legacy_tf_layers.

    That breaks tf_slim's batch_norm and therefore every DeepLabCut model load.
    Measured on the rig: ModuleNotFoundError: No module named
    'keras.legacy_tf_layers'.
    """
    extra = _extras_block(_text(), "tensorflow")
    assert re.search(r"tensorflow\s*[><=!,\s0-9.]*<\s*2\.13", extra), (
        "the tensorflow extra must cap below 2.13"
    )


def test_tensorflow_extra_carries_its_helper_packages():
    extra = _extras_block(_text(), "tensorflow")
    for package in ("tensorpack", "tf-slim"):
        assert package in extra, f"{package} is required by the DeepLabCut TF engine"


def test_torch_extra_covers_the_whole_gpu_fleet():
    """torch >= 2.7 on a CUDA 12.8+ build covers sm_75 through sm_120."""
    extra = _extras_block(_text(), "torch")
    match = re.search(r"torch\s*>=\s*([0-9.]+)", extra)
    assert match, "the torch extra must declare a floor"
    major, minor = (int(part) for part in match.group(1).split(".")[:2])
    assert (major, minor) >= (2, 7), (
        f"torch floor {match.group(1)} predates sm_120 support; the RTX 5060 Ti "
        f"needs >= 2.7 with CUDA 12.8+"
    )
