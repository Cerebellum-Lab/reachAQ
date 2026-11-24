"""What the DeepLabCut plugin distribution declares.

DeepLabCut, and the TensorFlow extra that only its TensorFlow engine needs,
left reachaq with this plugin. The pins and their reasons came
with them; see the comments beside the extra in this plugin's pyproject.toml.

DeepLabCut 3.x still ships both engines, verified against the installed 3.0.2,
so the 3.x floor does not strand the TensorFlow path.
"""

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _text():
    return PYPROJECT.read_text(encoding="utf-8")


def _dependencies_block(text):
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert match, "no dependencies block found"
    return match.group(1)


def _requirements(block):
    """The quoted requirement strings of a block, leaving its comments out."""
    return re.findall(r"""^\s*['"]([^'"]+)['"]""", block, re.M)


def _extras_block(text, name):
    match = re.search(
        r"^\[project\.optional-dependencies\](.*?)(?=^\[|\Z)", text, re.S | re.M
    )
    assert match, "no [project.optional-dependencies] table found"
    table = match.group(1)
    entry = re.search(rf"^{re.escape(name)}\s*=\s*\[(.*?)^\]", table, re.S | re.M)
    assert entry, f"no {name!r} extra found"
    return entry.group(1)


def test_the_distribution_is_named_for_the_install_message():
    """PoseBackendNotInstalled tells the operator to install this name."""
    assert re.search(r'^name\s*=\s*"reachaq-pose-deeplabcut"', _text(), re.M)


def test_it_depends_on_the_installed_application_not_a_sub_project():
    """reachaq is folded into reachaq and never installed alone."""
    requirements = _requirements(_dependencies_block(_text()))
    assert "reachaq" in requirements
    assert not [r for r in requirements if r.startswith("reachaq-")]


def test_deeplabcut_is_required_at_3_x():
    block = _dependencies_block(_text())
    match = re.search(r"deeplabcut\s*>=\s*([0-9.]+)\s*,\s*<\s*4", block)
    assert match, "deeplabcut >= 3, < 4 is not a dependency of the plugin"
    major = int(match.group(1).split(".")[0])
    assert major >= 3, f"deeplabcut floor is {match.group(1)}; the PyTorch engine needs 3.x"


def test_neither_framework_is_a_hard_dependency():
    block = _dependencies_block(_text())
    for package in ("tensorflow", "tensorpack", "tf-slim"):
        assert package not in block, f"{package} must stay in the tensorflow extra"


def test_the_entry_point_names_the_plugin_instance():
    text = _text()
    assert re.search(r'^\[project\.entry-points\."reachaq\.pose_backends"\]', text, re.M)
    assert re.search(
        r'^deeplabcut\s*=\s*"reachaq\.pose_plugins\.deeplabcut\.plugin:DLC_PLUGIN"',
        text, re.M)


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


def test_tensorflow_extra_carries_imgaug():
    """DeepLabCut 3's TensorFlow engine imports imgaug without declaring it."""
    assert "imgaug" in _extras_block(_text(), "tensorflow")
