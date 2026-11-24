"""The pose frameworks are extras, installed together by the portable installer.

They stay extras rather than hard dependencies because installing both naively
breaks one: their nvidia-* CUDA wheels share site-packages/nvidia, and on
christielab10 the later install overwrote the earlier. The installer installs
both deliberately, with TensorFlow's CUDA runtime kept apart.

DeepLabCut and the tensorflow extra moved to the reachaq-pose-deeplabcut
plugin, with the conditions recorded beside the extra; its own tests check
them (plugins/deeplabcut/tests/test_dependencies.py). Core keeps the torch
extra, which the built-in keypoint backend needs.
"""

import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


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


def test_deeplabcut_is_not_a_hard_dependency():
    """It ships in the reachaq-pose-deeplabcut plugin, installed on request.

    Requirement strings only: the comment saying where it went names it.
    """
    requirements = re.findall(r"""^\s*['"]([^'"]+)['"]""", _dependencies_block(_text()), re.M)
    assert requirements, "no requirements parsed"
    assert not [r for r in requirements if r.split()[0].lower() == "deeplabcut"]


def test_core_pandas_and_numpy_stay_required():
    block = _dependencies_block(_text())
    for package in ("pandas", "numpy"):
        assert package in block, f"{package} should remain a hard dependency"


def test_core_declares_what_core_imports():
    """scipy and scikit-learn are imported by core code (every
    `import reachaq.inference` reaches scipy; app_model reaches sklearn).
    They used to arrive only through deeplabcut, so a fresh install without
    the plugin lacked them."""
    requirements = re.findall(r"""^\s*['"]([^'"]+)['"]""", _dependencies_block(_text()), re.M)
    names = {re.split(r"[\s<>=!~;\[]", r, maxsplit=1)[0].lower() for r in requirements}
    # tables: pandas to_hdf/read_hdf (pose_result_process, h5_tools) need PyTables.
    for package in ("scipy", "scikit-learn", "tables"):
        assert package in names, f"{package} is imported by core but not a hard dependency"


def test_both_extras_are_declared():
    """Only torch is left: the tensorflow extra moved to the DeepLabCut plugin."""
    text = _text()
    assert _extras_block(text, "torch")
    with pytest.raises(AssertionError, match="no 'tensorflow' extra"):
        _extras_block(text, "tensorflow")


def test_core_has_no_yolo_extra():
    """ultralytics is AGPL-3.0: it lives in the reachaq-pose-yolo plugin,
    not in a core extra."""
    assert "ultralytics" not in _text()
    with pytest.raises(AssertionError, match="no 'yolo' extra"):
        _extras_block(_text(), "yolo")


def test_torch_extra_is_capped_at_the_build_shown_to_share_a_process():
    """2.7.1+cu128 runs beside TensorFlow 2.12; the CUDA 13 build aborted it."""
    extra = _extras_block(_text(), "torch")
    assert re.search(r"torch\s*>=\s*2\.7\s*,\s*<\s*2\.8", extra), (
        "the torch extra must stay below 2.8 until a newer build is checked "
        "against TensorFlow in one process"
    )


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


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "run_unit_tests.yaml"


def test_ci_never_installs_the_yolo_plugin():
    """The YOLO plugin is AGPL-3.0; CI neither installs nor tests it."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "plugins/yolo" not in text
    assert "ultralytics" not in text.lower()


def test_ci_tests_the_deeplabcut_plugin_in_its_own_job():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^  deeplabcut-plugin:", text, re.M), "no deeplabcut-plugin job"
    job = text[text.index("  deeplabcut-plugin:"):]
    assert "pip install -e plugins/deeplabcut" in job
    assert "plugins/deeplabcut[" not in job, "the tensorflow extra is not installed in CI"
    assert "check_licences --allowlist .github/licence-allowlist.txt" in job
    assert "pytest plugins/deeplabcut/tests" in job
