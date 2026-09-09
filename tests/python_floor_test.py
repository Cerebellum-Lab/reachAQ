"""Every package must declare the same Python floor.

DeepLabCut 3.x needs 3.10 and dlclive 1.1.0 needs 3.10-3.12. A package left at
3.8 installs on an interpreter the inference stack cannot use, and the failure
surfaces later as an unrelated import error.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECTS = [
    REPO / "pyproject.toml",
    REPO / "auto-trainer-behavior" / "pyproject.toml",
    REPO / "auto-trainer-core" / "pyproject.toml",
    REPO / "auto-trainer-device" / "pyproject.toml",
    REPO / "auto-trainer-inference" / "pyproject.toml",
    REPO / "auto-trainer-model" / "pyproject.toml",
    REPO / "auto-trainer-pyside" / "pyproject.toml",
    REPO / "auto-trainer-video" / "pyproject.toml",
]


@pytest.mark.parametrize("path", PYPROJECTS, ids=lambda p: p.parent.name)
def test_requires_python_floor_is_3_10(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
    assert match, f"{path} has no requires-python"
    declared = match.group(1)
    assert "3.10" in declared, (
        f"{path} declares {declared!r}; DeepLabCut 3.x needs >= 3.10"
    )
    assert "3.8" not in declared, f"{path} still allows 3.8"


def test_every_package_is_covered():
    """A new package must not silently escape the floor check."""
    found = {REPO / "pyproject.toml"}
    found.update(REPO.glob("auto-trainer-*/pyproject.toml"))
    assert found == set(PYPROJECTS), (
        "pyproject files on disk differ from the checked list: "
        f"{sorted(str(p.relative_to(REPO)) for p in found ^ set(PYPROJECTS))}"
    )
