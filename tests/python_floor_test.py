"""Every package must declare the same Python floor.

DeepLabCut 3.x needs 3.10 and dlclive 1.1.0 needs 3.10-3.12. A package left at
3.8 installs on an interpreter the inference stack cannot use, and the failure
surfaces later as an unrelated import error.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Discovered rather than listed. A hand-maintained list already missed
# tools/pyproject.toml, which stayed at ">= 3.8" while every other package moved
# to 3.10; the omission was invisible because the test only checked what it was
# told about. One level deep, so vendored trees are not swept in.
PYPROJECTS = sorted(REPO.glob("*/pyproject.toml")) + [REPO / "pyproject.toml"]


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
    """A package must not silently escape the floor check.

    Naming the packages that must be present, rather than re-deriving them with
    the same glob the list uses, which would assert nothing. The previous
    version globbed only "auto-trainer-*" and so was blind to tools/ in exactly
    the same way the hand-maintained list was.
    """
    covered = set(PYPROJECTS)
    required = {REPO / "pyproject.toml", REPO / "tools" / "pyproject.toml"}
    required.update(REPO.glob("auto-trainer-*/pyproject.toml"))
    missing = required - covered
    assert not missing, (
        "these declare a Python floor but are not checked: "
        f"{sorted(str(path.relative_to(REPO)) for path in missing)}"
    )
