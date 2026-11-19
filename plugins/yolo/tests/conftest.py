"""Fixtures for the YOLO plugin tests.

The repository root conftest.py (rootdir, via pytest.ini) already puts the
sub-project sources on sys.path; these tests need nothing else from it or from
tests/conftest.py. The plugin itself must be installed so that its entry point
is visible to discovery.
"""

import pytest

from autotrainer.inference.pose_plugins import reset_discovery


@pytest.fixture(autouse=True)
def _fresh_discovery():
    """Discovery is cached per process; start and end every test without it."""
    reset_discovery()
    yield
    reset_discovery()
