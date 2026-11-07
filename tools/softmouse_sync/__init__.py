"""Browserless SoftMouse export publication job."""

import sys
from pathlib import Path


# Allow the documented ``python -m tools.softmouse_sync.cli`` command to run
# directly from a source checkout, even before the monorepo packages have been
# installed in editable mode.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _component in (
    "auto-trainer-behavior",
    "auto-trainer-core",
    "auto-trainer-device",
    "auto-trainer-inference",
    "auto-trainer-model",
    "auto-trainer-pyside",
    "auto-trainer-video",
):
    _source = str(_REPOSITORY_ROOT / _component / "src")
    if _source not in sys.path:
        sys.path.insert(0, _source)

from .https_source import (
    SoftMouseCredentials,
    SoftMouseHttpsConfiguration,
    SoftMouseHttpsSource,
)
from .publisher import PublicationResult, SoftMouseExportPublisher

__all__ = [
    "PublicationResult",
    "SoftMouseCredentials",
    "SoftMouseExportPublisher",
    "SoftMouseHttpsConfiguration",
    "SoftMouseHttpsSource",
]
