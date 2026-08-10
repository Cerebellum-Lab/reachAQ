"""Platform-specific artifact selection kept outside runtime hardware code."""

from .spinnaker import SpinnakerArtifactError, select_spinnaker_wheel

__all__ = ["SpinnakerArtifactError", "select_spinnaker_wheel"]
