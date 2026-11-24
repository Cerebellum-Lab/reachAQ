"""The SoftMouse HTTPS client used by the SoftMouse animal metadata source."""

from .https_source import (
    SoftMouseCredentials,
    SoftMouseHttpsConfiguration,
    SoftMouseHttpsSource,
)

__all__ = [
    "SoftMouseCredentials",
    "SoftMouseHttpsConfiguration",
    "SoftMouseHttpsSource",
]
