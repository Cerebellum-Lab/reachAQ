from __future__ import annotations

from enum import IntEnum


class CaptureProcessStatus(IntEnum):
    """Valid VideoCaptureProcess states available through the status Value"""

    TERMINATED = -2
    """The capture loop is terminated"""

    FAILED = -1
    """Failed to configure or run process"""

    UNKNOWN = 0
    """Uninitialized value not yet set by capture process"""

    INITIALIZED = 1
    """The process is created, but not started"""

    RUNNING = 2
    """The process is running the capture loop"""

    RECORDING = 3
    """The process is recording the stream to disk"""
