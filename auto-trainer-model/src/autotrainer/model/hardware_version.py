import os
from enum import IntEnum
from typing import Optional, Union


class HardwareVersion(IntEnum):
    """
    Allow applications to identify hardware versions without knowing the specific techniques, function calls, or
    properties to make that identification.
    """
    UNKNOWN = 0
    ANSHUTZ = 1
    ALOGUS_V1 = 2

    def __str__(self):
        if self == HardwareVersion.ANSHUTZ:
            return "Anschutz"
        elif self == HardwareVersion.ALOGUS_V1:
            return "Alogus v1"
        else:
            return "Unknown"


_HARDWARE_VERSION_ENV = "AUTOTRAINER_HARDWARE_VERSION"


def normalize_hardware_version(
    value: Union[HardwareVersion, str, int],
) -> HardwareVersion:
    """Convert an explicit hardware profile value to ``HardwareVersion``."""
    if isinstance(value, HardwareVersion):
        return value
    if isinstance(value, int):
        return HardwareVersion(value)

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "anschutz": HardwareVersion.ANSHUTZ,
        "anshutz": HardwareVersion.ANSHUTZ,
        "alogus": HardwareVersion.ALOGUS_V1,
        "alogus_v1": HardwareVersion.ALOGUS_V1,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported hardware version {value!r}; expected one of: {supported}"
        ) from exc


def default_determine_hardware_version() -> HardwareVersion:
    """
    Return the explicitly configured hardware profile, defaulting to Alogus.

    CAN-library import availability is a transport capability and is not a
    reliable hardware-identity signal. In particular, a SocketCAN Alogus
    runtime does not require ``pyjerrycan``.

    :return: The `HardwareVersion` of the device.
    """
    configured: Optional[str] = os.getenv(_HARDWARE_VERSION_ENV)
    if configured is None:
        return HardwareVersion.ALOGUS_V1
    return normalize_hardware_version(configured)
