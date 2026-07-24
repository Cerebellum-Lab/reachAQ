import importlib.util
from typing import Optional, Union

from .hardware_version import (
    HardwareVersion,
    default_determine_hardware_version,
    normalize_hardware_version,
)

_spec_api = importlib.util.find_spec("autotrainer.api")


class EnvironmentProvider:
    """
    EnvironmentProvider provides access to static information that is generally defined outside any application
    implementation or choices.  A simple example is the style or version of the physical device that is being used.

    In addition, much of this information is typically needed across multiple, independent components of an
    application and would otherwise require repeatedly passing some instance of the information down a large
    hierarchy or similar.
    """
    _hardware_version_override: Optional[HardwareVersion] = None

    _allow_can_emulation = False

    _external_api_available = _spec_api is not None

    @staticmethod
    def hardware_version() -> HardwareVersion:
        override = EnvironmentProvider._hardware_version_override
        if override is not None:
            return override
        return default_determine_hardware_version()

    @staticmethod
    def set_hardware_version(
        value: Optional[Union[HardwareVersion, str, int]],
    ) -> None:
        """Set an explicit runtime profile, or clear it with ``None``."""
        EnvironmentProvider._hardware_version_override = (
            None if value is None else normalize_hardware_version(value)
        )

    @staticmethod
    def allow_can_emulation() -> bool:
        return EnvironmentProvider._allow_can_emulation

    @staticmethod
    def enable_can_emulation(enable: bool) -> None:
        EnvironmentProvider._allow_can_emulation = enable

    @staticmethod
    def is_external_api_available() -> bool:
        return EnvironmentProvider._external_api_available
