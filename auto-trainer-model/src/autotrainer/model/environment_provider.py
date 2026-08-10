import importlib.util

_spec_api = importlib.util.find_spec("autotrainer.api")


class EnvironmentProvider:
    """
    Provides process-wide opt-ins that are defined outside application state.
    """
    _allow_can_emulation = False

    _external_api_available = _spec_api is not None

    @staticmethod
    def allow_can_emulation() -> bool:
        return EnvironmentProvider._allow_can_emulation

    @staticmethod
    def enable_can_emulation(enable: bool) -> None:
        EnvironmentProvider._allow_can_emulation = enable

    @staticmethod
    def is_external_api_available() -> bool:
        return EnvironmentProvider._external_api_available
