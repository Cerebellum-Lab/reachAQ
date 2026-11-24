class EnvironmentProvider:
    """
    Provides process-wide opt-ins that are defined outside application state.
    """
    _allow_can_emulation = False

    @staticmethod
    def allow_can_emulation() -> bool:
        return EnvironmentProvider._allow_can_emulation

    @staticmethod
    def enable_can_emulation(enable: bool) -> None:
        EnvironmentProvider._allow_can_emulation = enable
