from typing import Protocol


class MeasurementMessageProtocol(Protocol):
    """
    MeasurementMessage defines the expected interface for measurement data provided by the hardware.  This is expected
    object passed as part of a SystemStatusMessageKind.MEASUREMENTS message.
    """

    @property
    def when(self) -> float:
        """Value of time.time() or equivalent absolute time provided by the hardware."""
        pass

    # Deprecated
    @property
    def timestamp(self) -> int:
        """
        A relative time index in nanoseconds. Provides finer resolution between measurements than absolute time may
        provide on some platforms.  By default, would be provided by time.perf_counter_ns().
        """
        pass

    @property
    def index(self) -> int:
        """
        A relative time index in nanoseconds. Provides finer resolution between measurements than absolute time may
        provide on some platforms.  By default, would be provided by time.perf_counter_ns().
        """
        pass

    @property
    def pressure(self) -> float:
        """The force detector value in the range of [0, 1024]"""
        pass

    # Deprecated
    @property
    def switch(self) -> float:
        """The head contact DIO value as 0 or 1."""
        pass

    @property
    def temperature(self) -> float:
        """The temperature in Celsius."""
        pass

    @property
    def humidity(self) -> float:
        """The relative humidity (%)."""
        pass

    @property
    def head_contact(self) -> bool:
        """The head contact DIO value state.  True if contact is made, False otherwise."""
        pass
