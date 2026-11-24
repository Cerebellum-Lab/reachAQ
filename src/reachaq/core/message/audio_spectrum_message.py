from typing import Protocol, List


class AudioSpectrumMessage(Protocol):

    @property
    def when(self) -> float:
        """Value of time.time() or equivalent absolute time provided by the hardware."""

    @property
    def index(self) -> int:
        """
        A relative time index in nanoseconds. Provides finer resolution between measurements than absolute time may
        provide on some platforms.  By default, would be provided by time.perf_counter_ns().
        """

    @property
    def magnitudes(self) -> List[float]:
        """The audio spectrum magnitudes in dB."""
