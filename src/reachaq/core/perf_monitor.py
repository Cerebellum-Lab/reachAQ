import logging
import time
from dataclasses import dataclass, field
from math import floor
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PerfMonitor:
    """
    Convenience class to measure and report performance metrics.  There are likely more robust or feature complete
    packages available.  This is intended as a simple implementation to not add to the dependency list of the project.

    The monitor tracks `cycles` or number of times something happens.  What this means is defined by the caller.
    """
    name: str = ""
    """User-friendly name for the performance monitor."""
    units: str = ""
    """Units for the performance monitor."""
    report_window: int = 15
    """Duration between each report"""
    enable_log: bool = True
    """Allow toggling reporting while running."""
    log_level: int = logging.DEBUG
    """Log level for reporting performance metrics."""
    cps: float = 0.0
    """Most recent cycles per second (CPS) value."""

    _cycle_count: Optional[int] = field(default=None, init=False, repr=False, compare=False)
    _start: float = field(default=0, init=False, repr=False, compare=False)
    _next_refresh: float = field(default=0, init=False, repr=False, compare=False)

    def reset(self):
        """
        Resets all cycle counts and timing.  Calculations will resume on the next `add_cycle` or `add_cycles` call.
        """
        logger.debug("%s: reset", self.name)
        self._cycle_count = None

    def add_cycle(self) -> bool:
        """
        Add a single cycle to the count.

        Returns:
            bool: True if the CPS value was reported. False otherwise.
        """
        return self.add_cycles(1)

    def add_cycles(self, cycles: int) -> bool:
        """
        Add a block of cycles to the count.

        Returns:
            bool: True if the CPS value was reported. False otherwise.
        """
        if cycles <= 0:
            return False

        t_perf_now = time.perf_counter()

        if self._cycle_count is None:
            self._cycle_count = 0
            self._next_refresh = self._start = t_perf_now
            self._next_refresh += self.report_window

        elif t_perf_now > self._next_refresh:
            self.cps = self._cycle_count / (t_perf_now - self._next_refresh + self.report_window)
            if self.enable_log:
                logger.log(self.log_level, f"{self.name}: {self.cps:.1f} {self.units}")
            self._next_refresh += self.report_window
            self._cycle_count = cycles
            return True

        self._cycle_count += cycles
        return False
