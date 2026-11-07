"""Small runtime analysis graph retained by reachAQ."""

from __future__ import annotations

from typing import Optional

from autotrainer.core.project import ProjectInfo

from .pellet_position_monitor import PelletMisplacedDetector
from .watchdog_monitor import WatchdogMonitor


class ReachAnalysis:
    """Own only operational detectors used by the reachAQ application.

    Post-session image analysis remains in the inference/behavior pipeline; it
    is intentionally not a sensor alarm and is not owned here.
    """

    def __init__(self):
        self._project_info: Optional[ProjectInfo] = None
        self._pellet_misplaced_monitor = PelletMisplacedDetector()
        self._watchdog_monitor = WatchdogMonitor()
        self._detectors = (
            self._pellet_misplaced_monitor,
            self._watchdog_monitor,
        )

    @property
    def project_info(self) -> Optional[ProjectInfo]:
        return self._project_info

    @project_info.setter
    def project_info(self, value: Optional[ProjectInfo]) -> None:
        self._project_info = value

    @property
    def pellet_misplaced_monitor(self) -> PelletMisplacedDetector:
        return self._pellet_misplaced_monitor

    @property
    def watchdog_monitor(self) -> WatchdogMonitor:
        return self._watchdog_monitor

    def start(self) -> None:
        for detector in self._detectors:
            detector.start()

    def stop(self) -> None:
        for detector in self._detectors:
            detector.stop()

    def restart(self) -> None:
        for detector in self._detectors:
            detector.restart()
