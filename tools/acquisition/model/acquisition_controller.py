"""Independent acquisition-domain readiness and lifecycle state."""

from __future__ import annotations

import threading

from tools.acquisition.model.subsystem_status import (
    SubsystemId,
    SubsystemStatusRegistry,
)


class AcquisitionController:
    """Own aggregate acquisition state without coupling subsystem lifecycles.

    Each camera, CAN, NI-DAQ, laser, and inference domain retains its own
    initialization/cleanup path. The registry records their independent state;
    synchronization readiness is a separate derived domain.
    """

    def __init__(self) -> None:
        self.subsystems = SubsystemStatusRegistry()
        self.lock = threading.RLock()
        self.starting = False
        self.started = False
        self.stopping = False

    def begin_start(self) -> bool:
        with self.lock:
            if self.started or self.starting:
                return False
            self.starting = True
            return True

    def mark_started(self) -> None:
        with self.lock:
            self.started = True
            self.starting = False
            self.stopping = False

    def begin_stop(self, *, force: bool = False) -> bool:
        with self.lock:
            if not self.started and not force:
                return False
            if self.stopping:
                return False
            self.stopping = True
            return True

    def mark_stopped(self) -> None:
        with self.lock:
            self.started = False
            self.starting = False
            self.stopping = False

    @property
    def synchronization_ready(self) -> bool:
        status = self.subsystems.get(SubsystemId.REACH_SYNCHRONIZATION)
        return status is not None and status.is_ready
