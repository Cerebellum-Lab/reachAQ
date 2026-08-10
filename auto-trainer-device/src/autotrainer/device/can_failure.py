"""Structured terminal CAN command and transport failures."""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any, Optional


class CanFailureKind(str, enum.Enum):
    COMMAND = "command_failure"
    TRANSPORT = "transport_failure"
    ACKNOWLEDGEMENT_TIMEOUT = "acknowledgement_timeout"


@dataclasses.dataclass(frozen=True)
class CanFailure:
    kind: CanFailureKind
    error: str
    command: Optional[Any] = None
    context: Optional[str] = None
    perf_time: float = dataclasses.field(default_factory=time.perf_counter)
    wall_time: float = dataclasses.field(default_factory=time.time)

