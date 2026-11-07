"""Structured terminal CAN command and transport failures."""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any, Dict, Optional


class CanFailureKind(str, enum.Enum):
    COMMAND = "command_failure"
    TRANSPORT = "transport_failure"
    ACKNOWLEDGEMENT_TIMEOUT = "acknowledgement_timeout"
    OPERATION_UNKNOWN = "operation_unknown"


@dataclasses.dataclass(frozen=True)
class CanFailure:
    kind: CanFailureKind
    error: str
    command: Optional[Any] = None
    context: Optional[str] = None
    perf_time: float = dataclasses.field(default_factory=time.perf_counter)
    wall_time: float = dataclasses.field(default_factory=time.time)
    category: Optional[str] = None
    error_code: Optional[int] = None
    exception_type: Optional[str] = None
    diagnostics: Dict[str, Any] = dataclasses.field(default_factory=dict)
