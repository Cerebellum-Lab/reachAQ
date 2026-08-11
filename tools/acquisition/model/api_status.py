"""ReachAQ-owned public status schema without retired trainer subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ReachAQSystemStatus:
    schema_version: int
    acquisition_state: str
    recording_state: str
    synchronization_ready: bool
    animal: Optional[Dict[str, Any]]
    project: Dict[str, Any]
    subsystems: Dict[str, Any]
    pellet_device: Dict[str, Any]
    session_counts: Dict[str, int]
    protocol: Dict[str, Any] = field(default_factory=dict)

