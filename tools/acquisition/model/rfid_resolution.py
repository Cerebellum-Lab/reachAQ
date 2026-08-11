from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from autotrainer.core import AnimalSubject
from autotrainer.core.animal.external_metadata import ExternalAnimalRecord


class RfidResolutionKind(str, enum.Enum):
    SELECTED = "selected"
    CREATED_AND_SELECTED = "created_and_selected"
    LINKED_AND_SELECTED = "linked_and_selected"
    SETUP_REQUIRED = "setup_required"
    UNKNOWN_RFID = "unknown_rfid"
    NEEDS_NAME = "needs_name"
    LINK_CONFLICT = "link_conflict"
    BUSY = "busy"


@dataclass(frozen=True)
class RfidResolution:
    kind: RfidResolutionKind
    rfid: str
    record: Optional[ExternalAnimalRecord] = None
    animal: Optional[AnimalSubject] = None
    message: str = ""
