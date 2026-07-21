from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class HardwareScanEntry:
    """One row of the most recent hardware refresh result."""

    info: str
    state: str = "idle"
