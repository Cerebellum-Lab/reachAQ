"""Whether the DCS configuration was measured against the active 3D calibration.

DCS (Tools > Calibrate Coordinate System) is measured in the oriented 3D frame of the
calibration active at the time. A new 3D calibration changes that frame, so DCS must be
redone before anything uses DCS-space coordinates again. A format-2 calibration has an id,
which DCS records; a legacy calibration has none, and any valid DCS is taken to fit it,
as before ids existed.
"""

from enum import Enum
from typing import Optional

from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.inference.multires_calibration.store import CalibrationIndex

#: Why Record is unavailable while DCS is stale.
STALE_RECORDING_BLOCKER = "Coordinate system out of date: redo Tools → Calibrate Coordinate System"
#: What the operator is told once, as DCS becomes stale.
STALE_NOTICE = (
    "Out of date after a new 3D calibration: redo Tools → Calibrate Coordinate System "
    "before running sessions."
)


class DcsState(Enum):
    VALID = "valid"
    STALE = "stale"  # measured against another 3D calibration than the active one
    MISSING = "missing"  # none loaded, or not fully valid


def dcs_state(cfg: Optional[DiamondTriangleOffsetConfig], active_index: Optional[CalibrationIndex]) -> DcsState:
    """The state of `cfg` against the active calibration, whose index is None for a legacy one."""
    if cfg is None or not cfg.fully_valid:
        return DcsState.MISSING
    if active_index is None:
        return DcsState.VALID
    return DcsState.VALID if cfg.calibration_id == active_index.id else DcsState.STALE
