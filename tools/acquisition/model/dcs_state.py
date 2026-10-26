"""Whether the DCS configuration was measured against the active 3D calibration.

DCS (Tools > Calibrate Coordinate System) is measured in the oriented 3D frame of the
calibration active at the time. A new 3D calibration changes that frame, so DCS must be
redone before anything uses DCS-space coordinates again. A format-2 calibration has an id,
which DCS records. A legacy calibration has none: a DCS without an id is taken to fit it,
as before ids existed, but a DCS that records an id was measured in a format-2 motor frame,
which is not the legacy frame, so it is stale under a legacy calibration (after a rollback
to a legacy backup, for example).
"""

from enum import Enum
from typing import Optional

from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.inference.multires_calibration.store import CalibrationIndex

#: Why DCS-space coordinates are unusable while DCS is stale, whichever calibration is active:
#: the reason Record is blocked.
STALE_RECORDING_BLOCKER = (
    "Coordinate system out of date: it was measured against a different 3D calibration. "
    "Redo Tools → Calibrate Coordinate System"
)
#: What the operator is told once, as DCS becomes stale.
STALE_NOTICE = f"{STALE_RECORDING_BLOCKER} before running sessions."


class DcsState(Enum):
    VALID = "valid"
    STALE = "stale"  # measured against another 3D calibration than the active one
    MISSING = "missing"  # none loaded, or not fully valid


def dcs_state(cfg: Optional[DiamondTriangleOffsetConfig], active_index: Optional[CalibrationIndex]) -> DcsState:
    """The state of `cfg` against the active calibration, whose index is None for a legacy one."""
    if cfg is None or not cfg.fully_valid:
        return DcsState.MISSING
    if active_index is None:
        # A recorded id means a format-2 frame, which a legacy calibration is not.
        return DcsState.STALE if cfg.calibration_id is not None else DcsState.VALID
    return DcsState.VALID if cfg.calibration_id == active_index.id else DcsState.STALE
