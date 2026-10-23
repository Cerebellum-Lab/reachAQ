"""When to suggest a new 3D calibration, and what to tell the operator.

Each rig is calibrated about once a month, so the app suggests a new calibration once the
active one is that old. It only suggests: a calibration moves the motors and needs the rig set
up, so the operator starts it, from the menu the text names.
"""

from datetime import datetime, timedelta
from typing import Optional


def calibration_due(created: Optional[datetime], now: datetime, days: int) -> bool:
    """Whether a calibration made at `created` is `days` old at `now`; no calibration is always due.

    `created` and `now` must both be naive or both be aware, or the subtraction raises.
    """
    return created is None or now - created >= timedelta(days=days)


def reminder_text(created: Optional[datetime]) -> str:
    """The message suggesting a new calibration, with when the last one was made (or "never")."""
    when = "never" if created is None else created.strftime("%Y-%m-%d")
    return (
        f"Last 3D calibration: {when}.\n\n"
        "A new 3D calibration is due. "
        "Choose Tools → Make 3D calibration when the rig is ready; "
        "reachAQ will not start one for you."
    )
