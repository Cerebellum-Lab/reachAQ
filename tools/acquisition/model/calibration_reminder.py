"""When to suggest a new 3D calibration, and what to tell the operator.

Each rig is calibrated about once a month, so the app suggests a new calibration once the
active one is that old. It only suggests: a calibration moves the motors and needs the rig set
up, so the operator starts it, from the menu the text names.
"""

from datetime import datetime, timedelta
from typing import Optional

MAKE_CALIBRATION_MENU_PATH = "Tools → Make 3D calibration"


def calibration_due(created: Optional[datetime], now: datetime, days: int) -> bool:
    """Whether a calibration made at `created` is `days` old at `now`; no calibration is always due.

    `created` and `now` must both be naive or both be aware, or the subtraction raises.
    """
    return created is None or now - created >= timedelta(days=days)


def reminder_text(created: Optional[datetime], *, unreadable: bool = False) -> str:
    """The message suggesting a new calibration, with when the last one was made.

    `created` None means there is no calibration ("never"), unless `unreadable`: the active
    calibration exists but its date could not be read, which "never" would misstate.
    """
    when = "unknown (could not be read)" if unreadable else _when(created)
    return (
        f"Last 3D calibration: {when}.\n\n"
        "A new 3D calibration is due. "
        f"Choose {MAKE_CALIBRATION_MENU_PATH} when the rig is ready; "
        "reachAQ will not start one for you."
    )


def due_label_text(created: Optional[datetime], *, unreadable: bool = False) -> str:
    """The status bar's short form of reminder_text, kept up until a new calibration is installed."""
    when = "unknown" if unreadable else _when(created)
    return f"3D calibration due (last: {when})"


def _when(created: Optional[datetime]) -> str:
    return "never" if created is None else created.strftime("%Y-%m-%d")
