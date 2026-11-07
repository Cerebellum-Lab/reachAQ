"""python-can's per-frame logging, off unless asked for.

python-can logs every frame it receives (level 9 on "can.bus") and every frame
it sends (the socketcan "tx" logger, at DEBUG). On christielab10 that was about
270 records a second, each formatted and pickled into the log queue on the CAN
reader thread. Holding the "can" logger at WARNING also quiets python-can's few
open and close INFO/DEBUG lines; its warnings still pass. The application's own
CAN lines (the autotrainer.* loggers, such as "CAN reader throughput") and the
kernel receive stamps in device.csv are not affected.

Set REACHAQ_CAN_FRAME_LOG=1 and restart to get the per-frame lines back for a
diagnostic run. Only the app entry points (reachaq, auto-trainer-local and
auto-trainer-headless) call this; can_console.py and the pellet-delivery tool
still log every frame.

Import-light on purpose: the entry points call this before setup_logging(),
and it must not pull in autotrainer.
"""

import logging
import os

CAN_FRAME_LOG_VARIABLE = "REACHAQ_CAN_FRAME_LOG"


def limit_can_frame_logging() -> bool:
    """Put python-can's logger at WARNING unless REACHAQ_CAN_FRAME_LOG=1.

    Call this before setup_logging(). With multiprocess logging, setup
    replaces Logger.setLevel with a relay to the listener process, so a level
    set afterwards in this process would change nothing here and the records
    would still be made, formatted and queued.

    Returns True when per-frame logging was asked for and left on.
    """
    requested = os.environ.get(CAN_FRAME_LOG_VARIABLE) == "1"
    if not requested:
        logging.getLogger("can").setLevel(logging.WARNING)
    return requested


def log_can_frame_logging(logger: logging.Logger, requested: bool) -> None:
    """Say which way python-can's frame logging was left, once logging is up."""
    if requested:
        logger.info("python-can per-frame logging is on (%s=1)", CAN_FRAME_LOG_VARIABLE)
    else:
        logger.info(
            "python-can per-frame logging is off; set %s=1 and restart to turn it on",
            CAN_FRAME_LOG_VARIABLE,
        )
