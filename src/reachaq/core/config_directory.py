"""The one place the per-computer settings directory is named.

The system configuration, motor and move files, animals, calibration and the
hardware tools' records all live under it. REACHAQ_CONFIG_DIR moves it; unset
or empty, it is ~/reachAQ. Like the paths built from it, it is not
expanded: callers expanduser() on what they use.
"""
import os
from pathlib import Path


CONFIG_DIR_ENV = "REACHAQ_CONFIG_DIR"

DEFAULT_CONFIG_DIR = Path(os.environ.get(CONFIG_DIR_ENV) or "~/reachAQ")
