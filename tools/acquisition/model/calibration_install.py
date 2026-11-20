"""Install a new 3D calibration into the configuration location, as its quality report allows.

A pass or warn report installs at once; a warn only informs. A fail keeps the previous
calibration unless the operator confirms an override. The override is a decision in the
library's decisions.jsonl, not an edit of the calibration: library entries never change.

An install adds the calibration to the library (<location>/calibrations) and makes it the
active entry. Until the readers use the library, it also keeps the 4mm_6r_8c_4x folder
current, which the app still loads.

That folder is never left half-copied: the new folder is copied in beside the
active one as <name>.partial, the active one is renamed to its dated backup, and the copy
is then renamed into place. A copy that fails leaves the previous calibration untouched,
and a final rename that fails puts it back.

camera_offsets.pkl is never installed with a calibration, nor carried over from the
previous one: it is measured by DCS against a calibration, so the previous offsets are
stale in the new frame, and DCS writes new ones once it is redone.
"""

import getpass
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.project.project_info import DATE_TIME_FORMAT
from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from autotrainer.inference.multires_calibration.library import (
    add_entry,
    discard_entry,
    library_root,
    record_decision,
    set_active,
)

logger = get_verbose_logger(__name__)

_PARTIAL_SUFFIX = ".partial"


def decide_install(verdict: str, ask_override: Callable[[], bool]) -> Tuple[bool, bool]:
    """(install, override) for a new calibration whose quality report has `verdict`.

    pass and warn install without asking. fail installs only when ask_override() confirms,
    and the install is then an override.
    """
    if verdict in ("pass", "warn"):
        return True, False
    if verdict == "fail":
        confirmed = bool(ask_override())
        return confirmed, confirmed
    raise ValueError(f"unknown quality verdict {verdict!r}; expected pass, warn or fail")


def override_record(user: str, now: datetime) -> dict:
    """The operator's confirmation to install a calibration that failed its quality checks."""
    return {"confirmed": True, "by": user, "at": now.isoformat()}


def install_calibration(
    new_dir: Path,
    configuration_location: Path,
    *,
    now: datetime,
    by: Optional[str] = None,
    override: Optional[dict] = None,
) -> Tuple[Path, Optional[Path]]:
    """Make `new_dir` the active calibration under `configuration_location`.

    `new_dir` is added to the library as a new entry, which is made the active one by `by`
    (the operating-system user when None); `override`, the operator's confirmation of a
    failed report, is appended to the library's decisions then. Returns (the installed
    4mm_6r_8c_4x folder, the previous one's dated backup or None). The previous folder is
    renamed to <name>_<now in DATE_TIME_FORMAT>, the suffix the store reads as the time it
    stopped being active. `new_dir` is copied, not moved, and stays as it is.

    When the folder swap fails the new entry is removed again and the previous calibration
    stays active, so the install can be retried. A decision that cannot be written after the
    install is logged, not raised: the calibration is installed by then.
    """
    new_dir = Path(new_dir)
    root = library_root(configuration_location)
    target = Path(configuration_location) / DEFAULT_3D_CALIB_DIR_NAME
    partial = target.with_name(target.name + _PARTIAL_SUFFIX)
    backup = target.with_name(f"{target.name}_{now.strftime(DATE_TIME_FORMAT)}") if target.exists() else None
    # Checked before anything changes: renaming onto an existing backup would replace it
    # (an empty one) or fail only after the copy.
    if backup is not None and backup.exists():
        raise FileExistsError(f"{backup} already exists; the previous calibration cannot be backed up there")

    if partial.exists():
        logger.warning("removing %s, left by an interrupted calibration install", partial)
        shutil.rmtree(partial)
    calibration_id = add_entry(root, new_dir)
    try:
        shutil.copytree(new_dir, partial, ignore=_without_camera_offsets)
        if backup is not None:
            target.rename(backup)
            logger.info("previous 3D calibration moved to %s", backup)
        try:
            os.replace(partial, target)
        except BaseException:
            if backup is not None:
                backup.rename(target)
                logger.warning("could not install %s; the previous calibration is back in place", new_dir)
            raise
        try:
            set_active(root, calibration_id, by=by or getpass.getuser())
        except BaseException:
            # The folder swap is done; undo it so the previous calibration stays the active one.
            shutil.rmtree(target)
            if backup is not None:
                backup.rename(target)
            raise
    except BaseException:
        discard_entry(root, calibration_id)
        raise
    if override is not None:
        try:
            record_decision(root, calibration_id, "override", dict(override))
        except OSError:
            logger.exception("calibration %s is installed, but its override could not be recorded: %s",
                             calibration_id, override)
    logger.success("installed the 3D calibration from %s into %s", new_dir, target)
    return target, backup


def _without_camera_offsets(directory: str, names: List[str]) -> List[str]:
    """copytree's ignore: skip camera_offsets.pkl, which only DCS writes."""
    if DEFAULT_CAM_OFFSET_FILE_NAME not in names:
        return []
    logger.warning("not installing %s: offsets come only from DCS", Path(directory, DEFAULT_CAM_OFFSET_FILE_NAME))
    return [DEFAULT_CAM_OFFSET_FILE_NAME]
