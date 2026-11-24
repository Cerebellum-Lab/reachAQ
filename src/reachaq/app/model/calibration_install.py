"""Install a new 3D calibration into the configuration location, as its quality report allows.

A pass or warn report offers Set as active (the default) or Keep, don't activate; a warn
only informs. Kept, the calibration is added to the library without moving the pointer, and
the library records a "kept" decision; it can be set active later from the library view. A
fail keeps the previous calibration unless the operator confirms an override. The override
is a decision in the library's decisions.jsonl, not an edit of the calibration: library
entries never change.

An install adds the calibration to the library (<location>/calibrations) and makes it the
active entry, which is what the app loads. The 4mm_6r_8c_4x folder and its dated backups
are no longer written: they stay as they are, read only, for the sessions recorded with
them and for an older app version. A pointer that cannot be written discards the new
entry again, so the previous calibration stays active and the install can be retried.

camera_offsets.pkl is never installed with a calibration, nor carried over from the
previous one: it is measured by DCS against a calibration, so the previous offsets are
stale in the new frame, and DCS writes new ones once it is redone, beside the library
entry rather than in it (AppModel.camera_offsets_path).
"""

import getpass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

from reachaq.core.logging import get_verbose_logger
from reachaq.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from reachaq.inference.multires_calibration.library import (
    add_entry,
    discard_entry,
    entry_path,
    library_root,
    record_decision,
    set_active,
)
from reachaq.inference.multires_calibration.store import load_index

logger = get_verbose_logger(__name__)


def decide_install(verdict: str, ask_override: Callable[[], bool],
                   ask_activate: Callable[[], bool] = lambda: True) -> Tuple[bool, bool, bool]:
    """(install, override, activate) for a new calibration whose quality report has `verdict`.

    pass and warn are added to the library, and made active when ask_activate() says so (Set
    as active); otherwise only kept. fail installs, active, only when ask_override() confirms,
    and the install is then an override; ask_activate is not asked for a fail.
    """
    if verdict in ("pass", "warn"):
        return True, False, bool(ask_activate())
    if verdict == "fail":
        confirmed = bool(ask_override())
        return confirmed, confirmed, confirmed
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
    activate: bool = True,
) -> Path:
    """Make `new_dir` the active calibration under `configuration_location`, or only keep it.

    `new_dir` is added to the library as a new entry, which is made the active one by `by`
    (the operating-system user when None); `override`, the operator's confirmation of a
    failed report, is appended to the library's decisions then. Returns the entry's folder,
    <location>/calibrations/<id>. `new_dir` is copied, not moved, and stays as it is. `now`
    is kept for the callers; the library stamps its own times.

    When the pointer or its decision cannot be written the new entry is removed again and the
    previous calibration stays active (set_active puts the old pointer back), so the install
    can be retried; the caller gets that write's error. A retry of the same calibration after
    a crash between adding and activating it finds the identical entry, keeps it and
    activates it (add_entry); an entry that was already there is never removed by a failed
    retry. A decision that cannot be written after the install is logged, not raised: the
    calibration is installed by then.

    With `activate` False the entry is only added (kept): the pointer is not moved, `override`
    is ignored (only a fail is overridden, and an override always activates), and a decision
    {"event": "kept", "detail": {"by"}} is appended.
    """
    new_dir = Path(new_dir)
    root = library_root(configuration_location)
    if Path(new_dir, DEFAULT_CAM_OFFSET_FILE_NAME).exists():
        raise ValueError(f"{new_dir} holds a {DEFAULT_CAM_OFFSET_FILE_NAME}; offsets come only from DCS")
    existed = root.joinpath(load_index(new_dir).id).exists()
    calibration_id = add_entry(root, new_dir)
    if not activate:
        # Kept: in the library, the pointer where it was. The decision is logged, not raised,
        # when it cannot be written: the entry is kept by then.
        try:
            record_decision(root, calibration_id, "kept", {"by": str(by or getpass.getuser())})
        except OSError:
            logger.exception("calibration %s is kept, but the decision could not be recorded", calibration_id)
        target = entry_path(root, calibration_id)
        logger.success("kept the 3D calibration from %s as %s, not active", new_dir, target)
        return target
    try:
        set_active(root, calibration_id, by=by or getpass.getuser())
    except BaseException:
        if not existed:
            try:
                discard_entry(root, calibration_id)
            except Exception:
                # The activation's own error is the one the caller needs, not the rollback's.
                logger.exception("could not remove calibration %s again after its activation failed", calibration_id)
        raise
    if override is not None:
        try:
            record_decision(root, calibration_id, "override", dict(override))
        except OSError:
            logger.exception("calibration %s is installed, but its override could not be recorded: %s",
                             calibration_id, override)
    target = entry_path(root, calibration_id)
    logger.success("installed the 3D calibration from %s as %s", new_dir, target)
    return target
