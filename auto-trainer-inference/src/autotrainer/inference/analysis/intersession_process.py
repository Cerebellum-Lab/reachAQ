import json
import math
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy

from autotrainer.core import ProjectInfo
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.pose_elements import SceneElement
from autotrainer.core.reach_event import ReachEvent

from autotrainer.core.config_directory import DEFAULT_CONFIG_DIR

from autotrainer.inference.analysis.prepare_jetson_data import process_raw_data
from autotrainer.inference.analysis.parse_pellet_presentations_jetson import segment_reaches
from autotrainer.inference.multires_calibration.store import (
    INDEX_FILE,
    CalibrationGeometryError,
    CalibrationResolutionError,
    LoadedPreset,
    calibration_created,
    find_calibration,
    load_index,
    load_preset,
    resolve_for_time,
)

from autotrainer.inference.analysis import IntersessionResponse

logger = get_verbose_logger(__name__)


AvailableShiftXYZ = numpy.array([[-5, 5], [-5, 5], [-5, 5]])


_segment_reach_debug: int = int(os.getenv("AUTOTRAINER_SEGMENT_REACH_DEBUG", 0))

_CALIB_DIR_HINT = "pass --calib-dir to choose the calibration folder"


def resolve_session_calibration(project: ProjectInfo, calib_root: Path) -> Path:
    """The calibration folder under `calib_root` that the session was recorded with.

    A session whose metadata names a calibration (hardware.configured.calibration3d.id) uses
    that calibration, wherever it now is: the active folder or a backup. Any other session
    uses the calibration that was active when it started (see _session_start for where that
    time comes from). When only the session's day is known, the calibration must not have
    changed that day. Raises CalibrationResolutionError, saying to pass --calib-dir, when the
    id is not found, the match is ambiguous, or no calibration qualifies.

    A session older than every surviving calibration resolves to the oldest one, with a
    warning: the calibration it was recorded with is gone, so its 3D will not reproduce.
    """
    calib_root = Path(calib_root).expanduser()
    metadata = _session_metadata(project)
    calibration_id = _calibration3d(metadata).get("id")
    window = _session_start(project, metadata)
    try:
        if calibration_id:
            folder = find_calibration(calib_root, calibration_id)
        elif window is None:
            raise CalibrationResolutionError(f"the start time of session {project.short_id} is unknown")
        else:
            earliest, latest = window
            folder = resolve_for_time(calib_root, earliest)
            if latest != earliest and resolve_for_time(calib_root, latest) != folder:
                raise CalibrationResolutionError(
                    f"only the day of session {project.short_id} is known, {earliest:%Y-%m-%d}, and the"
                    f" calibration under {calib_root} changed that day"
                )
    except CalibrationResolutionError as error:
        raise CalibrationResolutionError(f"{error}; {_CALIB_DIR_HINT}") from error
    if window is not None:
        _warn_if_made_after(folder, window[1], project)
    logger.info("session %s uses the calibration in %s", project.short_id, folder)
    return folder


def _session_metadata(project: ProjectInfo) -> dict:
    """The session's metadata JSON, or {} when it has none yet or it cannot be read."""
    path = Path(project.get_metadata_file() + ".json")
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as error:
        logger.warning("ignoring the session metadata %s: %s", path, error)
        return {}
    return data if isinstance(data, dict) else {}


def _calibration3d(metadata: dict) -> dict:
    """The session's hardware.configured.calibration3d, or {} for a session recorded without it."""
    value = ((metadata.get("hardware") or {}).get("configured") or {}).get("calibration3d")
    return value if isinstance(value, dict) else {}


def _session_start(project: ProjectInfo, metadata: dict) -> Optional[Tuple[datetime, datetime]]:
    """(earliest, latest) local start of the session; the two differ when only its day is known.

    The first of these that is set, each a unix time:
    - project.start_record_timestamp, which the app sets at Record;
    - the metadata's boundary.startWallTime (schema 2);
    - its sessionBoundary.startWallTime or start_record_timestamp (schema 1);
    - its createdUtc, which for a session is the time the app gave the session when it began.
    Otherwise project.when. The CLI builds that from a date alone, so a time of exactly
    midnight stands for the whole day.
    """
    for value in (
        project.start_record_timestamp,
        (metadata.get("boundary") or {}).get("startWallTime"),
        (metadata.get("sessionBoundary") or {}).get("startWallTime"),
        metadata.get("start_record_timestamp"),
        metadata.get("createdUtc"),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            start = datetime.fromtimestamp(value)
            return start, start
    when = project.when
    if when is None:
        return None
    if when.tzinfo is not None:
        when = when.astimezone().replace(tzinfo=None)
    if when.time() == time(0):
        return when, when + timedelta(days=1) - timedelta(microseconds=1)
    return when, when


def _warn_if_made_after(folder: Path, start: datetime, project: ProjectInfo) -> None:
    try:
        created = calibration_created(folder)
    except (OSError, ValueError) as error:
        logger.warning("cannot tell when the calibration in %s was made: %s", folder, error)
        return
    if start < created:
        logger.warning(
            "session %s started at %s, before the calibration in %s was made (%s): the calibration"
            " in use then is not kept, so this session's 3D will not reproduce its live 3D",
            project.short_id, start, folder, created,
        )


def _session_preset(folder: Path, metadata: dict, project: ProjectInfo) -> LoadedPreset:
    """The preset of the format-2 calibration in `folder` that the session's 2D points are in.

    That is the preset its live 3D used, named in its metadata. A session without one (live 3D
    was off, or no calibration3d) takes the base preset, the largest binning: the 2D pose files
    are in inference space, which is the base capture's pixels (spec, "Offline 3D input").
    """
    index = load_index(folder)
    name = _calibration3d(metadata).get("preset")
    if name is None:
        geometry = max(index.presets, key=lambda g: g.binning)
        logger.info("session %s names no preset; using the base preset %s of %s",
                    project.short_id, geometry.name, folder)
    else:
        matches = [g for g in index.presets if g.name == name]
        if not matches:
            available = ", ".join(g.name for g in index.presets)
            raise CalibrationGeometryError(
                f"session {project.short_id} was recorded at preset {name}, which the calibration in"
                f" {folder} does not have; it has {available}"
            )
        geometry, = matches
    return load_preset(folder, geometry)


def intersession_process(
    project: ProjectInfo,
    *,
    calib_dir: Optional[Path] = None,
    calib_root: Optional[Path] = None,
    frame_rate: int = 150,
    debug_level: int = _segment_reach_debug,
) -> IntersessionResponse:
    """
    Called after pose processing for intersession analysis.

    :param project: current project info for finding/defining file names
    :param calib_dir: calibration folder to use as given. When None, the session's own is
        resolved under calib_root with resolve_session_calibration.
    :param calib_root: where the calibration and its backups are, DEFAULT_CONFIG_DIR when None.
    :param debug_level: integer debug level.
    :return: information required to update behavior for future sessions
    """
    location, _, _ = project.get_session_path()
    logger.info("process intersession pose data using %s", location)
    if calib_dir is None:
        calib_src_dir = resolve_session_calibration(
            project, DEFAULT_CONFIG_DIR.expanduser() if calib_root is None else Path(calib_root))
    else:
        calib_src_dir = Path(calib_dir).expanduser()
    # A folder with a calibration.yaml is format 2, whether or not it reads: a legacy folder
    # never has one, so it is never read the legacy way (which would also fail obscurely).
    if calib_src_dir.joinpath(INDEX_FILE).exists():
        preset = _session_preset(calib_src_dir, _session_metadata(project), project)
    else:
        preset = None
        if not calib_src_dir.is_dir():
            logger.warning("calib_src_dir %s is not a directory",  calib_src_dir)
    calib_src_dir = calib_src_dir.as_posix()
    vid_tag = "." + project.video_write_ext
    dlc_seg = "_raw2D"
    center_method = (1, SceneElement.Diamond)
    #
    df_lr, centered_df_3d = process_raw_data(
        project,
        location, vid_tag, dlc_seg, calib_src_dir, center_method,
        frame_rate=frame_rate,
        preset=preset,
    )
    #
    results_dict = segment_reaches(
        project_info=project,
        session=location,
        center_method=center_method,
        df_lr=df_lr,
        df_3d=centered_df_3d,
        debug=debug_level,
        frame_rate=frame_rate,
    )
    logger.verbose("process intersession pose data complete %s", results_dict)
    # rename:
    results_dict["food_consumed"] = results_dict.pop("pellets_consumed")
    # all others keys are same than IntersessionResponse fields
    # convert to ReachEvent instances:
    results_dict["reach_events"] = [ReachEvent(**d) for d in results_dict["reach_events"]]
    results_dict["other_events"] = [
        # other events are pellet_events not associated with reach (by hand)
        ReachEvent(
            init=d['placed'],
            end=d['lost'],
            max=-1,
            method=d['method'],
            outcome=d['outcome'],
            delay_since_presented=(
                d['placed'] / frame_rate
                - project.get_first_pellet_presentation_offset()
            ),
        ) for d in results_dict["other_events"]
    ]
    return IntersessionResponse(**results_dict)
