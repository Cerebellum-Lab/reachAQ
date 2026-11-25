import json
import math
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy

from reachaq.core import ProjectInfo
from reachaq.core.logging import get_verbose_logger
from reachaq.core.pose_elements import SceneElement
from reachaq.core.reach_event import ReachEvent

from reachaq.core.config_directory import DEFAULT_CONFIG_DIR

from reachaq.inference.analysis.offline_3d import recorded_geometry, session_offline_3d
from reachaq.inference.analysis.prepare_jetson_data import process_raw_data
from reachaq.inference.analysis.parse_pellet_presentations_jetson import segment_reaches
from reachaq.inference.multires_calibration.geometry import PresetGeometry
from reachaq.inference.multires_calibration.library import entry_library, is_legacy_entry, library_root
from reachaq.inference.multires_calibration.products import stereo_pair_geometry
from reachaq.inference.multires_calibration.store import (
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

from reachaq.inference.analysis import IntersessionResponse

logger = get_verbose_logger(__name__)


AvailableShiftXYZ = numpy.array([[-5, 5], [-5, 5], [-5, 5]])


_segment_reach_debug: int = int(os.getenv("REACHAQ_SEGMENT_REACH_DEBUG", 0))

_CALIB_DIR_HINT = "pass --calib-dir to choose the calibration folder"
# The live3dDisabledReason the app records while the operator has switched live 3D off
# (src/reachaq/app/model/app_model.LIVE_3D_OFF). Live 3D off by choice says nothing
# about whether the calibration fits the session, so offline 3D does not refuse it.
LIVE_3D_OFF = "live 3D is off"
_LIVE_PAIR = ("left", "right")
# Where offline 3D derives the TOMLs of a calibration outside the library (--calib-dir), in
# the session folder: never in the operator's library.
OFFLINE_DERIVED_DIR = ".offline3d_derived"


def resolve_session_calibration(project: ProjectInfo, calib_root: Path) -> Path:
    """The calibration folder under `calib_root` that the session was recorded with.

    A session whose metadata names a calibration (hardware.configured.calibration3d.id) uses
    that calibration, wherever it now is: the library (<calib_root>/calibrations), the active
    folder or a backup. Any other session
    uses the calibration that was active when it started (see _session_start for where that
    time comes from). When only the session's day is known, the calibration must not have
    changed that day. Raises CalibrationResolutionError, saying to pass --calib-dir, when the
    id is not found, the match is ambiguous, or no calibration qualifies.

    A session older than every kept calibration is refused the same way (resolve_for_time):
    the calibration it was recorded with is gone, and the oldest kept one is no stand-in
    for it (spec, Old sessions). A session that names its calibration by id only gets a
    warning if it seems to predate it, since the id says which calibration it used.
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
    if calibration_id and window is not None:
        # By time, resolve_for_time has refused a calibration made after the session.
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
    if created.tzinfo is not None:
        # The start is local and naive; compare in the same terms, as resolve_for_time does.
        created = created.astimezone().replace(tzinfo=None)
    if start < created:
        logger.warning(
            "session %s started at %s, before the calibration in %s was made (%s): the calibration"
            " in use then is not kept, so this session's 3D will not reproduce its live 3D",
            project.short_id, start, folder, created,
        )


def _session_preset(folder: Path, metadata: dict, project: ProjectInfo) -> LoadedPreset:
    """The preset of the format-2 or 3 calibration in `folder` that the session's 2D points are in.

    That is the preset its live 3D used, named in its metadata. A session whose live 3D
    could not run (calibration3d records a live3dDisabledReason, for example a camera
    geometry that matched no preset) is refused: the pixels of its 2D points are unknown, and
    reading them at a preset that may not be theirs would be silently wrong (rulings R32,
    R35). A session recorded with live 3D switched off (LIVE_3D_OFF) is read like any other,
    unless its left or right camera was recorded as not in the calibration (a swapped body,
    say), which is refused; without a preset in its metadata it is read at the geometry
    left and right recorded at, from its configuration, when that is known, and refused
    with CalibrationGeometryError when that configuration cannot be read. Any other
    session without a preset takes the base preset, the largest binning, since its 2D pose
    files are in inference space, the base capture's pixels (spec, "Offline 3D input"). That
    is a session recorded before calibration3d existed, or one whose live 3D ran on a legacy
    calibration and reaches this format-2 folder; the latter is warned about.

    A format-3 calibration follows one order: the preset the metadata names, else its live
    preset set (the geometry the cameras recorded at) when that holds left and right at one
    geometry, else CalibrationGeometryError. It never falls back to the largest binning: the
    live set is what the cameras recorded, so any other preset would be a silent misread.
    """
    index = load_index(folder)
    calibration3d = _calibration3d(metadata)
    switched_off = _live_3d_switched_off(calibration3d)
    if "live3dDisabledReason" in calibration3d and not switched_off:
        raise CalibrationResolutionError(
            f"session {project.short_id} was recorded with live 3D off"
            f" ({calibration3d['live3dDisabledReason']}), so no preset of the calibration in {folder}"
            f" is known to match its 2D points; {_CALIB_DIR_HINT}"
        )
    if switched_off:
        _refuse_uncalibrated_pair(metadata, folder, project)
    name = calibration3d.get("preset")
    recorded = _recorded_pair_geometry(metadata, project) if switched_off and name is None else None
    if recorded is not None:
        logger.info("session %s was recorded with live 3D switched off; reading it at the geometry left and"
                    " right recorded at, %s", project.short_id, recorded.name)
        return load_preset(folder, recorded)
    if name is None:
        geometry = _live_pair_geometry(index) if index.format >= 3 else None
        if geometry is not None:
            logger.info("session %s names no preset; using the live preset set of %s (%s)",
                        project.short_id, folder, geometry.name)
        elif index.format >= 3:
            raise CalibrationGeometryError(
                f"session {project.short_id} names no preset, and the calibration in {folder} has no live"
                f" preset set with left and right at one geometry, so its 2D points have no known preset;"
                f" {_CALIB_DIR_HINT}"
            )
        elif not index.presets:
            raise CalibrationGeometryError(
                f"the calibration in {folder} has no left-right preset, so session {project.short_id}"
                f" has none to be read at; {_CALIB_DIR_HINT}"
            )
        else:
            geometry = max(index.presets, key=lambda g: g.binning)
            if calibration3d:
                logger.warning(
                    "session %s ran live 3D on a legacy calibration, not the multi-resolution one in %s;"
                    " reading it at the base preset %s", project.short_id, folder, geometry.name)
            else:
                logger.info("session %s has no calibration3d; using the base preset %s of %s",
                            project.short_id, geometry.name, folder)
    else:
        matches = [g for g in index.presets if g.name == name]
        if not matches:
            if not index.presets:
                raise CalibrationGeometryError(
                    f"the calibration in {folder} has no left-right preset, so session {project.short_id}"
                    f" (recorded at preset {name}) cannot be read from it; {_CALIB_DIR_HINT}"
                )
            available = ", ".join(g.name for g in index.presets)
            raise CalibrationGeometryError(
                f"session {project.short_id} was recorded at preset {name}, which the calibration in"
                f" {folder} does not have; it has {available}"
            )
        geometry, = matches
    return load_preset(folder, geometry)


def _live_3d_switched_off(calibration3d: dict) -> bool:
    """Whether the operator had switched live 3D off (schema 3's live3d.enabled, or its reason)."""
    live3d = calibration3d.get("live3d")
    if isinstance(live3d, dict) and live3d.get("enabled") is False:
        return True
    return calibration3d.get("live3dDisabledReason") == LIVE_3D_OFF


def _refuse_uncalibrated_pair(metadata: dict, folder: Path, project: ProjectInfo) -> None:
    """Raise CalibrationResolutionError when left or right was recorded as not in the calibration."""
    cameras = ((metadata.get("hardware") or {}).get("configured") or {}).get("cameras")
    if not isinstance(cameras, dict):
        return
    problems = [
        f"{name}: {cameras[name].get('uncalibratedReason') or 'not in the calibration'}"
        for name in _LIVE_PAIR
        if isinstance(cameras.get(name), dict) and cameras[name].get("inCalibration") is False
    ]
    if problems:
        raise CalibrationResolutionError(
            f"session {project.short_id} recorded cameras the calibration in {folder} does not hold as left and"
            f" right ({'; '.join(problems)}), so its left-right 3D cannot be read from it; {_CALIB_DIR_HINT}"
        )


def _recorded_pair_geometry(metadata: dict, project: ProjectInfo) -> Optional[PresetGeometry]:
    """The geometry left and right recorded at, from the session's configuration; None unless both share one.

    Raises CalibrationGeometryError when their configuration is there but cannot be read: the
    pixels of the session's 2D points are then unknown, and no preset is a safe guess (R32).
    """
    cameras = ((metadata.get("configuration") or {}).get("cameras")) or []
    params = {c.get("name"): c.get("params") or {} for c in cameras if isinstance(c, dict)}
    if not all(name in params for name in _LIVE_PAIR):
        return None
    try:
        geometries = {recorded_geometry(params[name]) for name in _LIVE_PAIR}
    except ValueError as error:
        raise CalibrationGeometryError(
            f"session {project.short_id} was recorded with live 3D off and names no preset, and the recorded"
            f" geometry of left and right is unknown: their configuration cannot be read ({error});"
            f" {_CALIB_DIR_HINT}"
        ) from error
    return geometries.pop() if len(geometries) == 1 else None


def _offline_3d(project: ProjectInfo, location: str, folder: Path, calib_root: Optional[Path],
                legacy_error: Optional[BaseException] = None) -> None:
    """Offline 3D from every calibrated camera of the session (offline_3d.session_offline_3d), for a format-3 calibration.

    It is an extra product: a failure is logged and the session's processing goes on. It
    does not depend on the earlier analysis (left-right 3D, segmentation): `legacy_error`, its failure, is logged
    with its reason and offline 3D runs regardless. A calibration outside the library
    (--calib-dir) has its derived TOMLs put in the session folder's OFFLINE_DERIVED_DIR,
    never in the operator's library; the check's settings are still the library's.
    """
    try:
        if not folder.joinpath(INDEX_FILE).exists() or is_legacy_entry(folder) or load_index(folder).format < 3:
            return
        if legacy_error is not None:
            logger.warning("the earlier analysis step of session %s failed: %s; offline 3D from every calibrated"
                           " camera runs regardless", project.short_id, str(legacy_error) or type(legacy_error).__name__)
        library_dir, derived_dir = entry_library(folder), None
        if library_dir is None:
            library_dir = library_root(DEFAULT_CONFIG_DIR.expanduser() if calib_root is None else Path(calib_root))
            derived_dir = Path(location) / OFFLINE_DERIVED_DIR
        session_offline_3d(project, folder, _session_metadata(project), library_dir=library_dir,
                           derived_dir=derived_dir)
    except Exception:
        logger.exception("offline 3D of session %s with the calibration in %s failed; its 2D processing goes on",
                         project.short_id, folder)


def _live_pair_geometry(index) -> Optional[PresetGeometry]:
    """The geometry left and right share in the calibration's live preset set, or None when it has no such set or pair."""
    for preset_set in index.preset_sets:
        if preset_set["name"] == "live":
            return stereo_pair_geometry({
                cam: PresetGeometry(g["binning"], g["width"], g["height"], g["offset_x"], g["offset_y"])
                for cam, g in preset_set["cameras"].items()
            })
    return None


def _session_calibration(project: ProjectInfo, calib_dir: Optional[Path], calib_root: Optional[Path]) -> Path:
    """`calib_dir` as given, else the session's own under calib_root (resolve_session_calibration)."""
    if calib_dir is None:
        return resolve_session_calibration(
            project, DEFAULT_CONFIG_DIR.expanduser() if calib_root is None else Path(calib_root))
    return Path(calib_dir).expanduser()


def intersession_process(
    project: ProjectInfo,
    *,
    calib_dir: Optional[Path] = None,
    calib_root: Optional[Path] = None,
    frame_rate: int = 150,
    debug_level: int = _segment_reach_debug,
    offline_3d: bool = True,
) -> IntersessionResponse:
    """
    Called after pose processing for intersession analysis.

    The detection response (reach segmentation, the response behaviour is updated from) is
    built first; offline 3D runs after it, since it takes minutes on a long session and the
    response does not depend on it. When the left-right step or the segmentation fails,
    offline 3D still runs, then the failure is raised.

    :param project: current project info for finding/defining file names
    :param calib_dir: calibration folder to use as given. When None, the session's own is
        resolved under calib_root with resolve_session_calibration.
    :param calib_root: where the calibration and its backups are, DEFAULT_CONFIG_DIR when None.
    :param debug_level: integer debug level.
    :param offline_3d: False leaves offline 3D to the caller (intersession_offline_3d), which
        then hands the response on without waiting for it.
    :return: information required to update behavior for future sessions
    """
    location, _, _ = project.get_session_path()
    logger.info("process intersession pose data using %s", location)
    calib_src_dir = _session_calibration(project, calib_dir, calib_root)
    calib_folder = calib_src_dir
    vid_tag = "." + project.video_write_ext
    dlc_seg = "_raw2D"
    center_method = (1, SceneElement.Diamond)
    try:
        # A folder with a calibration.yaml is format 2 or 3, whether or not it reads: a legacy folder
        # never has one, so it is never read the legacy way (which would also fail obscurely). The
        # library's copy of a legacy folder has a generated one, marked format "legacy".
        if calib_src_dir.joinpath(INDEX_FILE).exists() and not is_legacy_entry(calib_src_dir):
            preset = _session_preset(calib_src_dir, _session_metadata(project), project)
        else:
            preset = None
            if not calib_src_dir.is_dir():
                logger.warning("calib_src_dir %s is not a directory",  calib_src_dir)
        #
        df_lr, centered_df_3d = process_raw_data(
            project,
            location, vid_tag, dlc_seg, calib_src_dir.as_posix(), center_method,
            frame_rate=frame_rate,
            preset=preset,
        )
    except Exception as error:
        # The left-right step failed or was refused (no left-right pair, a swapped left, ...):
        # offline 3D from the cameras that do match still runs, then the failure stands.
        if offline_3d:
            _offline_3d(project, location, calib_folder, calib_root, legacy_error=error)
        raise
    try:
        response = _detection_response(project, location, center_method, df_lr, centered_df_3d,
                                       debug_level=debug_level, frame_rate=frame_rate)
    except Exception:
        if offline_3d:
            _offline_3d(project, location, calib_folder, calib_root)
        raise
    if offline_3d:
        _offline_3d(project, location, calib_folder, calib_root)
    return response


def intersession_offline_3d(
    project: ProjectInfo,
    *,
    calib_dir: Optional[Path] = None,
    calib_root: Optional[Path] = None,
    legacy_error: Optional[BaseException] = None,
) -> None:
    """Offline 3D alone, for a caller that ran intersession_process(offline_3d=False) and has
    handed its response on. `legacy_error` is that call's failure, if it failed. Never raises:
    a calibration that cannot be resolved was that call's failure already, and is only logged."""
    location, _, _ = project.get_session_path()
    try:
        folder = _session_calibration(project, calib_dir, calib_root)
    except Exception as err:
        logger.warning("no offline 3D for session %s: its calibration is not resolved: %s",
                       project.short_id, str(err) or type(err).__name__)
        return
    _offline_3d(project, location, folder, calib_root, legacy_error=legacy_error)


def _detection_response(project: ProjectInfo, location, center_method, df_lr, centered_df_3d, *,
                        debug_level: int, frame_rate: int) -> IntersessionResponse:
    """Reach segmentation, and the response behaviour is updated from."""
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
