"""Store a multi-resolution calibration on disk, read it back, and find the one in force at a time.

One folder per calibration, at format 3, for any 2 to 6 cameras:

    calibration.yaml        the index, written last: format, id, created, board, cameras
                            (name, serial, camera_id, calibrated, reason), preset_sets,
                            verdict, override, motor_scale and capture
    sensor/fit.yaml         the full motor-frame fit in sensor pixels, as plain lists
    calibration.toml        the anipose export of every calibrated camera, in sensor pixels
    presets/<set.name>/     one per preset set (geometry.PresetSet), e.g. presets/live/
        calibration.toml    the anipose export, each camera at its geometry in the set
        README.txt          how anipose and, with the extras, DeepLabCut 3D use the folder
      and, only when left and right are both in the set at the same geometry:
        camera_matrix/stereo_params.pickle            today's keys, so today's loaders read it
        camera_matrix/{left,right}_intrinsic_params.pickle   {cam: {"mtx", "dist"}}
        frame.yaml          the 3D frame: oriented = (raw - offsets) @ transform, in mm
    quality_report.json, quality_report.txt and any extra files (the coverage images and
    waypoints.yaml)

A camera the per-camera rule left out is listed in `cameras`, not calibrated, with its
reason; it is in no TOML and no preset set. A camera_matrix folder is a DeepLabCut 3D
camera_matrix folder: DLC's triangulate reads only stereo_params.pickle and keys the pair by
its project's camera names, so the project must name its cameras "left" and "right" to
match STEREO_KEY.

Format 2 (left and right only, one preset folder per geometry, presets/<g.name>/ with the
extras and a two-camera TOML, no top-level TOML) still reads: its index reads as cameras
left and right, both calibrated, with one preset set per preset.

A calibration is never written to after it is in place. Every reader here only reads;
load_preset_set derives a geometry no stored set has into a cache outside the folder.

The calibration library (library.py) keeps calibrations as immutable entries under
<config>/calibrations/; find_calibration and resolve_for_time look there first and fall back
to the active folder and its dated backups, which library.import_existing copies in.

A half-written folder never loads. The folder is written as <name>.partial and renamed into
place only when complete, and calibration.yaml is the last file written into it. A write
that raises removes its .partial folder; one a crash leaves is removed by the next write. A
folder
without calibration.yaml at format 2 or 3 is a legacy, single-geometry calibration, which is
loaded the old way.

Times are local and naive. Installing a calibration renames the active folder <name> to
<name>_<datetime.now() in DATE_TIME_FORMAT>, so a backup's suffix is the local time at which
it stopped being active, and `created` and a legacy folder's date use the same convention so
the two compare directly. The calibration in force at time t is the backup with the earliest
suffix after t, or the active folder when no backup ended after t.
"""

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.core.logging import get_verbose_logger
from reachaq.core.project.project_info import DATE_TIME_FORMAT
from reachaq.inference.calibration_FLIR import write_pickle
from reachaq.inference.config import StereoParams, load_calib_stereo_params
from reachaq.inference.multires_calibration.fit import BoardSpec, CameraModel, CaptureCamera, MotorFrameFit
from reachaq.inference.multires_calibration.geometry import PresetGeometry, PresetSet
from reachaq.inference.multires_calibration.products import (
    STEREO_KEY,
    anipose_toml,
    frame_transform,
    motor_to_inference_signs,
    preset_stereo_params,
    stereo_pair_geometry,
)
from reachaq.inference.multires_calibration.report import QualityReport

logger = get_verbose_logger(__name__)

CALIBRATION_FORMAT = 3
# The formats the readers accept: 2 (left and right only) and 3.
READABLE_FORMATS = (2, 3)
INDEX_FILE = "calibration.yaml"
# The format of the calibration.yaml the library generates for an imported legacy folder.
LEGACY_FORMAT = "legacy"

_SENSOR_FIT_FILE = Path("sensor", "fit.yaml")
_PRESETS_DIR = "presets"
_CAMERA_MATRIX_DIR = "camera_matrix"
_STEREO_FILE = "stereo_params.pickle"
_FRAME_FILE = "frame.yaml"
_ANIPOSE_FILE = "calibration.toml"
_README_FILE = "README.txt"
_REPORT_JSON = "quality_report.json"
_REPORT_TEXT = "quality_report.txt"
_PARTIAL_SUFFIX = ".partial"
# The pair the stereo parameters describe, in their order: cameraMatrix1 is the first.
_CAMERAS = tuple(STEREO_KEY.split("-"))
# The folders calibration_candidates has warned about as named like a backup but not one.
_WARNED_MISNAMED: Set[Path] = set()

_README = """\
{name}: one preset set of a reachAQ multi-resolution 3D calibration.
{geometries}
Every pixel value here is in these pixels: it applies only to video each camera
recorded at its binning and ROI.

anipose: calibration.toml, in aniposelib's CameraGroup format, with the cameras'
poses in the motor frame and translations in mm. Its tables are named after the
cameras, as their video files are.
"""
_README_DLC = """
DeepLabCut 3D: camera_matrix/ is a DLC camera_matrix folder for left and right.
Copy it to the 3D project's camera_matrix folder. stereo_params.pickle keys the pair
as "{stereo_key}", so the project's config.yaml must set camera_names: ["{first}", "{second}"].
"""
# Where load_preset_set derives a geometry no stored set has: <derived_root>/<id>/<key>/.
_DERIVED_KEY_LENGTH = 16


class CalibrationGeometryError(LookupError):
    """The calibration has no preset for the requested camera geometry."""


class CalibrationResolutionError(LookupError):
    """No single calibration matches the requested id or time."""


@dataclass(frozen=True)
class CalibrationIndex:
    id: str  # uuid4 hex
    created: datetime  # local, naive, to the second
    board: BoardSpec
    # The geometries load_preset serves: those at which left and right share a preset set,
    # each once, in set order. Empty when the calibration has not both.
    presets: Tuple[PresetGeometry, ...]
    verdict: str
    override: Optional[dict]
    motor_scale: float
    capture: dict
    format: int = CALIBRATION_FORMAT
    # Every capture camera, in capture order: {name, serial, camera_id, calibrated, reason}.
    cameras: List[dict] = field(default_factory=list)
    # {name, cameras: {camera: {binning, width, height, offset_x, offset_y}}}, the calibrated cameras only.
    preset_sets: List[dict] = field(default_factory=list)


@dataclass(frozen=True)
class LoadedPreset:
    geometry: PresetGeometry
    stereo_params: StereoParams
    frame_transform: np.ndarray


@dataclass(frozen=True)
class LoadedPresetSet:
    """An anipose calibration.toml in the pixels asked for, and the left-right extras when
    left and right were asked for at a geometry a stored set has them at."""

    toml_path: Path
    stereo: Optional[LoadedPreset]


@dataclass(frozen=True)
class CalibrationCandidate:
    path: Path
    active_until: Optional[datetime]  # None for the active folder
    calibration_id: Optional[str]  # None for a legacy folder


def write_calibration(
    folder: Path,
    *,
    fit: MotorFrameFit,
    board: BoardSpec,
    report: QualityReport,
    capture: dict,
    presets: Sequence[PresetGeometry] = (),
    preset_sets: Optional[Sequence[PresetSet]] = None,
    cameras: Optional[Sequence[CaptureCamera]] = None,
    extra_files: Optional[Mapping[str, Path]] = None,
) -> CalibrationIndex:
    """Write the calibration into `folder`, which must not exist yet, and return its index.

    `preset_sets` are the sets to write (geometry.presets_for_cameras); a camera in a set
    that the fit did not calibrate is dropped from it, a set that then repeats an earlier
    one's geometries is left out, and every set must give each calibrated camera a geometry.
    Without them, each of `presets` is a set of its own, named g.name, with every calibrated
    camera at g. `cameras` are the capture's cameras
    in capture order, and must include every calibrated one; each one's verdict comes from
    report.cameras. Without them the cameras are the fit's, with no serial or camera_id.
    `extra_files` maps a file name in the folder to a file to copy there, for the coverage
    images and the waypoint log, so they too are in place before the index is written.

    Any error while the files are written (a value the anipose export refuses, a full disk)
    removes the .partial folder before it is raised, so a refused calibration leaves nothing.
    """
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} already exists; a calibration folder is never overwritten")
    calibrated = list(fit.cameras)
    if cameras is None:
        cameras = [CaptureCamera(cam, None, None) for cam in calibrated]
    cameras = list(cameras)
    missing = [cam for cam in calibrated if cam not in {c.name for c in cameras}]
    if missing:
        raise ValueError(f"the fit's cameras {', '.join(missing)} are not among the capture's cameras")
    if preset_sets is None:
        preset_sets = [PresetSet(g.name, {cam: g for cam in calibrated}) for g in presets]
    sets = []
    for preset_set in preset_sets:
        lacking = [cam for cam in calibrated if cam not in preset_set.cameras]
        if lacking:
            raise ValueError(f"preset set {preset_set.name} has no geometry for {', '.join(lacking)}")
        kept = PresetSet(preset_set.name, {cam: preset_set.cameras[cam] for cam in calibrated})
        # Without its uncalibrated cameras a set can equal an earlier one; the first is kept.
        same = next((s for s in sets if dict(s.cameras) == dict(kept.cameras)), None)
        if same is not None:
            logger.info("preset set %s is the same as %s for the calibrated cameras; not written twice",
                        kept.name, same.name)
            continue
        sets.append(kept)
    names = [s.name for s in sets]
    if not sets or len(set(names)) != len(names):
        raise ValueError(f"expected preset sets with distinct names, got {names}")

    verdicts = report.cameras or {}
    camera_list = []
    for c in cameras:
        verdict = verdicts.get(c.name)
        camera_list.append({
            "name": str(c.name),
            "serial": None if c.serial is None else str(c.serial),
            "camera_id": None if c.camera_id is None else int(c.camera_id),
            "calibrated": c.name in fit.cameras,
            "reason": "" if c.name in fit.cameras else (verdict.reason if verdict is not None else "not fitted"),
        })
    index = CalibrationIndex(
        id=uuid.uuid4().hex,
        created=datetime.now().replace(microsecond=0),
        board=board,
        presets=_pair_geometries(sets),
        verdict=report.verdict,
        override=_plain(report.override),
        motor_scale=float(fit.motor_scale),
        capture=_plain(dict(capture)),
        format=CALIBRATION_FORMAT,
        cameras=camera_list,
        preset_sets=[_preset_set_to_plain(s) for s in sets],
    )
    # Serialised before anything is written, so a capture value YAML cannot hold fails here.
    index_text = _yaml_text(_index_to_plain(index))

    partial = folder.with_name(folder.name + _PARTIAL_SUFFIX)
    if partial.exists():
        logger.warning("removing %s, left by an interrupted calibration write", partial)
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    try:
        _write_text(partial / _SENSOR_FIT_FILE, _yaml_text(_fit_to_plain(fit)))
        toml_cameras = [c for c in cameras if c.name in fit.cameras]
        _write_text(
            partial / _ANIPOSE_FILE,
            anipose_toml(fit, _sensor_set(fit, calibrated), calibration_id=index.id, cameras=toml_cameras),
        )
        signs = motor_to_inference_signs()
        for s in sets:
            _write_preset(partial / _PRESETS_DIR / s.name, fit, s, signs, calibration_id=index.id,
                          cameras=toml_cameras)
        _write_text(partial / _REPORT_JSON, report.to_json())
        _write_text(partial / _REPORT_TEXT, report.to_text() + "\n")
        for name, source in (extra_files or {}).items():
            target = partial / name
            if target.exists():
                raise ValueError(f"extra file {name} would replace a calibration file")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # Last: until the index is in place the folder is not a calibration.
        _write_text(partial / INDEX_FILE, index_text)
        os.replace(partial, folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return index


def is_multires_calibration(folder: Path) -> bool:
    """True only when `folder` holds a readable calibration.yaml at format 2 or 3.

    An unreadable or corrupt index logs a warning and gives False; the loaders raise on it.
    """
    return _index_or_none(Path(folder)) is not None


def load_index(folder: Path) -> CalibrationIndex:
    """The index of `folder`. Raises ValueError when it has none at format 2 or 3 or it is corrupt.

    A format-2 index reads as cameras left and right, both calibrated, with no serial or
    camera_id, and one preset set per preset, named after it, with both cameras at it.
    """
    folder = Path(folder)
    index = _parse_index(folder)
    if index is None:
        formats = " or ".join(str(f) for f in READABLE_FORMATS)
        raise ValueError(f"{folder} is not a multi-resolution calibration: it has no {INDEX_FILE} at format {formats}")
    return index


def _index_from_data(data: dict) -> CalibrationIndex:
    calibration_id = data["id"]
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError(f"id must be a non-empty string, got {calibration_id!r}")
    created = data["created"]
    if not isinstance(created, datetime):
        created = datetime.fromisoformat(created)
    board = data["board"]
    fmt = int(data["format"])
    if fmt == 2:
        presets = tuple(_geometry_from_plain(p) for p in data["presets"])
        cameras = [
            {"name": cam, "serial": None, "camera_id": None, "calibrated": True, "reason": ""} for cam in _CAMERAS
        ]
        sets = [PresetSet(g.name, {cam: g for cam in _CAMERAS}) for g in presets]
    else:
        cameras = [
            {
                "name": str(c["name"]),
                "serial": None if c.get("serial") is None else str(c["serial"]),
                "camera_id": None if c.get("camera_id") is None else int(c["camera_id"]),
                "calibrated": bool(c["calibrated"]),
                "reason": str(c.get("reason") or ""),
            }
            for c in data["cameras"]
        ]
        sets = [
            PresetSet(str(s["name"]), {str(cam): _geometry_from_plain(g) for cam, g in s["cameras"].items()})
            for s in data["preset_sets"]
        ]
        _check_preset_sets(cameras, sets)
        presets = _pair_geometries(sets)
    return CalibrationIndex(
        id=calibration_id,
        created=created,
        board=BoardSpec(square_mm=float(board["square_mm"]), rows=int(board["rows"]), cols=int(board["cols"])),
        presets=presets,
        verdict=str(data["verdict"]),
        override=data.get("override"),
        motor_scale=float(data["motor_scale"]),
        capture=dict(data.get("capture") or {}),
        format=fmt,
        cameras=cameras,
        preset_sets=[_preset_set_to_plain(s) for s in sets],
    )


def _check_preset_sets(cameras: Sequence[dict], sets: Sequence[PresetSet]) -> None:
    """Raise ValueError unless every preset set holds exactly the calibrated cameras."""
    calibrated = [c["name"] for c in cameras if c["calibrated"]]
    for s in sets:
        lacking = [cam for cam in calibrated if cam not in s.cameras]
        if lacking:
            raise ValueError(f"preset set {s.name} has no geometry for calibrated {', '.join(lacking)}")
        extra = [cam for cam in s.cameras if cam not in calibrated]
        if extra:
            raise ValueError(f"preset set {s.name} holds {', '.join(extra)}, which the index does not list as calibrated")


def load_sensor_fit(folder: Path) -> MotorFrameFit:
    """The motor-frame fit in sensor/fit.yaml. Raises ValueError when, at format 3, its cameras
    are not the index's calibrated cameras: the two are written together, so one is damaged."""
    folder = Path(folder)
    index = load_index(folder)
    data = _read_yaml(folder / _SENSOR_FIT_FILE)
    if index.format >= 3:
        calibrated = sorted(c["name"] for c in index.cameras if c["calibrated"])
        fitted = sorted(str(cam) for cam in data["cameras"])
        if calibrated != fitted:
            raise ValueError(
                f"{folder}: the index calibrated {', '.join(calibrated)} but {_SENSOR_FIT_FILE.as_posix()} holds"
                f" {', '.join(fitted)}"
            )
    return MotorFrameFit(
        cameras={
            cam: CameraModel(
                K=np.array(m["K"], float),
                dist=np.array(m["dist"], float),
                rvec=np.array(m["rvec"], float),
                tvec=np.array(m["tvec"], float),
            )
            for cam, m in data["cameras"].items()
        },
        board_rvec=np.array(data["board_rvec"], float),
        board_tvec=np.array(data["board_tvec"], float),
        motor_scale=float(data["motor_scale"]),
        image_size=tuple(int(v) for v in data["image_size"]),
        rms_px={cam: float(v) for cam, v in data["rms_px"].items()},
        per_view_rms_px={
            cam: {int(i): float(v) for i, v in views.items()} for cam, views in data["per_view_rms_px"].items()
        },
        param_std={cam: {name: float(v) for name, v in std.items()} for cam, std in data["param_std"].items()},
    )


def load_preset(folder: Path, geometry: PresetGeometry) -> LoadedPreset:
    """The left-right stereo parameters and 3D frame at `geometry`, for both left and right.

    That is the preset whose geometry equals `geometry` at format 2, and a preset set
    holding left and right both at `geometry` at format 3. Raises CalibrationGeometryError
    when none matches: a different preset is never substituted, since its pixels would not
    be the frames' pixels.
    """
    folder = Path(folder)
    index = load_index(folder)
    if not index.presets:
        raise CalibrationGeometryError(
            f"the calibration in {folder} has no stereo preset: it does not hold left and right at a shared geometry"
        )
    if geometry not in index.presets:
        available = "; ".join(_describe(g) for g in index.presets)
        raise CalibrationGeometryError(
            f"the calibration in {folder} has no preset for {_describe(geometry)}; it has {available}"
        )
    return _load_stereo(folder, _stereo_set_name(index, geometry), geometry)


def load_preset_set(
    folder: Path, geometries: Mapping[str, PresetGeometry], *, derived_root: Path
) -> LoadedPresetSet:
    """An anipose calibration.toml holding each camera in `geometries` at its geometry there.

    A stored file serves when its set gives every camera asked for that geometry; it may
    hold more cameras, which anipose consumers select by name. The entry's top-level
    calibration.toml serves for cameras all asked for at full sensor resolution. Otherwise
    the file is derived from the sensor fit into
    <derived_root>/<id>/<key>/calibration.toml, where key is a hash of the geometries, and
    reused from there; derived_root (the library's .derived cache) must be outside the
    entry, which is never written to. `stereo` is the left-right preset when left and right
    are asked for at one geometry that load_preset serves, and None otherwise.

    Raises CalibrationGeometryError for a camera the calibration did not calibrate.
    """
    folder = Path(folder)
    index = load_index(folder)
    geometries = dict(geometries)
    if not geometries:
        raise ValueError("no camera geometries asked for")
    calibrated = [c["name"] for c in index.cameras if c["calibrated"]]
    unknown = [cam for cam in geometries if cam not in calibrated]
    if unknown:
        raise CalibrationGeometryError(
            f"the calibration in {folder} did not calibrate {', '.join(unknown)}; it calibrated {', '.join(calibrated)}"
        )
    pair = stereo_pair_geometry(geometries)
    stereo = load_preset(folder, pair) if pair is not None and pair in index.presets else None

    for s in index.preset_sets:
        if all(cam in s["cameras"] and _geometry_from_plain(s["cameras"][cam]) == g for cam, g in geometries.items()):
            return LoadedPresetSet(folder / _PRESETS_DIR / s["name"] / _ANIPOSE_FILE, stereo)
    sensor_toml = folder / _ANIPOSE_FILE
    if index.format >= 3 and sensor_toml.is_file():
        fit_size = _read_yaml(folder / _SENSOR_FIT_FILE)["image_size"]
        full = PresetGeometry(1, int(fit_size[0]), int(fit_size[1]), 0, 0)
        if all(g == full for g in geometries.values()):
            return LoadedPresetSet(sensor_toml, stereo)

    derived_root = Path(derived_root)
    if _is_within(derived_root, folder):
        raise ValueError(f"the derived cache {derived_root} is inside the calibration {folder}, which is never written to")
    ordered = {cam: geometries[cam] for cam in calibrated if cam in geometries}
    key = hashlib.sha256(
        json.dumps({cam: dataclasses.asdict(g) for cam, g in sorted(ordered.items())}, sort_keys=True).encode()
    ).hexdigest()[:_DERIVED_KEY_LENGTH]
    path = derived_root / index.id / key / _ANIPOSE_FILE
    cameras = [CaptureCamera(c["name"], c["serial"], c["camera_id"]) for c in index.cameras if c["calibrated"]]
    text = anipose_toml(load_sensor_fit(folder), PresetSet(key, ordered), calibration_id=index.id, cameras=cameras)
    if not (path.is_file() and path.read_text(encoding="utf-8") == text):
        path.parent.mkdir(parents=True, exist_ok=True)
        _replace_text(path, text)
        logger.info("derived the anipose calibration of %s at %s into %s", index.id,
                    "; ".join(f"{cam} {_describe(g)}" for cam, g in ordered.items()), path)
    return LoadedPresetSet(path, stereo)


def _load_stereo(folder: Path, set_name: str, geometry: PresetGeometry) -> LoadedPreset:
    preset_dir = folder / _PRESETS_DIR / set_name
    stereo = load_calib_stereo_params(preset_dir / _CAMERA_MATRIX_DIR / _STEREO_FILE)
    transform = np.array(_read_yaml(preset_dir / _FRAME_FILE)["transform"], float)
    if transform.shape != (3, 3):
        raise ValueError(f"{preset_dir / _FRAME_FILE}: expected a 3x3 transform, got shape {transform.shape}")
    return LoadedPreset(geometry=geometry, stereo_params=stereo, frame_transform=transform)


def calibration_created(folder: Path) -> datetime:
    """When the calibration in `folder` was made, as a local naive time to the second.

    A legacy folder records no date, so its stereo_params.pickle's mtime stands in: the
    legacy install copies the folder with shutil.copytree, which keeps file times. The
    library's copy of one (format "legacy") records that time as `created` when it is
    imported, which is read instead, whatever the copy's own file times are. A corrupt
    index raises rather than falling back to a pickle a format-2 folder does not have.
    """
    folder = Path(folder)
    index = _parse_index(folder)
    if index is not None:
        return index.created
    marker = _legacy_marker_created(folder)
    if marker is not None:
        return marker
    mtime = (folder / _CAMERA_MATRIX_DIR / _STEREO_FILE).stat().st_mtime
    return datetime.fromtimestamp(mtime).replace(microsecond=0)


def _legacy_marker_created(folder: Path) -> Optional[datetime]:
    """The `created` of an imported legacy entry's generated calibration.yaml, or None when
    `folder` has none or it cannot be read."""
    path = folder / INDEX_FILE
    if not path.is_file():
        return None
    try:
        data = _read_yaml(path)
        if not isinstance(data, dict) or data.get("format") != LEGACY_FORMAT:
            return None
        created = data["created"]
        created = created if isinstance(created, datetime) else datetime.fromisoformat(str(created))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as error:
        logger.warning("the legacy calibration marker %s has no readable created time: %s", path, error)
        return None
    return created.replace(microsecond=0)


def calibration_candidates(root: Path, name: str = DEFAULT_3D_CALIB_DIR_NAME) -> List[CalibrationCandidate]:
    """The backups of `root/name`, earliest end first, then the active folder if it exists.

    A backup is a folder named f"{name}_" plus a suffix that parses with DATE_TIME_FORMAT.
    Another folder starting with f"{name}_" is ignored with a warning (once per process,
    then at debug level); folders such as
    f"{name}.partial" or f"{name}.new" are not backups and are ignored silently.
    """
    root = Path(root)
    prefix = f"{name}_"
    backups = []
    children = sorted(root.iterdir()) if root.is_dir() else []
    for child in children:
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix):]
        try:
            active_until = datetime.strptime(suffix, DATE_TIME_FORMAT)
        except ValueError:
            # Every lookup lists the folders again (christielab10's 4mm_6r_8c_4x_crop256, for
            # every session processed), so the warning is given once per process per folder.
            level = logging.DEBUG if child in _WARNED_MISNAMED else logging.WARNING
            _WARNED_MISNAMED.add(child)
            logger.log(
                level, "ignoring %s: it is named like a backup of %s, but %r is not a %s time", child, name,
                suffix, DATE_TIME_FORMAT,
            )
            continue
        backups.append(CalibrationCandidate(child, active_until, _calibration_id(child)))
    backups.sort(key=lambda c: (c.active_until, c.path.name))
    active = root / name
    if active.is_dir():
        backups.append(CalibrationCandidate(active, None, _calibration_id(active)))
    return backups


def find_calibration(root: Path, calibration_id: str) -> Path:
    """The folder under `root` whose calibration has id `calibration_id`.

    The library (<root>/calibrations) is searched first, then the active folder and its dated
    backups, which hold calibrations the library has not imported (library.import_existing).
    A library entry whose index does not load (corrupt, half-written) does not shadow an old
    folder with the same id: that folder is used, and the entry is warned about.
    """
    from reachaq.inference.multires_calibration import library  # library imports this module

    broken: Optional[str] = None
    try:
        entry = library.entry_path(library.library_root(root), calibration_id)
    except CalibrationResolutionError as error:
        if library.library_root(root).joinpath(calibration_id).is_dir():
            broken = str(error)
    else:
        try:
            if not library.is_legacy_entry(entry):
                load_index(entry)
            return entry
        except (OSError, ValueError) as error:
            broken = f"{entry / INDEX_FILE} does not load: {error}"
            logger.warning("the library entry %s cannot be read; looking for calibration %s in the folders"
                           " under %s instead: %s", entry, calibration_id, root, error)
    matches = [c.path for c in calibration_candidates(root) if c.calibration_id == calibration_id]
    if not matches:
        detail = f" (the library's entry is unreadable: {broken})" if broken else ""
        raise CalibrationResolutionError(f"no calibration under {root} has id {calibration_id}{detail}")
    if len(matches) > 1:
        raise CalibrationResolutionError(
            f"{len(matches)} calibrations under {root} have id {calibration_id}: {', '.join(p.name for p in matches)}"
        )
    return matches[0]


def resolve_for_time(root: Path, when: datetime) -> Path:
    """The calibration folder under `root` that was active at `when`.

    That is the backup with the earliest end after `when`, or the active folder when no
    backup ended after it. Refuses when two backups end at the same time, since which one
    was active is then unknown, and when no folder qualifies. Refuses too when that folder's
    calibration was made after `when` (calibration_created): `when` is older than every kept
    calibration, the one in force then is gone, and the oldest kept one is no stand-in for
    it (spec, Old sessions). A folder that cannot be dated is warned about and returned; it
    fails when it is loaded.

    The library's own record comes first: when decisions.jsonl shows an entry made active at
    or before `when`, that entry is the one in force. Earlier than its first set_active, the
    folders above decide, as they did before the library.
    """
    from reachaq.inference.multires_calibration import library  # library imports this module

    root = Path(root)
    if when.tzinfo is not None:
        # The backup suffixes are local naive times; compare in the same terms.
        when = when.astimezone().replace(tzinfo=None)
    library_id = library.active_at(library.library_root(root), when)
    if library_id is not None:
        folder = library.entry_path(library.library_root(root), library_id)
        _check_made_by(folder, when)
        return folder
    candidates = calibration_candidates(root)
    if not candidates:
        raise CalibrationResolutionError(f"no calibration folder under {root}")
    backups = [c for c in candidates if c.active_until is not None]
    by_end: Dict[datetime, List[str]] = defaultdict(list)
    for c in backups:
        by_end[c.active_until].append(c.path.name)
    shared = sorted((end, names) for end, names in by_end.items() if len(names) > 1)
    if shared:
        detail = "; ".join(f"{' and '.join(names)} both end at {end}" for end, names in shared)
        raise CalibrationResolutionError(f"ambiguous calibration backups under {root}: {detail}")
    later = [c for c in backups if c.active_until > when]
    active = [c for c in candidates if c.active_until is None]
    if later:
        folder = min(later, key=lambda c: c.active_until).path
    elif active:
        folder = active[0].path
    else:
        raise CalibrationResolutionError(
            f"no calibration under {root} was active at {when}: every backup ended earlier and there is no"
            " active folder"
        )
    _check_made_by(folder, when)
    return folder


def _check_made_by(folder: Path, when: datetime) -> None:
    """Refuse `folder` for a time `when` (local, naive) before its calibration was made."""
    try:
        created = calibration_created(folder)
    except (OSError, ValueError) as error:
        logger.warning("cannot tell when the calibration in %s was made, so not whether it was in force at %s: %s",
                       folder, when, error)
        return
    if created.tzinfo is not None:
        created = created.astimezone().replace(tzinfo=None)
    if when < created:
        raise CalibrationResolutionError(
            f"no kept calibration under {folder.parent} was in force at {when}: {folder.name}, the one active"
            f" then by its end time, was made later, at {created}"
        )


def _write_preset(
    out: Path,
    fit: MotorFrameFit,
    preset_set: PresetSet,
    signs: Tuple[int, int, int],
    *,
    calibration_id: str,
    cameras: Sequence[CaptureCamera],
) -> None:
    """The anipose file and README of `preset_set`, and the left-right extras when left and
    right share a geometry in it."""
    out.mkdir(parents=True)
    _write_text(out / _ANIPOSE_FILE, anipose_toml(fit, preset_set, calibration_id=calibration_id, cameras=cameras))
    geometries = "\n".join(
        f"{c.name}: binning {g.binning}, {g.width}x{g.height} px, ROI offset ({g.offset_x}, {g.offset_y}) in binned px."
        for c in cameras
        for g in [preset_set.cameras.get(c.name)]
        if g is not None
    )
    readme = _README.format(name=preset_set.name, geometries=geometries)
    g = stereo_pair_geometry(preset_set.cameras)
    if g is not None:
        matrix_dir = out / _CAMERA_MATRIX_DIR
        matrix_dir.mkdir()
        stereo = preset_stereo_params(fit, g)
        write_pickle(matrix_dir / _STEREO_FILE, stereo.as_pickle_dict())
        for i, cam in enumerate(_CAMERAS, start=1):
            # The same arrays as the stereo file's, so the two can never disagree.
            intrinsics = {"mtx": stereo.matrix[f"cameraMatrix{i}"], "dist": stereo.matrix[f"distCoeffs{i}"]}
            write_pickle(matrix_dir / f"{cam}_intrinsic_params.pickle", {cam: intrinsics})
        frame = {
            "transform": frame_transform(fit, stereo, signs).tolist(),
            "units": "mm",
            "signs": [int(s) for s in signs],
        }
        _write_text(out / _FRAME_FILE, _yaml_text(frame))
        first, second = _CAMERAS
        readme += _README_DLC.format(stereo_key=STEREO_KEY, first=first, second=second)
    _write_text(out / _README_FILE, readme)


def _sensor_set(fit: MotorFrameFit, cameras: Sequence[str]) -> PresetSet:
    """Every camera at full sensor resolution, bin 1: the fit's own pixels."""
    width, height = (int(v) for v in fit.image_size)
    return PresetSet("sensor", {cam: PresetGeometry(1, width, height, 0, 0) for cam in cameras})


def _pair_geometries(sets: Sequence[PresetSet]) -> Tuple[PresetGeometry, ...]:
    """The geometries left and right share in `sets`, each once, in set order."""
    found: List[PresetGeometry] = []
    for s in sets:
        g = stereo_pair_geometry(s.cameras)
        if g is not None and g not in found:
            found.append(g)
    return tuple(found)


def _stereo_set_name(index: CalibrationIndex, geometry: PresetGeometry) -> str:
    """The first preset set holding left and right both at `geometry`: its folder has the extras."""
    for s in index.preset_sets:
        cameras = {cam: _geometry_from_plain(g) for cam, g in s["cameras"].items()}
        if stereo_pair_geometry(cameras) == geometry:
            return s["name"]
    raise CalibrationGeometryError(f"no preset set holds left and right at {_describe(geometry)}")


def _geometry_to_plain(g: PresetGeometry) -> dict:
    return {
        "binning": int(g.binning),
        "width": int(g.width),
        "height": int(g.height),
        "offset_x": int(g.offset_x),
        "offset_y": int(g.offset_y),
    }


def _geometry_from_plain(p: Mapping) -> PresetGeometry:
    return PresetGeometry(int(p["binning"]), int(p["width"]), int(p["height"]), int(p["offset_x"]), int(p["offset_y"]))


def _preset_set_to_plain(s: PresetSet) -> dict:
    return {"name": str(s.name), "cameras": {str(cam): _geometry_to_plain(g) for cam, g in s.cameras.items()}}


def _is_within(path: Path, folder: Path) -> bool:
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def _index_to_plain(index: CalibrationIndex) -> dict:
    return {
        "format": index.format,
        "id": index.id,
        # A string, so the file reads the same whatever YAML loader opens it.
        "created": index.created.isoformat(timespec="seconds"),
        "board": {
            "square_mm": float(index.board.square_mm),
            "rows": int(index.board.rows),
            "cols": int(index.board.cols),
        },
        "cameras": [dict(c) for c in index.cameras],
        "preset_sets": [dict(s) for s in index.preset_sets],
        "verdict": index.verdict,
        "override": index.override,
        "motor_scale": index.motor_scale,
        "capture": index.capture,
    }


def _fit_to_plain(fit: MotorFrameFit) -> dict:
    """The fit as lists and floats; tolist keeps each array's shape, so it reads back exactly."""
    return {
        "image_size": [int(v) for v in fit.image_size],
        "motor_scale": float(fit.motor_scale),
        "board_rvec": np.asarray(fit.board_rvec, float).tolist(),
        "board_tvec": np.asarray(fit.board_tvec, float).tolist(),
        "cameras": {
            cam: {field: np.asarray(getattr(m, field), float).tolist() for field in ("K", "dist", "rvec", "tvec")}
            for cam, m in fit.cameras.items()
        },
        "rms_px": {cam: float(v) for cam, v in fit.rms_px.items()},
        "per_view_rms_px": {
            cam: {int(i): float(v) for i, v in views.items()} for cam, views in fit.per_view_rms_px.items()
        },
        # Undetermined parameters are inf, which YAML writes as .inf and reads back.
        "param_std": {cam: {name: float(v) for name, v in std.items()} for cam, std in fit.param_std.items()},
    }


def _index_data(folder: Path) -> Optional[dict]:
    """The raw index of `folder`, or None when it has no calibration.yaml or one at a format not in READABLE_FORMATS.

    Raises ValueError when the file is not a YAML mapping, and OSError when it cannot be read.
    """
    path = folder / INDEX_FILE
    if not path.is_file():
        return None
    try:
        data = _read_yaml(path)
    except yaml.YAMLError as error:
        raise ValueError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a mapping")
    if data.get("format") not in READABLE_FORMATS:
        return None
    return data


def _parse_index(folder: Path) -> Optional[CalibrationIndex]:
    """The index of `folder`, or None when it has no calibration.yaml at format 2 or 3.

    Raises ValueError when the index is not valid YAML or a field is missing or garbled.
    """
    data = _index_data(folder)
    if data is None:
        return None
    try:
        return _index_from_data(data)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{folder / INDEX_FILE} is corrupt: {type(error).__name__}: {error}") from error


def _index_or_none(folder: Path) -> Optional[CalibrationIndex]:
    """_parse_index, with an unreadable or corrupt index logged and taken as no index.

    For the predicate and the candidate listing: one corrupt backup must not stop every
    other calibration from being found. It keeps its place in time by its folder name.
    """
    try:
        return _parse_index(folder)
    except (OSError, ValueError) as error:
        logger.warning("%s is not treated as a multi-resolution calibration: %s", folder, error)
        return None


def _calibration_id(folder: Path) -> Optional[str]:
    index = _index_or_none(folder)
    return None if index is None else index.id


def _describe(g: PresetGeometry) -> str:
    return f"{g.name} at ROI offset ({g.offset_x}, {g.offset_y})"


def _plain(value):
    """`value` with numpy values and tuples turned into the Python types yaml.safe_dump writes."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _yaml_text(data) -> str:
    # Flow style for the innermost lists keeps a matrix row on one line.
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None)


def _read_yaml(path: Path):
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _replace_text(path: Path, text: str) -> None:
    """Replace `path` whole, so a crash leaves the old file or the new one, never a mix.

    The temporary file's name is unique (<file>.<uuid4 hex>.tmp): the derived cache is shared
    by the live app and offline processes, which may write the same file at once.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
