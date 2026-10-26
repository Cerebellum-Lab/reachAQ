"""Store a multi-resolution calibration on disk, read it back, and find the one in force at a time.

One folder per calibration, at format 2:

    calibration.yaml        the index, written last: format, id, created, board, presets,
                            verdict, override, motor_scale and capture
    sensor/fit.yaml         the full motor-frame fit in sensor pixels, as plain lists
    presets/<g.name>/       one per preset, e.g. presets/256x256_bin4/
        camera_matrix/stereo_params.pickle            today's keys, so today's loaders read it
        camera_matrix/{left,right}_intrinsic_params.pickle   {cam: {"mtx", "dist"}}
        frame.yaml          the 3D frame: oriented = (raw - offsets) @ transform, in mm
        calibration.toml    the anipose export
        README.txt          how DeepLabCut 3D and anipose use the folder
    quality_report.json, quality_report.txt and any extra files (the coverage images)

Each preset folder is a DeepLabCut 3D camera_matrix folder: DLC's triangulate reads only
stereo_params.pickle and keys the pair by its project's camera names, so the project must
name its cameras "left" and "right" to match STEREO_KEY.

A half-written folder never loads. The folder is written as <name>.partial and renamed into
place only when complete, and calibration.yaml is the last file written into it. A folder
without calibration.yaml at format 2 is a legacy, single-geometry calibration, which is
loaded the old way.

Times are local and naive. Installing a calibration renames the active folder <name> to
<name>_<datetime.now() in DATE_TIME_FORMAT>, so a backup's suffix is the local time at which
it stopped being active, and `created` and a legacy folder's date use the same convention so
the two compare directly. The calibration in force at time t is the backup with the earliest
suffix after t, or the active folder when no backup ended after t.
"""

import logging
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.project.project_info import DATE_TIME_FORMAT
from autotrainer.inference.calibration_FLIR import write_pickle
from autotrainer.inference.config import StereoParams, load_calib_stereo_params
from autotrainer.inference.multires_calibration.fit import BoardSpec, CameraModel, MotorFrameFit
from autotrainer.inference.multires_calibration.geometry import PresetGeometry
from autotrainer.inference.multires_calibration.products import (
    STEREO_KEY,
    anipose_toml,
    frame_transform,
    motor_to_inference_signs,
    preset_stereo_params,
)
from autotrainer.inference.multires_calibration.report import QualityReport

logger = get_verbose_logger(__name__)

CALIBRATION_FORMAT = 2
INDEX_FILE = "calibration.yaml"

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
{name}: one preset of a reachAQ multi-resolution 3D calibration.
Binning {binning}, {width}x{height} px, ROI offset ({offset_x}, {offset_y}) in binned px.
Every pixel value here is in this preset's pixels: it applies only to video recorded
at this binning and ROI.

DeepLabCut 3D: camera_matrix/ is a DLC camera_matrix folder. Copy it to the 3D
project's camera_matrix folder. stereo_params.pickle keys the pair as "{stereo_key}",
so the project's config.yaml must set camera_names: ["{first}", "{second}"].

anipose: calibration.toml, in aniposelib's CameraGroup format, with the cameras'
poses in the motor frame and translations in mm.
"""


class CalibrationGeometryError(LookupError):
    """The calibration has no preset for the requested camera geometry."""


class CalibrationResolutionError(LookupError):
    """No single calibration matches the requested id or time."""


@dataclass(frozen=True)
class CalibrationIndex:
    id: str  # uuid4 hex
    created: datetime  # local, naive, to the second
    board: BoardSpec
    presets: Tuple[PresetGeometry, ...]
    verdict: str
    override: Optional[dict]
    motor_scale: float
    capture: dict


@dataclass(frozen=True)
class LoadedPreset:
    geometry: PresetGeometry
    stereo_params: StereoParams
    frame_transform: np.ndarray


@dataclass(frozen=True)
class CalibrationCandidate:
    path: Path
    active_until: Optional[datetime]  # None for the active folder
    calibration_id: Optional[str]  # None for a legacy folder


def write_calibration(
    folder: Path,
    *,
    fit: MotorFrameFit,
    presets: Sequence[PresetGeometry],
    board: BoardSpec,
    report: QualityReport,
    capture: dict,
    extra_files: Optional[Mapping[str, Path]] = None,
) -> CalibrationIndex:
    """Write the calibration into `folder`, which must not exist yet, and return its index.

    `extra_files` maps a file name in the folder to a file to copy there, for the coverage
    images, so they too are in place before the index is written.
    """
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} already exists; a calibration folder is never overwritten")
    presets = tuple(presets)
    names = [g.name for g in presets]
    if not presets or len(set(names)) != len(names):
        raise ValueError(f"expected presets with distinct names, got {names}")

    index = CalibrationIndex(
        id=uuid.uuid4().hex,
        created=datetime.now().replace(microsecond=0),
        board=board,
        presets=presets,
        verdict=report.verdict,
        override=_plain(report.override),
        motor_scale=float(fit.motor_scale),
        capture=_plain(dict(capture)),
    )
    # Serialised before anything is written, so a capture value YAML cannot hold fails here.
    index_text = _yaml_text(_index_to_plain(index))

    partial = folder.with_name(folder.name + _PARTIAL_SUFFIX)
    if partial.exists():
        logger.warning("removing %s, left by an interrupted calibration write", partial)
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    _write_text(partial / _SENSOR_FIT_FILE, _yaml_text(_fit_to_plain(fit)))
    signs = motor_to_inference_signs()
    for g in presets:
        _write_preset(partial / _PRESETS_DIR / g.name, fit, g, signs)
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
    return index


def is_multires_calibration(folder: Path) -> bool:
    """True only when `folder` holds a readable calibration.yaml at format 2.

    An unreadable or corrupt index logs a warning and gives False; the loaders raise on it.
    """
    return _index_or_none(Path(folder)) is not None


def load_index(folder: Path) -> CalibrationIndex:
    """The index of `folder`. Raises ValueError when it has none at format 2 or it is corrupt."""
    folder = Path(folder)
    index = _parse_index(folder)
    if index is None:
        raise ValueError(
            f"{folder} is not a multi-resolution calibration: it has no {INDEX_FILE} at format {CALIBRATION_FORMAT}"
        )
    return index


def _index_from_data(data: dict) -> CalibrationIndex:
    calibration_id = data["id"]
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError(f"id must be a non-empty string, got {calibration_id!r}")
    created = data["created"]
    if not isinstance(created, datetime):
        created = datetime.fromisoformat(created)
    board = data["board"]
    return CalibrationIndex(
        id=calibration_id,
        created=created,
        board=BoardSpec(square_mm=float(board["square_mm"]), rows=int(board["rows"]), cols=int(board["cols"])),
        presets=tuple(
            PresetGeometry(int(p["binning"]), int(p["width"]), int(p["height"]), int(p["offset_x"]), int(p["offset_y"]))
            for p in data["presets"]
        ),
        verdict=str(data["verdict"]),
        override=data.get("override"),
        motor_scale=float(data["motor_scale"]),
        capture=dict(data.get("capture") or {}),
    )


def load_sensor_fit(folder: Path) -> MotorFrameFit:
    folder = Path(folder)
    load_index(folder)
    data = _read_yaml(folder / _SENSOR_FIT_FILE)
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
    """The stereo parameters and 3D frame of the preset whose geometry equals `geometry`.

    Raises CalibrationGeometryError when no preset matches: a different preset is never
    substituted, since its pixels would not be the frames' pixels.
    """
    folder = Path(folder)
    index = load_index(folder)
    if geometry not in index.presets:
        available = "; ".join(_describe(g) for g in index.presets)
        raise CalibrationGeometryError(
            f"the calibration in {folder} has no preset for {_describe(geometry)}; it has {available}"
        )
    preset_dir = folder / _PRESETS_DIR / geometry.name
    stereo = load_calib_stereo_params(preset_dir / _CAMERA_MATRIX_DIR / _STEREO_FILE)
    transform = np.array(_read_yaml(preset_dir / _FRAME_FILE)["transform"], float)
    if transform.shape != (3, 3):
        raise ValueError(f"{preset_dir / _FRAME_FILE}: expected a 3x3 transform, got shape {transform.shape}")
    return LoadedPreset(geometry=geometry, stereo_params=stereo, frame_transform=transform)


def calibration_created(folder: Path) -> datetime:
    """When the calibration in `folder` was made, as a local naive time to the second.

    A legacy folder records no date, so its stereo_params.pickle's mtime stands in: the
    legacy install copies the folder with shutil.copytree, which keeps file times. A corrupt
    index raises rather than falling back to a pickle a format-2 folder does not have.
    """
    folder = Path(folder)
    index = _parse_index(folder)
    if index is not None:
        return index.created
    mtime = (folder / _CAMERA_MATRIX_DIR / _STEREO_FILE).stat().st_mtime
    return datetime.fromtimestamp(mtime).replace(microsecond=0)


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
    """The folder under `root` whose calibration has id `calibration_id`."""
    matches = [c.path for c in calibration_candidates(root) if c.calibration_id == calibration_id]
    if not matches:
        raise CalibrationResolutionError(f"no calibration under {root} has id {calibration_id}")
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
    """
    root = Path(root)
    if when.tzinfo is not None:
        # The backup suffixes are local naive times; compare in the same terms.
        when = when.astimezone().replace(tzinfo=None)
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


def record_override(folder: Path, override: dict) -> None:
    """Record the operator's override in calibration.yaml and in the quality report.

    Both new files are built before either is written, so an override JSON cannot hold
    changes nothing. Each file is replaced whole, and the index last.
    """
    folder = Path(folder)
    load_index(folder)  # refuses a folder that is not an intact calibration
    index_data = _index_data(folder)
    report = QualityReport.from_json((folder / _REPORT_JSON).read_text(encoding="utf-8"))
    report.override = dict(override)
    report_json = report.to_json()
    index_data["override"] = _plain(dict(override))
    index_text = _yaml_text(index_data)

    _replace_text(folder / _REPORT_JSON, report_json)
    _replace_text(folder / _REPORT_TEXT, report.to_text() + "\n")
    _replace_text(folder / INDEX_FILE, index_text)


def _write_preset(out: Path, fit: MotorFrameFit, g: PresetGeometry, signs: Tuple[int, int, int]) -> None:
    matrix_dir = out / _CAMERA_MATRIX_DIR
    matrix_dir.mkdir(parents=True)
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
    _write_text(out / _ANIPOSE_FILE, anipose_toml(fit, g))
    first, second = _CAMERAS
    readme = _README.format(
        name=g.name, binning=g.binning, width=g.width, height=g.height, offset_x=g.offset_x, offset_y=g.offset_y,
        stereo_key=STEREO_KEY, first=first, second=second,
    )
    _write_text(out / _README_FILE, readme)


def _index_to_plain(index: CalibrationIndex) -> dict:
    return {
        "format": CALIBRATION_FORMAT,
        "id": index.id,
        # A string, so the file reads the same whatever YAML loader opens it.
        "created": index.created.isoformat(timespec="seconds"),
        "board": {
            "square_mm": float(index.board.square_mm),
            "rows": int(index.board.rows),
            "cols": int(index.board.cols),
        },
        "presets": [
            {
                "name": g.name,
                "binning": int(g.binning),
                "width": int(g.width),
                "height": int(g.height),
                "offset_x": int(g.offset_x),
                "offset_y": int(g.offset_y),
            }
            for g in index.presets
        ],
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
    """The raw index of `folder`, or None when it has no calibration.yaml or one at another format.

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
    if data.get("format") != CALIBRATION_FORMAT:
        return None
    return data


def _parse_index(folder: Path) -> Optional[CalibrationIndex]:
    """The index of `folder`, or None when it has no calibration.yaml at format 2.

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
    """Replace `path` whole, so a crash leaves the old file or the new one, never a mix."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
