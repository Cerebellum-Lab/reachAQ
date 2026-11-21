"""The calibration library: every calibration kept as an immutable entry, one of them active.

Under <config>/calibrations/ (library_root):

    <id>/               one calibration, at format 3 (or a copy of a format-2 or legacy folder)
    active.yaml         the one mutable pointer: {id, by, at}
    decisions.jsonl     append-only: one JSON object per line, {event, id, at, ...}
    settings.yaml       library-wide switches; this module only names the file (settings_path)
    .derived/           the cache load_preset_set derives other geometries into, outside any entry
    .offsets/<id>/      the camera offsets DCS measured against entry <id> (offsets_dir)

A session names its calibration by id, so an entry never changes once it is in place: it is
copied into <id>.partial, its files and folders are flushed to disk, and it is renamed to
<id>; nothing here writes inside it afterwards. What happens to an entry afterwards (made
active, installed over a failed quality check) is a line in decisions.jsonl, not an edit of
the entry. The pointer is replaced whole (temp file, flushed, then os.replace), and
decisions are appended with one write per line. A decision that cannot be appended after
the pointer moved puts the previous pointer back, so the two never disagree.

A reader never raises on the pointer: active_id returns None, and says why in the log, when
active.yaml is missing, corrupt or names an entry that is not there.

import_existing brings in the folders that predate the library: the active 4mm_6r_8c_4x
folder and its dated backups. A legacy folder (no calibration.yaml) gets a generated
calibration.yaml at format "legacy", with an id made from its content (the sha256 of its
stereo_params.pickle and that file's mtime_ns, not the folder's name, so a folder renamed to
a dated backup is the same calibration) and the pickle's time as `created`; the file is
written before the entry is renamed into place. The old folders are only read.

Times are local and naive, as in store.py.
"""

import filecmp
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.multires_calibration.store import (
    INDEX_FILE,
    LEGACY_FORMAT,
    CalibrationResolutionError,
    calibration_candidates,
    load_index,
)

logger = get_verbose_logger(__name__)

LIBRARY_DIR_NAME = "calibrations"
ACTIVE_FILE = "active.yaml"
DECISIONS_FILE = "decisions.jsonl"
SETTINGS_FILE = "settings.yaml"
DERIVED_DIR_NAME = ".derived"
OFFSETS_DIR_NAME = ".offsets"

_PARTIAL_SUFFIX = ".partial"
_LEGACY_PICKLE = Path("camera_matrix", "stereo_params.pickle")
_LEGACY_NAMESPACE = uuid.UUID("6a1f3c52-8f0e-4c63-9a5d-2f3b7c1d9e40")


@dataclass(frozen=True)
class LibraryEntry:
    id: str
    created: datetime  # local, naive, to the second
    cameras: List[str]  # every capture camera's name, in capture order
    verdict: str
    active: bool
    uncalibrated: Dict[str, str]  # a 2D-only camera's name -> the reason
    path: Path


def library_root(config_dir: Path) -> Path:
    return Path(config_dir) / LIBRARY_DIR_NAME


def settings_path(root: Path) -> Path:
    return Path(root) / SETTINGS_FILE


def read_settings(root: Path) -> dict:
    """The library's settings.yaml as a dict; {} when it is missing, empty or corrupt. Never raises."""
    path = settings_path(root)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        logger.warning("ignoring the unreadable library settings %s: %s", path, err)
        return {}
    return dict(data) if isinstance(data, dict) else {}


def update_settings(root: Path, **changes) -> dict:
    """Set `changes` in settings.yaml, keeping every other key, and return the settings now stored.

    The file is replaced whole (temp file, then os.replace). A corrupt file is replaced by
    the changes alone.
    """
    settings = read_settings(root)
    settings.update(changes)
    _replace_text(settings_path(root), yaml.safe_dump(settings, sort_keys=False))
    return settings


def derived_root(root: Path) -> Path:
    """Where load_preset_set derives geometries no stored set has; never inside an entry."""
    return Path(root) / DERIVED_DIR_NAME


def offsets_dir(root: Path, calibration_id: str) -> Path:
    """Where the camera offsets DCS measured against entry `calibration_id` are kept, beside it."""
    return Path(root) / OFFSETS_DIR_NAME / calibration_id


def entry_library(folder: Path) -> Optional[Path]:
    """The library `folder` is an entry of, or None when it is not a library entry."""
    folder = Path(folder)
    if folder.parent.name == LIBRARY_DIR_NAME and _is_entry_name(folder.name) and (folder / INDEX_FILE).is_file():
        return folder.parent
    return None


def entry_path(root: Path, calibration_id: str) -> Path:
    """The folder of entry `calibration_id`. Raises CalibrationResolutionError when there is
    none, or when its calibration.yaml is empty (a write a power cut cut short)."""
    root = Path(root)
    if not _is_entry_name(calibration_id):
        raise CalibrationResolutionError(f"{calibration_id!r} is not a calibration id")
    path = root / calibration_id
    index = path / INDEX_FILE
    if not index.is_file():
        raise CalibrationResolutionError(f"the calibration library {root} has no calibration {calibration_id}")
    try:
        empty = index.stat().st_size == 0
    except OSError as error:
        raise CalibrationResolutionError(f"calibration {calibration_id} in {root} cannot be read: {error}") from error
    if empty:
        raise CalibrationResolutionError(f"calibration {calibration_id} in {root} has an empty {INDEX_FILE}")
    return path


def has_entries(root: Path) -> bool:
    """Whether the library holds any entry (a folder with a calibration.yaml), readable or not."""
    root = Path(root)
    try:
        children = list(root.iterdir()) if root.is_dir() else []
    except OSError:
        return False
    return any(c.is_dir() and _is_entry_name(c.name) and (c / INDEX_FILE).is_file() for c in children)


def add_entry(root: Path, built_folder: Path) -> str:
    """Copy the calibration in `built_folder` into the library and return its id.

    `built_folder` is a calibration already written at format 2 or 3 (store.write_calibration).
    It is copied to <id>.partial, flushed to disk, and renamed to <id>, so a power cut leaves
    no entry that loads, and it is left as it is. An entry with that id whose files are
    byte-identical is the same calibration added again (a retry after a crash before it was
    made active): it is kept and its id returned. Raises FileExistsError when the library
    has that id with other content: an entry is never replaced.
    """
    root = Path(root)
    calibration_id = load_index(built_folder).id
    if not _is_entry_name(calibration_id):
        raise ValueError(f"{built_folder} has an id that cannot name a folder: {calibration_id!r}")
    root.mkdir(parents=True, exist_ok=True)
    target = root / calibration_id
    if target.exists():
        if _same_tree(Path(built_folder), target):
            logger.info("calibration %s is already in the library %s with the same content; kept",
                        calibration_id, root)
            return calibration_id
        raise FileExistsError(f"the calibration library {root} already has calibration {calibration_id}")
    partial = _fresh_partial(root, calibration_id)
    try:
        shutil.copytree(built_folder, partial)
        _fsync_tree(partial)
        os.replace(partial, target)
        _fsync_dir(root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    logger.info("added calibration %s to the library %s", calibration_id, root)
    return calibration_id


def discard_entry(root: Path, calibration_id: str) -> None:
    """Delete an entry that was just added and never became active, to undo a failed install.

    Refuses the entry active.yaml names. Not for entries a session may already name.
    """
    path = entry_path(root, calibration_id)
    # Read quietly: a first install has no active.yaml, which is no reason to warn here.
    if _pointer_id(Path(root)) == calibration_id:
        raise ValueError(f"calibration {calibration_id} is the active one and is not discarded")
    shutil.rmtree(path)
    logger.warning("discarded calibration %s from the library %s", calibration_id, root)


def set_active(root: Path, calibration_id: str, *, by: str) -> None:
    """Make `calibration_id` the active calibration, and record who did it.

    active.yaml is written to a temp file and renamed over the old one, so a failure leaves
    the old pointer; the decision is appended only after the pointer is in place. When the
    append fails, the previous active.yaml is put back whole (or the new one removed when
    there was none) and the append's error raised, so the pointer never names an entry
    whose activation is not recorded.
    """
    root = Path(root)
    entry_path(root, calibration_id)
    pointer = root / ACTIVE_FILE
    try:
        previous: Optional[bytes] = pointer.read_bytes()
    except FileNotFoundError:
        previous = None
    at = _now()
    text = yaml.safe_dump({"id": calibration_id, "by": str(by), "at": at}, sort_keys=False)
    _replace_text(pointer, text)
    try:
        _append_decision(root, {"event": "set_active", "id": calibration_id, "by": str(by), "at": at})
    except BaseException:
        try:
            if previous is None:
                pointer.unlink(missing_ok=True)
            else:
                _replace_bytes(pointer, previous)
        except Exception:
            logger.exception("could not put back the previous %s after the decision for %s failed",
                             pointer, calibration_id)
        raise
    logger.info("calibration %s is now active in %s", calibration_id, root)


def record_decision(root: Path, calibration_id: str, kind: str, detail: dict) -> None:
    """Append {"event": kind, "id": calibration_id, "at": now, "detail": detail} to decisions.jsonl."""
    root = Path(root)
    entry_path(root, calibration_id)
    _append_decision(root, {"event": str(kind), "id": calibration_id, "at": _now(), "detail": detail})


def active_id(root: Path) -> Optional[str]:
    """The active calibration's id, or None.

    None when active.yaml is missing, is not a mapping with an id, or names an id that has no
    entry; each logs a warning saying which. Never raises.
    """
    root = Path(root)
    path = root / ACTIVE_FILE
    try:
        if not path.is_file():
            logger.warning("no active calibration: %s does not exist", path)
            return None
        calibration_id = _pointer_id(root, quiet=False)
        if calibration_id is None:
            logger.warning("no active calibration: %s does not name a calibration id", path)
            return None
        entry_path(root, calibration_id)
        return calibration_id
    except CalibrationResolutionError as error:
        logger.warning("no active calibration: %s names a calibration that is not in the library: %s", path, error)
    except Exception as error:
        # A YAML timestamp out of range (at: 2026-13-45) raises ValueError, binary garbage a
        # decode error; whatever it is, the pointer is unreadable, never a crash at start.
        logger.warning("no active calibration: %s cannot be read: %s: %s", path, type(error).__name__, error)
    return None


def pointer_id(root: Path) -> Optional[str]:
    """The id active.yaml names, whether or not that entry is usable; None when it names none
    or cannot be read. Never raises, and logs nothing (active_id says why)."""
    return _pointer_id(root)


def _pointer_id(root: Path, *, quiet: bool = True) -> Optional[str]:
    """The id active.yaml names, unchecked; None when it names none. With `quiet` (the
    default) any read error is None too; otherwise it raises."""
    try:
        data = yaml.safe_load((Path(root) / ACTIVE_FILE).read_text(encoding="utf-8"))
    except Exception:
        if quiet:
            return None
        raise
    calibration_id = data.get("id") if isinstance(data, dict) else None
    return calibration_id if isinstance(calibration_id, str) and calibration_id else None


def list_entries(root: Path) -> List[LibraryEntry]:
    """Every entry in the library, newest first.

    Skips .derived, any name starting with "." and any ending ".partial". An entry that
    cannot be read is warned about and left out.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    active = active_id(root) if (root / ACTIVE_FILE).exists() else None
    entries = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not _is_entry_name(child.name):
            continue
        try:
            entries.append(_read_entry(child, active))
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("skipping %s in the calibration library: %s", child, error)
    entries.sort(key=lambda e: (e.created, e.id), reverse=True)
    return entries


def is_legacy_entry(folder: Path) -> bool:
    """True when `folder` holds the generated calibration.yaml of an imported legacy calibration."""
    try:
        data = yaml.safe_load((Path(folder) / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and data.get("format") == LEGACY_FORMAT


def legacy_folder_name(folder: Path) -> str:
    """The name a legacy calibration's board is read from (calibration_FLIR.get_calibration_info):
    the folder it was imported from for a library entry, whose own name is its id; else its own."""
    folder = Path(folder)
    if is_legacy_entry(folder):
        try:
            source = yaml.safe_load((folder / INDEX_FILE).read_text(encoding="utf-8")).get("source")
        except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError):
            source = None
        if isinstance(source, str) and source:
            return source
    return folder.name


def active_at(root: Path, when: datetime) -> Optional[str]:
    """The entry that was active at `when` (local, naive): the last set_active at or before it.

    None when the log has no such line or the entry is gone. Unreadable lines are skipped.
    """
    root = Path(root)
    try:
        lines = (root / DECISIONS_FILE).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    best: Optional[Tuple[datetime, str]] = None
    for line in lines:
        try:
            record = json.loads(line)
            if record["event"] != "set_active":
                continue
            at = datetime.fromisoformat(record["at"])
            calibration_id = str(record["id"])
        except (ValueError, KeyError, TypeError):
            continue
        if at <= when and (best is None or at >= best[0]):
            best = (at, calibration_id)
    if best is None:
        return None
    try:
        entry_path(root, best[1])
    except CalibrationResolutionError:
        return None
    return best[1]


def latest_decision(root: Path, calibration_id: str, kind: str) -> Optional[dict]:
    """The detail of the last `kind` decision recorded for `calibration_id`, or None.

    None too when decisions.jsonl cannot be read. Unreadable lines are skipped.
    """
    try:
        lines = (Path(root) / DECISIONS_FILE).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    found = None
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("event") == kind and record.get("id") == calibration_id:
            detail = record.get("detail")
            found = detail if isinstance(detail, dict) else {}
    return found


def import_existing(config_dir: Path) -> List[str]:
    """Copy the 4mm_6r_8c_4x folder and every dated backup under `config_dir` into the library.

    Returns the ids this call added, oldest first; an id already in the library is skipped,
    so a second call returns []. A folder with a format-2 index keeps its id; a legacy folder
    gets an id made from its content (legacy_id), so it is imported once even after it is
    renamed to a dated backup, and the pickle's time as `created`. Each imported entry gets
    one "import" decision: one whose decision could not be written (or was imported without
    one) gets it on the next call. A folder that cannot be imported is warned about and left
    out. When the library has no readable active pointer, the active folder's entry becomes
    it. The old folders are not changed.
    """
    config_dir = Path(config_dir)
    root = library_root(config_dir)
    added: List[str] = []
    active_source: Optional[str] = None
    for candidate in calibration_candidates(config_dir):
        folder = candidate.path
        try:
            calibration_id, is_new = _import_folder(root, candidate.calibration_id, folder)
        except (OSError, ValueError) as error:
            logger.warning("not importing %s into the calibration library: %s", folder, error)
            continue
        if calibration_id is None:
            continue
        if latest_decision(root, calibration_id, "import") is None:
            try:
                record_decision(root, calibration_id, "import", {"source": folder.name})
            except (OSError, CalibrationResolutionError) as error:
                logger.warning("imported %s as %s, but its import decision could not be written; it is"
                               " retried on the next import: %s", folder, calibration_id, error)
        if candidate.active_until is None:
            active_source = calibration_id
        if is_new:
            added.append(calibration_id)
    if active_source is not None and active_id(root) is None:
        set_active(root, active_source, by="import")
    return added


def _import_folder(root: Path, known_id: Optional[str], folder: Path) -> Tuple[Optional[str], bool]:
    """(id, added) for one existing folder; (None, False) when it is not a calibration."""
    if known_id is not None:
        if (root / known_id).exists():
            return known_id, False
        return add_entry(root, folder), True
    if (folder / INDEX_FILE).exists():
        logger.warning("not importing %s: its %s is unreadable or at a format this version does not know",
                       folder, INDEX_FILE)
        return None, False
    pickle_path = folder / _LEGACY_PICKLE
    if not pickle_path.is_file():
        logger.warning("not importing %s: it has no %s", folder, _LEGACY_PICKLE.as_posix())
        return None, False
    calibration_id = legacy_id(pickle_path)
    if (root / calibration_id).exists():
        return calibration_id, False
    created = datetime.fromtimestamp(pickle_path.stat().st_mtime).replace(microsecond=0)
    root.mkdir(parents=True, exist_ok=True)
    partial = _fresh_partial(root, calibration_id)
    try:
        shutil.copytree(folder, partial)
        marker = {"format": LEGACY_FORMAT, "id": calibration_id, "created": created.isoformat(timespec="seconds"),
                  "source": folder.name}
        _replace_text(partial / INDEX_FILE, yaml.safe_dump(marker, sort_keys=False))
        _fsync_tree(partial)
        os.replace(partial, root / calibration_id)
        _fsync_dir(root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    logger.info("imported the legacy calibration %s as %s", folder, calibration_id)
    return calibration_id, True


def legacy_id(pickle_path: Path) -> str:
    """The library id of a legacy calibration: a uuid5 over the sha256 of its
    camera_matrix/stereo_params.pickle and that file's mtime_ns. The folder's name is not in
    it, so the active folder and the dated backup it is later renamed to are one calibration."""
    pickle_path = Path(pickle_path)
    digest = hashlib.sha256(pickle_path.read_bytes()).hexdigest()
    return uuid.uuid5(_LEGACY_NAMESPACE, f"{digest}|{pickle_path.stat().st_mtime_ns}").hex


def _read_entry(path: Path, active: Optional[str]) -> LibraryEntry:
    try:
        index = load_index(path)
    except ValueError:
        if not is_legacy_entry(path):
            raise
        data = yaml.safe_load((path / INDEX_FILE).read_text(encoding="utf-8"))
        created = data["created"]
        created = created if isinstance(created, datetime) else datetime.fromisoformat(created)
        calibration_id, cameras, verdict, uncalibrated = str(data["id"]), ["left", "right"], "unknown", {}
    else:
        calibration_id, created, verdict = index.id, index.created, index.verdict
        cameras = [c["name"] for c in index.cameras]
        uncalibrated = {c["name"]: c["reason"] for c in index.cameras if not c["calibrated"]}
    if calibration_id != path.name:
        raise ValueError(f"the folder is named {path.name} but its calibration id is {calibration_id}")
    return LibraryEntry(
        id=calibration_id, created=created, cameras=cameras, verdict=verdict,
        active=calibration_id == active, uncalibrated=uncalibrated, path=path,
    )


def _is_entry_name(name) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and not name.startswith(".")
        and not name.endswith(_PARTIAL_SUFFIX)
        and not any(sep in name for sep in ("/", "\\"))
    )


def _fresh_partial(root: Path, calibration_id: str) -> Path:
    partial = root / (calibration_id + _PARTIAL_SUFFIX)
    if partial.exists():
        logger.warning("removing %s, left by an interrupted write to the calibration library", partial)
        shutil.rmtree(partial)
    return partial


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _same_tree(a: Path, b: Path) -> bool:
    """Whether folders `a` and `b` hold the same files with the same bytes."""
    def files(folder: Path):
        return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())

    names = files(a)
    if names != files(b):
        return False
    return all(filecmp.cmp(a / name, b / name, shallow=False) for name in names)


def _fsync_dir(folder: Path) -> None:
    """Flush a folder's entries to disk. Windows cannot open a folder for that; it is skipped there."""
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(folder: Path) -> None:
    """Flush every file under `folder`, then every folder, so a rename after it publishes
    content that is on disk, not only in the page cache."""
    folders = [folder]
    for path in sorted(folder.rglob("*")):
        if path.is_dir():
            folders.append(path)
        elif path.is_file():
            with path.open("rb") as fh:
                os.fsync(fh.fileno())
    for path in reversed(folders):
        _fsync_dir(path)


def _replace_text(path: Path, text: str) -> None:
    _replace_bytes(path, text.encode("utf-8"))


def _replace_bytes(path: Path, data: bytes) -> None:
    """Replace `path` whole, so a crash leaves the old file or the new one, never a mix.

    The temporary file's name is unique (<file>.<uuid4 hex>.tmp), so two processes writing
    the same file at once do not write into each other's. It is flushed to disk before the
    rename, so the rename never publishes a file whose content is not there yet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _append_decision(root: Path, record: dict) -> None:
    """Append one JSON line. A last line a crash left without its newline is ended first."""
    path = root / DECISIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with path.open("a+b") as fh:
        if fh.seek(0, os.SEEK_END) > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                line = "\n" + line
        fh.write(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
