"""Rewrite the pickled SceneElement column labels in old pose h5 files.

Until upstream's 2025-06-25 change, pose DataFrames used SceneElement objects as
column labels. PyTables stores DataFrame column labels as pickled HDF5
attributes (``non_index_axes``, ``values_block_0_kind``), so those files name
the class by its module path, ``autotrainer.inference.pose_elements`` or
``autotrainer.core.pose_elements``. Neither module exists in reachaq, and
PyTables hides the failed unpickle by returning the raw bytes, so such files
load with broken columns.

Current code writes the labels as plain strings, and this tool rewrites old
files to that form. The labels compare equal either way, and the rewritten
file names no module at all. Files must be converted once before reachaq can
read them; ``--check`` finds them without writing anything.
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

LEGACY_MODULES = frozenset({
    "autotrainer.inference.pose_elements",
    "autotrainer.core.pose_elements",
})
LEGACY_CLASSES = frozenset({"SceneElement", "_BaseSceneElement"})
_MARKER = b"autotrainer"


class LegacyLabelError(Exception):
    """A file holds a module reference this tool does not know how to rewrite."""


class _LegacyLabel(str):
    """Stands in for an old SceneElement while unpickling."""


class _RecordingUnpickler(pickle.Unpickler):
    def __init__(self, data: bytes):
        super().__init__(io.BytesIO(data))
        self.legacy_refs: List[Tuple[str, str]] = []

    def find_class(self, module, name):
        if module in LEGACY_MODULES and name in LEGACY_CLASSES:
            self.legacy_refs.append((module, name))
            return _LegacyLabel
        if module.split(".")[0] == "autotrainer":
            raise LegacyLabelError(f"unknown legacy reference {module}.{name}")
        return super().find_class(module, name)


def _plain(value):
    if isinstance(value, _LegacyLabel):
        return str(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    return value


def _protocol(data: bytes) -> int:
    # Protocol 2+ starts with PROTO; protocols 0 and 1 have no header, and
    # PyTables writes protocol 0.
    return data[1] if data[:1] == b"\x80" else 0


def _pickled_attrs(h5file) -> Iterator[Tuple[object, str, bytes]]:
    """Yield (object, attribute name, raw bytes) for attributes that may hold
    a legacy pickle."""
    import numpy

    found = []

    def visit(_name, obj):
        for key, value in obj.attrs.items():
            if isinstance(value, numpy.bytes_) and _MARKER in value.tobytes():
                found.append((obj, key, value.tobytes()))

    for key, value in h5file.attrs.items():
        if isinstance(value, numpy.bytes_) and _MARKER in value.tobytes():
            found.append((h5file, key, value.tobytes()))
    h5file.visititems(visit)
    return iter(found)


def _convert_value(raw: bytes) -> Optional[bytes]:
    """Return the rewritten pickle, or None when the value names no legacy class."""
    try:
        unpickler = _RecordingUnpickler(raw)
        value = unpickler.load()
    except LegacyLabelError:
        raise
    except Exception:
        return None  # not a pickle; the marker is in ordinary text
    if not unpickler.legacy_refs:
        return None
    return pickle.dumps(_plain(value), protocol=_protocol(raw))


def needs_conversion(path: Path) -> bool:
    import h5py

    with h5py.File(path, "r") as h5file:
        return any(_convert_value(raw) is not None for _obj, _key, raw in _pickled_attrs(h5file))


def _rewrite(path: Path) -> List[Tuple[str, str, object]]:
    """Rewrite the legacy attributes of ``path`` and return (node, attribute,
    expected value) for each one."""
    import h5py
    import numpy

    done = []
    with h5py.File(path, "r+") as h5file:
        for obj, key, raw in _pickled_attrs(h5file):
            new = _convert_value(raw)
            if new is not None:
                obj.attrs[key] = numpy.bytes_(new)
                done.append((obj.name, key, pickle.loads(new)))
    return done


def _verify(path: Path, rewritten: List[Tuple[str, str, object]]) -> None:
    """Each rewritten attribute must load with the stock unpickler, equal what
    was written, and the file must hold no other legacy reference."""
    import h5py

    with h5py.File(path, "r") as h5file:
        for node, key, expected in rewritten:
            if pickle.loads(h5file[node].attrs[key].tobytes()) != expected:
                raise LegacyLabelError(f"{node}:{key} did not read back as written")
        for obj, key, raw in _pickled_attrs(h5file):
            if _convert_value(raw) is not None:
                raise LegacyLabelError(f"{obj.name}:{key} still names a legacy class")


def convert(source: Path, target: Path) -> int:
    """Write a converted copy of ``source`` to ``target`` and return the number
    of rewritten attributes. ``target`` may equal ``source``; it is replaced
    atomically only after the copy converts and verifies."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".converting")
    shutil.copy2(source, staging)
    try:
        rewritten = _rewrite(staging)
        _verify(staging, rewritten)
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()
    return len(rewritten)


def _h5_files(paths: Iterable[Path]) -> Iterator[Tuple[Path, Path]]:
    """Yield (file, path relative to the argument it came from)."""
    for path in paths:
        if path.is_dir():
            for file in sorted(path.rglob("*.h5")):
                yield file, file.relative_to(path)
        else:
            yield path, Path(path.name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reachaq-convert-pose-h5",
        description="Rewrite the autotrainer SceneElement column labels in pose h5 files "
                    "written before 2025-06-25 as plain strings, so reachaq can read them.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="only list the files that need converting; exit 1 if any do")
    mode.add_argument("--in-place", action="store_true",
                      help="convert each file in place, keeping the original as FILE.bak")
    mode.add_argument("--output-dir", type=Path,
                      help="write converted copies here, keeping each file's path below "
                           "the directory argument it was found in")
    parser.add_argument("paths", nargs="+", type=Path, help="h5 files or directories to search")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    legacy = errors = 0
    for file, relative in _h5_files(args.paths):
        try:
            if not needs_conversion(file):
                print(f"current    {file}")
                continue
            legacy += 1
            if args.check:
                print(f"legacy     {file}")
                continue
            if args.in_place:
                backup = file.with_name(file.name + ".bak")
                if backup.exists():
                    raise FileExistsError(f"backup {backup} already exists")
                shutil.copy2(file, backup)
                try:
                    count = convert(file, file)
                except Exception:
                    backup.unlink()  # the original is untouched; allow a retry
                    raise
                target = file
            else:
                target = args.output_dir / relative
                if target.exists():
                    raise FileExistsError(f"{target} already exists")
                count = convert(file, target)
            print(f"converted  {file} -> {target} ({count} attributes)")
        except Exception as exc:  # report and carry on with the other files
            errors += 1
            print(f"error      {file}: {exc}", file=sys.stderr)
    if errors:
        return 2
    return 1 if args.check and legacy else 0


if __name__ == "__main__":
    sys.exit(main())
