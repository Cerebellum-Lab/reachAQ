from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4


def fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        descriptor = os.open(
            path.as_posix(),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
                raise
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def session_staging_path(
    session_dir: Path,
    generation_id: str,
    final_path: Path,
) -> Path:
    relative = final_path.relative_to(session_dir)
    return session_dir / ".staging" / generation_id / relative


def atomic_publish_file(
    final_path: Path,
    write: Callable[[Path], None],
    *,
    session_dir: Optional[Path] = None,
    generation_id: str = "default",
    validate: Optional[Callable[[Path], None]] = None,
) -> Path:
    final_path = Path(final_path)
    unique_suffix = uuid4().hex
    if session_dir is None:
        staged_path = final_path.with_name(
            f".{final_path.name}.{unique_suffix}.tmp"
        )
    else:
        staged_base = session_staging_path(
            Path(session_dir),
            str(generation_id),
            final_path,
        )
        staged_path = staged_base.with_name(
            f".{staged_base.name}.{unique_suffix}.tmp"
        )
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(staged_path)
        if not staged_path.is_file():
            raise RuntimeError(f"staged writer did not create {staged_path}")
        fsync_file(staged_path)
        if validate is not None:
            validate(staged_path)
        os.replace(staged_path, final_path)
        fsync_directory(final_path.parent)
    except Exception:
        # A failed staged file is intentionally retained for post-mortem and
        # bounded finalization retry. A failed sibling temporary file has no
        # session recovery location and is safe to remove.
        if session_dir is None:
            staged_path.unlink(missing_ok=True)
        raise
    return final_path


def atomic_write_json(
    final_path: Path,
    value,
    *,
    session_dir: Optional[Path] = None,
    generation_id: str = "default",
    encoder=None,
) -> Path:
    def write(path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                cls=encoder,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def validate(path: Path) -> None:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)

    return atomic_publish_file(
        final_path,
        write,
        session_dir=session_dir,
        generation_id=generation_id,
        validate=validate,
    )


def file_manifest_entry(path: Path, *, relative_to: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sizeBytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
