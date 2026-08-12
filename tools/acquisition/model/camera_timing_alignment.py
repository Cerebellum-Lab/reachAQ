from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas

from tools.acquisition.model.atomic_session_io import (
    atomic_publish_file,
    atomic_write_json,
)


TIMESTAMP_FIELDS = (
    "frame_time",
    "fps",
    "frame_when",
    "frame_perf",
    "frame_id",
)


class CameraTimestampIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraTimestampInput:
    name: str
    field_name: str
    path: Path


def _load_timestamp_table(source: CameraTimestampInput) -> pandas.DataFrame:
    if not source.path.is_file():
        raise CameraTimestampIntegrityError(
            f"{source.name} timestamp file is missing: {source.path}"
        )
    table = pandas.read_csv(
        source.path,
        names=TIMESTAMP_FIELDS,
        sep=",",
        skipinitialspace=True,
    )
    if table.empty:
        raise CameraTimestampIntegrityError(
            f"{source.name} timestamp file is empty"
        )
    numeric_ids = pandas.to_numeric(table["frame_id"], errors="coerce")
    if numeric_ids.isna().any():
        rows = tuple(int(index) + 1 for index in numeric_ids[numeric_ids.isna()].index)
        raise CameraTimestampIntegrityError(
            f"{source.name} has non-numeric frame IDs at row(s) {rows}"
        )
    integer_ids = numeric_ids.astype("int64")
    if not (numeric_ids == integer_ids).all():
        raise CameraTimestampIntegrityError(
            f"{source.name} has non-integer frame IDs"
        )
    duplicate_ids = tuple(
        int(value) for value in integer_ids[integer_ids.duplicated()].unique()
    )
    if duplicate_ids:
        raise CameraTimestampIntegrityError(
            f"{source.name} has duplicate frame IDs: {duplicate_ids}"
        )
    deltas = integer_ids.diff().dropna()
    if (deltas <= 0).any():
        bad_rows = tuple(int(index) + 1 for index in deltas[deltas <= 0].index)
        raise CameraTimestampIntegrityError(
            f"{source.name} frame IDs are not strictly increasing at row(s) {bad_rows}"
        )
    table["frame_id"] = integer_ids
    return table.set_index("frame_id", drop=False)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def align_camera_timestamp_files(
    sources: Sequence[CameraTimestampInput],
    *,
    output_path: Path,
    diagnostics_path: Path,
    primary_frame_id: int,
    primary_fps: float,
) -> dict:
    if not sources:
        raise ValueError("at least one camera timestamp source is required")
    if not math.isfinite(primary_fps) or primary_fps <= 0:
        raise ValueError(f"primary camera FPS is invalid: {primary_fps}")

    loaded = []
    load_errors = {}
    for source in sources:
        try:
            loaded.append(_load_timestamp_table(source))
        except CameraTimestampIntegrityError as error:
            load_errors[source.name] = str(error)
            loaded.append(None)
    if load_errors:
        diagnostics = {
            "primaryCamera": sources[0].name,
            "primaryFrameCount": 0,
            "synchronizationComplete": False,
            "cameras": {
                source.name: {
                    "integrityError": load_errors.get(source.name),
                }
                for source in sources
            },
        }
        atomic_write_json(diagnostics_path, diagnostics)
        raise CameraTimestampIntegrityError(
            "; ".join(
                f"{name}: {error}" for name, error in load_errors.items()
            )
        )
    tables = tuple(loaded)
    primary = tables[0]
    primary_ids = tuple(int(value) for value in primary["frame_id"])
    if primary_ids[0] != int(primary_frame_id):
        raise CameraTimestampIntegrityError(
            f"primary timing starts at frame {primary_ids[0]}, expected {primary_frame_id}"
        )
    primary_id_set = set(primary_ids)
    primary_first = primary_ids[0]
    primary_last = primary_ids[-1]
    frame_duration = 1.0 / float(primary_fps)
    first_frame_time = float(primary.iloc[0]["frame_time"])

    camera_diagnostics: Dict[str, dict] = {}
    synchronization_complete = True
    for source, table in zip(sources, tables):
        ids = tuple(int(value) for value in table["frame_id"])
        id_set = set(ids)
        gaps = sum(max(0, current - previous - 1) for previous, current in zip(ids, ids[1:]))
        missing = tuple(frame_id for frame_id in primary_ids if frame_id not in id_set)
        extra = tuple(frame_id for frame_id in ids if frame_id not in primary_id_set)
        early = tuple(frame_id for frame_id in extra if frame_id < primary_first)
        late = tuple(frame_id for frame_id in extra if frame_id > primary_last)
        camera_diagnostics[source.name] = {
            "firstFrameId": ids[0],
            "lastFrameId": ids[-1],
            "timestampRows": len(ids),
            "frameIdGapCount": gaps,
            "missingPrimaryFrameIds": list(missing),
            "extraFrameIds": list(extra),
            "earlyFrameIds": list(early),
            "lateFrameIds": list(late),
        }
        if gaps or missing or extra:
            synchronization_complete = False

    camera_fields = [source.field_name for source in sources]
    output_fields = [
        "frame_id",
        "frame_when",
        "frame_present_primary",
        "frame_present_secondary",
        *(
            field
            for name in camera_fields
            for field in (f"frame_when_{name}", f"frame_present_{name}")
        ),
        "utc_when",
    ]
    def write_timing(path):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, output_fields)
            writer.writeheader()
            for frame_id in primary_ids:
                primary_row = primary.loc[frame_id]
                primary_when = primary_row["frame_when"]
                secondary_when = (
                    math.nan
                    if len(tables) < 2 or frame_id not in tables[1].index
                    else tables[1].loc[frame_id]["frame_when"]
                )
                recorded_frame_time = primary_row["frame_time"]
                utc_when = (
                    float(recorded_frame_time)
                    if _finite(recorded_frame_time)
                    else first_frame_time + (frame_id - primary_first) * frame_duration
                )
                row = {
                    "frame_id": frame_id,
                    "frame_when": primary_when if _finite(primary_when) else "",
                    "frame_present_primary": 1,
                    "frame_present_secondary": 1 if _finite(secondary_when) else 0,
                    "utc_when": utc_when,
                }
                for table, field_name in zip(tables, camera_fields):
                    frame_when = (
                        math.nan
                        if frame_id not in table.index
                        else table.loc[frame_id]["frame_when"]
                    )
                    row[f"frame_when_{field_name}"] = (
                        frame_when if _finite(frame_when) else ""
                    )
                    row[f"frame_present_{field_name}"] = 1 if _finite(frame_when) else 0
                writer.writerow(row)

    atomic_publish_file(output_path, write_timing)

    diagnostics = {
        "primaryCamera": sources[0].name,
        "primaryFrameCount": len(primary_ids),
        "synchronizationComplete": synchronization_complete,
        "cameras": camera_diagnostics,
    }
    atomic_write_json(diagnostics_path, diagnostics)
    return diagnostics
