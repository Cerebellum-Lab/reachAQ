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
        # The frame_present_<camera> column's total: a blank the capture
        # inserted for a dropped frame has a row but no arrival time.
        present = sum(
            1 for frame_when in table["frame_when"].reindex(primary_ids)
            if _finite(frame_when)
        )
        camera_diagnostics[source.name] = {
            "firstFrameId": ids[0],
            "lastFrameId": ids[-1],
            "timestampRows": len(ids),
            "presentFrames": present,
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

    # Each value as table.loc[frame_id][field] returns it: cast to the dtype
    # pandas gives a whole row, so integer stamps still print as floats. A
    # .loc per value took 41 s of Stop on a 45-minute session; these lists and
    # the row positions below take about one.
    def column(table, field):
        return table[field].astype(table.iloc[0].dtype).tolist()

    primary_times = column(primary, "frame_time")
    camera_whens = [column(table, "frame_when") for table in tables]
    # Each camera's row for each primary frame, or -1 when it has none.
    camera_rows = [table.index.get_indexer(primary_ids).tolist() for table in tables]

    def write_timing(path):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(output_fields)
            for position, frame_id in enumerate(primary_ids):
                whens = [
                    math.nan if rows[position] < 0 else values[rows[position]]
                    for values, rows in zip(camera_whens, camera_rows)
                ]
                primary_when = whens[0]
                secondary_when = math.nan if len(tables) < 2 else whens[1]
                recorded_frame_time = primary_times[position]
                utc_when = (
                    float(recorded_frame_time)
                    if _finite(recorded_frame_time)
                    else first_frame_time + (frame_id - primary_first) * frame_duration
                )
                row = [
                    frame_id,
                    primary_when if _finite(primary_when) else "",
                    1,
                    1 if _finite(secondary_when) else 0,
                ]
                for frame_when in whens:
                    present = _finite(frame_when)
                    row.extend((frame_when if present else "", 1 if present else 0))
                row.append(utc_when)
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
