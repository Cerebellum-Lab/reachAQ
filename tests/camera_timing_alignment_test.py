import csv
import json

import pytest

from tools.acquisition.model.camera_timing_alignment import (
    CameraTimestampInput,
    CameraTimestampIntegrityError,
    align_camera_timestamp_files,
)


def _write_timestamps(path, frame_ids):
    with path.open("w", encoding="utf-8") as stream:
        for index, frame_id in enumerate(frame_ids):
            stream.write(
                f"{1000 + index / 150}, 150, {2000 + index}, "
                f"{3000 + index / 150}, {frame_id}\n"
            )


def test_alignment_joins_by_actual_frame_id_and_marks_missing(tmp_path):
    primary = tmp_path / "left.txt"
    secondary = tmp_path / "right.txt"
    output = tmp_path / "frame_timing.csv"
    diagnostics_path = tmp_path / "streams" / "camera_alignment.json"
    _write_timestamps(primary, (100, 101, 102))
    _write_timestamps(secondary, (99, 100, 102, 103))

    diagnostics = align_camera_timestamp_files(
        (
            CameraTimestampInput("left", "left", primary),
            CameraTimestampInput("right", "right", secondary),
        ),
        output_path=output,
        diagnostics_path=diagnostics_path,
        primary_frame_id=100,
        primary_fps=150,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frame_id"]) for row in rows] == [100, 101, 102]
    assert [int(row["frame_present_right"]) for row in rows] == [1, 0, 1]
    assert rows[1]["frame_when_right"] == ""
    assert not diagnostics["synchronizationComplete"]
    right = diagnostics["cameras"]["right"]
    assert right["missingPrimaryFrameIds"] == [101]
    assert right["earlyFrameIds"] == [99]
    assert right["lateFrameIds"] == [103]
    assert json.loads(diagnostics_path.read_text()) == diagnostics


@pytest.mark.parametrize(
    ("frame_ids", "message"),
    (
        ((1, 1, 2), "duplicate"),
        ((1, 3, 2), "strictly increasing"),
        ((1, 2.5, 3), "non-integer"),
    ),
)
def test_alignment_rejects_unprovable_frame_identity(
    tmp_path,
    frame_ids,
    message,
):
    primary = tmp_path / "left.txt"
    _write_timestamps(primary, frame_ids)

    with pytest.raises(CameraTimestampIntegrityError, match=message):
        align_camera_timestamp_files(
            (CameraTimestampInput("left", "left", primary),),
            output_path=tmp_path / "frame_timing.csv",
            diagnostics_path=tmp_path / "camera_alignment.json",
            primary_frame_id=1,
            primary_fps=150,
        )
