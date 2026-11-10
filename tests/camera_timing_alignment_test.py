import csv
import json
import time

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


def test_alignment_counts_each_cameras_present_frames(tmp_path):
    """The exact count the session metadata derives frames dropped from.

    A frame is present when the camera has an arrival time for it. A frame
    the capture filled with a blank has a row but no arrival time, and one
    the camera never reported has no row at all.
    """
    primary = tmp_path / "left.txt"
    secondary = tmp_path / "right.txt"
    _write_timestamps(primary, (100, 101, 102, 103))
    _write_timestamps(secondary, (100, 101, 103))
    lines = secondary.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(", ")
    fields[2] = "nan"
    lines[1] = ", ".join(fields)
    secondary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    diagnostics = align_camera_timestamp_files(
        (
            CameraTimestampInput("left", "left", primary),
            CameraTimestampInput("right", "right", secondary),
        ),
        output_path=tmp_path / "frame_timing.csv",
        diagnostics_path=tmp_path / "camera_alignment.json",
        primary_frame_id=100,
        primary_fps=150,
    )

    assert diagnostics["cameras"]["left"]["presentFrames"] == 4
    assert diagnostics["cameras"]["right"]["presentFrames"] == 2


def test_the_timing_file_keeps_its_exact_text(tmp_path):
    """frame_timing.csv byte for byte, as the row-at-a-time merge wrote it.

    Integer arrival stamps print as floats. A blank the capture inserted and a
    frame the secondary never wrote both leave empty fields, and a frame with
    no frame_time takes its UTC time from the first frame and the frame rate.
    """
    primary = tmp_path / "left.txt"
    secondary = tmp_path / "right.txt"
    output = tmp_path / "frame_timing.csv"
    primary.write_text(
        "1793995561.4592035, 150, 88836625845888, 259875.39, 1760\n"
        "1793995561.4658704, 150, nan, 259875.3966, 1761\n"
        "nan, 150, 88836639180536, 259875.4033, 1762\n",
        encoding="utf-8",
    )
    secondary.write_text(
        "1793995561.4525366, 150, 88222181270000, 259875.3833, 1759\n"
        "1793995561.4592035, 150, 88222187937592, 259875.39, 1760\n"
        "1793995561.4725371, 150, 88222201272080, 259875.4033, 1762\n"
        "1793995561.479204, 150, 88222207940000, 259875.41, 1763\n",
        encoding="utf-8",
    )

    align_camera_timestamp_files(
        (
            CameraTimestampInput("left", "left", primary),
            CameraTimestampInput("right", "right", secondary),
        ),
        output_path=output,
        diagnostics_path=tmp_path / "camera_alignment.json",
        primary_frame_id=1760,
        primary_fps=150,
    )

    assert output.read_bytes() == (
        b"frame_id,frame_when,frame_present_primary,frame_present_secondary,"
        b"frame_when_left,frame_present_left,frame_when_right,frame_present_right,"
        b"utc_when\r\n"
        b"1760,88836625845888.0,1,1,88836625845888.0,1,88222187937592.0,1,"
        b"1793995561.4592035\r\n"
        b"1761,,1,0,,0,,0,1793995561.4658704\r\n"
        b"1762,88836639180536.0,1,1,88836639180536.0,1,88222201272080.0,1,"
        b"1793995561.4725368\r\n"
    )


def test_the_merge_stays_fast_on_a_long_session(tmp_path):
    # 150,000 frames, under 17 minutes at 150 fps. Looking each value up with
    # DataFrame.loc took 41 s of Stop on a 45-minute session on christielab10,
    # about 11 s at this size.
    primary = tmp_path / "left.txt"
    secondary = tmp_path / "right.txt"
    _write_timestamps(primary, range(150_000))
    _write_timestamps(secondary, range(150_000))

    started = time.perf_counter()
    align_camera_timestamp_files(
        (
            CameraTimestampInput("left", "left", primary),
            CameraTimestampInput("right", "right", secondary),
        ),
        output_path=tmp_path / "frame_timing.csv",
        diagnostics_path=tmp_path / "camera_alignment.json",
        primary_frame_id=0,
        primary_fps=150,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"the merge took {elapsed:.1f} s on 150,000 frames"


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
    diagnostics = json.loads(
        (tmp_path / "camera_alignment.json").read_text(encoding="utf-8")
    )
    assert diagnostics["synchronizationComplete"] is False
    assert message in diagnostics["cameras"]["left"]["integrityError"]
