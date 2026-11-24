import numpy as np

from reachaq.inference.analysis.prepare_jetson_data import identify_dropped_frames


# The header camera_timing_alignment writes for a three-camera rig.
WIDE_HEADER = (
    "frame_id,frame_when,frame_present_primary,frame_present_secondary,"
    "frame_when_left,frame_present_left,frame_when_right,frame_present_right,"
    "frame_when_camera4,frame_present_camera4,utc_when\n"
)


def test_wide_timing_file_reads_the_named_columns(tmp_path):
    timing = tmp_path / "frame_timing.csv"
    timing.write_text(
        WIDE_HEADER
        + "1,1.0,1,1,1.0,1,1.0,1,1.05,1,100\n"
        + "2,2.0,1,0,2.0,1,,0,2.05,1,101\n"
        + "3,3.0,1,1,3.0,1,3.0,1,,0,102\n",
        encoding="utf-8",
    )

    primary = identify_dropped_frames(str(timing), 150, cam_col_present_name="frame_present_primary")
    secondary = identify_dropped_frames(str(timing), 150, cam_col_present_name="frame_present_secondary")

    assert primary.tolist() == [0, 0, 0]
    assert secondary.tolist() == [0, 1, 0]


def test_five_column_timing_file_still_reads(tmp_path):
    timing = tmp_path / "frame_timing.csv"
    timing.write_text(
        "frame_id,frame_when,frame_present_primary,frame_present_secondary,utc_when\n"
        "1,1.0,1,0,100\n2,2.0,1,1,101\n",
        encoding="utf-8",
    )

    secondary = identify_dropped_frames(str(timing), 150, cam_col_present_name="frame_present_secondary")

    assert np.array_equal(secondary, [1, 0])
