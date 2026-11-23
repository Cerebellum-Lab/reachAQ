import shutil
from pathlib import Path

import h5py
import pandas as pd
import pytest

from reachaq.core import pose_h5_labels
from reachaq.core.pose_elements import SceneElement

# The 2025-06-23 agx001 session 2 recording, written before upstream switched
# the labels to plain strings. Its raw2D files pickle
# autotrainer.inference.pose_elements.SceneElement.
SESSION = (Path(__file__).parents[1] / "inference" / "fp-and-xp-not-same"
           / "20250623" / "agx001" / "session002")
LEFT = SESSION / "20250623_agx001_session002_left_raw2D.h5"


@pytest.fixture
def legacy(tmp_path):
    path = tmp_path / LEFT.name
    shutil.copy2(LEFT, path)
    return path


def _columns(path):
    return pd.read_hdf(path).columns


def test_the_2025_session_needs_converting(legacy):
    assert pose_h5_labels.needs_conversion(legacy)
    # PyTables swallows the failed unpickle, so the labels come back unusable.
    with pytest.raises(Exception):
        list(_columns(legacy))


def test_check_lists_legacy_files_and_writes_nothing(legacy, capsys):
    before = legacy.read_bytes()
    assert pose_h5_labels.main(["--check", str(legacy.parent)]) == 1
    assert "legacy" in capsys.readouterr().out
    assert legacy.read_bytes() == before
    assert sorted(p.name for p in legacy.parent.iterdir()) == [legacy.name]


def test_in_place_converts_and_keeps_a_backup(legacy):
    original = legacy.read_bytes()
    assert pose_h5_labels.main(["--in-place", str(legacy)]) == 0

    backup = legacy.with_name(legacy.name + ".bak")
    assert backup.read_bytes() == original
    assert not pose_h5_labels.needs_conversion(legacy)
    assert pose_h5_labels.main(["--check", str(legacy)]) == 0

    columns = _columns(legacy)
    labels = {label for level in range(columns.nlevels) for label in columns.get_level_values(level)}
    assert SceneElement.Pellet in labels
    assert all(type(label) is str for level in range(columns.nlevels)
               for label in columns.get_level_values(level))
    with h5py.File(legacy, "r") as h5file:
        assert b"autotrainer" not in h5file["df_with_missing"].attrs["non_index_axes"].tobytes()


def test_in_place_refuses_to_overwrite_a_backup(legacy):
    backup = legacy.with_name(legacy.name + ".bak")
    backup.write_bytes(b"earlier backup")
    original = legacy.read_bytes()
    assert pose_h5_labels.main(["--in-place", str(legacy)]) == 2
    assert backup.read_bytes() == b"earlier backup"
    assert legacy.read_bytes() == original


def test_output_dir_writes_a_converted_copy(tmp_path):
    out = tmp_path / "out"
    assert pose_h5_labels.main(["--output-dir", str(out), str(SESSION)]) == 0

    copy = out / LEFT.name
    assert not pose_h5_labels.needs_conversion(copy)
    assert pose_h5_labels.needs_conversion(LEFT)  # the source is untouched
    # Only the files that needed converting are written.
    assert sorted(p.name for p in out.iterdir()) == [
        "20250623_agx001_session002_left_raw2D.h5",
        "20250623_agx001_session002_right_raw2D.h5",
    ]
    right = pd.read_hdf(out / "20250623_agx001_session002_right_raw2D.h5")
    assert list(pd.read_hdf(copy).columns) == list(right.columns)


def test_converted_frame_matches_the_legacy_values(legacy, tmp_path):
    out = tmp_path / "out"
    assert pose_h5_labels.main(["--output-dir", str(out), str(legacy)]) == 0
    with h5py.File(legacy, "r") as old, h5py.File(out / legacy.name, "r") as new:
        assert (old["df_with_missing/table"][()] == new["df_with_missing/table"][()]).all()


def test_current_files_are_left_alone(legacy, tmp_path):
    converted = tmp_path / "out"
    pose_h5_labels.main(["--output-dir", str(converted), str(legacy)])
    again = tmp_path / "again"
    assert pose_h5_labels.main(["--output-dir", str(again), str(converted)]) == 0
    assert not again.exists()
