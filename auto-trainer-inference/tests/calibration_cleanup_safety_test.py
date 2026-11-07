from pathlib import Path

import pytest

from autotrainer.inference.calibration_FLIR import create_or_clean_directory


@pytest.mark.parametrize("name", ("corners", "gray", "rejected", "camera_matrix"))
def test_cleanup_removes_only_contents_of_approved_generated_directory(tmp_path, name):
    target = tmp_path / name
    nested = target / "nested"
    nested.mkdir(parents=True)
    (target / "generated.txt").write_text("generated")
    (nested / "generated.txt").write_text("generated")
    source = tmp_path / "camera0.png"
    source.write_text("source")

    create_or_clean_directory(target, calibration_root=tmp_path)

    assert target.is_dir()
    assert tuple(target.iterdir()) == ()
    assert source.read_text() == "source"


def test_cleanup_rejects_unapproved_child_without_deleting_it(tmp_path):
    target = tmp_path / "source_images"
    target.mkdir()
    source = target / "frame.png"
    source.write_text("keep")

    with pytest.raises(ValueError, match="unapproved"):
        create_or_clean_directory(target, calibration_root=tmp_path)

    assert source.read_text() == "keep"


def test_cleanup_rejects_target_outside_selected_root(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    outside = tmp_path / "corners"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")

    with pytest.raises(ValueError, match="outside"):
        create_or_clean_directory(outside, calibration_root=root)

    assert (outside / "keep.txt").read_text() == "keep"


def test_cleanup_rejects_symlink_target_and_preserves_destination(tmp_path):
    root = tmp_path / "selected"
    root.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    keep = destination / "keep.txt"
    keep.write_text("keep")
    (root / "corners").symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        create_or_clean_directory(root / "corners", calibration_root=root)

    assert keep.read_text() == "keep"


def test_cleanup_rejects_symlink_source_root(tmp_path):
    destination = tmp_path / "real-root"
    destination.mkdir()
    root = tmp_path / "selected"
    root.symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError, match="source root"):
        create_or_clean_directory(root / "gray", calibration_root=root)
