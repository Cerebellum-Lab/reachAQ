"""The pose training dataset: storage, provenance hash, session split, DLC import
and frame picking.

The session split is what keeps the evaluation honest: the two cameras of a
recording see the same reach, so holding out one camera while training on the
other would score the model on frames it has effectively seen.
"""

import hashlib
import os

import numpy
import pytest

cv2 = pytest.importorskip("cv2")

from autotrainer.inference.keypoint.dataset import (  # noqa: E402
    FRAME_SIZE,
    PoseDataset,
    import_dlc,
    pick_diverse,
    session_of,
)

PARTS = ["Paw", "Pellet", "Nose"]


def _tree_digest(root):
    digest = hashlib.sha256()
    for folder, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            path = os.path.join(folder, name)
            digest.update(path.encode())
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def _write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (frames.shape[2], frames.shape[1]), isColor=False)
    for frame in frames:
        writer.write(frame)
    writer.release()


def _video(tmp_path, count=20, size=64, name="20260101_rig_session001_left.mp4"):
    rng = numpy.random.default_rng(0)
    # Distinct, smooth frames so that decoding noise does not merge them.
    frames = numpy.zeros((count, size, size), "uint8")
    for i in range(count):
        frames[i] = (i * 12) % 256
        frames[i, : (i % size) + 1, :] = 255 - (i * 12) % 256
    frames = (frames.astype(int) + rng.integers(0, 3, frames.shape)).clip(0, 255).astype("uint8")
    path = tmp_path / name
    _write_video(path, frames)
    return path


def _dataset_with_rows(tmp_path):
    data = PoseDataset.create(str(tmp_path / "sets"), "demo", PARTS)
    image = numpy.full((FRAME_SIZE, FRAME_SIZE), 7, "uint8")
    for session in ("s1_left", "s1_right", "s2_left", "s3_left"):
        frame = data.add_image(image, session_of(session), f"{session}_000001")
        data.set_points(frame, numpy.array([[1, 2], [numpy.nan, numpy.nan], [5, 6]], "float32"))
        data.set_status(frame, "labelled")
    data.save()
    return data


def test_session_of_strips_the_camera_from_a_recording_name():
    assert session_of("20251202_agx001_session007_left") == "20251202_agx001_session007"
    assert session_of("20251202_agx001_session007_right") == "20251202_agx001_session007"
    assert session_of("20251124_christie2P_session001_frontCam-0000") == \
        "20251124_christie2P_session001"
    assert session_of("plain") == "plain"


def test_points_status_and_split_survive_a_save_and_reopen(tmp_path):
    data = _dataset_with_rows(tmp_path)
    data.set_val("s2", True)
    data.save()
    again = PoseDataset.open(data.path)
    assert again.bodyparts == PARTS
    assert again.val_sessions == ["s2"]
    assert [row.status for row in again.rows] == ["labelled"] * 4
    numpy.testing.assert_array_equal(again.rows[0].points, data.rows[0].points)
    assert numpy.isnan(again.rows[0].points[1]).all()


def test_save_leaves_no_temporary_files(tmp_path):
    data = _dataset_with_rows(tmp_path)
    data.save()
    leftovers = [n for n in os.listdir(data.path) if n.endswith(".tmp")]
    assert leftovers == []


def test_content_hash_tracks_labelled_points_only(tmp_path):
    data = _dataset_with_rows(tmp_path)
    before = data.content_hash()
    data.rows.reverse()
    assert data.content_hash() == before
    pending = data.add_image(numpy.zeros((FRAME_SIZE, FRAME_SIZE), "uint8"), "s4", "s4_000001")
    data.set_points(pending, numpy.zeros((3, 2), "float32"))
    assert data.content_hash() == before
    data.set_points(data.rows[0].frame, numpy.array([[1, 3], [numpy.nan] * 2, [5, 6]], "float32"))
    assert data.content_hash() != before


def test_default_split_holds_out_both_cameras_of_a_recording(tmp_path):
    data = _dataset_with_rows(tmp_path)
    data.default_split()
    assert len(data.val_sessions) == 1
    held = [row.session for row in data.rows if row.session in data.val_sessions]
    if data.val_sessions == ["s1"]:
        assert len(held) == 2
    train_frames, _ = data.arrays("train")
    val_frames, _ = data.arrays("val")
    assert len(train_frames) + len(val_frames) == 4


def test_arrays_leave_out_pending_frames(tmp_path):
    data = _dataset_with_rows(tmp_path)
    data.add_image(numpy.zeros((FRAME_SIZE, FRAME_SIZE), "uint8"), "s1", "s1_left_000099")
    frames, points = data.arrays("train")
    assert frames.shape == (4, FRAME_SIZE, FRAME_SIZE)
    assert points.shape == (4, 3, 2)


def _dlc_project(tmp_path):
    root = tmp_path / "dlc"
    (root / "labeled-data" / "rec1_left").mkdir(parents=True)
    (root / "labeled-data" / "rec1_right").mkdir(parents=True)
    (root / "config.yaml").write_text(
        "Task: t\nbodyparts:\n- Paw\n- Pellet\n- Nose\n", encoding="utf-8")
    cv2.imwrite(str(root / "labeled-data" / "rec1_left" / "img001.png"),
                numpy.full((256, 256), 50, "uint8"))
    cv2.imwrite(str(root / "labeled-data" / "rec1_left" / "img002.png"),
                numpy.full((256, 256), 60, "uint8"))
    cv2.imwrite(str(root / "labeled-data" / "rec1_right" / "img003.png"),
                numpy.full((512, 512, 3), 70, "uint8"))
    header = ("scorer,,,me,me,me,me,me,me\n"
              "bodyparts,,,Paw,Paw,Pellet,Pellet,Nose,Nose\n"
              "coords,,,x,y,x,y,x,y\n")
    (root / "labeled-data" / "rec1_left" / "CollectedData_me.csv").write_text(
        header
        + "labeled-data,rec1_left,img001.png,10.5,20.5,,,30,40\n"
        + "labeled-data,rec1_left,img002.png,,,,,,\n", encoding="utf-8")
    # The older single-column index form.
    (root / "labeled-data" / "rec1_right" / "CollectedData_me.csv").write_text(
        "scorer,me,me,me,me,me,me\n"
        "bodyparts,Paw,Paw,Pellet,Pellet,Nose,Nose\n"
        "coords,x,y,x,y,x,y\n"
        "labeled-data/rec1_right/img003.png,100,200,,,300,400\n", encoding="utf-8")
    return root


def test_import_dlc_reads_both_index_forms_and_leaves_the_source_alone(tmp_path):
    project = _dlc_project(tmp_path)
    digest = _tree_digest(project)
    data = import_dlc(str(project), str(tmp_path / "sets"), "imported")
    assert _tree_digest(project) == digest
    assert data.bodyparts == PARTS
    # The all-empty row is not a labelled frame.
    assert len(data.rows) == 2
    assert {row.session for row in data.rows} == {"rec1"}
    left = next(row for row in data.rows if "rec1_left" in row.frame)
    numpy.testing.assert_allclose(left.points[0], [10.5, 20.5])
    assert numpy.isnan(left.points[1]).all()
    right = next(row for row in data.rows if "rec1_right" in row.frame)
    # 512 px frame stored at 256. Pixel centres sit on integers (the
    # trainer's convention), so a coordinate maps as (p + 0.5) / 2 - 0.5.
    numpy.testing.assert_allclose(right.points[0], [49.75, 99.75])
    assert data.image(right.frame).shape == (FRAME_SIZE, FRAME_SIZE)
    assert all(row.status == "labelled" for row in data.rows)


def test_import_dlc_refuses_a_non_square_frame(tmp_path):
    project = _dlc_project(tmp_path)
    cv2.imwrite(str(project / "labeled-data" / "rec1_left" / "img001.png"),
                numpy.zeros((200, 256), "uint8"))
    with pytest.raises(ValueError, match="square"):
        import_dlc(str(project), str(tmp_path / "sets"), "imported")


def test_add_frames_stores_resized_frames_as_pending(tmp_path):
    video = _video(tmp_path)
    data = PoseDataset.create(str(tmp_path / "sets"), "demo", PARTS)
    added = data.add_frames(str(video), [0, 5])
    assert len(added) == 2
    assert {row.session for row in data.rows} == {"20260101_rig_session001"}
    assert all(row.status == "pending" for row in data.rows)
    assert data.image(added[0]).shape == (FRAME_SIZE, FRAME_SIZE)
    assert numpy.isnan(data.rows[0].points).all()


def test_add_frames_past_the_end_adds_nothing(tmp_path):
    video = _video(tmp_path)
    data = PoseDataset.create(str(tmp_path / "sets"), "demo", PARTS)
    with pytest.raises(IndexError):
        data.add_frames(str(video), [3, 500])
    assert data.rows == []


def test_pick_diverse_is_deterministic_and_skips_excluded_frames(tmp_path):
    video = _video(tmp_path)
    first = pick_diverse(str(video), 5, exclude=(0, 1))
    assert first == pick_diverse(str(video), 5, exclude=(0, 1))
    assert len(set(first)) == 5
    assert not {0, 1} & set(first)


def test_read_frames_seeks_to_the_same_frames_sequential_decoding_gives(tmp_path):
    from autotrainer.inference.keypoint.dataset import read_frames
    video = _video(tmp_path)
    sequential = {}
    capture = cv2.VideoCapture(str(video))
    for index in range(20):
        sequential[index] = capture.read()[1]
    capture.release()
    found = read_frames(str(video), [17, 3, 9])
    for index in (3, 9, 17):
        numpy.testing.assert_array_equal(found[index], sequential[index])


def test_read_frames_can_shrink_as_it_decodes(tmp_path):
    from autotrainer.inference.keypoint.dataset import read_frames
    video = _video(tmp_path)
    found = read_frames(str(video), [0, 5], size=32)
    assert found[5].shape == (32, 32)
    assert found[5].dtype == numpy.uint8
