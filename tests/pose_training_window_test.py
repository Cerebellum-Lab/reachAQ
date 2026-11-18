"""The pose training window, offscreen: labelling clicks land on the pixel the
operator clicked, Done stores the frame as labelled, and training cannot be
started while a session records."""

import multiprocessing.dummy
import os

import numpy
import pytest

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.inference.keypoint.dataset import FRAME_SIZE, LABELLED, PENDING, PoseDataset  # noqa: E402
from tools.acquisition.model.pose_training_model import PoseTrainingModel  # noqa: E402
from tools.acquisition.view.keypoint_label_view import KeypointLabelView  # noqa: E402
from tools.acquisition.view.pose_training_window import PoseTrainingWindow  # noqa: E402

PARTS = ["Paw", "Pellet", "Nose"]


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class _Inference:
    model_location = ""


class _FakeAppModel:
    """The parts of AppModel the window uses."""

    def __init__(self, tmp_path, ready=True):
        self.ready = ready
        self.inference = _Inference()
        self.output_location = str(tmp_path / "out")
        self.pose_training = PoseTrainingModel(
            str(tmp_path / "models"), is_ready=lambda: self.ready,
            context=multiprocessing.dummy, target=lambda *args: None)
        self.pose_training.datasets_root = str(tmp_path / "sets")
        self.deployed = []

        from autotrainer.core import ObservableObject
        self._events = ObservableObject()
        self.property_changed = self._events.property_changed

    @property
    def session_ready(self):
        return self.ready

    def deploy_pose_model(self, path):
        self.deployed.append(path)


def _dataset(tmp_path):
    data = PoseDataset.create(str(tmp_path / "sets"), "demo", PARTS)
    for i in range(2):
        data.add_image(numpy.full((FRAME_SIZE, FRAME_SIZE), 40 * (i + 1), "uint8"),
                       "s1", f"s1_left_{i:06d}")
    data.save()
    return data


def _click(view, x, y, button=Qt.MouseButton.LeftButton):
    point = view.mapFromScene(QPointF(x, y))
    QTest.mouseClick(view.viewport(), button, Qt.KeyboardModifier.NoModifier, point)


def test_a_click_places_the_selected_part_and_selects_the_next(qapp):
    view = KeypointLabelView()
    view.resize(600, 600)
    view.show()
    changed = []
    view.points_changed.connect(lambda points: changed.append(points.copy()))
    view.set_frame(numpy.zeros((FRAME_SIZE, FRAME_SIZE), "uint8"),
                   numpy.full((3, 2), numpy.nan, "float32"), PARTS, pending=False)
    view.selected_part = 1
    _click(view, 100.0, 60.0)
    assert changed, "no points_changed signal"
    numpy.testing.assert_allclose(changed[-1][1], [100.0, 60.0], atol=0.5)
    assert numpy.isnan(changed[-1][0]).all()
    assert view.selected_part == 2


def test_a_right_click_on_a_point_clears_it(qapp):
    view = KeypointLabelView()
    view.resize(600, 600)
    view.show()
    points = numpy.full((3, 2), numpy.nan, "float32")
    points[0] = [30, 40]
    view.set_frame(numpy.zeros((FRAME_SIZE, FRAME_SIZE), "uint8"), points, PARTS, pending=False)
    _click(view, 30.0, 40.0, Qt.MouseButton.RightButton)
    assert numpy.isnan(view.points()[0]).all()


def test_done_stores_the_frame_as_labelled(qapp, tmp_path):
    data = _dataset(tmp_path)
    window = PoseTrainingWindow(_FakeAppModel(tmp_path))
    window.open_dataset(data.path)
    window.show_frame(data.rows[0].frame)
    window.label_view.selected_part = 0
    window.label_view.place(0, 12.0, 34.0)
    window.mark_done()
    stored = PoseDataset.open(data.path)
    row = stored.row(data.rows[0].frame)
    assert row.status == LABELLED
    numpy.testing.assert_allclose(row.points[0], [12.0, 34.0])
    # Done moves on to the next pending frame.
    assert window.current_frame == data.rows[1].frame
    assert stored.row(data.rows[1].frame).status == PENDING


def test_training_cannot_start_while_a_session_records(qapp, tmp_path):
    data = _dataset(tmp_path)
    app_model = _FakeAppModel(tmp_path, ready=False)
    window = PoseTrainingWindow(app_model)
    window.open_dataset(data.path)
    window.refresh_training_controls()
    assert not window.start_button.isEnabled()
    assert "recording" in window.training_banner.text().lower()
    app_model.ready = True
    window.refresh_training_controls()
    # Still disabled: nothing is labelled yet, and the banner says so.
    assert not window.start_button.isEnabled()
    assert "label" in window.training_banner.text().lower()


def test_the_dataset_list_shows_datasets_under_the_root(qapp, tmp_path):
    _dataset(tmp_path)
    window = PoseTrainingWindow(_FakeAppModel(tmp_path))
    names = [window.dataset_combo.itemText(i) for i in range(window.dataset_combo.count())]
    assert names == ["demo"]


def test_session_videos_are_found_under_the_output_location(qapp, tmp_path):
    folder = tmp_path / "out" / "20261118" / "rig" / "session001"
    folder.mkdir(parents=True)
    for camera in ("left", "right"):
        (folder / f"20261118_rig_session001_{camera}.mp4").write_bytes(b"")
    window = PoseTrainingWindow(_FakeAppModel(tmp_path))
    videos = window.find_session_videos()
    assert [os.path.basename(v) for v in videos] == [
        "20261118_rig_session001_left.mp4", "20261118_rig_session001_right.mp4"]
