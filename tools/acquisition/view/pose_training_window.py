"""Tools -> Pose Training: label frames, train a live pose model, check it, deploy it.

One window, four steps, one dataset at a time:

    Frames    pick frames from recorded session videos (diverse or by hand),
              optionally pre-placed by the live model
    Label     place each part, Done to accept the frame
    Train     choose held-out sessions, start / pause / resume / cancel
    Evaluate  compare a trained model with the live one, check its speed,
              use it live, revert, export

Everything that touches the GPU (training, pre-labelling, evaluation, the
speed check) runs in a child process and is refused while a session records.
Model events arrive from worker threads; the _Bridge signal carries them to
the Qt thread.
"""

from __future__ import annotations

import glob
import os
import shutil
import threading
import time
import typing

import numpy
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.keypoint.dataset import (
    DATASET_FILE,
    LABELLED,
    PENDING,
    PoseDataset,
    import_dlc,
    pick_diverse,
    read_frames,
    video_frame_count,
)
from autotrainer.inference.keypoint.evaluate import COVERAGES, FRAME_BUDGET_MS
from tools.acquisition.model.pose_training_model import (
    DEFAULT_EPOCHS,
    TrainingState,
    evaluate_models,
    measure_model_speed,
    prelabel_frames,
    run_job,
)
from tools.acquisition.view.keypoint_label_view import KeypointLabelView, part_colours

logger = get_verbose_logger(__name__)

SESSION_RECORDING_STATUS = "session_recording_status"
RECORDING_BANNER = "Training is unavailable while a session is recording."


class _Bridge(QObject):
    """Carries callbacks from worker threads to the Qt thread."""
    call = Signal(object)


class PoseTrainingWindow(QMainWindow):

    def __init__(self, app_model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pose Training")
        self.resize(1200, 800)
        self._app_model = app_model
        self._training = app_model.pose_training
        self._dataset: typing.Optional[PoseDataset] = None
        self.current_frame = ""
        self._video_preview: typing.Optional[str] = None
        self._busy = False
        self._started_at = 0.0
        self._first_epoch: typing.Optional[int] = None
        self._curve: typing.List[typing.Tuple[int, float]] = []

        self._bridge = _Bridge()
        self._bridge.call.connect(lambda function: function(),
                                  Qt.ConnectionType.QueuedConnection)
        self._training.property_changed += self._on_training_changed
        app_model.property_changed += self._on_app_changed

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(self._build_dataset_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_frames_tab(), "1  Frames")
        self.tabs.addTab(self._build_label_tab(), "2  Label")
        self.tabs.addTab(self._build_train_tab(), "3  Train")
        self.tabs.addTab(self._build_evaluate_tab(), "4  Evaluate")
        self.tabs.currentChanged.connect(lambda _: self._refresh())
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.reload_datasets()
        self._refresh()

    # -- plumbing ---------------------------------------------------------

    def _in_ui(self, function: typing.Callable[[], None]) -> None:
        self._bridge.call.emit(function)

    def _run_in_background(self, describe: str, job: typing.Callable[[], typing.Any],
                           done: typing.Callable[[typing.Any], None]) -> None:
        """Run job() on a thread, then done(result) on the Qt thread."""
        if self._busy:
            self.statusBar().showMessage("Another job is still running", 4000)
            return
        self._busy = True
        self.statusBar().showMessage(describe + "…")

        def work():
            try:
                result, error = job(), None
            except Exception as err:  # noqa: BLE001 - reported in the window
                logger.exception("%s failed", describe)
                result, error = None, f"{type(err).__name__}: {err}"

            def finish():
                self._busy = False
                if error:
                    self.statusBar().showMessage(f"{describe} failed", 8000)
                    QMessageBox.warning(self, "Pose Training", f"{describe} failed:\n\n{error}")
                else:
                    self.statusBar().showMessage(f"{describe}: done", 4000)
                    done(result)
                self._refresh()
            self._in_ui(finish)

        threading.Thread(target=work, name="pose-training-job", daemon=True).start()

    def _gpu_ready(self) -> bool:
        if self._training.is_ready():
            return True
        QMessageBox.information(self, "Pose Training", RECORDING_BANNER)
        return False

    def _on_training_changed(self, name, value, previous) -> None:
        self._in_ui(lambda: self._training_changed(name, value))

    def _on_app_changed(self, name, value, previous) -> None:
        if name == SESSION_RECORDING_STATUS:
            self._in_ui(self.refresh_training_controls)

    def closeEvent(self, event):  # noqa: N802 - Qt's name
        # Training keeps running in the app; only the view goes away.
        self._save()
        super().closeEvent(event)

    # -- dataset bar ------------------------------------------------------

    def _build_dataset_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(220)
        self.dataset_combo.activated.connect(
            lambda index: self.open_dataset(self.dataset_combo.itemData(index)))
        bar.addWidget(self.dataset_combo)
        new_button = QPushButton("New…")
        new_button.clicked.connect(self._new_dataset)
        bar.addWidget(new_button)
        import_button = QPushButton("Import DeepLabCut…")
        import_button.clicked.connect(self._import_dlc)
        bar.addWidget(import_button)
        self.counts_label = QLabel("")
        bar.addWidget(self.counts_label, 1)
        return bar

    def reload_datasets(self, select: str = "") -> None:
        root = self._training.datasets_root
        paths = sorted(glob.glob(os.path.join(root, "*", DATASET_FILE)))
        self.dataset_combo.clear()
        for path in paths:
            folder = os.path.dirname(path)
            self.dataset_combo.addItem(os.path.basename(folder), folder)
        if select:
            index = self.dataset_combo.findData(select)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        if self.dataset_combo.count() and self._dataset is None:
            self.open_dataset(self.dataset_combo.currentData())

    def open_dataset(self, path: str) -> None:
        self._save()
        try:
            self._dataset = PoseDataset.open(path)
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Pose Training", f"Cannot open {path}:\n\n{err}")
            return
        index = self.dataset_combo.findData(path)
        if index >= 0:
            self.dataset_combo.setCurrentIndex(index)
        self.current_frame = ""
        self._reload_frame_list()
        self._reload_parts()
        self._reload_sessions()
        self._reload_models()
        self._refresh()

    def _save(self) -> None:
        if self._dataset is not None:
            self._dataset.save()

    def _live_parts(self) -> typing.List[str]:
        """The live model's part names, to start a new dataset with."""
        location = self._app_model.inference.model_location
        sidecar = os.path.join(location or "", "keypoint_pose.yaml")
        if location and os.path.isfile(sidecar):
            from autotrainer.inference.keypoint.train import read_bodyparts
            return read_bodyparts(sidecar)
        return list(self._dataset.bodyparts) if self._dataset else []

    def _new_dataset(self) -> None:
        name, ok = QInputDialog.getText(self, "New dataset", "Name:")
        if not ok or not name.strip():
            return
        parts_text, ok = QInputDialog.getMultiLineText(
            self, "New dataset", "Body parts, one per line, in labelling order:",
            "\n".join(self._live_parts()))
        if not ok:
            return
        parts = [line.strip() for line in parts_text.splitlines() if line.strip()]
        try:
            data = PoseDataset.create(self._training.datasets_root, name.strip(), parts)
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "New dataset", str(err))
            return
        self.reload_datasets(select=data.path)
        self.open_dataset(data.path)

    def _import_dlc(self) -> None:
        project = QFileDialog.getExistingDirectory(self, "DeepLabCut project folder")
        if not project:
            return
        name, ok = QInputDialog.getText(self, "Import DeepLabCut", "Dataset name:",
                                        text=os.path.basename(project.rstrip("/\\")))
        if not ok or not name.strip():
            return
        root = self._training.datasets_root

        def done(data):
            self.reload_datasets(select=data.path)
            self.open_dataset(data.path)

        self._run_in_background("Importing the DeepLabCut project",
                                lambda: import_dlc(project, root, name.strip()), done)

    # -- frames tab -------------------------------------------------------

    def _build_frames_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QVBoxLayout()
        left.addWidget(QLabel("Session videos"))
        self.video_list = QListWidget()
        self.video_list.currentItemChanged.connect(lambda item, _: self._preview_video(item))
        left.addWidget(self.video_list, 1)
        buttons = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._reload_videos)
        buttons.addWidget(refresh)
        browse = QPushButton("Add video…")
        browse.clicked.connect(self._browse_video)
        buttons.addWidget(browse)
        left.addLayout(buttons)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.preview = QLabel("Choose a video")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(384, 384)
        right.addWidget(self.preview, 1)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self._show_preview_frame)
        right.addWidget(self.frame_slider)
        row = QHBoxLayout()
        self.add_frame_button = QPushButton("Add this frame")
        self.add_frame_button.clicked.connect(
            lambda: self._add_frames([self.frame_slider.value()]))
        row.addWidget(self.add_frame_button)
        row.addStretch(1)
        row.addWidget(QLabel("Pick"))
        self.pick_count = QSpinBox()
        self.pick_count.setRange(1, 200)
        self.pick_count.setValue(20)
        row.addWidget(self.pick_count)
        self.pick_button = QPushButton("diverse frames")
        self.pick_button.clicked.connect(self._pick_diverse)
        row.addWidget(self.pick_button)
        right.addLayout(row)
        self.prelabel_check = QCheckBox("Pre-label new frames with the live model")
        self.prelabel_check.setChecked(True)
        right.addWidget(self.prelabel_check)
        layout.addLayout(right, 2)
        self._reload_videos()
        return tab

    def find_session_videos(self) -> typing.List[str]:
        """Recorded camera videos under the output location, newest session first."""
        root = os.path.expanduser(str(self._app_model.output_location or ""))
        if not root or not os.path.isdir(root):
            return []
        videos = glob.glob(os.path.join(root, "*", "*", "*", "*.mp4"))
        return sorted(videos, key=lambda path: (os.path.dirname(path), path),
                      reverse=False)

    def _reload_videos(self) -> None:
        self.video_list.clear()
        for path in reversed(self.find_session_videos()):
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.video_list.addItem(item)

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Session video", "", "Video (*.mp4 *.avi)")
        if path:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.video_list.insertItem(0, item)
            self.video_list.setCurrentItem(item)

    def _preview_video(self, item) -> None:
        self._video_preview = item.data(Qt.ItemDataRole.UserRole) if item else None
        count = video_frame_count(self._video_preview) if self._video_preview else 0
        self.frame_slider.setRange(0, max(0, count - 1))
        self.frame_slider.setValue(0)
        self._show_preview_frame(0)

    def _show_preview_frame(self, index: int) -> None:
        if not self._video_preview:
            return
        frame = read_frames(self._video_preview, [index]).get(index)
        if frame is None:
            self.preview.setText("Frame not readable")
            return
        from PySide6.QtGui import QImage
        grey = frame.mean(axis=2).astype("uint8") if frame.ndim == 3 else frame
        grey = numpy.ascontiguousarray(grey)
        image = QImage(grey.data, grey.shape[1], grey.shape[0], grey.shape[1],
                       QImage.Format.Format_Grayscale8).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def _pick_diverse(self) -> None:
        if not (self._dataset and self._video_preview):
            return
        video, count = self._video_preview, self.pick_count.value()
        stem = os.path.splitext(os.path.basename(video))[0]
        taken = [int(row.frame.rsplit("_", 1)[-1][:-4]) for row in self._dataset.rows
                 if os.path.basename(row.frame).startswith(stem + "_")]
        self._run_in_background(
            "Picking diverse frames",
            lambda: pick_diverse(video, count, exclude=taken),
            lambda indices: self._add_frames(indices))

    def _add_frames(self, indices: typing.Sequence[int]) -> None:
        if not (self._dataset and self._video_preview and len(indices)):
            return
        try:
            added = self._dataset.add_frames(self._video_preview, indices)
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Add frames", str(err))
            return
        self._save()
        self._reload_frame_list()
        self.statusBar().showMessage(f"Added {len(added)} frame(s)", 4000)
        live = self._app_model.inference.model_location
        if added and live and self.prelabel_check.isChecked() and self._training.is_ready():
            path = self._dataset.path
            self._run_in_background(
                "Pre-labelling with the live model",
                lambda: run_job(prelabel_frames, live, path, added),
                self._apply_prelabels)
        self._refresh()

    def _apply_prelabels(self, predictions: typing.Dict[str, list]) -> None:
        for frame, points in predictions.items():
            try:
                row = self._dataset.row(frame)
            except KeyError:
                continue
            if row.status == PENDING and numpy.isnan(row.points).all():
                self._dataset.set_points(frame, numpy.asarray(points, "float32"))
        self._save()
        if self.current_frame in predictions:
            self.show_frame(self.current_frame)

    # -- label tab --------------------------------------------------------

    def _build_label_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.frame_filter = QComboBox()
        self.frame_filter.addItems(["To label", "Labelled", "All"])
        self.frame_filter.currentIndexChanged.connect(lambda _: self._reload_frame_list())
        left_layout.addWidget(self.frame_filter)
        self.frame_list = QListWidget()
        self.frame_list.currentItemChanged.connect(
            lambda item, _: item and self.show_frame(item.data(Qt.ItemDataRole.UserRole)))
        left_layout.addWidget(self.frame_list, 1)
        remove = QPushButton("Remove frame")
        remove.clicked.connect(self._remove_frame)
        left_layout.addWidget(remove)
        splitter.addWidget(left)

        self.label_view = KeypointLabelView()
        self.label_view.points_changed.connect(self._points_changed)
        self.label_view.selected_part_changed.connect(self._select_part_row)
        splitter.addWidget(self.label_view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Parts (click to place, right-click to clear)"))
        self.part_list = QListWidget()
        self.part_list.currentRowChanged.connect(self._part_row_changed)
        right_layout.addWidget(self.part_list, 1)
        self.frame_status = QLabel("")
        self.frame_status.setWordWrap(True)
        right_layout.addWidget(self.frame_status)
        done = QPushButton("Done  (Enter)")
        done.clicked.connect(self.mark_done)
        right_layout.addWidget(done)
        reopen = QPushButton("Back to To label")
        reopen.clicked.connect(self._mark_pending)
        right_layout.addWidget(reopen)
        right_layout.addWidget(QLabel(
            "Keys: Enter done, Delete clear part, Up/Down part,\n"
            "PgUp/PgDn frame, 0 fit, wheel zoom"))
        splitter.addWidget(right)
        splitter.setSizes([220, 700, 260])
        layout.addWidget(splitter)

        for key, slot in ((Qt.Key.Key_Return, self.mark_done), (Qt.Key.Key_Enter, self.mark_done),
                          (Qt.Key.Key_PageDown, lambda: self._step_frame(1)),
                          (Qt.Key.Key_PageUp, lambda: self._step_frame(-1))):
            shortcut = QShortcut(QKeySequence(key), tab)
            shortcut.activated.connect(slot)
        return tab

    def _reload_parts(self) -> None:
        self.part_list.clear()
        if not self._dataset:
            return
        for name, colour in zip(self._dataset.bodyparts,
                                part_colours(len(self._dataset.bodyparts))):
            swatch = QPixmap(12, 12)
            swatch.fill(colour)
            self.part_list.addItem(QListWidgetItem(QIcon(swatch), name))
        self.part_list.setCurrentRow(0)

    def _filtered_rows(self):
        if not self._dataset:
            return []
        wanted = {0: (PENDING,), 1: (LABELLED,), 2: (PENDING, LABELLED)}[
            self.frame_filter.currentIndex()]
        return [row for row in self._dataset.rows if row.status in wanted]

    def _reload_frame_list(self) -> None:
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for row in self._filtered_rows():
            item = QListWidgetItem(("✓ " if row.status == LABELLED else "· ") + row.frame)
            item.setData(Qt.ItemDataRole.UserRole, row.frame)
            self.frame_list.addItem(item)
            if row.frame == self.current_frame:
                self.frame_list.setCurrentItem(item)
        self.frame_list.blockSignals(False)
        if not self.current_frame and self.frame_list.count():
            self.frame_list.setCurrentRow(0)

    def show_frame(self, frame: str) -> None:
        if not self._dataset or not frame:
            return
        row = self._dataset.row(frame)
        self.current_frame = frame
        self.label_view.set_frame(self._dataset.image(frame), row.points,
                                  self._dataset.bodyparts, pending=row.status == PENDING)
        self.label_view.selected_part = self._first_unplaced(row.points)
        self._update_frame_status()

    @staticmethod
    def _first_unplaced(points) -> int:
        missing = numpy.flatnonzero(numpy.isnan(points).any(axis=1))
        return int(missing[0]) if len(missing) else 0

    def _update_frame_status(self) -> None:
        if not (self._dataset and self.current_frame):
            self.frame_status.setText("")
            return
        row = self._dataset.row(self.current_frame)
        placed = int((~numpy.isnan(row.points).any(axis=1)).sum())
        held = " (held out)" if row.session in self._dataset.val_sessions else ""
        self.frame_status.setText(
            f"{row.frame}\nsession {row.session}{held}\n{row.status}, "
            f"{placed} of {len(self._dataset.bodyparts)} parts placed")

    def _points_changed(self, points) -> None:
        if self._dataset and self.current_frame:
            self._dataset.set_points(self.current_frame, points)
            self._save()
            self._update_frame_status()

    def _select_part_row(self, index: int) -> None:
        self.part_list.blockSignals(True)
        self.part_list.setCurrentRow(index)
        self.part_list.blockSignals(False)

    def _part_row_changed(self, index: int) -> None:
        if index >= 0:
            self.label_view.selected_part = index
            self.label_view.setFocus()

    def mark_done(self) -> None:
        if not (self._dataset and self.current_frame):
            return
        self._dataset.set_points(self.current_frame, self.label_view.points())
        self._dataset.set_status(self.current_frame, LABELLED)
        self._save()
        following = self._next_pending(self.current_frame)
        self.current_frame = following or self.current_frame
        self._reload_frame_list()
        if following:
            self.show_frame(following)
        else:
            self.show_frame(self.current_frame)
            self.statusBar().showMessage("No frames left to label", 4000)
        self._refresh()

    def _mark_pending(self) -> None:
        if self._dataset and self.current_frame:
            self._dataset.set_status(self.current_frame, PENDING)
            self._save()
            self.label_view.set_pending(True)
            self._reload_frame_list()
            self._refresh()

    def _next_pending(self, after: str) -> str:
        rows = self._dataset.rows
        start = next((i for i, row in enumerate(rows) if row.frame == after), -1)
        for row in rows[start + 1:] + rows[:start + 1]:
            if row.status == PENDING and row.frame != after:
                return row.frame
        return ""

    def _step_frame(self, step: int) -> None:
        count = self.frame_list.count()
        if count:
            self.frame_list.setCurrentRow((self.frame_list.currentRow() + step) % count)

    def _remove_frame(self) -> None:
        if not (self._dataset and self.current_frame):
            return
        if QMessageBox.question(self, "Remove frame",
                                f"Remove {self.current_frame} from the dataset?") \
                != QMessageBox.StandardButton.Yes:
            return
        self._dataset.remove(self.current_frame)
        self._save()
        self.current_frame = ""
        self._reload_frame_list()
        self._refresh()

    # -- train tab --------------------------------------------------------

    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QVBoxLayout()
        left.addWidget(QLabel("Held-out sessions (checked = used only to evaluate)"))
        self.session_list = QListWidget()
        self.session_list.itemChanged.connect(self._session_toggled)
        left.addWidget(self.session_list, 1)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.training_banner = QLabel("")
        self.training_banner.setWordWrap(True)
        right.addWidget(self.training_banner)
        row = QHBoxLayout()
        row.addWidget(QLabel("Epochs"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        row.addWidget(self.epochs_spin)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_training)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._pause_training)
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self._resume_training)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_training)
        for button in (self.start_button, self.pause_button, self.resume_button,
                       self.cancel_button):
            row.addWidget(button)
        row.addStretch(1)
        right.addLayout(row)
        self.train_progress = QProgressBar()
        right.addWidget(self.train_progress)
        self.train_status = QLabel("")
        right.addWidget(self.train_status)
        self.curve_plot = pg.PlotWidget()
        self.curve_plot.setLabel("left", "held-out median error", units="px")
        self.curve_plot.setLabel("bottom", "epoch")
        self._curve_item = self.curve_plot.plot([], [], symbol="o")
        right.addWidget(self.curve_plot, 1)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(500)
        self.train_log.setMaximumHeight(160)
        right.addWidget(self.train_log)
        layout.addLayout(right, 2)
        return tab

    def _reload_sessions(self) -> None:
        self.session_list.blockSignals(True)
        self.session_list.clear()
        if self._dataset:
            for session in self._dataset.sessions():
                count = len(self._dataset.frames_from(session))
                item = QListWidgetItem(f"{session}  ({count} frames)")
                item.setData(Qt.ItemDataRole.UserRole, session)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if session in self._dataset.val_sessions
                                   else Qt.CheckState.Unchecked)
                self.session_list.addItem(item)
        self.session_list.blockSignals(False)

    def _session_toggled(self, item) -> None:
        if self._dataset:
            self._dataset.set_val(item.data(Qt.ItemDataRole.UserRole),
                                  item.checkState() == Qt.CheckState.Checked)
            self._save()
            self._refresh()

    def refresh_training_controls(self) -> None:
        state = self._training.state
        ready = self._training.is_ready()
        counts = self._dataset.counts() if self._dataset else {"train": 0, "val": 0}
        if not ready:
            banner = RECORDING_BANNER
        elif self._dataset is None:
            banner = "Create or import a dataset first."
        elif counts["train"] == 0:
            banner = "Label some frames first: no labelled frames are outside the held-out sessions."
        elif counts["val"] == 0:
            banner = ("No held-out frames: the model can be trained but not evaluated. "
                      "Check at least one session on the left.")
        else:
            banner = f"{counts['train']} frames to train on, {counts['val']} held out."
        if state is TrainingState.PAUSED and not ready:
            banner = "Paused for the recording. Resume once the session has stopped."
        elif state is TrainingState.FAILED:
            banner = f"Training failed: {self._training.error}"
        self.training_banner.setText(banner)
        idle = state in (TrainingState.IDLE, TrainingState.FINISHED, TrainingState.FAILED)
        self.start_button.setEnabled(idle and ready and counts["train"] > 0)
        self.pause_button.setEnabled(state is TrainingState.RUNNING)
        self.resume_button.setEnabled(state is TrainingState.PAUSED and ready)
        self.cancel_button.setEnabled(state in (TrainingState.RUNNING, TrainingState.PAUSED))
        self.epochs_spin.setEnabled(idle)
        self.session_list.setEnabled(idle)

    def _start_training(self) -> None:
        if not self._dataset or not self._gpu_ready():
            return
        self._save()
        self._curve = []
        self._curve_item.setData([], [])
        self.train_log.clear()
        try:
            self._training.start(self._dataset.path, self.epochs_spin.value())
        except RuntimeError as err:
            QMessageBox.warning(self, "Pose Training", str(err))
        self._started_at, self._first_epoch = time.monotonic(), None
        self._refresh()

    def _pause_training(self) -> None:
        self._run_in_background("Pausing training", self._training.pause, lambda _: None)

    def _resume_training(self) -> None:
        if not self._gpu_ready():
            return
        try:
            self._training.resume()
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Pose Training", str(err))
        self._started_at, self._first_epoch = time.monotonic(), None
        self._refresh()

    def _cancel_training(self) -> None:
        if QMessageBox.question(self, "Cancel training",
                                "Stop this run and discard it?") \
                == QMessageBox.StandardButton.Yes:
            self._run_in_background("Cancelling training", self._training.cancel,
                                    lambda _: None)

    def _training_changed(self, name: str, value) -> None:
        if name == self._training.PROGRESS:
            epoch, epochs, line = value
            self.train_progress.setRange(0, epochs)
            self.train_progress.setValue(epoch)
            # The ETA counts only epochs run since this start or resume.
            if self._first_epoch is None:
                self._first_epoch = epoch - 1
            elapsed = time.monotonic() - self._started_at
            remaining = ""
            if elapsed > 0 and self._started_at:
                done_here = max(1, epoch - self._first_epoch)
                left = elapsed / done_here * (epochs - epoch)
                remaining = f", about {left / 60:.0f} min left" if left >= 60 \
                    else f", about {left:.0f} s left"
            self.train_status.setText(f"epoch {epoch} of {epochs}{remaining}")
            if "median_px" in line:
                self._curve.append((epoch, line["median_px"]))
                self._curve_item.setData(*zip(*self._curve))
        elif name == self._training.LOG:
            self.train_log.appendPlainText(str(value).rstrip())
        elif name == self._training.STATE:
            if value is TrainingState.FINISHED:
                self.train_status.setText("Finished. Open the Evaluate step to compare it.")
                self._reload_models(select=self._training.run_dir)
            self._refresh()

    # -- evaluate tab -----------------------------------------------------

    def _build_evaluate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("Trained model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(320)
        row.addWidget(self.model_combo, 1)
        browse = QPushButton("Other…")
        browse.clicked.connect(self._browse_model)
        row.addWidget(browse)
        self.evaluate_button = QPushButton("Evaluate")
        self.evaluate_button.clicked.connect(self._evaluate)
        row.addWidget(self.evaluate_button)
        self.speed_button = QPushButton("Measure speed")
        self.speed_button.clicked.connect(self._measure_speed)
        row.addWidget(self.speed_button)
        layout.addLayout(row)

        self.live_label = QLabel("")
        layout.addWidget(self.live_label)
        self.summary_table = QTableWidget(0, 3)
        self.summary_table.setHorizontalHeaderLabels(["", "This model", "Live model"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary_table.setMaximumHeight(230)
        layout.addWidget(self.summary_table)
        self.parts_table = QTableWidget(0, 5)
        self.parts_table.setHorizontalHeaderLabels(
            ["Part", "Held-out labels", "Detected", "Median px", "Live median px"])
        self.parts_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parts_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.parts_table, 1)
        self.speed_label = QLabel("")
        layout.addWidget(self.speed_label)

        row = QHBoxLayout()
        self.deploy_button = QPushButton("Use as live model")
        self.deploy_button.clicked.connect(self._deploy)
        row.addWidget(self.deploy_button)
        revert = QPushButton("Revert to previous live model")
        revert.clicked.connect(self._revert)
        row.addWidget(revert)
        export = QPushButton("Export…")
        export.clicked.connect(self._export)
        row.addWidget(export)
        row.addStretch(1)
        layout.addLayout(row)
        return tab

    def _reload_models(self, select: str = "") -> None:
        self.model_combo.clear()
        if not self._dataset:
            return
        folder = os.path.join(self._training.models_root, self._dataset.name)
        for sidecar in sorted(glob.glob(os.path.join(folder, "*", "keypoint_pose.yaml")),
                              reverse=True):
            path = os.path.dirname(sidecar)
            self.model_combo.addItem(os.path.basename(path), path)
        if select:
            index = self.model_combo.findData(select)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

    def _browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Model folder")
        if path:
            self.model_combo.insertItem(0, path, path)
            self.model_combo.setCurrentIndex(0)

    def _selected_model(self) -> str:
        return self.model_combo.currentData() or ""

    def _evaluate(self) -> None:
        model = self._selected_model()
        if not (self._dataset and model) or not self._gpu_ready():
            return
        live = self._app_model.inference.model_location
        paths = [model] + ([live] if live and os.path.normpath(live) != os.path.normpath(model)
                           else [])
        dataset = self._dataset.path
        self._run_in_background("Evaluating on the held-out frames",
                                lambda: run_job(evaluate_models, dataset, paths),
                                self._show_evaluation)

    def _show_evaluation(self, results: typing.List[dict]) -> None:
        candidate = results[0]
        live = results[1] if len(results) > 1 else None
        live_path = self._app_model.inference.model_location
        self.live_label.setText(f"Live model: {live_path or 'none set'}"
                                + ("  (this model)" if live is None and live_path else ""))

        def cell(result, key):
            if result is None:
                return ""
            if "error" in result:
                return result["error"] if key == "n" else ""
            if key == "n":
                return f"{result['n']} keypoints on {result['frames']} frames"
            if key == "detected":
                return f"{100 * result['detected']:.1f}%"
            return f"{result['coverage_median'][key]:.2f}"

        rows = [("Held-out", "n")] + [
            (f"Median px, most confident {c}%", c) for c in COVERAGES] + [
            (f"Detected at gate {candidate.get('gate', 0):.2f}", "detected")]
        self.summary_table.setRowCount(len(rows))
        for index, (title, key) in enumerate(rows):
            self.summary_table.setItem(index, 0, QTableWidgetItem(title))
            self.summary_table.setItem(index, 1, QTableWidgetItem(cell(candidate, key)))
            self.summary_table.setItem(index, 2, QTableWidgetItem(cell(live, key)))

        parts = candidate.get("parts", {})
        live_parts = (live or {}).get("parts", {})
        self.parts_table.setRowCount(len(parts))
        for index, (name, part) in enumerate(parts.items()):
            note = "  (few labels)" if part["indicative"] else ""
            values = [name + note, str(part["n"]),
                      "" if not part["n"] else f"{100 * part['detected']:.0f}%",
                      "" if not part["n"] else f"{part['median_px']:.2f}",
                      "" if not live_parts.get(name, {}).get("n")
                      else f"{live_parts[name]['median_px']:.2f}"]
            for column, value in enumerate(values):
                self.parts_table.setItem(index, column, QTableWidgetItem(value))
        if candidate.get("n") == 0:
            QMessageBox.information(self, "Evaluate",
                                    "There are no labelled frames in the held-out sessions.")

    def _measure_speed(self) -> None:
        model = self._selected_model()
        if not model or not self._gpu_ready():
            return

        def show(result):
            verdict = "within" if result["p99_ms"] < result["budget_ms"] else "OVER"
            self.speed_label.setText(
                f"Two-camera batch: p50 {result['p50_ms']:.2f} ms, p99 {result['p99_ms']:.2f} ms; "
                f"{verdict} the {FRAME_BUDGET_MS:.2f} ms frame budget. Measured on this "
                f"computer's GPU while nothing else used it.")

        self._run_in_background("Measuring speed",
                                lambda: run_job(measure_model_speed, model), show)

    def _deploy(self) -> None:
        model = self._selected_model()
        if not model:
            return
        if QMessageBox.question(
                self, "Use as live model",
                f"Make this the live pose model?\n\n{model}\n\n"
                "It is used from the next time live inference starts.") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            self._app_model.deploy_pose_model(model)
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Use as live model", str(err))
            return
        self.statusBar().showMessage("Live model changed", 6000)

    def _revert(self) -> None:
        try:
            previous = self._app_model.revert_pose_model()
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Revert", str(err))
            return
        self.statusBar().showMessage(f"Live model back to {previous}", 6000)

    def _export(self) -> None:
        model = self._selected_model()
        if not model:
            return
        target = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not target:
            return
        name = f"{self._dataset.name if self._dataset else 'pose'}-{os.path.basename(model)}"
        destination = os.path.join(target, name)
        try:
            shutil.copytree(model, destination,
                            ignore=shutil.ignore_patterns("checkpoint.pt*"))
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, "Export", str(err))
            return
        self.statusBar().showMessage(f"Exported to {destination}", 6000)

    # -- shared -----------------------------------------------------------

    def _refresh(self) -> None:
        if self._dataset:
            counts = self._dataset.counts()
            self.counts_label.setText(
                f"{len(self._dataset.bodyparts)} parts · {counts['labelled']} labelled, "
                f"{counts['pending']} to label · {counts['val']} held out "
                f"({len(self._dataset.val_sessions)} of {counts['sessions']} sessions)")
        else:
            self.counts_label.setText("No dataset")
        has_video = self._dataset is not None and bool(self._video_preview)
        self.add_frame_button.setEnabled(has_video)
        self.pick_button.setEnabled(has_video and not self._busy)
        self.prelabel_check.setEnabled(bool(self._app_model.inference.model_location))
        ready = self._training.is_ready()
        has_model = bool(self._selected_model())
        self.evaluate_button.setEnabled(has_model and ready and not self._busy)
        self.speed_button.setEnabled(has_model and ready and not self._busy)
        self.deploy_button.setEnabled(has_model and ready)
        self.refresh_training_controls()
