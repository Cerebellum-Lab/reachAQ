"""Set library and experiment builder beside the trial protocol table."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ProtocolSetSidebar(QWidget):
    """Choose a set to work on, or an experiment to compile into a protocol."""

    selection_changed = Signal(str, str)

    def __init__(self, app_model, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app_model = app_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Sets"))
        self.set_list = QListWidget()
        self.set_list.currentRowChanged.connect(self._set_selected)
        layout.addWidget(self.set_list, stretch=1)

        layout.addWidget(QLabel("Experiments"))
        self.experiment_list = QListWidget()
        self.experiment_list.currentRowChanged.connect(self._experiment_selected)
        layout.addWidget(self.experiment_list, stretch=1)

        buttons = QHBoxLayout()
        self.compile_button = QPushButton("Compile and save")
        self.compile_button.setToolTip(
            "Flatten the selected experiment into a protocol you can run"
        )
        self.compile_button.clicked.connect(self.compile_selected_experiment)
        buttons.addWidget(self.compile_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #5f6772;")
        layout.addWidget(self.status_label)

        self.refresh()

    def current_selection(self) -> Tuple[str, str]:
        item = self.experiment_list.currentItem()
        if item is not None:
            return "experiment", item.data(Qt.ItemDataRole.UserRole)
        item = self.set_list.currentItem()
        if item is not None:
            return "set", item.data(Qt.ItemDataRole.UserRole)
        return "", ""

    def refresh(self) -> None:
        self._fill(
            self.set_list,
            [
                (
                    item.set_id,
                    "{} ({} trials, rev {})".format(
                        item.name, item.trial_count, item.revision
                    ),
                )
                for item in self._app_model.trial_protocol_sets
            ],
        )
        self._fill(
            self.experiment_list,
            [
                (
                    item.experiment_id,
                    "{} ({} entries, rev {})".format(
                        item.name, len(item.entries), item.revision
                    ),
                )
                for item in self._app_model.experiment_compositions
            ],
        )

    def compile_selected_experiment(self) -> None:
        item = self.experiment_list.currentItem()
        if item is None:
            self.status_label.setText("Select an experiment to compile.")
            return
        experiment_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            saved = self._app_model.compile_and_save_experiment(experiment_id)
        except Exception as error:
            logger.exception("Could not compile experiment %s", experiment_id)
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Compiled {} into protocol {} revision {}, {} trials.".format(
                experiment_id, saved.protocol_id, saved.revision, saved.trial_count
            )
        )
        self.refresh()

    @staticmethod
    def _fill(widget: QListWidget, rows) -> None:
        """Repopulate a list, keeping the previous selection where it survives."""
        previous = widget.currentItem()
        previous_key = (
            None if previous is None else previous.data(Qt.ItemDataRole.UserRole)
        )
        widget.blockSignals(True)
        widget.clear()
        for key, label in rows:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            widget.addItem(item)
        widget.blockSignals(False)
        if previous_key is None:
            return
        for index in range(widget.count()):
            if widget.item(index).data(Qt.ItemDataRole.UserRole) == previous_key:
                widget.setCurrentRow(index)
                return

    def _set_selected(self, row: int) -> None:
        if row < 0:
            return
        self.experiment_list.setCurrentRow(-1)
        self.selection_changed.emit(
            "set", self.set_list.item(row).data(Qt.ItemDataRole.UserRole)
        )

    def _experiment_selected(self, row: int) -> None:
        if row < 0:
            return
        self.set_list.setCurrentRow(-1)
        self.selection_changed.emit(
            "experiment",
            self.experiment_list.item(row).data(Qt.ItemDataRole.UserRole),
        )
