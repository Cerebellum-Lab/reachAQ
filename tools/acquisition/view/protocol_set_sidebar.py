"""Set library and experiment builder beside the trial protocol table."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tools.acquisition.view.experiment_entry_editor import ExperimentEntryEditor

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

        experiment_buttons = QHBoxLayout()
        self.new_experiment_button = QPushButton("New experiment")
        self.new_experiment_button.setToolTip(
            "Start an empty experiment to build from your sets"
        )
        self.new_experiment_button.clicked.connect(self._new_experiment)
        experiment_buttons.addWidget(self.new_experiment_button)
        experiment_buttons.addStretch(1)
        layout.addLayout(experiment_buttons)

        self.entry_editor = ExperimentEntryEditor()
        self.entry_editor.changed.connect(self._entries_changed)
        layout.addWidget(self.entry_editor, stretch=1)

        self.save_entries_button = QPushButton("Save experiment")
        self.save_entries_button.setToolTip(
            "Store this set list against the selected experiment"
        )
        self.save_entries_button.clicked.connect(self.save_selected_experiment)
        layout.addWidget(self.save_entries_button)

        buttons = QHBoxLayout()
        self.compile_button = QPushButton("Compile and save")
        self.compile_button.setToolTip(
            "Flatten the selected experiment into a protocol you can run"
        )
        self.compile_button.clicked.connect(self.compile_selected_experiment)
        buttons.addWidget(self.compile_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        # A set's trials are authored through the protocol editor rather than a
        # second one: capture the protocol you built, or open a set back up to
        # revise it and capture it again.
        authoring = QHBoxLayout()
        self.capture_button = QPushButton("Set from protocol")
        self.capture_button.setToolTip(
            "Save the protocol selected for the session as a reusable set"
        )
        self.capture_button.clicked.connect(self._capture_selected_protocol)
        authoring.addWidget(self.capture_button)
        self.open_set_button = QPushButton("Open set")
        self.open_set_button.setToolTip(
            "Publish the selected set as an editable protocol"
        )
        self.open_set_button.clicked.connect(self._open_selected_set)
        authoring.addWidget(self.open_set_button)
        authoring.addStretch(1)
        layout.addLayout(authoring)

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
        self.entry_editor.set_available_sets(self._app_model.trial_protocol_sets)
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

    def new_experiment(self, experiment_id: str, name: str) -> None:
        try:
            created = self._app_model.create_experiment(experiment_id, name)
        except Exception as error:
            logger.exception("Could not create experiment %s", experiment_id)
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Created experiment {}. Add sets, then save.".format(
                created.experiment_id
            )
        )
        self.refresh()
        self._select(self.experiment_list, created.experiment_id)

    def save_selected_experiment(self) -> None:
        item = self.experiment_list.currentItem()
        if item is None:
            self.status_label.setText("Select an experiment to save.")
            return
        experiment_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            saved = self._app_model.save_experiment_entries(
                experiment_id, self.entry_editor.entries()
            )
        except Exception as error:
            logger.exception("Could not save experiment %s", experiment_id)
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Saved experiment {} revision {}, {} entries.".format(
                saved.experiment_id, saved.revision, len(saved.entries)
            )
        )
        self.refresh()

    def _new_experiment(self) -> None:
        identity = self._ask_identity("New experiment")
        if identity is not None:
            self.new_experiment(*identity)

    def _entries_changed(self) -> None:
        self.status_label.setText("Unsaved changes to this experiment.")

    def _select(self, widget: QListWidget, key: str) -> None:
        for index in range(widget.count()):
            if widget.item(index).data(Qt.ItemDataRole.UserRole) == key:
                widget.setCurrentRow(index)
                return

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

    def save_selected_protocol_as_set(self, set_id: str, name: str) -> None:
        protocol = self._app_model.selected_ordered_protocol
        if protocol is None:
            self.status_label.setText("Select a session protocol to capture.")
            return
        try:
            saved = self._app_model.save_protocol_as_set(
                protocol.protocol_id, set_id, name
            )
        except Exception as error:
            logger.exception("Could not capture protocol as a set")
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Saved set {} revision {}.".format(saved.set_id, saved.revision)
        )
        self.refresh()

    def open_selected_set_in_editor(self, protocol_id: str, name: str) -> None:
        item = self.set_list.currentItem()
        if item is None:
            self.status_label.setText("Select a set to open.")
            return
        set_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            saved = self._app_model.open_set_as_protocol(set_id, protocol_id, name)
        except Exception as error:
            logger.exception("Could not open set %s", set_id)
            self.status_label.setText("{}: {}".format(type(error).__name__, error))
            return
        self.status_label.setText(
            "Opened set {} as protocol {}. Edit it, then capture it again.".format(
                set_id, saved.protocol_id
            )
        )
        self.refresh()

    def _capture_selected_protocol(self) -> None:
        identity = self._ask_identity("New set")
        if identity is not None:
            self.save_selected_protocol_as_set(*identity)

    def _open_selected_set(self) -> None:
        identity = self._ask_identity("Open set as protocol")
        if identity is not None:
            self.open_selected_set_in_editor(*identity)

    def _ask_identity(self, title: str):
        identifier, accepted = QInputDialog.getText(self, title, "Identifier:")
        if not accepted or not identifier.strip():
            return None
        name, accepted = QInputDialog.getText(self, title, "Display name:")
        if not accepted or not name.strip():
            return None
        return identifier.strip(), name.strip()

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
        self.entry_editor.load(None)
        self.selection_changed.emit(
            "set", self.set_list.item(row).data(Qt.ItemDataRole.UserRole)
        )

    def _experiment_selected(self, row: int) -> None:
        if row < 0:
            self.entry_editor.load(None)
            return
        self.set_list.setCurrentRow(-1)
        experiment_id = self.experiment_list.item(row).data(
            Qt.ItemDataRole.UserRole
        )
        self.entry_editor.load(self._experiment(experiment_id))
        self.selection_changed.emit("experiment", experiment_id)

    def _experiment(self, experiment_id: str):
        for item in self._app_model.experiment_compositions:
            if item.experiment_id == experiment_id:
                return item
        return None
