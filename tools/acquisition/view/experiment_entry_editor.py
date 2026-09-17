"""Arrange the ordered set list an experiment compiles from.

One row per appearance: which set, how many times in a row, and whether the
trials inside each appearance are shuffled. Order is the row order.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tools.acquisition.model.trial_protocol_set import (
    MAX_SET_REPEAT,
    ExperimentComposition,
    ExperimentSetEntry,
)


class ExperimentEntryEditor(QWidget):
    """Add, order, repeat and shuffle the sets an experiment is built from."""

    changed = Signal()

    SET_COLUMN = 0
    REPEAT_COLUMN = 1
    SHUFFLE_COLUMN = 2

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._available = ()
        self._composition: Optional[ExperimentComposition] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Set", "x", "Shuffle"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(
            self.SET_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        header.setSectionResizeMode(
            self.REPEAT_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            self.SHUFFLE_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.table, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(2)
        self.set_selector = QComboBox()
        self.set_selector.setToolTip("Set to append to this experiment")
        controls.addWidget(self.set_selector, stretch=1)
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self._add_selected_set)
        controls.addWidget(self.add_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        controls.addWidget(self.remove_button)
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        controls.addWidget(self.up_button)
        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(lambda: self.move_selected(1))
        controls.addWidget(self.down_button)
        layout.addLayout(controls)

    def set_available_sets(self, sets: Sequence) -> None:
        """Offer these sets for appending, remembering each one's revision."""
        self._available = tuple(sets)
        self.set_selector.clear()
        for item in self._available:
            self.set_selector.addItem(
                "{} (rev {})".format(item.name, item.revision), item.set_id
            )

    def load(self, composition: Optional[ExperimentComposition]) -> None:
        self._composition = composition
        entries = () if composition is None else composition.entries
        self._fill(entries)

    def entries(self) -> Tuple[ExperimentSetEntry, ...]:
        """Read the rows back as entries, in row order."""
        result = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.SET_COLUMN)
            if item is None:
                continue
            set_id = item.data(Qt.ItemDataRole.UserRole)
            revision = item.data(Qt.ItemDataRole.UserRole + 1)
            repeat_widget = self.table.cellWidget(row, self.REPEAT_COLUMN)
            shuffle_widget = self.table.cellWidget(row, self.SHUFFLE_COLUMN)
            seed = item.data(Qt.ItemDataRole.UserRole + 2)
            result.append(
                ExperimentSetEntry(
                    set_id=set_id,
                    set_revision=int(revision),
                    repeat=int(repeat_widget.value()),
                    shuffle_trials=bool(shuffle_widget.isChecked()),
                    shuffle_seed=seed,
                )
            )
        return tuple(result)

    def add_entry(self, set_id: str) -> None:
        """Append one appearance of a set, pinned at its current revision."""
        revision = self._current_revision(set_id)
        if revision is None:
            return
        entries = self.entries() + (
            ExperimentSetEntry(set_id=set_id, set_revision=revision),
        )
        self._fill(entries)
        self.changed.emit()

    def remove_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        entries = list(self.entries())
        del entries[row]
        self._fill(tuple(entries))
        self.table.setCurrentCell(min(row, self.table.rowCount() - 1), 0)
        self.changed.emit()

    def move_selected(self, offset: int) -> None:
        row = self.table.currentRow()
        target = row + int(offset)
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        entries = list(self.entries())
        entries[row], entries[target] = entries[target], entries[row]
        self._fill(tuple(entries))
        self.table.setCurrentCell(target, 0)
        self.changed.emit()

    def _add_selected_set(self) -> None:
        set_id = self.set_selector.currentData()
        if set_id:
            self.add_entry(set_id)

    def _current_revision(self, set_id: str) -> Optional[int]:
        for item in self._available:
            if item.set_id == set_id:
                return int(item.revision)
        return None

    def _fill(self, entries: Tuple[ExperimentSetEntry, ...]) -> None:
        self.table.setRowCount(0)
        for entry in entries:
            row = self.table.rowCount()
            self.table.insertRow(row)

            label = QTableWidgetItem(entry.set_id)
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            label.setData(Qt.ItemDataRole.UserRole, entry.set_id)
            # The pinned revision travels with the row. The editor never
            # repins it: an entry built against an older set must keep saying
            # so, and the compiler is what reports the mismatch.
            label.setData(Qt.ItemDataRole.UserRole + 1, int(entry.set_revision))
            label.setData(Qt.ItemDataRole.UserRole + 2, entry.shuffle_seed)
            current = self._current_revision(entry.set_id)
            if current is not None and current != int(entry.set_revision):
                label.setText(
                    "{} (pinned rev {}, library {})".format(
                        entry.set_id, entry.set_revision, current
                    )
                )
            self.table.setItem(row, self.SET_COLUMN, label)

            repeat = QSpinBox()
            repeat.setRange(1, MAX_SET_REPEAT)
            repeat.setValue(int(entry.repeat))
            repeat.valueChanged.connect(self.changed.emit)
            self.table.setCellWidget(row, self.REPEAT_COLUMN, repeat)

            shuffle = QCheckBox()
            shuffle.setChecked(bool(entry.shuffle_trials))
            shuffle.setToolTip(
                "Shuffle the trial order inside each appearance of this set"
            )
            shuffle.toggled.connect(self.changed.emit)
            self.table.setCellWidget(row, self.SHUFFLE_COLUMN, shuffle)
