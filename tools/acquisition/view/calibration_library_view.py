"""Tools → Calibration library…: every calibration in the library, which one is active, and
which cameras each left 2D only. The operator sets another entry active or reads its report.
"""

import html
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from autotrainer.core.logging import get_verbose_logger
from autotrainer.inference.multires_calibration.library import LibraryEntry, list_entries
from autotrainer.inference.multires_calibration.report import QualityReport
from tools.acquisition.view.calibration_dialog import UNCALIBRATED_COLOR, result_page

logger = get_verbose_logger(__name__)

COLUMNS = ("ID", "Created", "Cameras", "Verdict", "Active", "2D only")
SHORT_ID_LENGTH = 8
# The report files store.write_calibration puts in every format-3 entry.
_REPORT_JSON = "quality_report.json"
_REPORT_TEXT = "quality_report.txt"


def report_page(entry: LibraryEntry) -> str:
    """The entry's quality report as the result page shows it, or why there is none."""
    try:
        report = QualityReport.from_json((entry.path / _REPORT_JSON).read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("cannot read the quality report of %s: %s", entry.path, error)
    else:
        return result_page(report)
    try:
        return f"<pre>{html.escape((entry.path / _REPORT_TEXT).read_text(encoding='utf-8'))}</pre>"
    except OSError:
        # A legacy entry, imported from before reports were kept.
        return f"<p>Calibration {html.escape(entry.id)}: no quality report.</p>"


class CalibrationLibraryView(QDialog):
    """The library's entries, newest first.

    `set_active(id)` makes an entry active and loads it; it returns None, or why it did not.
    """

    def __init__(self, root: Path, *, set_active: Callable[[str], Optional[str]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration library")
        self._root = Path(root)
        self._set_active = set_active
        self._entries: List[LibraryEntry] = []

        layout = QVBoxLayout(self)
        table = self.table = QTableWidget(0, len(COLUMNS))
        table.setHorizontalHeaderLabels(list(COLUMNS))
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(table)

        buttons = QHBoxLayout()
        self.set_active_button = QPushButton("Set active")
        self.set_active_button.clicked.connect(self._on_set_active)
        self.show_report_button = QPushButton("Show report")
        self.show_report_button.clicked.connect(self._on_show_report)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(self.set_active_button)
        buttons.addWidget(self.show_report_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.resize(820, 360)
        self.refresh()

    def refresh(self) -> None:
        selected = self._selected()
        self._entries = list_entries(self._root)
        table = self.table
        table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            uncalibrated = "; ".join(f"{name}: {reason}" for name, reason in entry.uncalibrated.items())
            values = (
                entry.id[:SHORT_ID_LENGTH],
                entry.created.isoformat(sep=" ", timespec="seconds"),
                ", ".join(entry.cameras),
                entry.verdict.upper(),
                "active" if entry.active else "",
                uncalibrated,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                table.setItem(row, column, item)
            table.item(row, 0).setToolTip(entry.id)
            if uncalibrated:
                item = table.item(row, COLUMNS.index("2D only"))
                item.setForeground(QColor(UNCALIBRATED_COLOR))
                item.setToolTip(uncalibrated)
            if entry.active:
                font = table.item(row, 0).font()
                font.setBold(True)
                for column in range(len(COLUMNS)):
                    table.item(row, column).setFont(font)
        if selected is not None:
            rows = [row for row, entry in enumerate(self._entries) if entry.id == selected.id]
            if rows:
                table.selectRow(rows[0])
        self._update_buttons()

    def _selected(self) -> Optional[LibraryEntry]:
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._entries[row] if row < len(self._entries) else None

    def _update_buttons(self) -> None:
        entry = self._selected()
        self.set_active_button.setEnabled(entry is not None and not entry.active)
        self.show_report_button.setEnabled(entry is not None)

    def _on_set_active(self) -> None:
        entry = self._selected()
        if entry is None or entry.active:
            return
        refusal = self._set_active(entry.id)
        if refusal:
            QMessageBox.warning(self, "Calibration not set active", refusal)
        self.refresh()

    def _on_show_report(self) -> None:
        entry = self._selected()
        if entry is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle(f"Calibration {entry.id[:SHORT_ID_LENGTH]}")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(report_page(entry))
        box.setIcon(QMessageBox.Icon.Information)
        box.show()
