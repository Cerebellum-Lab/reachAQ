"""Check protocol: every finding of the protocol check, errors first.

The Record tooltip names a few of the errors; this lists all of them, and the
warnings, by trial, and copies them for a report.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

_SEVERITY = {"error": "Error", "warning": "Warning"}


def _ordered(findings):
    """Errors, then warnings; the whole protocol's first, then by trial."""
    return sorted(findings, key=lambda item: (
        item.severity != "error",
        item.trial_id is not None,
        item.trial_id or 0,
    ))


class ProtocolCheckDialog(QDialog):
    def __init__(self, protocol_name: str, findings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Check protocol")
        findings = _ordered(findings)
        errors = sum(1 for item in findings if item.severity == "error")
        warnings = len(findings) - errors
        if not findings:
            summary = f"{protocol_name}: nothing found that would stop it."
        else:
            summary = (
                f"{protocol_name}: {errors} error{'' if errors == 1 else 's'}, "
                f"{warnings} warning{'' if warnings == 1 else 's'}. "
                + ("Errors hold Record back until they are fixed. " if errors else "")
                + ("Warnings do not." if warnings else "")
            )
        heading = QLabel(summary)
        heading.setWordWrap(True)

        table = self._table = QTableWidget(len(findings), 3)
        table.setHorizontalHeaderLabels(("Severity", "Trial", "Finding"))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setWordWrap(True)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._rows = []
        for index, finding in enumerate(findings):
            row = (
                _SEVERITY.get(finding.severity, finding.severity),
                "Protocol" if finding.trial_id is None else f"Trial {finding.trial_id}",
                finding.message,
            )
            self._rows.append(row)
            for column, text in enumerate(row):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(index, column, item)
        table.resizeRowsToContents()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self._copy_button = QPushButton("Copy")
        self._copy_button.setToolTip("Copy every finding, one per line")
        self._copy_button.setEnabled(bool(findings))
        self._copy_button.clicked.connect(self._copy)
        buttons.addButton(self._copy_button, QDialogButtonBox.ButtonRole.ActionRole)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(table, stretch=1)
        layout.addWidget(buttons)
        self.resize(820, 420)

    def _copy(self) -> None:
        QApplication.clipboard().setText("\n".join("\t".join(row) for row in self._rows))
