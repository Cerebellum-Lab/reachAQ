from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method


class ProtocolContent(ContentWidget):
    """Ordered per-trial editor with immutable active/completed rows."""

    COLUMNS = (
        ("Trial", None),
        ("Pellet delivery", "pellet_behavior"),
        ("Shift X", "shift_x_mm"),
        ("Shift Y", "shift_y_mm"),
        ("Shift Z", "shift_z_mm"),
        ("Cover", "cover"),
        ("Tone", "tone"),
        ("Laser", "laser"),
        ("State", None),
    )

    def __init__(self, app_model):
        super().__init__()
        self._app_model = app_model
        self._updating = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._card_widget = CardWidget(title="Trial Protocol")
        self._card_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        note = QLabel(
            "Each row configures one logical pellet trial. Placeholder delivery "
            "settings are snapshotted with the trial; future execution support "
            "can consume the same row schema."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6772;")
        content_layout.addWidget(note)
        self._analysis_status = QLabel()
        self._analysis_status.setWordWrap(True)
        self._analysis_status.setStyleSheet("color: #5f6772;")
        content_layout.addWidget(self._analysis_status)

        table = self._table = QTableWidget()
        table.setColumnCount(len(self.COLUMNS))
        table.setHorizontalHeaderLabels([label for label, _ in self.COLUMNS])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.itemChanged.connect(self._item_changed)
        content_layout.addWidget(table, stretch=1)
        self._card_widget.setContentWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._card_widget, stretch=1)

        app_model.property_changed += self._on_app_model_property_changed
        self._refresh(app_model.trial_protocol_state)

    @staticmethod
    def _display_value(field, value) -> str:
        if field in {"shift_x_mm", "shift_y_mm", "shift_z_mm"}:
            return f"{float(value):.1f}"
        if field == "cover":
            return "Yes" if value else "No"
        return str(value)

    def _refresh(self, state: dict) -> None:
        self._updating = True
        try:
            rows = tuple(state.get("rows", ()))
            analysis = state.get("analysis", {})
            if not analysis.get("enabled"):
                analysis_text = "Live trial analysis: Disabled"
            else:
                reason = analysis.get("send_block_reason")
                analysis_text = (
                    f"Live trial analysis: {analysis.get('pending_attempts', 0)} pending; "
                    f"{analysis.get('estimate', 'estimating')} per 1 s tracking"
                )
                if reason:
                    analysis_text += f" — {reason}"
            self._analysis_status.setText(analysis_text)
            active = state.get("active_trial_id")
            completed = set(state.get("completed_trial_ids", ()))
            self._table.setRowCount(len(rows))
            for row_index, record in enumerate(rows):
                trial_id = int(record["trial_id"])
                locked = trial_id == active or trial_id in completed
                row_state = (
                    "Active" if trial_id == active
                    else "Completed" if trial_id in completed
                    else "Future"
                )
                values = [trial_id]
                values.extend(record[field] for _, field in self.COLUMNS[1:-1])
                values.append(row_state)
                for column, ((_, field), value) in enumerate(zip(self.COLUMNS, values)):
                    item = self._table.item(row_index, column) or QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, (trial_id, field))
                    item.setText(self._display_value(field, value))
                    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    if field is not None and not locked:
                        flags |= Qt.ItemFlag.ItemIsEditable
                    item.setFlags(flags)
                    color = (
                        QColor("#fff0b3") if trial_id == active
                        else QColor("#e5e7eb") if trial_id in completed
                        else QColor("white")
                    )
                    item.setBackground(color)
                    self._table.setItem(row_index, column, item)
            self._table.resizeColumnsToContents()
        finally:
            self._updating = False

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        trial_id, field = item.data(Qt.ItemDataRole.UserRole) or (None, None)
        if trial_id is None or field is None:
            return
        value = item.text()
        if field == "cover":
            value = value.strip().lower() in {"1", "yes", "true", "on"}
        try:
            accepted = self._app_model.update_trial_protocol_row(
                trial_id,
                field,
                value,
            )
        except (TypeError, ValueError):
            accepted = False
        if not accepted:
            self._refresh(self._app_model.trial_protocol_state)

    @invoke_method
    def _on_app_model_property_changed(self, name, value, _previous):
        if name == self._app_model.Props.TRIAL_PROTOCOL_STATE:
            self._refresh(value)
