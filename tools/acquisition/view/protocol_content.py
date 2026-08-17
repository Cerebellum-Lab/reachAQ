from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QColor, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget, invoke_method
from tools.acquisition.model.trial_protocol_schedule import (
    ActionPhase,
    AutomaticWindowMethod,
    CoverPolicy,
    LaserTriggerRoute,
    PelletBehavior,
    PelletLane,
    PelletPositionMode,
    RetryAssignment,
    StimulusAssignment,
    StimulusTrigger,
)


class _EnumDelegate(QStyledItemDelegate):
    def __init__(self, enum_type, parent=None):
        super().__init__(parent)
        self._values = tuple(enum_type)

    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        for value in self._values:
            editor.addItem(value.label, value.value)
        return editor

    def setEditorData(self, editor, index):
        value = index.data(Qt.ItemDataRole.EditRole)
        selected = editor.findData(value)
        editor.setCurrentIndex(max(0, selected))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class _BooleanDelegate(QStyledItemDelegate):
    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        editor.addItem("Enabled", True)
        editor.addItem("Disabled", False)
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(0 if bool(index.data(Qt.ItemDataRole.EditRole)) else 1)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class _FloatDelegate(QStyledItemDelegate):
    def __init__(self, minimum, maximum, decimals=2, suffix="", parent=None):
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._decimals = decimals
        self._suffix = suffix

    def createEditor(self, parent, _option, _index):
        editor = QDoubleSpinBox(parent)
        editor.setRange(self._minimum, self._maximum)
        editor.setDecimals(self._decimals)
        editor.setSuffix(self._suffix)
        return editor

    def setEditorData(self, editor, index):
        editor.setValue(float(index.data(Qt.ItemDataRole.EditRole)))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)


class _IntegerDelegate(QStyledItemDelegate):
    def __init__(self, minimum, maximum, parent=None):
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum

    def createEditor(self, parent, _option, _index):
        editor = QSpinBox(parent)
        editor.setRange(self._minimum, self._maximum)
        return editor

    def setEditorData(self, editor, index):
        editor.setValue(int(index.data(Qt.ItemDataRole.EditRole)))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)


class _IdentifierDelegate(QStyledItemDelegate):
    def createEditor(self, parent, _option, _index):
        editor = QLineEdit(parent)
        editor.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"[a-z0-9][a-z0-9._-]{0,63}|"),
            editor,
        ))
        return editor


@dataclass(frozen=True)
class _Column:
    label: str
    field: str | None
    delegate: object = None


class ProtocolContent(ContentWidget):
    """Typed editor for reusable per-pellet-trial protocol revisions."""

    COLUMNS = (
        _Column("Trial", None),
        _Column("Run", "enabled", _BooleanDelegate),
        _Column("Pellet cycle", "pellet_behavior", PelletBehavior),
        _Column("Position", "position_mode", PelletPositionMode),
        _Column("Lane", "position_lane", PelletLane),
        _Column("X (mm)", "shift_x_mm", "shift"),
        _Column("Y (mm)", "shift_y_mm", "shift"),
        _Column("Z (mm)", "shift_z_mm", "shift"),
        _Column("Auto window", "automatic_window_method", AutomaticWindowMethod),
        _Column("Window N", "automatic_window_size", "window"),
        _Column("Cover", "cover_policy", CoverPolicy),
        _Column("Tone profile", "tone_profile_id", "identifier"),
        _Column("Tone phase", "tone_phase", ActionPhase),
        _Column("Laser profile", "laser_profile_id", "identifier"),
        _Column("Laser phase", "laser_phase", ActionPhase),
        _Column("Laser route", "laser_trigger_route", LaserTriggerRoute),
        _Column("Assignment", "stimulus_assignment", StimulusAssignment),
        _Column("Stim %", "stimulus_probability_percent", "percent"),
        _Column("Trigger", "stimulus_trigger", StimulusTrigger),
        _Column("Retry", "retry_assignment", RetryAssignment),
        _Column("State", None),
        _Column("Sources", None),
    )

    def __init__(self, app_model):
        super().__init__()
        self._app_model = app_model
        self._updating = False
        self._copied_values = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._card_widget = CardWidget(title="Pellet-trial protocol")
        self._card_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.addLayout(self._build_library_bar())

        note = QLabel(
            "Select one saved protocol or No protocol. Values are typed and "
            "versioned; yellow is active, gray is completed, and bulk edits skip "
            "locked trials. ROI 1/2 remain visible as future, non-runnable triggers."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6772;")
        content_layout.addWidget(note)

        content_layout.addLayout(self._build_edit_bar())
        self._edit_status = QLabel()
        self._edit_status.setWordWrap(True)
        self._edit_status.setStyleSheet("color: #5f6772;")
        content_layout.addWidget(self._edit_status)

        self._analysis_status = QLabel()
        self._analysis_status.setWordWrap(True)
        self._analysis_status.setStyleSheet("color: #5f6772;")
        content_layout.addWidget(self._analysis_status)
        resolution_layout = QHBoxLayout()
        self._retry_analysis = QPushButton("Retry analysis")
        self._continue_without_analysis = QPushButton("Continue without result")
        self._retry_analysis.clicked.connect(app_model.retry_pending_intertrial_analysis)
        self._continue_without_analysis.clicked.connect(
            app_model.continue_without_pending_intertrial_result
        )
        resolution_layout.addWidget(self._retry_analysis)
        resolution_layout.addWidget(self._continue_without_analysis)
        resolution_layout.addStretch(1)
        content_layout.addLayout(resolution_layout)

        table = self._table = QTableWidget()
        table.setColumnCount(len(self.COLUMNS))
        table.setHorizontalHeaderLabels([column.label for column in self.COLUMNS])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setAlternatingRowColors(True)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.itemChanged.connect(self._item_changed)
        self._install_delegates()
        content_layout.addWidget(table, stretch=1)
        self._card_widget.setContentWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._card_widget, stretch=1)

        app_model.property_changed += self._on_app_model_property_changed
        self._refresh(app_model.trial_protocol_state)

    def _build_library_bar(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Session protocol:"))
        self._protocol_selector = QComboBox()
        self._protocol_selector.setMinimumContentsLength(18)
        self._protocol_selector.currentIndexChanged.connect(self._protocol_selected)
        layout.addWidget(self._protocol_selector)
        for label, slot in (
            ("New", self._new_protocol),
            ("Duplicate", self._duplicate_protocol),
            ("Rename", self._rename_protocol),
            ("Import", self._import_protocol),
            ("Export", self._export_protocol),
            ("Revert", self._reload_protocols),
        ):
            button = QToolButton()
            button.setText(label)
            button.clicked.connect(slot)
            layout.addWidget(button)
        layout.addStretch(1)
        return layout

    def _build_edit_bar(self):
        layout = QHBoxLayout()
        self._copy_button = QPushButton("Copy row")
        self._paste_button = QPushButton("Paste to selected")
        self._fill_button = QPushButton("Fill selected field")
        self._epoch_button = QPushButton("Create/update epoch")
        self._block_button = QPushButton("Create/update block")
        self._copy_button.clicked.connect(self._copy_row)
        self._paste_button.clicked.connect(self._paste_rows)
        self._fill_button.clicked.connect(self._fill_selected_field)
        self._epoch_button.clicked.connect(lambda: self._apply_named_scope("epoch"))
        self._block_button.clicked.connect(lambda: self._apply_named_scope("block"))
        for button in (
            self._copy_button,
            self._paste_button,
            self._fill_button,
            self._epoch_button,
            self._block_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)
        return layout

    def _install_delegates(self):
        for index, column in enumerate(self.COLUMNS):
            kind = column.delegate
            if kind is None:
                continue
            if kind is _BooleanDelegate:
                delegate = _BooleanDelegate(self._table)
            elif kind == "shift":
                delegate = _FloatDelegate(-50.0, 50.0, 2, " mm", self._table)
            elif kind == "percent":
                delegate = _FloatDelegate(0.0, 100.0, 1, " %", self._table)
            elif kind == "window":
                delegate = _IntegerDelegate(1, 10_000, self._table)
            elif kind == "identifier":
                delegate = _IdentifierDelegate(self._table)
            else:
                delegate = _EnumDelegate(kind, self._table)
            self._table.setItemDelegateForColumn(index, delegate)

    @staticmethod
    def _display_value(field, value) -> str:
        if field in {"shift_x_mm", "shift_y_mm", "shift_z_mm"}:
            return f"{float(value):.2f}"
        if field == "stimulus_probability_percent":
            return f"{float(value):.1f}"
        if field == "enabled":
            return "Enabled" if value else "Disabled"
        return str(value)

    @staticmethod
    def _legacy_value(record, field):
        if field == "cover_policy" and field not in record:
            return "cover" if record.get("cover", True) else "reveal"
        if field == "tone_profile_id" and field not in record:
            value = record.get("tone", "None")
            return "" if str(value).lower() == "none" else value
        if field == "laser_profile_id" and field not in record:
            value = record.get("laser", "None")
            return "" if str(value).lower() == "none" else value
        defaults = {
            "enabled": False,
            "position_mode": "base",
            "position_lane": "center",
            "automatic_window_method": "legacy_batch",
            "automatic_window_size": 15,
            "tone_phase": "none",
            "laser_phase": "none",
            "laser_trigger_route": "none",
            "stimulus_assignment": "disabled",
            "stimulus_probability_percent": 100.0,
            "stimulus_trigger": "none",
            "retry_assignment": "repeat",
        }
        return record.get(field, defaults.get(field, ""))

    def _refresh(self, state: dict) -> None:
        self._updating = True
        try:
            selected = state.get("selected_protocol")
            selected_id = None if selected is None else selected.get("protocol_id")
            self._protocol_selector.clear()
            self._protocol_selector.addItem("No protocol", None)
            for protocol in state.get("protocols", ()):
                label = f"{protocol['name']} (r{protocol['revision']})"
                self._protocol_selector.addItem(label, protocol["protocol_id"])
            selected_index = self._protocol_selector.findData(selected_id)
            self._protocol_selector.setCurrentIndex(max(0, selected_index))

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
                if analysis.get("resolution_reason"):
                    analysis_text += f"\n{analysis['resolution_reason']}"
            self._analysis_status.setText(analysis_text)
            resolution_required = bool(analysis.get("resolution_required"))
            self._retry_analysis.setVisible(resolution_required)
            self._continue_without_analysis.setVisible(resolution_required)
            self._retry_analysis.setEnabled(resolution_required)
            self._continue_without_analysis.setEnabled(resolution_required)

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
                for column_index, column in enumerate(self.COLUMNS):
                    if column_index == 0:
                        value = trial_id
                    elif column.label == "State":
                        value = row_state
                    elif column.label == "Sources":
                        sources = record.get("value_sources", {})
                        value = ", ".join(sorted(set(sources.values()))) or "defaults"
                    else:
                        value = self._legacy_value(record, column.field)
                    item = self._table.item(row_index, column_index) or QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, (trial_id, column.field))
                    item.setData(Qt.ItemDataRole.EditRole, value)
                    item.setText(self._display_value(column.field, value))
                    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    if column.field is not None and not locked and selected_id is not None:
                        flags |= Qt.ItemFlag.ItemIsEditable
                    item.setFlags(flags)
                    color = (
                        QColor("#fff0b3") if trial_id == active
                        else QColor("#e5e7eb") if trial_id in completed
                        else QColor("white")
                    )
                    item.setBackground(color)
                    self._table.setItem(row_index, column_index, item)
            self._table.resizeColumnsToContents()
            has_protocol = selected_id is not None
            for button in (
                self._copy_button,
                self._paste_button,
                self._fill_button,
                self._epoch_button,
                self._block_button,
            ):
                button.setEnabled(has_protocol)
            errors = state.get("repository_errors", {})
            if errors:
                self._edit_status.setText(
                    f"{len(errors)} protocol file(s) could not be loaded; other "
                    "valid protocols remain available."
                )
        finally:
            self._updating = False

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        trial_id, field = item.data(Qt.ItemDataRole.UserRole) or (None, None)
        if trial_id is None or field is None:
            return
        value = item.data(Qt.ItemDataRole.EditRole)
        patch = {field: value}
        row = self._row_record(int(trial_id))
        if field == "tone_profile_id":
            patch["tone_phase"] = "before_send" if value else "none"
        elif field == "laser_profile_id" and not value:
            patch.update(laser_phase="none", laser_trigger_route="none")
        elif field == "stimulus_assignment" and value == "disabled":
            patch.update(stimulus_trigger="none", pre_reveal_ms=0)
        elif field == "position_mode" and value != "fixed_manual":
            patch.update(shift_x_mm=0.0, shift_y_mm=0.0, shift_z_mm=0.0)
        try:
            accepted = self._app_model.apply_ordered_protocol_values(
                (trial_id,), patch, scope_kind="trials"
            ) if hasattr(self._app_model, "apply_ordered_protocol_values") else (
                self._app_model.update_trial_protocol_row(trial_id, field, value)
            )
            if isinstance(accepted, dict):
                accepted = bool(accepted.get("changed_trial_ids"))
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            accepted = False
            self._edit_status.setText(str(error))
        if not accepted:
            self._refresh(self._app_model.trial_protocol_state)

    def _row_record(self, trial_id):
        for record in self._app_model.trial_protocol_state.get("rows", ()):
            if int(record["trial_id"]) == int(trial_id):
                return record
        return {}

    def _selected_trial_ids(self):
        return tuple(sorted({
            int(self._table.item(index.row(), 0).text())
            for index in self._table.selectionModel().selectedRows()
        }))

    def _copy_row(self):
        rows = self._selected_trial_ids()
        if not rows:
            return
        record = self._row_record(rows[0])
        self._copied_values = {
            column.field: self._legacy_value(record, column.field)
            for column in self.COLUMNS if column.field is not None
        }
        self._edit_status.setText(f"Copied resolved values from trial {rows[0]}.")

    def _paste_rows(self):
        rows = self._selected_trial_ids()
        if not rows or self._copied_values is None:
            return
        self._apply_values(rows, self._copied_values, "bulk", "paste")

    def _fill_selected_field(self):
        rows = self._selected_trial_ids()
        current = self._table.currentItem()
        if not rows or current is None:
            return
        _trial_id, field = current.data(Qt.ItemDataRole.UserRole) or (None, None)
        if field is None:
            return
        self._apply_values(
            rows,
            {field: current.data(Qt.ItemDataRole.EditRole)},
            "bulk",
            f"fill-{field}",
        )

    def _apply_named_scope(self, kind):
        rows = self._selected_trial_ids()
        if not rows:
            self._edit_status.setText("Select one or more future trials first.")
            return
        name, accepted = QInputDialog.getText(self, f"{kind.title()} name", "Name:")
        if not accepted or not name.strip():
            return
        parent = ""
        if kind == "block":
            parent, accepted = QInputDialog.getText(
                self, "Parent epoch", "Existing epoch name:"
            )
            if not accepted:
                return
        current = self._table.currentItem()
        _trial, field = current.data(Qt.ItemDataRole.UserRole) if current else (None, None)
        if field is None:
            self._edit_status.setText("Select a value cell to apply to the scope.")
            return
        self._apply_values(
            rows,
            {field: current.data(Qt.ItemDataRole.EditRole)},
            kind,
            name,
            parent_epoch=parent,
        )

    def _apply_values(self, rows, values, kind, name, *, parent_epoch=""):
        try:
            result = self._app_model.apply_ordered_protocol_values(
                rows,
                values,
                scope_kind=kind,
                scope_name=name,
                parent_epoch=parent_epoch,
            )
            changed = result.get("changed_trial_ids", ())
            skipped = result.get("skipped_trial_ids", ())
            message = f"Changed trials: {', '.join(map(str, changed)) or 'none'}"
            if skipped:
                message += f"; locked/skipped: {', '.join(map(str, skipped))}"
            self._edit_status.setText(message)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self._edit_status.setText(str(error))

    def _protocol_selected(self):
        if self._updating or not hasattr(self._app_model, "select_ordered_protocol"):
            return
        try:
            self._app_model.select_ordered_protocol(self._protocol_selector.currentData())
        except (KeyError, RuntimeError, ValueError) as error:
            self._edit_status.setText(str(error))
            self._refresh(self._app_model.trial_protocol_state)

    @staticmethod
    def _ask_identity(parent, title, default_id=""):
        protocol_id, accepted = QInputDialog.getText(
            parent, title, "Protocol ID:", text=default_id
        )
        if not accepted or not protocol_id.strip():
            return None
        name, accepted = QInputDialog.getText(parent, title, "Display name:")
        return (protocol_id.strip(), name.strip()) if accepted and name.strip() else None

    def _new_protocol(self):
        identity = self._ask_identity(self, "New protocol")
        if identity:
            count, accepted = QInputDialog.getInt(
                self, "New protocol", "Trial rows:", 15, 1, 100_000
            )
            if accepted:
                self._run_library_action(
                    self._app_model.create_ordered_protocol,
                    *identity,
                    trial_count=count,
                )

    def _selected_protocol_id(self):
        return self._protocol_selector.currentData()

    def _duplicate_protocol(self):
        source = self._selected_protocol_id()
        if not source:
            return
        identity = self._ask_identity(self, "Duplicate protocol", f"{source}-copy")
        if identity:
            self._run_library_action(
                self._app_model.duplicate_ordered_protocol,
                source,
                protocol_id=identity[0],
                name=identity[1],
            )

    def _rename_protocol(self):
        source = self._selected_protocol_id()
        if not source:
            return
        identity = self._ask_identity(self, "Rename protocol", source)
        if identity:
            self._run_library_action(
                self._app_model.rename_ordered_protocol,
                source,
                protocol_id=identity[0],
                name=identity[1],
            )

    def _import_protocol(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import protocol", "", "Protocol JSON (*.json)"
        )
        if path:
            self._run_library_action(self._app_model.import_ordered_protocol, Path(path))

    def _export_protocol(self):
        source = self._selected_protocol_id()
        if not source:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export protocol", f"{source}.json", "Protocol JSON (*.json)"
        )
        if path:
            self._run_library_action(
                self._app_model.export_ordered_protocol,
                source,
                Path(path),
            )

    def _reload_protocols(self):
        if hasattr(self._app_model, "reload_ordered_protocols"):
            self._run_library_action(self._app_model.reload_ordered_protocols)

    def _run_library_action(self, action, *args, **kwargs):
        try:
            action(*args, **kwargs)
        except (FileExistsError, KeyError, RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Protocol action failed", str(error))

    @invoke_method
    def _on_app_model_property_changed(self, name, value, _previous):
        if name == self._app_model.Props.TRIAL_PROTOCOL_STATE:
            self._refresh(value)
