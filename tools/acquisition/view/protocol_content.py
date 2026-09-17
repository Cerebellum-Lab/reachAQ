from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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


class _ProfileDelegate(QStyledItemDelegate):
    def __init__(self, content, kind, parent=None):
        super().__init__(parent)
        self._content = content
        self._kind = kind

    def createEditor(self, parent, _option, _index):
        editor = QComboBox(parent)
        editor.addItem("None", "")
        for profile_id, summary in self._content._profile_options(self._kind):
            editor.addItem(f"{profile_id} — {summary}", profile_id)
        return editor

    def setEditorData(self, editor, index):
        selected = editor.findData(index.data(Qt.ItemDataRole.EditRole))
        editor.setCurrentIndex(max(0, selected))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


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
        _Column("Auto policy", "automatic_shift_policy_id", "automatic_shift_profile"),
        _Column("Auto window", "automatic_window_method", AutomaticWindowMethod),
        _Column("Window N", "automatic_window_size", "window"),
        _Column("Cover", "cover_policy", CoverPolicy),
        _Column("Tone profile", "tone_profile_id", "tone_profile"),
        _Column("Tone phase", "tone_phase", ActionPhase),
        # Tone 2 and the interval that separates it from Tone 1. A fixed
        # interval and an interval profile are alternatives; the compiler
        # rejects a row that sets both.
        _Column("Cue tone", "cue_tone_profile_id", "tone_profile"),
        _Column("Cue interval", "cue_interval_profile_id", "cue_interval_profile"),
        _Column("Cue fixed (ms)", "cue_interval_fixed_ms", "cue_interval"),
        _Column("Lock timing", "cue_lock_timing", _BooleanDelegate),
        _Column("Post-clear (ms)", "cue_post_clear_delay_ms", "post_clear"),
        _Column("Laser profile", "laser_profile_id", "laser_profile"),
        _Column("Laser phase", "laser_phase", ActionPhase),
        _Column("Laser route", "laser_trigger_route", LaserTriggerRoute),
        _Column("Assignment", "stimulus_assignment", StimulusAssignment),
        _Column("Stim %", "stimulus_probability_percent", "percent"),
        _Column("Trigger", "stimulus_trigger", StimulusTrigger),
        # Randomized assignment draws from this profile instead of using the
        # single trigger above.
        _Column("Trigger profile", "stimulus_trigger_profile_id",
                "stimulus_trigger_profile"),
        _Column("Pre-reveal (ms)", "pre_reveal_ms", "pre_reveal"),
        _Column("Retry", "retry_assignment", RetryAssignment),
        _Column("State", None),
        _Column("Sources", None),
    )

    def __init__(self, app_model):
        super().__init__()
        self._app_model = app_model
        self._updating = False
        self._copied_values = None
        self._undo_history = []
        self._redo_history = []
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
            ("New tone", self._new_tone_profile),
            ("New laser", self._new_laser_profile),
            ("New auto shift", self._new_automatic_shift_profile),
            ("Delete profile", self._delete_profile),
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
        self._repeat_button = QPushButton("Repeat copied pattern")
        self._undo_button = QPushButton("Undo")
        self._redo_button = QPushButton("Redo")
        self._random_preview_button = QPushButton("Preview random assignment")
        self._epoch_button = QPushButton("Create/update epoch")
        self._block_button = QPushButton("Create/update block")
        self._copy_button.clicked.connect(self._copy_row)
        self._paste_button.clicked.connect(self._paste_rows)
        self._fill_button.clicked.connect(self._fill_selected_field)
        self._repeat_button.clicked.connect(self._paste_rows)
        self._undo_button.clicked.connect(self._undo)
        self._redo_button.clicked.connect(self._redo)
        self._random_preview_button.clicked.connect(self._preview_random_assignment)
        self._epoch_button.clicked.connect(lambda: self._apply_named_scope("epoch"))
        self._block_button.clicked.connect(lambda: self._apply_named_scope("block"))
        for button in (
            self._copy_button,
            self._paste_button,
            self._fill_button,
            self._repeat_button,
            self._undo_button,
            self._redo_button,
            self._random_preview_button,
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
            elif kind == "pre_reveal":
                delegate = _IntegerDelegate(0, 60_000, self._table)
            elif kind == "cue_interval":
                # Zero means "use the interval profile instead"; the schema
                # rejects anything between 1 and 304 ms.
                delegate = _IntegerDelegate(0, 60_000, self._table)
            elif kind == "post_clear":
                delegate = _IntegerDelegate(0, 60_000, self._table)
            elif kind in {
                "tone_profile", "laser_profile", "automatic_shift_profile",
                "cue_interval_profile", "stimulus_trigger_profile",
            }:
                delegate = _ProfileDelegate(self, kind, self._table)
            else:
                delegate = _EnumDelegate(kind, self._table)
            self._table.setItemDelegateForColumn(index, delegate)

    def _profile_options(self, kind):
        state = self._app_model.trial_protocol_state
        key = {
            "tone_profile": "tone_profiles",
            "laser_profile": "laser_profiles",
            "automatic_shift_profile": "automatic_shift_profiles",
            "cue_interval_profile": "cue_interval_profiles",
            "stimulus_trigger_profile": "stimulus_trigger_profiles",
        }[kind]
        return tuple(
            (item["profile_id"], item.get("summary", ""))
            for item in state.get(key, ())
        )

    @staticmethod
    def _display_value(field, value) -> str:
        if field in {"shift_x_mm", "shift_y_mm", "shift_z_mm"}:
            return f"{float(value):.2f}"
        if field == "stimulus_probability_percent":
            return f"{float(value):.1f}"
        if field == "enabled":
            return "Enabled" if value else "Disabled"
        if field == "cue_lock_timing":
            # Named for what it does at the deadline, not True/False.
            return "Locked" if value else "Unlocked"
        if field == "cue_interval_fixed_ms":
            # Zero is not a zero-length interval; it defers to the profile.
            return "From profile" if not value else f"{int(value)} ms"
        if field == "cue_post_clear_delay_ms":
            return "None" if not value else f"{int(value)} ms"
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
            "automatic_shift_policy_id": "default",
            "tone_phase": "none",
            "laser_phase": "none",
            "laser_trigger_route": "none",
            "stimulus_assignment": "disabled",
            "stimulus_probability_percent": 100.0,
            "stimulus_trigger": "none",
            "retry_assignment": "repeat",
            "pre_reveal_ms": 0,
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
                self._repeat_button,
                self._random_preview_button,
                self._epoch_button,
                self._block_button,
            ):
                button.setEnabled(has_protocol)
            self._undo_button.setEnabled(has_protocol and bool(self._undo_history))
            self._redo_button.setEnabled(has_protocol and bool(self._redo_history))
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
            before = self._protocol_snapshot()
            accepted = self._app_model.apply_ordered_protocol_values(
                (trial_id,), patch, scope_kind="trials"
            ) if hasattr(self._app_model, "apply_ordered_protocol_values") else (
                self._app_model.update_trial_protocol_row(trial_id, field, value)
            )
            if isinstance(accepted, dict):
                accepted = bool(accepted.get("changed_trial_ids"))
            if accepted:
                self._remember_edit(before)
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
        self._copied_values = tuple({
            column.field: self._legacy_value(self._row_record(trial_id), column.field)
            for column in self.COLUMNS if column.field is not None
        } for trial_id in rows)
        self._edit_status.setText(
            f"Copied {len(rows)} row(s); paste repeats this pattern across the selection."
        )

    def _paste_rows(self):
        rows = self._selected_trial_ids()
        if not rows or not self._copied_values:
            return
        before = self._protocol_snapshot()
        try:
            result = self._app_model.apply_ordered_protocol_row_patches({
                trial_id: self._copied_values[index % len(self._copied_values)]
                for index, trial_id in enumerate(rows)
            })
            changed = result.get("changed_trial_ids", ())
            skipped = result.get("skipped_trial_ids", ())
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self._edit_status.setText(str(error))
            return
        if changed:
            self._remember_edit(before)
        self._edit_status.setText(
            f"Repeated {len(self._copied_values)}-row pattern across "
            f"{len(changed)} trial(s)"
            + (f"; skipped {', '.join(map(str, skipped))}" if skipped else "")
        )

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
            before = self._protocol_snapshot()
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
            if changed:
                self._remember_edit(before)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self._edit_status.setText(str(error))

    def _protocol_snapshot(self):
        document = getattr(self._app_model, "selected_ordered_protocol", None)
        return document

    def _remember_edit(self, before):
        after = self._protocol_snapshot()
        if before is None or after is None or before == after:
            return
        self._undo_history.append((before, after))
        self._redo_history.clear()
        self._undo_button.setEnabled(True)
        self._redo_button.setEnabled(False)

    def _undo(self):
        if not self._undo_history:
            return
        before, after = self._undo_history.pop()
        try:
            self._app_model.restore_ordered_protocol_document(before)
        except (RuntimeError, ValueError) as error:
            self._undo_history.append((before, after))
            self._edit_status.setText(str(error))
            return
        self._redo_history.append((before, after))
        self._edit_status.setText("Restored the previous values as a new revision.")

    def _redo(self):
        if not self._redo_history:
            return
        before, after = self._redo_history.pop()
        try:
            self._app_model.restore_ordered_protocol_document(after)
        except (RuntimeError, ValueError) as error:
            self._redo_history.append((before, after))
            self._edit_status.setText(str(error))
            return
        self._undo_history.append((before, after))
        self._edit_status.setText("Reapplied the values as a new revision.")

    def _preview_random_assignment(self):
        rows = self._selected_trial_ids()
        if not rows:
            self._edit_status.setText("Select trials to preview first.")
            return
        seed, accepted = QInputDialog.getInt(
            self, "Randomized assignment preview", "Seed:", 1, 0, 2_147_483_647
        )
        if not accepted:
            return
        selected = []
        for trial_id in rows:
            record = self._row_record(trial_id)
            probability = float(record.get("stimulus_probability_percent", 100.0))
            payload = json.dumps((seed, trial_id), separators=(",", ":")).encode()
            draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
            if draw * 100.0 < probability:
                selected.append(trial_id)
        self._edit_status.setText(
            f"Preview seed {seed}: stimulus on trials "
            f"{', '.join(map(str, selected)) or 'none'}. The actual draw is frozen at preparation."
        )

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

    def _new_tone_profile(self):
        profile_id, accepted = QInputDialog.getText(
            self, "Tone profile", "Profile ID:"
        )
        if not accepted or not profile_id.strip():
            return
        frequency, accepted = QInputDialog.getInt(
            self, "Tone profile", "Frequency (Hz):", 5000, 1, 100_000
        )
        if not accepted:
            return
        duration, accepted = QInputDialog.getInt(
            self, "Tone profile", "Duration (ms):", 100, 1, 60_000
        )
        if accepted:
            self._run_library_action(
                self._app_model.save_tone_profile,
                profile_id.strip(), frequency, duration,
            )

    def _new_laser_profile(self):
        configured_channels = tuple(
            int(channel.channel_id)
            for channel in self._app_model.laser.configuration.channels
        )
        if not configured_channels:
            self._app_model.on_error(
                "Laser profile unavailable",
                "Configure at least one laser channel in Edit DAQ Ports first.",
            )
            return
        profile_id, accepted = QInputDialog.getText(
            self, "Laser pulse profile", "Profile ID:"
        )
        if not accepted or not profile_id.strip():
            return
        channel_label, accepted = QInputDialog.getItem(
            self,
            "Laser pulse profile",
            "Configured laser channel:",
            tuple(f"Laser {channel}" for channel in configured_channels),
            0,
            False,
        )
        if not accepted:
            return
        channel = int(channel_label.rsplit(" ", 1)[-1])
        amplitude, accepted = QInputDialog.getDouble(
            self, "Laser pulse profile", "Amplitude (V):", 1.0, -100.0, 100.0, 4
        )
        if not accepted:
            return
        pulse_ms, accepted = QInputDialog.getDouble(
            self, "Laser pulse profile", "Pulse duration (ms):", 5.0, 0.001, 60_000.0, 3
        )
        if not accepted:
            return
        count, accepted = QInputDialog.getInt(
            self, "Laser pulse profile", "Pulse count:", 1, 1, 100_000
        )
        if not accepted:
            return
        frequency = None
        if count > 1:
            frequency, accepted = QInputDialog.getDouble(
                self, "Laser pulse profile", "Pulse frequency (Hz):", 20.0, 0.001, 100_000.0, 3
            )
            if not accepted:
                return
        route_label, accepted = QInputDialog.getItem(
            self,
            "Laser pulse profile",
            "Trigger route:",
            ("Hardware STIM3", "Direct NI software start"),
            0,
            False,
        )
        if not accepted:
            return
        route = (
            "hardware_stim3"
            if route_label == "Hardware STIM3"
            else "direct_ni_software"
        )
        terminal = ""
        if route == "hardware_stim3":
            terminals = tuple(
                self._app_model.laser.configuration.trigger_listener_inputs
            )
            if not terminals:
                self._app_model.on_error(
                    "Hardware trigger unavailable",
                    "Select a hardware trigger input in Edit DAQ Ports first.",
                )
                return
            terminal, accepted = QInputDialog.getItem(
                self,
                "Laser pulse profile",
                "Configured NI trigger terminal:",
                terminals,
                0,
                False,
            )
            if not accepted or not terminal.strip():
                return
        self._run_library_action(
            self._app_model.save_laser_profile,
            profile_id=profile_id.strip(),
            channel_id=channel,
            amplitude_volts=amplitude,
            pulse_duration_ms=pulse_ms,
            pulse_count=count,
            frequency_hz=frequency,
            trigger_route=route,
            trigger_terminal=terminal.strip(),
        )

    def _ask_xyz(self, title, label, defaults, minimum, maximum):
        values = []
        for axis, default in zip("XYZ", defaults):
            value, accepted = QInputDialog.getDouble(
                self,
                title,
                f"{label} {axis} (mm):",
                float(default),
                float(minimum),
                float(maximum),
                3,
            )
            if not accepted:
                return None
            values.append(value)
        return tuple(values)

    def _new_automatic_shift_profile(self):
        policy_id, accepted = QInputDialog.getText(
            self, "Automatic shift policy", "Policy ID:"
        )
        if not accepted or not policy_id.strip():
            return
        reduction, accepted = QInputDialog.getItem(
            self,
            "Automatic shift policy",
            "Reach reduction:",
            ("Mean", "Median"),
            0,
            False,
        )
        if not accepted:
            return
        eligibility, accepted = QInputDialog.getItem(
            self,
            "Automatic shift policy",
            "Eligible reaches:",
            ("Failed reaches", "Successful reaches", "Both"),
            0,
            False,
        )
        if not accepted:
            return
        target = self._ask_xyz(
            "Automatic shift policy", "Desired reach offset", (1.5, -3.0, 1.0),
            -50.0, 50.0,
        )
        deadbands = self._ask_xyz(
            "Automatic shift policy", "Deadband", (0.5, 1.0, 0.5), 0.0, 50.0,
        )
        maximum_update = self._ask_xyz(
            "Automatic shift policy", "Maximum update", (2.0, 2.0, 2.0),
            0.0, 50.0,
        )
        maximum_absolute = self._ask_xyz(
            "Automatic shift policy", "Maximum absolute shift", (5.0, 5.0, 5.0),
            0.0, 50.0,
        )
        if None in (target, deadbands, maximum_update, maximum_absolute):
            return
        application, accepted = QInputDialog.getItem(
            self,
            "Automatic shift policy",
            "Use recommendation:",
            ("Apply automatically", "Recommend only"),
            0,
            False,
        )
        if not accepted:
            return
        eligible = {
            "Failed reaches": frozenset({"failure"}),
            "Successful reaches": frozenset({"success"}),
            "Both": frozenset({"success", "failure"}),
        }[eligibility]
        self._run_library_action(
            self._app_model.save_automatic_shift_policy,
            policy_id=policy_id.strip(),
            eligible_outcomes=eligible,
            reduction_method=reduction.lower(),
            target_reach_offset_dcs=target,
            deadbands_mm=deadbands,
            maximum_update_mm=maximum_update,
            maximum_absolute_mm=maximum_absolute,
            apply_automatically=application == "Apply automatically",
        )

    def _delete_profile(self):
        state = self._app_model.trial_protocol_state
        choices = [
            *(f"tone: {item['profile_id']}" for item in state.get("tone_profiles", ())),
            *(f"laser: {item['profile_id']}" for item in state.get("laser_profiles", ())),
            *(f"automatic_shift: {item['policy_id']}" for item in state.get("automatic_shift_profiles", ())),
        ]
        if not choices:
            return
        selected, accepted = QInputDialog.getItem(
            self, "Delete stimulus profile", "Unused profile:", choices, 0, False
        )
        if accepted:
            kind, profile_id = selected.split(": ", 1)
            self._run_library_action(
                self._app_model.delete_stimulus_profile, kind, profile_id
            )

    def _run_library_action(self, action, *args, **kwargs):
        try:
            action(*args, **kwargs)
        except (FileExistsError, KeyError, RuntimeError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "Protocol action failed", str(error))

    @invoke_method
    def _on_app_model_property_changed(self, name, value, _previous):
        if name == self._app_model.Props.TRIAL_PROTOCOL_STATE:
            self._refresh(value)
