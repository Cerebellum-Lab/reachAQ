from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autotrainer.core.logging import get_verbose_logger
from tools.acquisition.model.animal_sources.base import available_source_kinds
from tools.acquisition.model.animal_sources.config import load_source_configuration
from tools.acquisition.model.animal_sources.credentials import CredentialStore
from tools.acquisition.model.animal_sources.mapping import ColumnMapping
from tools.acquisition.model.animal_sources.publication import PublicationSource
from tools.acquisition.model.animal_sources.source_setup import (
    list_choices,
    read_columns,
    save_setup,
    suggest_mapping,
)


logger = get_verbose_logger(__name__)

_NONE = "(none)"


class AnimalSourceSetupDialog(QDialog):
    """Choose this computer's animal metadata source, log in, and map its columns.

    The CLI (``reachaq-animal-metadata setup``) runs the same steps.
    """

    def __init__(
        self,
        preferences,
        parent=None,
        *,
        credentials: Optional[CredentialStore] = None,
        kinds=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Animal metadata source")
        self.resize(620, 640)
        self._preferences = preferences
        self._credentials = credentials or CredentialStore()
        self._kinds = available_source_kinds() if kinds is None else kinds
        self._existing = load_source_configuration(preferences)
        self._field_widgets: Dict[str, QWidget] = {}
        self.saved = None

        layout = QVBoxLayout(self)
        top = QFormLayout()
        self.kind_combo = QComboBox()
        for name, kind in self._kinds.items():
            self.kind_combo.addItem(kind.label, name)
        top.addRow("Source:", self.kind_combo)
        layout.addLayout(top)

        self._settings_group = QGroupBox("Settings")
        self._settings_form = QFormLayout(self._settings_group)
        layout.addWidget(self._settings_group)

        columns_row = QHBoxLayout()
        self.read_columns_button = QPushButton("Read columns")
        self.read_columns_button.setToolTip("Fetch the source once to list its columns")
        self.read_columns_button.clicked.connect(self.read_columns)
        columns_row.addWidget(self.read_columns_button)
        columns_row.addStretch(1)
        layout.addLayout(columns_row)

        mapping_group = QGroupBox("Columns")
        mapping_form = QFormLayout(mapping_group)
        self._mapping_combos: Dict[str, QComboBox] = {}
        for name, label, optional in ColumnMapping.COLUMN_FIELDS:
            combo = QComboBox()
            combo.setEditable(True)
            if optional:
                combo.addItem(_NONE)
            mapping_form.addRow(f"{label}:", combo)
            self._mapping_combos[name] = combo
        self.require_rfid = QCheckBox("Skip animals without a valid RFID")
        mapping_form.addRow("", self.require_rfid)
        self.ended_states = QLineEdit("ended")
        self.ended_states.setToolTip("State values that mean the animal is gone, comma-separated")
        mapping_form.addRow("Ended states:", self.ended_states)
        layout.addWidget(mapping_group)

        self._publish_group = QGroupBox("Shared publication (optional)")
        publish_form = QFormLayout(self._publish_group)
        self.publish_directory = QLineEdit()
        self.publish_directory.setPlaceholderText("Not published")
        self.publish_directory.setToolTip(
            "After each refresh, also write a copy here for other computers to read"
        )
        publish_form.addRow("Publish to:", self.publish_directory)
        self.publish_mount = QLineEdit()
        self.publish_mount.setPlaceholderText("No mount check")
        publish_form.addRow("Only if mounted:", self.publish_mount)
        layout.addWidget(self._publish_group)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.kind_combo.currentIndexChanged.connect(self._kind_changed)
        if self._existing is not None:
            index = self.kind_combo.findData(self._existing.kind)
            if index >= 0:
                self.kind_combo.setCurrentIndex(index)
        self._kind_changed()
        if self._existing is not None and self._existing.kind == self.kind_name:
            self._load_existing()

    @property
    def kind_name(self) -> str:
        return self.kind_combo.currentData()

    @property
    def kind(self):
        return self._kinds[self.kind_name]

    def _kind_changed(self, *_args) -> None:
        while self._settings_form.rowCount():
            self._settings_form.removeRow(0)
        self._field_widgets = {}
        for item in self.kind.setup_fields():
            if item.kind == "secret":
                widget = QLineEdit()
                widget.setEchoMode(QLineEdit.EchoMode.Password)
                if self._existing is not None and self._existing.kind == self.kind_name:
                    widget.setPlaceholderText("Stored; leave empty to keep")
                row = widget
            elif item.kind == "multiline":
                widget = QPlainTextEdit(item.default)
                widget.setFixedHeight(80)
                row = widget
            elif item.kind == "choice":
                widget = QComboBox()
                widget.setEditable(True)
                widget.addItems(list(item.choices))
                row = widget
                if item.dynamic_choices:
                    row = QWidget()
                    line = QHBoxLayout(row)
                    line.setContentsMargins(0, 0, 0, 0)
                    line.addWidget(widget, 1)
                    load = QPushButton("List")
                    load.setToolTip("Log in and list the available choices")
                    load.clicked.connect(lambda _=False, n=item.name: self.load_choices(n))
                    line.addWidget(load)
            else:
                widget = QLineEdit(item.default)
                row = widget
                if item.kind == "path":
                    row = QWidget()
                    line = QHBoxLayout(row)
                    line.setContentsMargins(0, 0, 0, 0)
                    line.addWidget(widget, 1)
                    browse = QPushButton("Browse…")
                    browse.clicked.connect(lambda _=False, w=widget: self._browse(w))
                    line.addWidget(browse)
            widget.setToolTip(item.help)
            self._field_widgets[item.name] = widget
            self._settings_form.addRow(
                f"{item.label}{'' if item.required else ' (optional)'}:", row
            )
        self._publish_group.setVisible(self.kind_name != PublicationSource.kind)
        default = self.kind.default_mapping()
        if default is not None:
            self._show_mapping(default)
            self.require_rfid.setChecked(default.require_rfid)
        else:
            # Columns differ per lab: none carried over from another kind.
            for combo in self._mapping_combos.values():
                self._select(combo, None if combo.findText(_NONE) >= 0 else "")
            self.require_rfid.setChecked(False)

    def _load_existing(self) -> None:
        existing = self._existing
        for name, value in existing.settings.items():
            self._set_value(name, value)
        if existing.mapping is not None:
            self._show_mapping(existing.mapping)
            self.require_rfid.setChecked(existing.mapping.require_rfid)
            self.ended_states.setText(", ".join(existing.mapping.ended_states))
        self.publish_directory.setText(existing.publish_directory)
        self.publish_mount.setText(existing.publish_mount)

    def _show_mapping(self, mapping: ColumnMapping) -> None:
        for name, combo in self._mapping_combos.items():
            self._select(combo, getattr(mapping, name))

    @staticmethod
    def _select(combo: QComboBox, value: Optional[str]) -> None:
        text = value if value is not None else _NONE
        if combo.findText(text) < 0:
            combo.addItem(text)
        combo.setCurrentIndex(combo.findText(text))

    def _browse(self, widget: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a file", widget.text())
        if path:
            widget.setText(path)

    def _set_value(self, name: str, value: str) -> None:
        widget = self._field_widgets.get(name)
        if isinstance(widget, QComboBox):
            self._select(widget, value)
        elif isinstance(widget, QPlainTextEdit):
            widget.setPlainText(value)
        elif isinstance(widget, QLineEdit) and widget.echoMode() != QLineEdit.EchoMode.Password:
            widget.setText(value)

    def values(self) -> Dict[str, str]:
        result = {}
        for name, widget in self._field_widgets.items():
            if isinstance(widget, QComboBox):
                result[name] = widget.currentText().strip()
            elif isinstance(widget, QPlainTextEdit):
                result[name] = widget.toPlainText().strip()
            else:
                result[name] = widget.text()
        return result

    def mapping(self) -> ColumnMapping:
        columns = {}
        for name, combo in self._mapping_combos.items():
            text = combo.currentText().strip()
            columns[name] = None if text in {"", _NONE} else text
        base = self.kind.default_mapping()
        return ColumnMapping(
            **columns,
            require_rfid=self.require_rfid.isChecked(),
            ended_states=tuple(
                part.strip() for part in self.ended_states.text().split(",") if part.strip()
            ),
            subject_id_kind=(base.subject_id_kind if base else ColumnMapping.subject_id_kind),
            profile_id=(base.profile_id if base else "custom"),
            version=(base.version if base else 1),
        )

    def _busy(self, action):
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return action()
        except Exception as exc:
            logger.warning("Animal metadata source setup step failed: %s", exc)
            self.status.setText(f"Failed: {exc}")
            return None
        finally:
            QGuiApplication.restoreOverrideCursor()

    def load_choices(self, field_name: str) -> None:
        options = self._busy(
            lambda: list_choices(self.kind, self.values(), field_name, self._credentials)
        )
        if options is None:
            return
        combo = self._field_widgets[field_name]
        current = combo.currentText()
        combo.clear()
        combo.addItems(options)
        if current in options:
            combo.setCurrentText(current)
        self.status.setText(f"{len(options)} choices found")

    def read_columns(self) -> None:
        table = self._busy(lambda: read_columns(self.kind, self.values(), self._credentials))
        if table is None:
            return
        headers = [header for header in table.headers if header]
        suggestion = suggest_mapping(self.kind, headers)
        optional = {name for name, _, is_optional in ColumnMapping.COLUMN_FIELDS if is_optional}
        for name, combo in self._mapping_combos.items():
            current = combo.currentText().strip()
            combo.clear()
            if name in optional:
                combo.addItem(_NONE)
            combo.addItems(headers)
            keep = current if current in headers else None
            self._select(combo, keep or suggestion.get(name))
        self.status.setText(
            f"{len(table.rows)} rows; columns: {', '.join(headers)}"
        )

    def save(self) -> None:
        try:
            self.saved = save_setup(
                self._preferences,
                self._credentials,
                self.kind,
                self.values(),
                self.mapping(),
                publish_directory=(
                    "" if self.kind_name == PublicationSource.kind else self.publish_directory.text()
                ),
                publish_mount=(
                    "" if self.kind_name == PublicationSource.kind else self.publish_mount.text()
                ),
            )
        except Exception as exc:
            logger.warning("Animal metadata source setup not saved: %s", exc)
            self.status.setText(f"Not saved: {exc}")
            return
        save = getattr(self._preferences, "save", None)
        if save is not None:
            save()
        logger.info(
            "Animal metadata source saved: kind=%s secrets=%s",
            self.saved.configuration.kind,
            ", ".join(self.saved.secret_locations) or "unchanged",
        )
        self.accept()
