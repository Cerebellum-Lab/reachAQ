from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tools.acquisition.model.animal_reconciliation import AnimalReconciliationChoices
from tools.acquisition.model.app_model_status import SessionRecordingStatus


class AnimalMetadataDialog(QDialog):
    """Explicit manual-link and two-record condensation workflow."""

    def __init__(self, app_model, parent=None):
        super().__init__(parent)
        self.app_model = app_model
        self.setWindowTitle("Animal RFID links and reconciliation")
        self.resize(650, 420)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._link_tab(), "Manual link")
        tabs.addTab(self._reconcile_tab(), "Condense duplicates")
        layout.addWidget(tabs)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _animal_combo(self) -> QComboBox:
        combo = QComboBox()
        for animal in self.app_model.animals:
            link = (
                "unlinked"
                if animal.external_identity is None
                else animal.external_identity.subject_id
            )
            combo.addItem(f"{animal.name} · {link} · {animal.id}", animal.id)
        selected = self.app_model.selected_animal
        if selected is not None:
            combo.setCurrentIndex(combo.findData(selected.id))
        return combo

    def _link_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.link_animal = self._animal_combo()
        form.addRow("Local animal:", self.link_animal)
        self.external_record = QComboBox()
        self._external_records = self.app_model.current_external_animal_records()
        for record in self._external_records:
            self.external_record.addItem(
                f"{record.identity.subject_id} · {record.physical_rfid} · "
                f"{record.new_animal_name_candidate or '(no name)'}",
                record.identity.key,
            )
        form.addRow("SoftMouse animal:", self.external_record)
        note = QLabel(
            "Linking updates only external identity/metadata. The existing local "
            "animal name, training progress, limits, and reach position are preserved."
        )
        note.setWordWrap(True)
        form.addRow("", note)
        button = QPushButton("Link selected records")
        button.setEnabled(bool(self._external_records) and self._ready())
        button.clicked.connect(self._link)
        form.addRow("", button)
        return tab

    def _reconcile_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.survivor = self._animal_combo()
        self.loser = self._animal_combo()
        if self.loser.count() > 1:
            survivor_index = self.loser.findData(self.survivor.currentData())
            self.loser.setCurrentIndex(0 if survivor_index != 0 else 1)
        form.addRow("Surviving UUID:", self.survivor)
        form.addRow("Losing UUID:", self.loser)
        self.choice_combos = {}
        for key, label in (
            ("name", "Display name from:"),
            ("pellet", "Reach/pellet position from:"),
            ("training", "Training state from:"),
            ("limit", "Target limit from:"),
            ("external", "SoftMouse link/metadata from:"),
        ):
            combo = self._animal_combo()
            if key == "external" and self.loser.count() > 1:
                combo.setCurrentIndex(combo.findData(self.loser.currentData()))
            self.choice_combos[key] = combo
            form.addRow(label, combo)
        note = QLabel(
            "Both original JSON files are backed up. The losing UUID is archived "
            "with a redirect; historical sessions are not changed."
        )
        note.setWordWrap(True)
        form.addRow("", note)
        button = QPushButton("Review and condense")
        button.setEnabled(len(self.app_model.animals) >= 2 and self._ready())
        button.clicked.connect(self._condense)
        form.addRow("", button)
        return tab

    def _ready(self) -> bool:
        return self.app_model.session_recording_status is SessionRecordingStatus.READY

    def _link(self):
        try:
            record = self._external_records[self.external_record.currentIndex()]
            self.app_model.manually_link_animal(self.link_animal.currentData(), record)
            self.status.setText("Manual link saved.")
        except Exception as exc:
            self.status.setText(f"Link failed: {exc}")

    def _condense(self):
        survivor_id = self.survivor.currentData()
        loser_id = self.loser.currentData()
        if survivor_id == loser_id:
            self.status.setText("Choose two different local animals.")
            return
        sources = {
            key: combo.currentData() for key, combo in self.choice_combos.items()
        }
        summary = "\n".join(
            f"{field}: {source_id}" for field, source_id in sources.items()
        )
        answer = QMessageBox.question(
            self,
            "Confirm animal condensation",
            f"Survivor: {survivor_id}\nLoser: {loser_id}\n\n{summary}\n\n"
            "Both originals will be backed up. Continue?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.app_model.condense_animals(
                survivor_id,
                loser_id,
                AnimalReconciliationChoices(
                    name_from=sources["name"],
                    pellet_position_from=sources["pellet"],
                    training_from=sources["training"],
                    target_limit_from=sources["limit"],
                    external_identity_from=sources["external"],
                ),
            )
            self.status.setText("Animals condensed; close and reopen to review lists.")
        except Exception as exc:
            self.status.setText(f"Condensation failed: {exc}")
