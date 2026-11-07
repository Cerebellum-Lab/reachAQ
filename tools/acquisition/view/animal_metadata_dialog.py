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

from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.view.animal_details_dialog import AnimalDetailsDialog


class AnimalMetadataDialog(QDialog):
    """Simple manual-link and duplicate-condensation workflow."""

    def __init__(self, app_model, parent=None):
        super().__init__(parent)
        self.app_model = app_model
        self.setWindowTitle("Animal RFID links and duplicates")
        self.resize(610, 330)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._edit_tab(), "Edit animal")
        tabs.addTab(self._link_tab(), "Link RFID")
        tabs.addTab(self._reconcile_tab(), "Condense duplicate")
        layout.addWidget(tabs)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _all_animal_combo(self) -> QComboBox:
        combo = QComboBox()
        for animal in self.app_model.animals:
            combo.addItem(animal.name, animal.id)
        selected = self.app_model.selected_animal
        if selected is not None:
            combo.setCurrentIndex(combo.findData(selected.id))
        return combo

    def _unlinked_animal_combo(self) -> QComboBox:
        combo = QComboBox()
        for animal in self.app_model.animals:
            if animal.external_identity is None:
                combo.addItem(animal.name, animal.id)
        selected = self.app_model.selected_animal
        if selected is not None and selected.external_identity is None:
            combo.setCurrentIndex(combo.findData(selected.id))
        return combo

    def _linked_animal_combo(self) -> QComboBox:
        combo = QComboBox()
        records_by_key = {
            record.identity.key: record
            for record in self.app_model.current_external_animal_records()
        }
        for animal in self.app_model.animals:
            if animal.external_identity is None:
                continue
            metadata = animal.external_metadata
            record = records_by_key.get(animal.external_identity.key)
            linked_name = (
                record.new_animal_name_candidate
                if record is not None and record.new_animal_name_candidate
                else animal.external_identity.subject_id
            )
            rfid = "unknown" if metadata is None or not metadata.rfid else metadata.rfid
            combo.addItem(f"{linked_name} · RFID {rfid}", animal.id)
        return combo

    def _link_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.link_animal = self._unlinked_animal_combo()
        form.addRow("Animal JSON name:", self.link_animal)
        self.external_record = QComboBox()
        self._external_records = tuple(
            record
            for record in self.app_model.current_external_animal_records()
            if record.identity.key not in self.app_model.external_link_conflicts
            and self.app_model.get_animal_by_external_identity(record.identity) is None
        )
        for record in self._external_records:
            linked_name = (
                record.new_animal_name_candidate or record.identity.subject_id
            )
            self.external_record.addItem(
                f"{linked_name} · RFID {record.physical_rfid}",
                record.identity.key,
            )
        form.addRow("Scanned RFID name:", self.external_record)
        note = QLabel(
            "This attaches the selected SoftMouse/RFID identity to the existing "
            "animal JSON, selects it, and then opens its editable details."
        )
        note.setWordWrap(True)
        form.addRow("", note)
        button = QPushButton("Link and edit")
        button.setEnabled(
            self.link_animal.count() > 0
            and bool(self._external_records)
            and self._ready()
        )
        button.clicked.connect(self._link)
        form.addRow("", button)
        return tab

    def _edit_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.edit_animal = self._all_animal_combo()
        form.addRow("Animal JSON name:", self.edit_animal)
        note = QLabel("Edit the local subject name and persistent animal notes.")
        note.setWordWrap(True)
        form.addRow("", note)
        button = QPushButton("Edit selected animal")
        button.setEnabled(self.edit_animal.count() > 0 and self._ready())
        button.clicked.connect(self._edit_selected)
        form.addRow("", button)
        return tab

    def _reconcile_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.survivor = self._all_animal_combo()
        self.linked_duplicate = self._linked_animal_combo()
        form.addRow("Keep animal JSON:", self.survivor)
        form.addRow("Merge linked RFID animal:", self.linked_duplicate)
        note = QLabel(
            "The kept JSON retains its name, notes, training, reach position, and "
            "limits. The RFID/SoftMouse link is transferred; the duplicate JSON "
            "is archived with a redirect. The kept animal opens for editing next."
        )
        note.setWordWrap(True)
        form.addRow("", note)
        button = QPushButton("Review and condense")
        button.setEnabled(
            self.survivor.count() > 0
            and self.linked_duplicate.count() > 0
            and self._ready()
        )
        button.clicked.connect(self._condense)
        form.addRow("", button)
        return tab

    def _ready(self) -> bool:
        return self.app_model.session_recording_status is SessionRecordingStatus.READY

    def _edit(self, animal) -> None:
        AnimalDetailsDialog(self.app_model, self, animal=animal).exec()

    def _edit_selected(self) -> None:
        animal = self.app_model.get_animal_by_id(self.edit_animal.currentData())
        if animal is not None:
            self._edit(animal)

    def _link(self) -> None:
        try:
            record = self._external_records[self.external_record.currentIndex()]
            animal = self.app_model.manually_link_animal(
                self.link_animal.currentData(), record
            )
        except Exception as exc:
            self.status.setText(f"Link failed: {exc}")
            return
        self.status.setText("RFID link saved.")
        self._edit(animal)

    def _condense(self) -> None:
        survivor_id = self.survivor.currentData()
        duplicate_id = self.linked_duplicate.currentData()
        if survivor_id == duplicate_id:
            self.status.setText("Choose two different animal JSON records.")
            return
        survivor = self.app_model.get_animal_by_id(survivor_id)
        if survivor is None or self.app_model.get_animal_by_id(duplicate_id) is None:
            self.status.setText("One of the selected animal JSON records no longer exists.")
            return
        answer = QMessageBox.question(
            self,
            "Confirm animal condensation",
            f"Keep animal JSON: {survivor.name}\n"
            f"Merge linked RFID animal: {self.linked_duplicate.currentText()}\n\n"
            "The duplicate JSON will be archived. Continue?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            merged = self.app_model.condense_animal_duplicate(
                survivor_id,
                duplicate_id,
            )
        except Exception as exc:
            self.status.setText(f"Condensation failed: {exc}")
            return
        self.status.setText("Animals condensed and RFID link transferred.")
        self._edit(merged)
