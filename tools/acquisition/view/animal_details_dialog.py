from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

from autotrainer.core import AnimalSubject, ExternalAnimalRecord


class AnimalDetailsDialog(QDialog):
    """Create, link, or edit the local JSON record for one animal."""

    def __init__(
        self,
        app_model,
        parent=None,
        *,
        record: Optional[ExternalAnimalRecord] = None,
        scanned_rfid: str = "",
        animal: Optional[AnimalSubject] = None,
    ):
        super().__init__(parent)
        if (record is None) == (animal is None):
            raise ValueError("Provide either a first-scan record or an existing animal")
        self.app_model = app_model
        self.record = record
        self.scanned_rfid = scanned_rfid
        self.animal = animal
        self.saved_animal: Optional[AnimalSubject] = None
        self.setWindowTitle(
            "Set up scanned animal" if record is not None else "Edit animal details"
        )
        self.resize(540, 390)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.target = None
        if record is not None:
            self.target = QComboBox()
            self.target.addItem("Create new animal JSON", None)
            for candidate in app_model.animals:
                if candidate.external_identity is None:
                    self.target.addItem(candidate.name, candidate.id)
            self.target.currentIndexChanged.connect(self._target_changed)
            form.addRow("Save as:", self.target)
            self._add_read_only_record(form, record, scanned_rfid)
        else:
            self._add_read_only_animal(form, animal)

        self.name = QLineEdit()
        self.name.setPlaceholderText("Required local subject name")
        form.addRow("Subject name:", self.name)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Persistent notes for this animal")
        self.notes.setMinimumHeight(100)
        form.addRow("Animal notes:", self.notes)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._target_changed()

    @staticmethod
    def _value_label(value) -> QLabel:
        label = QLabel(str(value or "—"))
        label.setWordWrap(True)
        return label

    def _add_read_only_record(
        self,
        form: QFormLayout,
        record: ExternalAnimalRecord,
        scanned_rfid: str,
    ) -> None:
        form.addRow("RFID:", self._value_label(scanned_rfid))
        form.addRow(
            "SoftMouse linked name:",
            self._value_label(record.new_animal_name_candidate),
        )
        form.addRow("SoftMouse subject ID:", self._value_label(record.identity.subject_id))
        form.addRow("Sex:", self._value_label(record.sex))
        form.addRow("Date of birth:", self._value_label(record.date_of_birth))
        form.addRow("Cage:", self._value_label(record.cage))

    def _add_read_only_animal(
        self, form: QFormLayout, animal: AnimalSubject
    ) -> None:
        metadata = animal.external_metadata
        form.addRow(
            "RFID:",
            self._value_label(None if metadata is None else metadata.rfid),
        )
        form.addRow(
            "SoftMouse subject ID:",
            self._value_label(
                None
                if animal.external_identity is None
                else animal.external_identity.subject_id
            ),
        )

    def _target_changed(self) -> None:
        animal = self.animal
        if self.target is not None:
            animal_id = self.target.currentData()
            animal = (
                None
                if animal_id is None
                else self.app_model.get_animal_by_id(animal_id)
            )
        if animal is None:
            default_name = (
                self.record.new_animal_name_candidate
                or self.record.identity.subject_id
            )
            self.name.setText(default_name)
            self.notes.setPlainText("")
        else:
            self.name.setText(animal.name)
            self.notes.setPlainText(animal.notes)

    def _save(self) -> None:
        name = self.name.text().strip()
        if not name:
            self.status.setText("Subject name is required.")
            return
        try:
            if self.record is not None:
                self.saved_animal = self.app_model.complete_rfid_animal_setup(
                    self.record,
                    scanned_rfid=self.scanned_rfid,
                    name=name,
                    notes=self.notes.toPlainText(),
                    existing_animal_id=self.target.currentData(),
                ).animal
            else:
                self.saved_animal = self.app_model.update_animal_details(
                    self.animal.id,
                    name=name,
                    notes=self.notes.toPlainText(),
                )
        except Exception as exc:
            self.status.setText(f"Not saved: {exc}")
            return
        self.accept()
