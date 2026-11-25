import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from softmouse_https_source_test import COLONY, Session  # noqa: E402

from reachaq.app.model.animal_sources.config import load_source_configuration
from reachaq.app.model.animal_sources.credentials import KEYRING_SERVICE, CredentialStore
from reachaq.app.model.animal_sources.softmouse import SoftMouseSource
from reachaq.app.view.animal_source_setup_dialog import AnimalSourceSetupDialog


TAG = "360002353933099"


class Preferences:
    animal_metadata_source = ""
    softmouse_manifest_path = ""
    softmouse_name_column = "Physical Tag"
    saved = 0

    def save(self):
        self.saved += 1


class FakeKeyring:
    priority = 1

    def __init__(self):
        self.values = {}

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[(service, account)] = value


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def choose_kind(dialog, kind):
    dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData(kind))


def test_softmouse_setup_lists_colonies_and_saves_the_password_to_the_keyring(
    qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(SoftMouseSource, "session", Session())
    keyring = FakeKeyring()
    preferences = Preferences()
    dialog = AnimalSourceSetupDialog(
        preferences,
        credentials=CredentialStore(keyring_backend=keyring, fallback_path=tmp_path / "s.json"),
    )
    choose_kind(dialog, "softmouse")
    dialog._field_widgets["username"].setText("ben")
    dialog._field_widgets["password"].setText("secret")

    dialog.load_choices("colony")
    colony = dialog._field_widgets["colony"]
    assert [colony.itemText(i) for i in range(colony.count())] == [COLONY, "Other Colony"]
    dialog.read_columns()
    assert dialog._mapping_combos["rfid_column"].currentText() == "Plate ID"
    assert dialog.require_rfid.isChecked()
    dialog.save()

    assert dialog.result() == dialog.DialogCode.Accepted
    assert preferences.saved == 1
    configuration = load_source_configuration(preferences)
    assert configuration.settings == {"username": "ben", "colony": COLONY}
    assert configuration.mapping == SoftMouseSource.default_mapping()
    assert keyring.values == {(KEYRING_SERVICE, "softmouse/password/ben"): "secret"}
    assert "secret" not in preferences.animal_metadata_source


def test_spreadsheet_setup_maps_the_files_columns(qapp, tmp_path):
    sheet = tmp_path / "animals.csv"
    sheet.write_text(f"Mouse ID,RFID,Status\nM-1,{TAG},alive\n", encoding="utf-8")
    preferences = Preferences()
    dialog = AnimalSourceSetupDialog(
        preferences,
        credentials=CredentialStore(keyring_backend=False, fallback_path=tmp_path / "s.json"),
    )
    choose_kind(dialog, "spreadsheet")
    assert dialog._publish_group.isVisibleTo(dialog)
    dialog._field_widgets["path"].setText(str(sheet))

    dialog.read_columns()
    assert "1 rows" in dialog.status.text()
    assert dialog._mapping_combos["external_identity_column"].currentText() == "Mouse ID"
    # Only the tag, name and RFID columns are chosen; the rest are filled in by name.
    assert set(dialog._mapping_combos) == {
        "external_identity_column", "new_animal_name_column", "rfid_column"
    }
    assert "State = Status" in dialog.other_columns.text()
    dialog.save()

    configuration = load_source_configuration(preferences)
    assert configuration.kind == "spreadsheet"
    assert configuration.mapping.new_animal_name_column == "Mouse ID"
    assert configuration.mapping.rfid_column == "RFID"
    assert configuration.mapping.state_column == "Status"
    assert not configuration.mapping.require_rfid


def test_setup_saves_a_name_column_apart_from_the_physical_tag(qapp, tmp_path):
    sheet = tmp_path / "animals.csv"
    sheet.write_text(f"Physical Tag,Name,RFID\nM-1,Mouse One,{TAG}\n", encoding="utf-8")
    preferences = Preferences()
    dialog = AnimalSourceSetupDialog(
        preferences,
        credentials=CredentialStore(keyring_backend=False, fallback_path=tmp_path / "s.json"),
    )
    choose_kind(dialog, "spreadsheet")
    dialog._field_widgets["path"].setText(str(sheet))

    dialog.read_columns()
    assert dialog._mapping_combos["new_animal_name_column"].currentText() == "Name"
    dialog.save()

    mapping = load_source_configuration(preferences).mapping
    assert mapping.external_identity_column == "Physical Tag"
    assert mapping.new_animal_name_column == "Name"


def test_errors_stay_in_the_dialog_and_nothing_is_saved(qapp, tmp_path):
    preferences = Preferences()
    dialog = AnimalSourceSetupDialog(
        preferences,
        credentials=CredentialStore(keyring_backend=False, fallback_path=tmp_path / "s.json"),
    )
    choose_kind(dialog, "spreadsheet")
    dialog._field_widgets["path"].setText(str(tmp_path / "missing.csv"))

    dialog.read_columns()
    assert dialog.status.text().startswith("Failed:")
    dialog._field_widgets["path"].setText("")
    dialog.save()

    assert dialog.status.text().startswith("Not saved:")
    assert preferences.animal_metadata_source == ""


def test_an_existing_setup_is_shown_again_without_its_secret(qapp, tmp_path):
    preferences = Preferences()
    preferences.animal_metadata_source = (
        '{"kind": "database", "settings": {"url": "/srv/colony.sqlite", "query": "SELECT 1"},'
        ' "mapping": {"external_identity_column": "id", "new_animal_name_column": "id"}}'
    )
    dialog = AnimalSourceSetupDialog(
        preferences,
        credentials=CredentialStore(keyring_backend=False, fallback_path=tmp_path / "s.json"),
    )

    assert dialog.kind_name == "database"
    assert dialog._field_widgets["url"].text() == "/srv/colony.sqlite"
    assert dialog._field_widgets["query"].toPlainText() == "SELECT 1"
    password = dialog._field_widgets["password"]
    assert password.text() == "" and "Stored" in password.placeholderText()
    assert dialog._mapping_combos["external_identity_column"].currentText() == "id"
    assert dialog._mapping_combos["rfid_column"].currentText() == "(none)"


def test_a_publication_reader_cannot_also_publish(qapp, tmp_path):
    dialog = AnimalSourceSetupDialog(
        Preferences(),
        credentials=CredentialStore(keyring_backend=False, fallback_path=tmp_path / "s.json"),
    )
    choose_kind(dialog, "publication")

    assert not dialog._publish_group.isVisibleTo(dialog)
    assert isinstance(dialog._field_widgets["manifest"], QLineEdit)
    assert isinstance(dialog.kind_combo, QComboBox)
