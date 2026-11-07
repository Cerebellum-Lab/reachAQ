import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QTabWidget

from autotrainer.core import AnimalSubject
from tools.acquisition.view.animal_metadata_dialog import AnimalMetadataDialog
from tools.acquisition.view.main_window import MainWindow


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_manual_link_and_reconciliation_workflows_are_visible(qapp, app_model):
    app_model.add_animal("one")
    app_model.add_animal("two")
    dialog = AnimalMetadataDialog(app_model)

    tabs = dialog.findChild(QTabWidget)
    assert tabs.count() == 3
    assert tabs.tabText(0) == "Edit animal"
    assert tabs.tabText(1) == "Link RFID"
    assert tabs.tabText(2) == "Condense duplicate"

    dialog.close()


def test_new_animal_is_populated_in_editable_subject_widget(qapp):
    class WindowHarness:
        def __init__(self):
            self._animal_dropdown_combo = QComboBox()
            self._animal_dropdown_combo.setEditable(True)
            self._animal_dropdown_combo.addItem("", None)

        def _show_selected_animal_identity(self, animal):
            pass

        def _refresh_prev_next_phases(self):
            pass

    window = WindowHarness()
    animal = AnimalSubject(name="new mouse")

    MainWindow._sync_selected_animal_widget(window, animal)

    assert window._animal_dropdown_combo.currentData() == animal.id
    assert window._animal_dropdown_combo.currentText() == "new mouse"
