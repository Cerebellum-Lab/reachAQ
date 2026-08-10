import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QTabWidget

from tools.acquisition.view.animal_metadata_dialog import AnimalMetadataDialog


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_manual_link_and_reconciliation_workflows_are_visible(qapp, app_model):
    app_model.add_animal("one")
    app_model.add_animal("two")
    dialog = AnimalMetadataDialog(app_model)

    tabs = dialog.findChild(QTabWidget)
    assert tabs.count() == 2
    assert tabs.tabText(0) == "Manual link"
    assert tabs.tabText(1) == "Condense duplicates"

    dialog.close()
