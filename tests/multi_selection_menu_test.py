import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tools.acquisition.view.multi_selection_menu import MultiSelectionMenu


def _app():
    return QApplication.instance() or QApplication([])


def test_checkable_action_keeps_multi_selection_menu_open():
    app = _app()
    menu = MultiSelectionMenu()
    action = QAction("Enabled", menu)
    action.setCheckable(True)
    menu.addAction(action)
    menu.popup(QPoint(100, 100))
    app.processEvents()

    point = menu.actionGeometry(action).center()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()

    assert action.isChecked()
    assert menu.isVisible()
    menu.close()


def test_ordinary_action_closes_multi_selection_menu():
    app = _app()
    menu = MultiSelectionMenu()
    action = QAction("Configure…", menu)
    menu.addAction(action)
    menu.popup(QPoint(100, 100))
    app.processEvents()

    point = menu.actionGeometry(action).center()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=point)
    app.processEvents()

    assert not menu.isVisible()
