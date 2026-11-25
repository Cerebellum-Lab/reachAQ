from PySide6.QtCore import QSize, Qt, QKeyCombination
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QStatusBar, QWidget, QVBoxLayout, QApplication
from reachaq.pyside import Separator

from reachaq.pellet_delivery.model.app_model import AppModel
from reachaq.pellet_delivery.view.main_content import MainContent

_MIN_WIDTH = 600
_DEFAULT_WIDTH = 800
_HEIGHT_WITH_DIAGNOSTICS = 480
_HEIGHT_WITHOUT_DIAGNOSTICS = 360


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication, app_view_model: AppModel):
        super().__init__()

        self._app = app

        self._app_view_model = app_view_model

        self.setWindowTitle("Pellet Module")

        self.setMinimumSize(QSize(_DEFAULT_WIDTH, _HEIGHT_WITH_DIAGNOSTICS))

        self.main_content = MainContent(app_view_model)

        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(Separator("#b9b9b9"))
        layout.addWidget(self.main_content)
        layout.addWidget(Separator("#b9b9b9"))
        layout.setStretch(1, 1)
        container.setLayout(layout)

        self.setCentralWidget(container)

        self._configure_actions()

        self._configure_menubar()

        self.setStatusBar(QStatusBar(self))

        self.main_content.on_disconnect(lambda: self.update_status(""))

        self.update_status("")

        self._app_view_model.property_changed += self._model_property_changed

    def update_status(self, value: str):
        self.statusBar().showMessage(value)

    def on_activated(self):
        self._app_view_model.on_activated()
        self.main_content.on_activated()

    def closeEvent(self, event):
        self._app_view_model.on_close()

        event.accept()

    def _configure_actions(self):
        self.quit_action = QAction("Quit")
        self.quit_action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_Q))
        self.quit_action.triggered.connect(lambda: self._app.quit())

        self.view_diagnostics_action = QAction("Diagnostics", self)
        self.view_diagnostics_action.setToolTip("Show or hide diagnostics panel")
        self.view_diagnostics_action.setShortcut(QKeyCombination(Qt.Modifier.CTRL, Qt.Key.Key_D))
        self.view_diagnostics_action.setCheckable(True)
        self.view_diagnostics_action.setChecked(self.main_content.is_diagnostics_visible)
        self.view_diagnostics_action.triggered.connect(lambda: self._toggle_diagnostics_view())

    def _configure_menubar(self):
        menu_bar = self.menuBar()

        menu_bar.setObjectName("MenuBar")
        menu_bar.setStyleSheet("#MenuBar {background-color: #eee}")

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.quit_action)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(self.view_diagnostics_action)

    def _toggle_diagnostics_view(self):
        width = self.width()
        height = self.height()
        if not self.main_content.is_diagnostics_visible:
            self.resize(width, max(_HEIGHT_WITH_DIAGNOSTICS, height))
            self.setMinimumSize(_MIN_WIDTH, _HEIGHT_WITH_DIAGNOSTICS)
        else:
            self.resize(width, max(_HEIGHT_WITHOUT_DIAGNOSTICS, height))
            self.setMinimumSize(_MIN_WIDTH, _HEIGHT_WITHOUT_DIAGNOSTICS)

        self.main_content.set_diagnostics_visible(not self.main_content.is_diagnostics_visible)
        self.view_diagnostics_action.setChecked(self.main_content.is_diagnostics_visible)

    def _model_property_changed(self, name: str, value, _old_value):
        if name == "hardware_configuration":
            self.update_status(value)
