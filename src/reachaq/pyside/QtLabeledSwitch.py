from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel

from .QtSwitch import QSwitch


class QLabeledSwitch(QWidget):
    stateChanged = Signal(int)

    def __init__(self, parent=None):
        super(QLabeledSwitch, self).__init__(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

        self._label = QLabel("Off")
        layout.addWidget(self._label)

        self._switch = QSwitch()
        self._switch.stateChanged.connect(self._state_changed)

        layout.addWidget(self._switch)

        self.setLayout(layout)

    def isChecked(self) -> bool:
        return self._switch.isChecked()

    def setChecked(self, state: bool) -> None:
        self._switch.setChecked(state)

    def _state_changed(self, state):
        self._label.setText("On" if state != 0 else "Off")
        self.stateChanged.emit(state)
