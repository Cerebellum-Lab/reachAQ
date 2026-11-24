from __future__ import annotations

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class CardFooter(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("CardFooter")
        self.setStyleSheet(
            "#CardFooter {"
            "background-color: #eef0f2; "
            "padding: 0px; "
            "border-top: 1px solid #cfd3d8; "
            "border-bottom-left-radius: 4px; "
            "border-bottom-right-radius: 4px"
            "}"
            "#CardFooter QLabel {color: #4b5563;}"
        )
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._layout = None

    def setContent(self, widget: QWidget):
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        self._layout.setContentsMargins(4, 2, 4, 3)
        self._layout.addWidget(widget)
        widget.setContentsMargins(4, 1, 2, 1)
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setLayout(self._layout)
