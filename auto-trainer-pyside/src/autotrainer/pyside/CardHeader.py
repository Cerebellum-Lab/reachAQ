from __future__ import annotations

from typing import Optional, Union

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLayout, QSizePolicy


class CardHeader(QWidget):
    DEFAULT_BACKGROUND_COLOR = "#eef1f4"
    DEFAULT_TITLE_COLOR = "#20242a"

    @staticmethod
    def _make_main_layout():
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)
        return layout

    def __init__(self, title: str = "", background_color: Optional[str] = None, title_color: Optional[str] = None):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)

        if background_color is None:
            background_color = CardHeader.DEFAULT_BACKGROUND_COLOR

        if title_color is None:
            title_color = CardHeader.DEFAULT_TITLE_COLOR

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("CardHeader")
        style = (
            f"""#CardHeader {{
            background-color: {background_color};
            padding: 0px;
            border-bottom: 1px solid #c9cdd3;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            }}""")
        style += f"#CardHeader QLabel {{color: {title_color};}}"
        self.setStyleSheet(style)

        layout = self._layout = self._make_main_layout()

        sub_l = QHBoxLayout()
        sub_l.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label = self._title_label = QLabel(f"<b>{title}</b>")
        sub_l.addWidget(label)
        if title_color is not None:
            label.setStyleSheet(f"color: {title_color}")
        layout.addLayout(sub_l)

        self._right_content: Optional[QWidget, QLayout] = None

        layout.addStretch(1)  # this allow the right content to be put on to the right side of this hbox layout
        self.setLayout(layout)

    def setTitle(self, title: str, color: Optional[str] = None):
        title = f"<b>{title}</b>"
        self._title_label.setText(title)
        if color is not None:
            self._title_label.setStyleSheet(f"color: {color}")

    def setRightContent(self, content: Optional[Union[QWidget, QLayout]] = None):
        prev = self._right_content
        if prev is not None:
            self._layout.removeWidget(prev)
            prev.setParent(None)  # required

        if content is not None:
            self._right_content = content
            if isinstance(content, QWidget):
                self._layout.addWidget(content, alignment=Qt.AlignmentFlag.AlignRight)
            else:
                self._layout.addLayout(content)
                content.setAlignment(Qt.AlignmentFlag.AlignRight)

    def setContent(self, widget: QWidget):
        layout = self._layout = self._make_main_layout()
        self.setLayout(layout)
        layout.addStretch(1)  # this allow the right content to be put on to the right side of this hbox layout
