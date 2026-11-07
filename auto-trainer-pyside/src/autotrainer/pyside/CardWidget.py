from __future__ import annotations

from typing import Optional, Union

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLayout, QBoxLayout, QVBoxLayout

from .CardFooter import CardFooter
from .CardHeader import CardHeader
from .StackedContent import StackedWidget

_DEFAULT_STYLE = "border-color: #c9cdd3; border-width: 1px; border-style: solid; border-radius: 4px;"


class CardWidget(QWidget):

    def __init__(
        self,
        background_color: str = "white",
        title: str = "",
        header_background_color: Optional[str] = None,
        content_layout: Optional[QLayout] = None,
        header_right_layout: Optional[Union[QWidget, QLayout, QBoxLayout]] = None,
    ):
        super().__init__()

        # self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setContentsMargins(0, 0, 0, 0)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("CardWidget")

        style = _DEFAULT_STYLE

        if background_color is not None:
            style = f"background-color: {background_color}; {style}"

        self.setStyleSheet(f"#CardWidget {{{style}}}")

        layout = self._layout = QVBoxLayout()  # QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = self._header = CardHeader(title=title, background_color=header_background_color)

        if header_right_layout is not None:
            header.setRightContent(header_right_layout)

        # for info: setting the alignment here is preventing the header to expand itself ...
        layout.addWidget(header)

        content_widget = self._content_widget = StackedWidget()
        # content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(content_widget)

        self._footer = CardFooter()
        layout.addWidget(self._footer)

        self.setLayout(layout)

        self._last_widget_or_layout = None

        if content_layout is not None:
            self.setContentLayout(content_layout)

    @property
    def header(self) -> CardHeader:
        return self._header

    @property
    def footer(self) -> CardFooter:
        return self._footer

    def setFooterVisible(self, visible: bool):
        self._footer.setVisible(visible)

    def setContentWidget(self, widget: QWidget,
                         *,
                         alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         ):
        last = self._last_widget_or_layout
        if last is not None:
            self._content_widget.removeWidget(last)
            last.setParent(None)  # THIS IS REQUIRED,
            # otherwise the last widget will continue display itself on top of any other
            # already selected previously.
            last.hide()
            last.update()  # force update to ensure widget is hidden

        if widget is not None:
            self._content_widget.addWidget(widget)
            self._content_widget.setCurrentWidget(widget)
            # widget.setParent(self)  # not required, this is implicit with addWidget()

        self._last_widget_or_layout = widget
        self._layout.update()  # force update to ensure layout is refreshed

    def setContentLayout(
        self,
        layout: QLayout,
    ):
        last = self._last_widget_or_layout
        if last is not None:
            self._content_widget.removeWidget(last)
        widget = QWidget()
        widget.setLayout(layout)
        self._content_widget.addWidget(widget)
        self._content_widget.setCurrentWidget(widget)
        self._last_widget_or_layout = widget
        self._layout.update()  # force update to ensure layout is refreshed
