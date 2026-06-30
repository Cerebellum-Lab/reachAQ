from __future__ import annotations

from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from autotrainer.pyside import CardWidget
from autotrainer.pyside.content_widget import ContentWidget


class ProtocolContent(ContentWidget):
    """Placeholder for the new reachAQ protocol workflow."""

    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._card_widget = CardWidget(title="Protocol")
        self._card_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._card_widget.setContentWidget(content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._card_widget, stretch=1)
        self.setLayout(layout)
