from __future__ import annotations

from typing import Iterable, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget


# High-contrast colors on the light graph background.  The ordering is shared
# by selectors, curves, and legends so the first option is always blue, the
# second green, and subsequent options remain visually distinct.
STREAM_SIGNAL_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (23, 105, 224),   # blue
    (18, 138, 67),    # green
    (214, 107, 0),    # orange
    (123, 63, 198),   # purple
    (198, 40, 40),    # red
    (0, 131, 143),    # teal
    (173, 20, 87),    # magenta
    (69, 90, 100),    # blue gray
)


def stream_signal_color(index: int) -> Tuple[int, int, int]:
    return STREAM_SIGNAL_COLORS[index % len(STREAM_SIGNAL_COLORS)]


def color_hex(color: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def color_code_checkbox(checkbox, color: Tuple[int, int, int]) -> None:
    """Give a stream option a visible swatch while retaining native disabled state."""
    hex_color = color_hex(color)
    checkbox.setProperty("signalColor", hex_color)
    checkbox.setText(f"■  {checkbox.text()}")
    checkbox.setStyleSheet(
        f"QCheckBox {{ color: {hex_color}; font-weight: 600; spacing: 5px; }}"
        f"QCheckBox:disabled {{ color: {hex_color}; }}"
    )


class StreamGraphLegend(QWidget):
    """Compact below-graph legend shared by Analysis and laser streams."""

    def __init__(self, *, columns: int = 3, parent=None):
        super().__init__(parent)
        self._columns = max(1, columns)
        self._entries = tuple()
        self.setObjectName("StreamGraphLegend")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setHorizontalSpacing(12)
        self._layout.setVerticalSpacing(2)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    @property
    def entries(self):
        return self._entries

    def set_entries(
        self,
        entries: Iterable[Tuple[str, Tuple[int, int, int], bool]],
    ) -> None:
        entries = tuple(entries)
        if entries == self._entries:
            return
        self._entries = entries
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for index, (label_text, color, _is_digital) in enumerate(entries):
            item_widget = QWidget(self)
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)

            swatch = QFrame(item_widget)
            swatch.setObjectName("StreamLegendSwatch")
            swatch.setFixedSize(24, 8)
            hex_color = color_hex(color)
            swatch.setStyleSheet(
                f"QFrame#StreamLegendSwatch {{ background: {hex_color}; border: 0px; }}"
            )

            label = QLabel(label_text, item_widget)
            label.setStyleSheet("color: #2f343a; font-size: 10px;")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            item_layout.addWidget(swatch)
            item_layout.addWidget(label)
            self._layout.addWidget(item_widget, index // self._columns, index % self._columns)

        self.setVisible(bool(entries))
