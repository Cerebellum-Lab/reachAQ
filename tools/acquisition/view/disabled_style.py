"""Make disabled controls obvious without changing any control's size.

Under Fusion a disabled button, field or box looked almost the same as an
enabled one (Ben, christielab10, 2026-10-02). A style sheet rule for
:disabled would replace Fusion's frame with a box model of its own and can
change a control's size as it is enabled and disabled, which moves the tightly
fitted Laser Control layout. This draws on top of Fusion instead: a dashed grey
outline inside the frame Fusion already drew, and a muted, flat disabled
palette. Sizes and layouts are untouched.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QPen
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle

DISABLED_TEXT = QColor("#a3a3a3")
DISABLED_FILL = QColor("#f4f4f4")
DISABLED_BORDER = QColor("#8f8f8f")

#: Frames that get the dashed outline when their control is disabled.
_OUTLINED_PRIMITIVES = frozenset({
    QStyle.PrimitiveElement.PE_PanelButtonCommand,  # QPushButton
    QStyle.PrimitiveElement.PE_PanelButtonTool,     # QToolButton with a panel
    QStyle.PrimitiveElement.PE_FrameLineEdit,       # QLineEdit
    QStyle.PrimitiveElement.PE_IndicatorCheckBox,   # QCheckBox's box
})
_OUTLINED_CONTROLS = frozenset({
    QStyle.ComplexControl.CC_ComboBox,
    QStyle.ComplexControl.CC_SpinBox,
})


def _is_disabled(option) -> bool:
    return not (option.state & QStyle.StateFlag.State_Enabled)


def _outline(painter, rect) -> None:
    painter.save()
    pen = QPen(DISABLED_BORDER)
    pen.setStyle(Qt.PenStyle.DashLine)
    pen.setWidth(1)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(rect.adjusted(0, 0, -1, -1))
    painter.restore()


class DisabledVisibleStyle(QProxyStyle):
    """Fusion, with a dashed outline drawn over disabled controls' frames."""

    def drawPrimitive(self, element, option, painter, widget=None):
        super().drawPrimitive(element, option, painter, widget)
        if element in _OUTLINED_PRIMITIVES and _is_disabled(option):
            _outline(painter, option.rect)

    def drawComplexControl(self, control, option, painter, widget=None):
        super().drawComplexControl(control, option, painter, widget)
        if control in _OUTLINED_CONTROLS and _is_disabled(option):
            _outline(painter, option.rect)


def disabled_palette(base: QPalette) -> QPalette:
    """`base` with muted text and a flat light fill for disabled controls."""
    palette = QPalette(base)
    disabled = QPalette.ColorGroup.Disabled
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(disabled, role, DISABLED_TEXT)
    for role in (QPalette.ColorRole.Button, QPalette.ColorRole.Base):
        palette.setColor(disabled, role, DISABLED_FILL)
    return palette


def apply_disabled_style(app: QApplication, base_style: str = "Fusion") -> DisabledVisibleStyle:
    """Install the style and palette on `app`; returns the style it set."""
    style = DisabledVisibleStyle(base_style)
    app.setStyle(style)
    app.setPalette(disabled_palette(style.standardPalette()))
    return style
