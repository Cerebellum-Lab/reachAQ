"""Small widgets for a docked panel that must fit without scrolling.

Laser Control sits in the right-hand panel, about 440 px wide on
christielab10's 1920x1080 screen, and every control, both laser graphs and
the builder preview had to fit there at once. These pieces make that
possible without hiding anything: a section that folds away, a one-line
label that elides instead of wrapping or widening its panel, and graph axes
drawn in the panel's smaller font.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

#: The panel font, about 20% under the rig's 9 pt Qt default.
COMPACT_FONT_POINT_SIZE = 7.5
#: Graph tick and axis-label font; pyqtgraph draws these itself.
COMPACT_AXIS_POINT_SIZE = 7.0
#: Past these, a field only gets emptier; the extra width goes to the graphs.
FIELD_MAXIMUM_WIDTH = 170
WIDE_FIELD_MAXIMUM_WIDTH = 400
#: Grid field columns grow first; an empty last column takes what is left
#: once the fields reach their maximum width.
FIELD_COLUMN_STRETCH = 10


def compact_font_style_sheet(object_name: str) -> str:
    """The panel font for a widget and everything inside it."""
    return (
        f"#{object_name}, #{object_name} QWidget "
        f"{{font-size: {COMPACT_FONT_POINT_SIZE:g}pt;}}"
    )


def compact_button_style_sheet(object_name: str) -> str:
    """Push buttons a little shorter than the style's, inside one widget."""
    return f"#{object_name} QPushButton {{min-height: 18px; padding: 1px 8px;}}"


def compact_combo_box() -> QComboBox:
    """A combo box sized to a few characters rather than its longest entry.

    Sized to its longest entry, a board route or a long profile name widened
    the whole panel.
    """
    combo_box = QComboBox()
    combo_box.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo_box.setMinimumContentsLength(6)
    return combo_box


def form_label(text: str) -> QLabel:
    """A field's label, right-aligned against the field."""
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def field_grid(parent: QWidget, fields) -> QGridLayout:
    """Label and field pairs, two to a row, in a grid on ``parent``.

    Each field grows up to FIELD_MAXIMUM_WIDTH, and then an empty fifth
    column takes the rest, so a wide detached panel does not stretch them.
    Returns the grid, for any rows under the fields.
    """
    grid = QGridLayout(parent)
    grid.setContentsMargins(4, 0, 2, 2)
    grid.setHorizontalSpacing(4)
    grid.setVerticalSpacing(2)
    for index, (text, field) in enumerate(fields):
        row, column = divmod(index, 2)
        field.setMaximumWidth(FIELD_MAXIMUM_WIDTH)
        grid.addWidget(form_label(text), row, 2 * column)
        grid.addWidget(field, row, 2 * column + 1)
    grid.setColumnStretch(1, FIELD_COLUMN_STRETCH)
    grid.setColumnStretch(3, FIELD_COLUMN_STRETCH)
    grid.setColumnStretch(4, 1)
    return grid


def compact_plot_axes(plot_widget, point_size: float = COMPACT_AXIS_POINT_SIZE) -> None:
    """Draw a pyqtgraph plot's ticks and axis labels in the compact font.

    A widget style sheet does not reach them: the axes are graphics items
    with their own font. Call after the axis labels are set.
    """
    font = QFont(plot_widget.font())
    font.setPointSizeF(point_size)
    for name in ("left", "bottom"):
        axis = plot_widget.getAxis(name)
        axis.setStyle(tickFont=font, tickTextOffset=2)
        if axis.labelText or axis.labelUnits:
            axis.setLabel(
                axis.labelText,
                units=axis.labelUnits or None,
                unitPrefix=axis.labelUnitPrefix or None,
                **{**axis.labelStyle, "font-size": f"{point_size:g}pt"},
            )


class ElidedLabel(QLabel):
    """One line of text that ends in an ellipsis when it does not fit.

    A plain label either wraps, which makes a panel's height depend on its
    text, or sets the panel's minimum width to the whole line. This one keeps
    one line at any width, and its tooltip shows the whole text whenever any
    of it is cut, followed by whatever tooltip it was given.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setWordWrap(False)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        ellipsis = self.fontMetrics().horizontalAdvance("…")
        margins = self.contentsMargins()
        width = min(hint.width(), 3 * ellipsis + margins.left() + margins.right())
        return QSize(width, hint.height())

    def elided_text(self) -> str:
        return self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width())

    def is_elided(self) -> bool:
        return self.elided_text() != self.text()

    def full_tooltip(self) -> str:
        """What hovering shows: the whole text when cut, then the set tooltip."""
        parts = (self.text() if self.is_elided() else "", self.toolTip())
        return "\n".join(part for part in parts if part)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        self.style().drawItemText(
            painter,
            self.contentsRect(),
            int(self.alignment()),
            self.palette(),
            self.isEnabled(),
            self.elided_text(),
            self.foregroundRole(),
        )

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            tooltip = self.full_tooltip()
            if tooltip:
                QToolTip.showText(event.globalPos(), tooltip, self)
            else:
                QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)


class CollapsibleSection(QWidget):
    """A titled group whose contents fold away under an arrow header.

    Styled like the Hardware Status categories. Put the contents in a layout
    on ``content``. A section marked ``stretch`` takes the spare height while
    it is open; closed, every section is its header only, so the space goes
    to whatever else is open.
    """

    expanded_changed = Signal(bool)

    def __init__(
        self,
        title: str,
        name: Optional[str] = None,
        *,
        expanded: bool = True,
        stretch: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        #: The key an owner remembers this section's state under.
        self.name = name or title
        self.setObjectName(self.name)
        self._stretch = stretch

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self.header = QWidget(self)
        self.header.setObjectName("CollapsibleSectionHeader")
        self.header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.header.setStyleSheet(
            "#CollapsibleSectionHeader {"
            "background: #eef1f4; border: 1px solid #d1d5db; border-radius: 3px;"
            "}"
            "#CollapsibleSectionHeader QToolButton {"
            "color: #20242a; font-weight: 600; border: none; text-align: left;"
            "padding: 0px;"
            "}"
        )
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(3, 0, 4, 0)
        header_layout.setSpacing(4)

        self.toggle_button = QToolButton(self.header)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setIconSize(QSize(8, 8))
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header_layout.addWidget(self.toggle_button)
        layout.addWidget(self.header)

        self.content = QWidget(self)
        self.content.setObjectName("CollapsibleSectionContent")
        layout.addWidget(self.content, stretch=1)

        self.toggle_button.setChecked(expanded)
        self._apply_expanded(expanded)
        self.toggle_button.toggled.connect(self._on_toggled)

    @property
    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(bool(expanded))

    def setToolTip(self, text: str) -> None:
        # On the header, where the pointer is when deciding to open it.
        self.toggle_button.setToolTip(text)

    def _on_toggled(self, expanded: bool) -> None:
        self._apply_expanded(expanded)
        self.expanded_changed.emit(expanded)

    def _apply_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)
        if not expanded:
            vertical = QSizePolicy.Policy.Fixed
        elif self._stretch:
            vertical = QSizePolicy.Policy.Expanding
        else:
            vertical = QSizePolicy.Policy.Preferred
        self.setSizePolicy(QSizePolicy.Policy.Preferred, vertical)
