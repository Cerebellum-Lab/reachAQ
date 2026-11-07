"""Legend and chooser for the live pose overlay, in one window.

The two belong together: the question "what is the cyan dot?" and the question
"stop drawing the cyan dot" are asked at the same moment, by the same person,
about the same list. Splitting them into a legend and a settings page would
mean reading one to act on the other.

The colours come from the painter itself rather than being restated here, so
the legend cannot drift out of step with what is on screen.
"""

from typing import Iterable, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autotrainer.pyside.capture.QtGLImageView import (
    is_subsumed_by_composite,
    overlay_colour_for,
)

SWATCH = 12


def _swatch(name: str) -> QLabel:
    """A filled dot in the colour this part is drawn in."""
    pixmap = QPixmap(SWATCH, SWATCH)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    colour = QColor(overlay_colour_for(name))
    painter.setPen(QPen(colour))
    painter.setBrush(QBrush(colour))
    painter.drawEllipse(1, 1, SWATCH - 3, SWATCH - 3)
    painter.end()
    label = QLabel()
    label.setPixmap(pixmap)
    label.setFixedWidth(SWATCH + 6)
    return label


class OverlayPartsDialog(QDialog):
    """Show which colour means which part, and choose what is drawn."""

    def __init__(self, parts: Sequence[str], shown: Iterable[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay parts")
        self._boxes = {}

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Colours are the ones the overlay paints. Parts left unticked are "
            "not drawn."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid);")
        layout.addWidget(intro)

        # Empty configuration means "everything the model reports, except the
        # hand orientations a composite already stands for". Reproduce that
        # here so the dialog opens showing what is actually on screen.
        chosen = set(shown)
        listing = QWidget()
        rows = QVBoxLayout(listing)
        rows.setContentsMargins(0, 0, 0, 0)
        for name in parts:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(_swatch(name))
            box = QCheckBox(str(name))
            box.setChecked(name in chosen if chosen
                           else not is_subsumed_by_composite(name))
            if is_subsumed_by_composite(name):
                box.setToolTip(
                    "A hand is drawn as one dot from whichever orientation the "
                    "model saw most confidently, so this is off unless you want "
                    "to see the orientation itself.")
            self._boxes[name] = box
            row.addWidget(box, 1)
            holder = QWidget()
            holder.setLayout(row)
            rows.addWidget(holder)
        rows.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(listing)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(260)
        layout.addWidget(scroll)

        presets = QHBoxLayout()
        all_button = QPushButton("All")
        all_button.clicked.connect(lambda: self._set_all(True))
        none_button = QPushButton("None")
        none_button.clicked.connect(lambda: self._set_all(False))
        default_button = QPushButton("Default")
        default_button.setToolTip(
            "Every part the model reports, with each hand as a single dot")
        default_button.clicked.connect(self._restore_default)
        presets.addWidget(all_button)
        presets.addWidget(none_button)
        presets.addWidget(default_button)
        presets.addStretch(1)
        layout.addLayout(presets)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, checked: bool) -> None:
        for box in self._boxes.values():
            box.setChecked(checked)

    def _restore_default(self) -> None:
        for name, box in self._boxes.items():
            box.setChecked(not is_subsumed_by_composite(name))

    def selected_parts(self) -> Tuple[str, ...]:
        """The parts to draw.

        Returns every ticked part, including when that is all of them: the
        caller stores this, and storing an explicit list is what lets a part a
        composite normally hides stay visible across a restart.
        """
        return tuple(name for name, box in self._boxes.items()
                     if box.isChecked())
