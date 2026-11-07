"""Disabled controls are obvious, and nothing changes size (Ben, 2026-10-02)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton,
)

from tools.acquisition.view.disabled_style import (
    DISABLED_BORDER, DISABLED_FILL, DISABLED_TEXT, DisabledVisibleStyle, disabled_palette,
)


@pytest.fixture
def style():
    app = QApplication.instance() or QApplication([])
    previous_style, previous_palette = app.style().name(), app.palette()
    style = DisabledVisibleStyle("Fusion")
    yield style
    app.setStyle(previous_style)
    app.setPalette(previous_palette)


def _make(cls):
    widget = cls()
    if isinstance(widget, (QPushButton, QCheckBox)):
        widget.setText("Run Pulse")
    if isinstance(widget, QComboBox):
        widget.addItems(["laser1", "burst_10hz_5s"])
    if isinstance(widget, QLineEdit):
        widget.setText("PXI1Slot5/ai3")
    return widget


CONTROLS = [QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, QCheckBox]


@pytest.mark.parametrize("cls", CONTROLS, ids=lambda c: c.__name__)
def test_disabling_a_control_does_not_change_its_size(style, cls):
    widget = _make(cls)
    widget.setStyle(style)
    enabled = (widget.sizeHint(), widget.minimumSizeHint())
    widget.setEnabled(False)
    assert (widget.sizeHint(), widget.minimumSizeHint()) == enabled


@pytest.mark.parametrize("cls", CONTROLS, ids=lambda c: c.__name__)
def test_a_disabled_control_draws_the_dashed_outline_and_an_enabled_one_does_not(style, cls):
    widget = _make(cls)
    widget.setStyle(style)
    widget.resize(widget.sizeHint())

    def outline_pixels():
        image = widget.grab().toImage()
        border = DISABLED_BORDER.rgb()
        return sum(
            1
            for y in range(image.height())
            for x in range(image.width())
            if image.pixel(x, y) == border
        )

    # Fusion's own frame or arrow can share a pixel or two with the outline's
    # grey, so compare: the dashed outline adds dozens around the frame.
    enabled = outline_pixels()
    widget.setEnabled(False)
    assert outline_pixels() >= enabled + 20


def test_the_disabled_palette_mutes_text_and_flattens_the_fill():
    palette = disabled_palette(QPalette())
    disabled = QPalette.ColorGroup.Disabled
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        assert palette.color(disabled, role) == DISABLED_TEXT
    for role in (QPalette.ColorRole.Button, QPalette.ColorRole.Base):
        assert palette.color(disabled, role) == DISABLED_FILL
    # The active group is left as it was.
    assert palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.Text) == QPalette().color(
        QPalette.ColorGroup.Active, QPalette.ColorRole.Text)
