"""The legend and the chooser are one window, and the legend must not lie.

The colours shown have to be the colours painted, which is why the dialog
imports them from the painter instead of restating them.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from autotrainer.pyside.capture.QtGLImageView import (  # noqa: E402
    is_subsumed_by_composite,
    overlay_colour_for,
)

PARTS = ("Pellet", "Nose", "Mouth", "Tongue_mid", "L_Hand", "R_Hand",
         "LH_flat", "RH_grab", "Star")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _dialog(qapp, shown=()):
    from tools.acquisition.view.overlay_parts_dialog import OverlayPartsDialog
    return OverlayPartsDialog(PARTS, shown)


def test_every_part_is_listed(qapp):
    dialog = _dialog(qapp)
    assert tuple(dialog._boxes) == PARTS


def test_no_configuration_shows_what_the_overlay_actually_draws(qapp):
    """Open with the live default: everything but the hand orientations."""
    dialog = _dialog(qapp)
    assert dialog.selected_parts() == tuple(
        p for p in PARTS if not is_subsumed_by_composite(p))
    assert "LH_flat" not in dialog.selected_parts()
    assert "L_Hand" in dialog.selected_parts()


def test_an_existing_configuration_is_reflected(qapp):
    dialog = _dialog(qapp, shown=("Pellet", "LH_flat"))
    assert dialog.selected_parts() == ("Pellet", "LH_flat")


def test_none_then_ok_means_draw_nothing(qapp):
    dialog = _dialog(qapp)
    dialog._set_all(False)
    assert dialog.selected_parts() == ()


def test_default_restores_the_composite_only_hands(qapp):
    dialog = _dialog(qapp, shown=("LH_flat",))
    dialog._restore_default()
    chosen = dialog.selected_parts()
    assert "LH_flat" not in chosen
    assert {"L_Hand", "R_Hand", "Pellet", "Nose"} <= set(chosen)


def test_the_legend_colour_is_the_painted_colour(qapp):
    """A legend maintained separately from the painter would drift."""
    from PySide6.QtGui import QColor
    for part in PARTS:
        assert QColor(overlay_colour_for(part)).isValid()
    # Distinct parts that share a colour would make the legend ambiguous for
    # the ones that matter most.
    important = ("Pellet", "Nose", "Mouth", "Star")
    colours = {QColor(overlay_colour_for(p)).name() for p in important}
    assert len(colours) == len(important)
