"""Laser Control fits the docked right-hand panel without scrolling.

On christielab10's 1920x1080 screen the saved main splitter gives the right
panel 440 px. At the application font the laser Pulse page needed a scroll
bar there and its board trigger graph was squeezed (2026-09-24). With every
section open, the whole panel - its card header, tab bar and footer included -
must fit 440 x 860 px.

The rig's desktop session gives Qt Ubuntu 11 pt, while the offscreen platform
measures with DejaVu Sans 9 pt, so every test here uses Ubuntu 11 pt when it
is installed. The checks compare against those targets rather than exact
pixel counts, so a slightly different font on another machine does not fail
them.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QHelpEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QInputDialog,
    QLabel,
    QMessageBox,
    QScrollArea,
    QToolTip,
    QWidget,
)

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
)
from tools.acquisition.model.trial_action import LaserPulseProfile  # noqa: E402
from tools.acquisition.view.compact_panel import (  # noqa: E402
    COMPACT_FONT_POINT_SIZE,
    CollapsibleSection,
    ElidedLabel,
)
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    LaserControlContent,
)

PANEL_WIDTH = 440
PANEL_HEIGHT = 860
#: What each graph must keep on the docked panel with every section open.
MINIMUM_OUTPUT_STREAM_HEIGHT = 140
MINIMUM_BOARD_TRIGGER_HEIGHT = 60
MINIMUM_PREVIEW_HEIGHT = 120


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


#: christielab10's desktop font, as its xcb session gives it to Qt.
RIG_FONT_FAMILY = "Ubuntu"
RIG_FONT_POINT_SIZE = 11


@pytest.fixture(autouse=True)
def rig_font(qapp):
    """Measure with the rig's font when it is installed; restored afterwards."""
    previous = QFont(QApplication.font())
    if QFontDatabase.hasFamily(RIG_FONT_FAMILY):
        QApplication.setFont(QFont(RIG_FONT_FAMILY, RIG_FONT_POINT_SIZE))
    try:
        yield
    finally:
        QApplication.setFont(previous)


def _christielab10_lasers():
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="PXI1Slot4/ao0",
                diode_input="PXI1Slot5/ai0",
                shutter_output="PXI1Slot4/port0/line0",
                command_copy_input="PXI1Slot5/ai2",
                trigger_source="/PXI1Slot4/PXI_Trig0",
                board_stim_line=3,
            ),
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_2,
                analog_output="PXI1Slot4/ao1",
                diode_input="PXI1Slot5/ai1",
                shutter_output="PXI1Slot4/port0/line1",
                trigger_source="/PXI1Slot4/PXI_Trig2",
                board_stim_line=2,
            ),
        ),
        backend="null",
        sample_rate_hz=10000.0,
    )


def _settle(qapp):
    for _ in range(5):
        qapp.processEvents()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)


SAVED_PROFILES = (
    LaserPulseProfile("stim-a", 3, 2.5, 5.0, pulse_count=20, frequency_hz=20.0,
                      baseline_ms=500.0, post_stim_ms=500.0),
    LaserPulseProfile("burst-40hz-2s-long-name", 1, 1.25, 3.0, pulse_count=80,
                      frequency_hz=40.0, baseline_ms=1000.0, post_stim_ms=1000.0),
)


@pytest.fixture
def panel(qapp, app_model, monkeypatch):
    # trial_protocol_state is a read-only property, so patch it on the class.
    monkeypatch.setattr(type(app_model), "trial_protocol_state", property(
        lambda _self: {"laser_profiles": tuple(
            {"profile_id": profile.profile_id, "revision": profile.revision,
             "summary": profile.summary()}
            for profile in SAVED_PROFILES)}))
    monkeypatch.setattr(type(app_model), "laser_profile", lambda _self, profile_id: next(
        (profile for profile in SAVED_PROFILES if profile.profile_id == profile_id), None))
    # Connected, as while the system runs, so every control is enabled.
    app_model.laser.configure_null(_christielab10_lasers())
    content = LaserControlContent(app_model)
    content.resize(PANEL_WIDTH, PANEL_HEIGHT)
    content.show()
    _settle(qapp)
    try:
        yield content
    finally:
        content.on_close()
        content.close()
        content.deleteLater()
        app_model.laser.close()


def _open_every_section(panel, qapp):
    # Whatever holds a folding section, none may stay closed: the worst case.
    for widget in panel.findChildren(QWidget):
        if callable(getattr(widget, "set_expanded", None)):
            widget.set_expanded(True)
    _settle(qapp)


def _show_pulse_page(panel, tab, qapp) -> QScrollArea:
    panel._tabs.setCurrentWidget(tab)
    tab._mode_tabs.setCurrentIndex(0)
    _settle(qapp)
    scroll = tab._mode_tabs.currentWidget()
    assert isinstance(scroll, QScrollArea)
    return scroll


def test_every_laser_pulse_page_fits_the_docked_panel_with_every_section_open(panel, qapp):
    _open_every_section(panel, qapp)

    assert (panel.width(), panel.height()) == (PANEL_WIDTH, PANEL_HEIGHT)
    for tab in panel._channel_tabs:
        tab.stim_profile_selector.setCurrentIndex(
            tab.stim_profile_selector.findData("burst-40hz-2s-long-name"))
        # A realistic outcome, two lines at this width.
        tab.stim_test_result.setText(
            "Stim test stim-a on laser 1: STIM3 pulse 1000 us started the "
            "waveform on /PXI1Slot4/PXI_Trig0")
        scroll = _show_pulse_page(panel, tab, qapp)
        page = scroll.widget()

        assert scroll.verticalScrollBar().maximum() == 0, f"laser {tab.channel_id_value}"
        assert page.minimumSizeHint().height() <= scroll.viewport().height()
        assert page.minimumSizeHint().width() <= scroll.viewport().width()
        assert tab._trace_plot.height() >= MINIMUM_OUTPUT_STREAM_HEIGHT
        assert tab.trigger_plot.height() >= MINIMUM_BOARD_TRIGGER_HEIGHT
        assert tab.trigger_status.isVisible()
        # Its second line holds what to do about it: wire the board line into
        # an NI input and set it in Edit DAQ Ports. One line cut that off.
        assert not tab.trigger_status.is_elided(), tab.trigger_status.text()
        assert all(checkbox.isVisible() for checkbox in tab._trace_signal_checkboxes.values())
        assert tab.stim_test_button.isVisible()


def test_the_calibration_page_fits_the_docked_panel(panel, qapp):
    _open_every_section(panel, qapp)
    tab = panel._channel_tabs[0]
    panel._tabs.setCurrentWidget(tab)
    tab._mode_tabs.setCurrentIndex(1)
    _settle(qapp)
    page = tab._mode_tabs.currentWidget()

    assert page.minimumSizeHint().height() <= page.height()
    assert page.minimumSizeHint().width() <= page.width()
    assert tab._run_ramp_button.isVisible()


def test_the_pulse_builder_fits_the_docked_panel_with_every_section_open(panel, qapp):
    _open_every_section(panel, qapp)
    builder = panel._builder
    selector = builder._profile_selector
    selector.setCurrentIndex(selector.findData("burst-40hz-2s-long-name"))
    panel._tabs.setCurrentWidget(builder)
    _settle(qapp)

    assert builder.minimumSizeHint().height() <= builder.height()
    # Its profile picker once sized itself to the longest saved profile and
    # made the builder wider than the panel.
    assert builder.minimumSizeHint().width() <= builder.width()
    assert builder._preview_plot.height() >= MINIMUM_PREVIEW_HEIGHT
    assert builder._preview_status.isVisible()
    assert builder._preview_status.geometry().bottom() < builder.height()
    # The picker shows only the start of "name — summary" at this width;
    # hovering it shows the whole entry.
    assert selector.itemText(selector.currentIndex()) in selector.toolTip()


def test_a_dialog_from_the_builder_keeps_the_application_font(panel, qapp):
    # The panel's smaller font once reached the Save profile and Delete
    # dialogs too, as they are parented to the builder.
    builder = panel._builder
    application_size = QFontInfo(QApplication.font()).pointSizeF()
    message_box = QMessageBox(
        QMessageBox.Icon.Question, "Delete laser profile", "Delete it?", parent=builder)
    name_dialog = QInputDialog(builder)
    name_dialog.setLabelText("Profile name:")
    try:
        for dialog in (message_box, name_dialog):
            dialog.ensurePolished()
            assert QFontInfo(dialog.font()).pointSizeF() == pytest.approx(
                application_size, abs=0.3)
            labels = dialog.findChildren(QLabel)
            assert labels
            for label in labels:
                assert QFontInfo(label.font()).pointSizeF() == pytest.approx(
                    application_size, abs=0.3)
    finally:
        message_box.deleteLater()
        name_dialog.deleteLater()

    assert QFontInfo(builder.findChild(QLabel).font()).pointSizeF() == pytest.approx(
        COMPACT_FONT_POINT_SIZE, abs=0.3)


def test_a_folded_section_stays_folded_on_every_laser_across_a_rebuild(panel, qapp):
    # Every system Run/Stop rebuilds the laser tabs.
    first, second = panel._channel_tabs[:2]
    assert not first.sections["signals"].is_expanded
    assert first.sections["board_trigger"].is_expanded

    first.sections["board_trigger"].set_expanded(False)
    first.sections["signals"].set_expanded(True)
    assert not second.sections["board_trigger"].is_expanded
    assert second.sections["signals"].is_expanded

    panel._refresh_from_model()

    for tab in panel._channel_tabs:
        assert tab is not first
        assert not tab.sections["board_trigger"].is_expanded
        assert tab.sections["signals"].is_expanded
        assert tab.sections["output_stream"].is_expanded


def test_a_folded_section_gives_its_height_to_the_output_stream(panel, qapp):
    tab = panel._channel_tabs[0]
    _show_pulse_page(panel, tab, qapp)
    before = tab._trace_plot.height()

    tab.sections["run_pulse"].set_expanded(False)
    _settle(qapp)

    assert not tab._run_pulse_button.isVisible()
    assert tab._trace_plot.height() > before


def test_a_cut_line_keeps_its_whole_text_in_the_tooltip(qapp):
    text ="No trigger readback input is configured for this laser"
    label = ElidedLabel(text)
    label.resize(60, label.sizeHint().height())

    assert label.text() == text
    assert label.is_elided()
    assert label.elided_text().endswith("…")
    assert label.full_tooltip() == text
    label.setToolTip("detail")
    assert label.full_tooltip() == f"{text}\ndetail"

    label.resize(label.sizeHint().width() + 10, label.sizeHint().height())
    assert not label.is_elided()
    assert label.full_tooltip() == "detail"


def test_hovering_a_cut_line_shows_its_whole_text(qapp):
    text = "No trigger readback input is configured for this laser"
    label = ElidedLabel(text)
    label.setToolTip("detail")
    label.resize(60, label.sizeHint().height())
    label.show()
    _settle(qapp)
    point = QPoint(5, 5)
    try:
        hover = QHelpEvent(QEvent.Type.ToolTip, point, label.mapToGlobal(point))
        assert QApplication.sendEvent(label, hover)
        assert QToolTip.text() == f"{text}\ndetail"

        # Nothing cut and no tooltip: nothing to show, so the event goes on.
        label.setToolTip("")
        label.resize(label.sizeHint().width() + 10, label.sizeHint().height())
        quiet = QHelpEvent(QEvent.Type.ToolTip, point, label.mapToGlobal(point))
        QApplication.sendEvent(label, quiet)
        assert not quiet.isAccepted()
    finally:
        QToolTip.hideText()
        label.close()
        label.deleteLater()


def test_a_two_line_label_wraps_and_cuts_only_after_its_second_line(qapp):
    text = (
        "No trigger readback input is configured. Wire the board stimulus "
        "line into an NI input and set it in Edit DAQ Ports."
    )
    label = ElidedLabel(text, max_lines=2)
    one_line = ElidedLabel(text)
    # The same height whatever the text, so the page does not move when it changes.
    assert label.sizeHint().height() == (
        one_line.sizeHint().height() + label.fontMetrics().lineSpacing())
    full_width = label.fontMetrics().horizontalAdvance(text)

    label.resize(full_width * 2 // 3, label.sizeHint().height())
    lines = label.elided_text().split("\n")
    assert not label.is_elided()
    assert len(lines) == 2
    assert " ".join(lines) == text
    assert label.full_tooltip() == ""

    label.resize(full_width // 3, label.sizeHint().height())
    lines = label.elided_text().split("\n")
    assert label.is_elided()
    assert len(lines) == 2
    assert lines[1].endswith("…")
    assert label.full_tooltip() == text


def test_a_section_tooltip_goes_on_its_header(qapp):
    section = CollapsibleSection("Signals", "signals")
    section.set_header_tooltip("Choose the signals")
    assert section.toggle_button.toolTip() == "Choose the signals"

    # Qt's own setToolTip is left alone, so toolTip() reads back what it set.
    section.setToolTip("The whole section")
    assert section.toolTip() == "The whole section"
