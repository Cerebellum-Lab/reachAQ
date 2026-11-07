"""Streaming the board's stimulus line back as a trace beside the laser output."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
)
from tools.acquisition.model.laser_plot_process import (  # noqa: E402
    LaserPlotProcess,
    STREAM_CURVE_NAMES,
)
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    _LaserChannelTab,
)


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_channel(**overrides):
    values = dict(
        channel_id=LaserChannelId.LASER_1,
        analog_output="/Dev1/ao0",
        diode_input="/Dev1/ai0",
        shutter_output="/Dev1/port0/line0",
    )
    values.update(overrides)
    return LaserChannelConfiguration(**values)


def make_tab(app_model, channel):
    return _LaserChannelTab(
        app_model,
        channel,
        True,
        None,
        lambda status, operation: None,
        lambda message, is_error: None,
    )


def test_a_channel_has_no_trigger_monitor_by_default():
    assert make_channel().trigger_monitor_input is None


def test_a_channel_can_name_a_trigger_monitor_input():
    channel = make_channel(trigger_monitor_input="/Dev1/ai7")

    assert channel.trigger_monitor_input == "/Dev1/ai7"


def test_the_trigger_curve_is_one_of_the_streamed_curves():
    assert "trigger" in STREAM_CURVE_NAMES
    # command comes from the pulse trace, not from the NI-DAQ ring.
    assert "command" not in STREAM_CURVE_NAMES


def test_every_channel_gets_its_own_trigger_slot():
    slots = {
        LaserPlotProcess.curve_slot(channel_id, "trigger")
        for channel_id in (1, 2, 3, 4)
    }

    assert len(slots) == 4


def test_curve_slots_do_not_collide_across_curves_and_channels():
    slots = [
        LaserPlotProcess.curve_slot(channel_id, curve)
        for channel_id in (1, 2, 3, 4)
        for curve in ("command", "diode", "copy", "trigger")
    ]

    assert len(set(slots)) == len(slots)


def test_no_trigger_candidate_without_a_configured_input(qapp, app_model):
    tab = make_tab(app_model, make_channel())

    assert tab._make_trace_signal_candidate("trigger") is None


def test_an_analog_trigger_input_streams_as_analog(qapp, app_model):
    tab = make_tab(app_model, make_channel(trigger_monitor_input="/Dev1/ai7"))

    candidate = tab._make_trace_signal_candidate("trigger")

    assert candidate.physical_channel == "/Dev1/ai7"
    assert candidate.kind == "analog"
    assert candidate.name == "laser1_trigger"


def test_a_digital_trigger_input_streams_as_digital(qapp, app_model):
    tab = make_tab(
        app_model, make_channel(trigger_monitor_input="/Dev1/port0/line3")
    )

    candidate = tab._make_trace_signal_candidate("trigger")

    assert candidate.kind == "digital"


def test_the_tab_offers_a_trigger_plot_and_selection(qapp, app_model):
    tab = make_tab(app_model, make_channel(trigger_monitor_input="/Dev1/ai7"))

    assert tab.trigger_plot is not None
    assert "trigger" in tab._trace_curves
    assert "trigger" in tab._trace_signal_checkboxes
