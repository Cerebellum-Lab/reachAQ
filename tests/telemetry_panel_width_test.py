"""The panel must not resize as its numbers change.

Elapsed passing a minute, a percentage reaching three figures, a latency
gaining a digit - each changed a label's width, which changed the panel's, and
the strip twitched several times a second while anyone was watching it.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class _Event:
    """Stands in for the observable the panel subscribes to."""

    def __iadd__(self, handler):
        return self

    def __isub__(self, handler):
        return self


class _Telemetry:
    """Just enough of SessionTelemetry for the panel to render."""

    def __init__(self, **values):
        self.property_changed = _Event()
        self.__dict__.update(dict(
            elapsed_seconds=0.0, frames_acquired=0, frames_inferenced=0,
            inferenced_percent=0.0, dropped_frames=0, has_dropped_frames=False,
            is_active=True, inference_mean_ms=float("nan"),
            inference_max_ms=float("nan"),
            sensor_to_result_mean_ms=float("nan"),
            sensor_to_result_max_ms=float("nan"),
            stim_p99_ms=float("nan"),
            latency_status="",
        ), **values)


def _panel(qapp, telemetry):
    from tools.acquisition.view.capture_telemetry_panel import (
        CaptureTelemetryPanel)
    panel = CaptureTelemetryPanel(telemetry)
    panel.set_expanded(True)
    return panel


def _show(panel, telemetry):
    """Point the panel at these readings and repaint it."""
    panel._telemetry = telemetry
    panel._refresh()


# Readings that would each have been wider than the one before.
GROWING = [
    _Telemetry(elapsed_seconds=9.0, frames_acquired=8, frames_inferenced=8,
               inferenced_percent=8.0, inference_mean_ms=4.2,
               inference_max_ms=5.0, sensor_to_result_mean_ms=6.1,
               sensor_to_result_max_ms=6.9, stim_p99_ms=3.4),
    _Telemetry(elapsed_seconds=90.0, frames_acquired=13500,
               frames_inferenced=13500, inferenced_percent=100.0,
               inference_mean_ms=44.2, inference_max_ms=55.0,
               sensor_to_result_mean_ms=66.1, sensor_to_result_max_ms=99.9,
               stim_p99_ms=44.4, latency_status="partial (software)"),
    _Telemetry(elapsed_seconds=7200.0, frames_acquired=1080000,
               frames_inferenced=1079000, inferenced_percent=99.9,
               dropped_frames=1234, has_dropped_frames=True,
               inference_mean_ms=444.2, inference_max_ms=555.5,
               sensor_to_result_mean_ms=666.1, sensor_to_result_max_ms=999.9,
               stim_p99_ms=1234.5, latency_status="complete (hardware)"),
]


def _effective_widths(panel):
    """What each readout will actually occupy.

    A label is laid out at least its minimum width and otherwise at its text
    width, so this is the quantity that moves the panel when it changes.
    """
    labels = (panel._elapsed, panel._dropped, panel._inferenced,
              panel._inference_ms, panel._e2e_ms, panel._stim_p99,
              panel._latency, panel._collapsed_summary)
    return tuple(max(label.sizeHint().width(), label.minimumWidth())
                 for label in labels)


def test_no_readout_changes_width_as_the_numbers_grow(qapp):
    panel = _panel(qapp, GROWING[0])
    seen = set()
    for telemetry in GROWING:
        _show(panel, telemetry)
        seen.add(_effective_widths(panel))

    assert len(seen) == 1, f"readout widths moved across readings: {seen}"


def test_the_reservation_covers_the_widest_reading(qapp):
    """A reading wider than the reservation would still move the panel."""
    panel = _panel(qapp, GROWING[-1])
    _show(panel, GROWING[-1])

    for label in (panel._elapsed, panel._dropped, panel._inferenced,
                  panel._inference_ms, panel._e2e_ms, panel._stim_p99,
                  panel._latency, panel._collapsed_summary):
        assert label.sizeHint().width() <= label.minimumWidth(), (
            f"{label.text()!r} is wider than its reserved width")


def test_a_readout_still_shows_its_value(qapp):
    """Reserving width must not stop the number being displayed."""
    panel = _panel(qapp, GROWING[1])
    _show(panel, GROWING[1])

    assert "100%" in panel._inferenced.text()
    assert "13500" in panel._inferenced.text()


def test_the_stim_p99_readout_says_what_it_spans(qapp):
    """The figure starts at host arrival, so it must not be read as sensor-to-output."""
    panel = _panel(qapp, GROWING[0])

    tip = panel._stim_p99.parentWidget().toolTip()
    assert "host arrival" in tip
    assert "DAQmx start return" in tip
    assert "excludes exposure" in tip


# The five readouts that were the whole expanded row at bbf065d8, before the
# latency record added two (caption, widest reading). Kept here, not read off
# the panel, so the baseline cannot drift along with what is being checked.
_PRE_LATENCY_RECORD_READOUTS = (
    ("Elapsed", "0:00:00"),
    ("Dropped frames", "0000000"),
    ("Inferenced", "000%  0000000/0000000"),
    ("Inference call", "000.0 ms  max 0000.0"),
    ("Sensor → result", "000.0 ms  max 0000.0"),
)


def _pre_latency_record_expanded_width(qapp):
    """The expanded body's width before the latency record: one row, these five."""
    from PySide6.QtWidgets import QFrame, QHBoxLayout
    from tools.acquisition.view.capture_telemetry_panel import (
        CaptureTelemetryPanel)
    body = QFrame()
    row = QHBoxLayout(body)
    row.setContentsMargins(10, 4, 10, 6)
    row.setSpacing(18)
    for caption, widest in _PRE_LATENCY_RECORD_READOUTS:
        CaptureTelemetryPanel._add_readout(row, caption, widest)
    row.addStretch(1)
    return body.sizeHint().width()


def test_the_expanded_panel_is_no_wider_than_before_the_latency_record(qapp):
    """Two expanded panels share the camera row and neither can shrink below
    its minimum, so a wider panel is paid for twice, in the window's minimum
    width. The stim and record-status readouts therefore sit on a second line.
    """
    panel = _panel(qapp, GROWING[-1])
    _show(panel, GROWING[-1])

    before = _pre_latency_record_expanded_width(qapp)
    assert panel.sizeHint().width() <= before, (
        f"expanded panel is {panel.sizeHint().width()} px wide, "
        f"it was {before} px before the latency record")
    assert panel.minimumSizeHint().width() <= before
