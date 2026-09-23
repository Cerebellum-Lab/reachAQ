"""The two wiring phases that do not hold a board output high.

Observing: cam_frames and barcode are sourced by a running camera and a
session, so the driven phase can only ever report them untested. It cannot
drive them; it can watch while an operator does.

Tones: holding STIM0 high with a digital write proves the cable and nothing
more. Asking the board for a tone and seeing which line follows proves the
mechanism the rig depends on, and it is gated on the frequency matching the
one the devicetree assigns to that pin.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.acquisition.model.nidaq_wiring_verification import WiringPoint

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hardware" / "verify_nidaq_wiring.py"
_spec = importlib.util.spec_from_file_location("verify_nidaq_wiring", _TOOL)
verify = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("verify_nidaq_wiring", verify)
_spec.loader.exec_module(verify)


class _Task:
    """A port that reads a scripted sequence of words."""

    def __init__(self, words):
        self._words = list(words)
        self.closed = False

    def read(self):
        return self._words[0] if len(self._words) == 1 else self._words.pop(0)

    def close(self):
        self.closed = True


class _Nidaqmx:
    def __init__(self, ports):
        self._ports = ports
        self.opened = []

    def Task(self, name):  # noqa: N802 - mirrors the nidaqmx name
        self.opened.append(name)
        return _Builder(self, name)


class _Builder:
    """What nidaqmx hands back: a task you add a channel to."""

    def __init__(self, parent, name):
        self._parent = parent
        self._name = name
        self._task = None
        self.di_channels = SimpleNamespace(add_di_chan=self._add)

    def _add(self, channel):
        words = self._parent._ports.get(channel)
        if words is None:
            raise RuntimeError(f"{channel} does not exist")
        self._task = _Task(words)

    def read(self):
        return self._task.read()

    def close(self):
        if self._task is not None:
            self._task.close()


def _point(name, physical_channel):
    return WiringPoint(name=name, physical_channel=physical_channel,
                       role="digital stream input")


def _observe(ports, points, undriven, seconds=0.01):
    nidaqmx = _Nidaqmx(ports)
    confirmed = {}
    verify.observe_undriven(
        nidaqmx, points, seconds, undriven,
        lambda point, detail: confirmed.__setitem__(point.name, detail))
    return confirmed


def test_a_line_that_goes_high_while_watched_is_confirmed():
    points = [_point("cam_frames", "PXI1Slot5/port0/line2")]
    undriven = {points[0].fingerprint}
    # The tool probes the port when it opens it and reads a baseline before
    # polling, so the edge has to arrive after two low reads.
    ports = {"PXI1Slot5/port0": [0, 0, 0b0100], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}

    confirmed = _observe(ports, points, undriven, seconds=0.05)

    assert "cam_frames" in confirmed
    assert "nothing in this run drove it" in confirmed["cam_frames"]


def test_a_line_that_stays_put_is_left_untested():
    """Silence here is not evidence: nothing was driving it either."""
    points = [_point("cam_frames", "PXI1Slot5/port0/line2")]
    ports = {"PXI1Slot5/port0": [0], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}

    assert _observe(ports, points, {points[0].fingerprint}) == {}


def test_only_the_undriven_points_are_watched():
    """A point the run drove has a real answer already; do not overwrite it."""
    watched = _point("cam_frames", "PXI1Slot5/port0/line2")
    driven = _point("tone1", "PXI1Slot5/port0/line0")
    ports = {"PXI1Slot5/port0": [0, 0, 0b0101], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}

    confirmed = _observe(ports, [watched, driven], {watched.fingerprint},
                         seconds=0.05)

    assert set(confirmed) == {"cam_frames"}


def test_nothing_to_watch_is_not_an_error():
    points = [_point("tone1", "PXI1Slot5/port0/line0")]

    assert _observe({"PXI1Slot5/port0": [0]}, points, set()) == {}


def test_a_port_that_will_not_open_is_skipped_rather_than_fatal():
    """A board without this port simply has nothing to watch."""
    points = [_point("cam_frames", "PXI1Slot5/port0/line2")]
    ports = {"PXI1Slot5/port0": [0, 0, 0b0100]}  # port1 and port2 absent

    confirmed = _observe(ports, points, {points[0].fingerprint}, seconds=0.05)

    assert "cam_frames" in confirmed


def test_no_port_at_all_reports_rather_than_raising(capsys):
    points = [_point("cam_frames", "PXI1Slot9/port0/line2")]

    assert _observe({}, points, {points[0].fingerprint}) == {}
    assert "no digital port" in capsys.readouterr().out


# ------------------------------------------------- the tone confirmations


class _Interface:
    """A board that accepts tones and records which were asked for."""

    def __init__(self, accepts=True):
        self.tones = []
        self._accepts = accepts

    def emit_tone(self, frequency_hz, duration_ms):
        self.tones.append((frequency_hz, duration_ms))
        return self._accepts


def _configuration(tone1="PXI1Slot5/port0/line0", tone2="PXI1Slot5/port0/line1"):
    return SimpleNamespace(nidaq_ports=SimpleNamespace(tone1=tone1, tone2=tone2))


def _check_tones(ports, points, configuration=None, accepts=True):
    nidaqmx = _Nidaqmx(ports)
    interface = _Interface(accepts=accepts)
    confirmed = {}
    verify.check_tones(
        nidaqmx, interface, configuration or _configuration(), points,
        lambda point, detail: confirmed.__setitem__(point.name, detail))
    return confirmed, interface


def test_each_mapped_frequency_confirms_its_own_line():
    """5 kHz raises tone1 and 6 kHz raises tone2; the board decides which."""
    points = [_point("tone1", "PXI1Slot5/port0/line0"),
              _point("tone2", "PXI1Slot5/port0/line1")]
    # Probe, baseline, then line0 for the first tone and line1 for the second.
    ports = {"PXI1Slot5/port0": [0, 0, 0b0001, 0b0000, 0b0010],
             "PXI1Slot5/port1": [0], "PXI1Slot5/port2": [0]}

    confirmed, interface = _check_tones(ports, points)

    assert [frequency for frequency, _ms in interface.tones] == [5000, 6000]
    assert "5000 Hz tone" in confirmed["tone1"]


def test_a_tone_the_board_refuses_confirms_nothing():
    points = [_point("tone1", "PXI1Slot5/port0/line0")]
    ports = {"PXI1Slot5/port0": [0, 0, 0b0001], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}

    confirmed, _interface = _check_tones(ports, points, accepts=False)

    assert confirmed == {}


def test_a_tone_that_raises_nothing_is_not_confirmed():
    """Silence here is real evidence: the tone was played and asked for."""
    points = [_point("tone1", "PXI1Slot5/port0/line0")]
    ports = {"PXI1Slot5/port0": [0], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}

    confirmed, interface = _check_tones(ports, points)

    assert confirmed == {} and interface.tones


def test_a_line_the_configuration_does_not_name_is_not_played_for():
    points = [_point("tone1", "PXI1Slot5/port0/line0")]
    ports = {"PXI1Slot5/port0": [0, 0, 0b0001], "PXI1Slot5/port1": [0],
             "PXI1Slot5/port2": [0]}

    confirmed, interface = _check_tones(
        ports, points, configuration=_configuration(tone2=None))

    assert [frequency for frequency, _ms in interface.tones] == [5000]
    assert set(confirmed) == {"tone1"}


def test_no_tone_lines_configured_is_not_an_error(capsys):
    confirmed, interface = _check_tones(
        {}, [], configuration=_configuration(tone1=None, tone2=None))

    assert confirmed == {} and interface.tones == []
    assert "no tone confirmation lines" in capsys.readouterr().out
