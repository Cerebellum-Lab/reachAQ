"""The protocol check where the operator meets it: Record and the Protocol tab.

Errors are Record blockers, one line each in its tooltip; warnings leave
Record available. Every test runs on the app fixture: christielab10's laser
block on a null controller standing in for the NI-DAQ one, and the NI-DAQ
stream's flags set as a running stream sets them. No hardware is touched.
"""

import dataclasses
import io
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import SystemConfiguration  # noqa: E402
from autotrainer.device.laser import NullLaserController  # noqa: E402

from nidaq_timing_test import _pinned_christielab10  # noqa: E402
from tools.acquisition.model.nidaq_channel_plan import (  # noqa: E402
    build_nidaq_acquisition_configuration,
)

HERE = Path(__file__).parent
BENCH_PROTOCOL = HERE / "bench-laser-validation-2026-10-01.json"
CHRISTIELAB10 = SystemConfiguration.load_yaml(io.StringIO(
    f"!SystemConfiguration\nversion: {SystemConfiguration.version}\n"
    + (HERE / "christielab10_nidaq_blocks.yaml").read_text(encoding="utf-8")
    + (HERE / "christielab10_laser_block.yaml").read_text(encoding="utf-8")))


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _without_laser_2s_stim_line(laser):
    return dataclasses.replace(laser, channels=tuple(
        dataclasses.replace(channel, board_stim_line=None)
        if int(channel.channel_id) == 2 else channel
        for channel in laser.channels))


@pytest.fixture
def rig(app_model):
    """System Mode running on christielab10, as far as the check can see.

    The bench protocol is imported last, so the check that runs on its
    selection sees everything else already in place.
    """
    monitor = app_model.nidaq_signal_monitor

    def make(laser=CHRISTIELAB10.laser):
        app_model.save_laser_profile(
            profile_id="laser1", amplitude_volts=2.0, pulse_duration_ms=23.0,
            pulse_count=31, frequency_hz=29.0)
        app_model.laser.set_controller(NullLaserController(laser))
        monitor._configuration = build_nidaq_acquisition_configuration(
            CHRISTIELAB10.nidaq_stream, CHRISTIELAB10.nidaq_ports, laser)
        monitor._hardware_enabled = True
        monitor._set_timing_plan(_pinned_christielab10())
        monitor._set_running(True)
        app_model._acquisition.started = True
        app_model.import_ordered_protocol(BENCH_PROTOCOL)
        return app_model

    try:
        yield make
    finally:
        # Stopped first: a laser that closes while System Mode runs is a
        # failure, and this one is only the test ending.
        app_model._acquisition.started = False
        monitor._set_running(False)
        app_model.laser.close()


class _Event:
    def __iadd__(self, _callback):
        return self


def _behavior_content(app_model):
    """Behavior's card on the real app model.

    The fixture's inference stand-in has no model location, which the card
    reads as it is made; a stand-in with one takes its place. Record's state
    comes from the app model alone.
    """
    from tools.acquisition.view.behavior_content import BehaviorContent

    inference = SimpleNamespace(
        IS_ENABLED="is_enabled", STATUS="status", MODEL_LOCATION="model_location",
        status="disabled", model_location="", property_changed=_Event())
    return BehaviorContent(app_model, app_model.behavior, inference)


def _protocol_lines(app_model):
    return [line for line in app_model.recording_blockers if line.startswith("Protocol check:")]


def test_recording_blockers_list_the_protocols_errors(rig):
    # christielab10 without laser 2's boardStimLine: trials 6-8 cannot fire.
    app_model = rig(_without_laser_2s_stim_line(CHRISTIELAB10.laser))

    assert _protocol_lines(app_model) == [
        "Protocol check: trials 6-8: Laser 2 has no board STIM line; set "
        "boardStimLine for it in the system configuration",
    ]


def test_choosing_no_protocol_clears_its_lines(rig):
    app_model = rig(_without_laser_2s_stim_line(CHRISTIELAB10.laser))
    assert _protocol_lines(app_model)

    app_model.select_ordered_protocol(None)

    assert _protocol_lines(app_model) == []


def test_records_tooltip_shows_the_protocols_errors(rig, qapp):
    app_model = rig(_without_laser_2s_stim_line(CHRISTIELAB10.laser))
    content = _behavior_content(app_model)
    try:
        qapp.processEvents()
        assert not content._record_button.isEnabled()
        assert "Protocol check: trials 6-8: Laser 2 has no board STIM line" in (
            content._record_button.toolTip())
    finally:
        content.deleteLater()


def test_a_protocol_with_only_warnings_leaves_record_available(rig, qapp):
    # The 2026-10-01 bench protocol on christielab10 as it is: its trigger
    # readbacks are not configured, and no laser tab has a profile picked.
    app_model = rig()
    content = _behavior_content(app_model)
    try:
        qapp.processEvents()
        assert app_model.recording_blockers == ()
        assert content._record_button.isEnabled()
        warnings = [item for item in app_model.protocol_readiness
                    if item.severity == "warning"]
        assert warnings
        assert app_model.protocol_check_notice == (
            f"Protocol check: {len(warnings)} warnings (see Check protocol)")
    finally:
        content.deleteLater()


def test_a_laser_tab_pick_is_what_the_check_reads(rig):
    app_model = rig()

    def tab_warnings():
        return [item.message for item in app_model.protocol_readiness
                if "Laser Control tab" in item.message]

    assert len(tab_warnings()) == 2
    app_model.note_laser_tab_profile(2, "laser1")
    assert len(tab_warnings()) == 1 and tab_warnings()[0].startswith("laser 1 ")


def test_check_protocol_lists_every_finding_and_copies_them(rig, qapp, monkeypatch):
    from tools.acquisition.view import protocol_check_dialog
    from tools.acquisition.view.protocol_content import ProtocolContent

    app_model = rig(_without_laser_2s_stim_line(CHRISTIELAB10.laser))
    shown = []
    monkeypatch.setattr(
        protocol_check_dialog.ProtocolCheckDialog, "exec", lambda self: shown.append(self))
    content = ProtocolContent(app_model)
    try:
        content._check_protocol_button.click()

        dialog, = shown
        table = dialog._table
        rows = [tuple(table.item(row, column).text() for column in range(3))
                for row in range(table.rowCount())]
        # Errors first, each on its own trial, then the warnings.
        assert rows[:3] == [
            ("Error", f"Trial {trial}", "Laser 2 has no board STIM line; set "
             "boardStimLine for it in the system configuration")
            for trial in (6, 7, 8)
        ]
        assert {row[0] for row in rows[3:]} == {"Warning"}

        dialog._copy_button.click()
        copied = QApplication.clipboard().text()
        assert "Error\tTrial 6\tLaser 2 has no board STIM line" in copied
        assert copied.count("\n") == len(rows) - 1
    finally:
        content.deleteLater()
