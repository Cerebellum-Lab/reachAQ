"""Laser Control's own footer shows the errors the panel reports.

They reached only the log and the main window's status bar, and with the
right-hand panel detached that status bar is in the other window (Ben,
2026-09-30). Each error now also shows on the footer, in the application's
error red, and stays there until the panel's next status or error replaces
it: the next operation's "Running ..." line and its outcome, or the Ready
line a Run/Stop rebuild ends on. There is no timeout. It is still logged
once, and that record is what puts it on the status bar.
"""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
)
from tools.acquisition.model.app_model import AppModel  # noqa: E402
from tools.acquisition.model.trial_action import LaserPulseProfile  # noqa: E402
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    _ERROR_STATUS_COLOR,
    LaserControlContent,
)

PROFILE = LaserPulseProfile("short", 1, 1.0, 2.0)
NO_PROFILE = "Laser 1: pick a saved profile or the builder draft"
RUNNING = "Running laser 1 pulse train"


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _lasers():
    # Laser 1 only; lasers 2 to 4 are the unmapped placeholders.
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line2",
            ),
        ),
        backend="null",
        sample_rate_hz=10000.0,
    )


@pytest.fixture
def listed(app_model, monkeypatch):
    """The saved laser profiles; a test empties it to delete them."""
    profiles = [PROFILE]
    # trial_protocol_state is a read-only property, so patch it on the class.
    monkeypatch.setattr(type(app_model), "trial_protocol_state", property(
        lambda _self: {"laser_profiles": tuple(
            {"profile_id": profile.profile_id, "revision": profile.revision,
             "summary": profile.summary()}
            for profile in profiles)}))
    monkeypatch.setattr(type(app_model), "laser_profile", lambda _self, profile_id: next(
        (profile for profile in profiles if profile.profile_id == profile_id), None))
    return profiles


@pytest.fixture
def panel(qapp, app_model, listed, caplog):
    caplog.set_level("ERROR")
    # Connected, on the null backend, so Run Pulse and Test stim are enabled.
    app_model.laser.configure_null(_lasers())
    content = LaserControlContent(app_model)
    qapp.processEvents()
    try:
        yield content
    finally:
        content.on_close()
        content.deleteLater()
        app_model.laser.close()


def _footer(panel):
    return panel._status_label


def _shows_error(panel, text) -> bool:
    footer = _footer(panel)
    return footer.text() == text and _ERROR_STATUS_COLOR in footer.styleSheet()


def _logged(caplog, text):
    return [record for record in caplog.records if text in record.getMessage()]


def _pick(tab, profile_id):
    tab.stim_profile_selector.setCurrentIndex(tab.stim_profile_selector.findData(profile_id))


def _wait_for_operation(panel, qapp):
    deadline = time.monotonic() + 10.0
    while panel._operation_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert panel._operation_thread is None


def test_a_refused_run_pulse_stays_on_the_footer_until_the_next_status(
    panel, app_model, qapp, caplog,
):
    tab = panel._channel_tabs[0]
    assert tab._run_pulse_button.isEnabled(), tab._run_pulse_button.toolTip()

    tab._run_pulse_button.click()

    assert _shows_error(panel, NO_PROFILE), (_footer(panel).text(), _footer(panel).styleSheet())
    # Logged once, which puts it on the status bar; the footer logs nothing more.
    assert len(_logged(caplog, NO_PROFILE)) == 1

    # The refreshes that follow the stream and System Mode report nothing,
    # so it stays.
    app_model.property_changed(AppModel.Props.SUBSYSTEM_STATUSES, None, None)
    qapp.processEvents()
    assert _shows_error(panel, NO_PROFILE)

    # The next operation's own lines replace it, in the ordinary colour.
    _pick(tab, "short")
    tab._run_pulse_button.click()
    assert _footer(panel).text() == RUNNING
    assert _ERROR_STATUS_COLOR not in _footer(panel).styleSheet()
    _wait_for_operation(panel, qapp)
    assert _footer(panel).text() == "Pulse complete: laser 1"
    assert _footer(panel).styleSheet() == ""
    assert len(_logged(caplog, NO_PROFILE)) == 1


def test_the_running_line_stays_until_the_operation_ends(panel, app_model, qapp, monkeypatch):
    # Nothing can be fired while an operation runs, and the panel said why in
    # place of "Running ...": the first refusal it found, an unmapped
    # laser's ("Laser 3 has no hardware channel ..." on christielab10), for
    # the whole operation.
    release = threading.Event()
    run_pulse_train = app_model.laser.run_pulse_train
    monkeypatch.setattr(
        app_model.laser, "run_pulse_train",
        lambda pulse_train: release.wait(10.0) and run_pulse_train(pulse_train))
    tab = panel._channel_tabs[0]
    _pick(tab, "short")
    try:
        tab._run_pulse_button.click()
        assert _footer(panel).text() == RUNNING
        # What the model announces as System Mode or the subsystems change
        # meanwhile; the first of these announces refusals, the second not.
        panel.set_is_capture_active(False)
        app_model.property_changed(AppModel.Props.SUBSYSTEM_STATUSES, None, None)
        qapp.processEvents()
        assert panel._operation_thread is not None
        assert _footer(panel).text() == RUNNING
    finally:
        release.set()
    _wait_for_operation(panel, qapp)

    assert _footer(panel).text() == "Pulse complete: laser 1"


def _test_stim_with_no_profile(panel, _listed):
    panel._channel_tabs[0].stim_test_button.click()
    return NO_PROFILE


def _run_ramp_on_an_unmapped_laser(panel, _listed):
    # Its button is greyed out; a press that gets through is refused.
    panel._channel_tabs[2]._run_calibration_ramp()
    return "Laser 3 has no hardware channel mapping"


def _deleting_a_picked_profile(panel, listed):
    _pick(panel._channel_tabs[0], "short")
    listed.clear()
    # What the Pulse Builder announces after its Delete.
    panel._builder.profiles_changed.emit()
    return "Laser 1: profile 'short' is no longer saved; pick a profile"


def _saving_an_invalid_builder_draft(panel, _listed):
    builder = panel._builder
    builder._pulse_count.setValue(2)
    builder._frequency_hz.setValue(1000.0)
    builder._duration_ms.setValue(10.0)
    reason = builder._preview_status.text()
    assert reason
    builder._save_button.click()
    return reason


@pytest.mark.parametrize("refuse", (
    _test_stim_with_no_profile,
    _run_ramp_on_an_unmapped_laser,
    _deleting_a_picked_profile,
    _saving_an_invalid_builder_draft,
))
def test_every_refusal_the_panel_reports_shows_on_its_footer(panel, listed, caplog, refuse):
    refusal = refuse(panel, listed)

    assert _shows_error(panel, refusal), (_footer(panel).text(), _footer(panel).styleSheet())
    assert len(_logged(caplog, refusal)) == 1


def test_a_failed_operation_is_shown_in_red_and_logged_once(panel, qapp, caplog):
    # Test stim is refused inside its operation on the null backend. The
    # footer already said "Laser operation failed: ...", but as an ordinary
    # status.
    tab = panel._channel_tabs[0]
    _pick(tab, "short")

    tab.stim_test_button.click()
    _wait_for_operation(panel, qapp)

    refusal = "Stim test needs the nidaq laser backend; this rig is configured for 'null'."
    assert _shows_error(panel, f"Laser operation failed: {refusal}"), _footer(panel).text()
    # The whole error on hover; the footer's line is its first line, cut.
    assert _footer(panel).toolTip() == refusal
    # The operation's own record; showing it on the footer adds none.
    assert len(_logged(caplog, refusal)) == 1

    # The next status takes the error's tooltip away with it.
    tab._run_pulse_button.click()
    _wait_for_operation(panel, qapp)
    assert _footer(panel).text() == "Pulse complete: laser 1"
    assert _footer(panel).toolTip() == ""


def test_a_profile_gone_across_a_rebuild_is_left_on_the_footer(panel, listed, caplog):
    # A rebuild picks each laser's profile again, and the Ready line it ended
    # on replaced the refusal for one no longer saved at once.
    _pick(panel._channel_tabs[0], "short")
    listed.clear()

    panel._refresh_from_model()

    refusal = "Laser 1: profile 'short' is no longer saved; pick a profile"
    assert _shows_error(panel, refusal), _footer(panel).text()
    assert len(_logged(caplog, refusal)) == 1
