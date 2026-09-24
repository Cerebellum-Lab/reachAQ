import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
)
from tools.acquisition.model.trial_action import (  # noqa: E402
    LaserPulseProfile,
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


def make_tab(app_model, channel_id=LaserChannelId.LASER_1, trigger_source=None,
             board_stim_line=None, draft_provider=None):
    channel = LaserChannelConfiguration(
        channel_id=channel_id,
        analog_output="/Dev1/ao0",
        diode_input="/Dev1/ai0",
        shutter_output="/Dev1/port0/line0",
        trigger_source=trigger_source,
        board_stim_line=board_stim_line,
    )
    started = []
    statuses = []
    tab = _LaserChannelTab(
        app_model,
        channel,
        True,
        None,
        lambda status, operation: started.append((status, operation)),
        lambda message, is_error: statuses.append((message, is_error)),
        draft_provider=draft_provider,
    )
    return tab, started, statuses


@pytest.fixture
def channel_tab(qapp, app_model):
    return make_tab(app_model)


def test_the_channel_tab_offers_a_stim_test_control(channel_tab):
    tab, _started, _statuses = channel_tab

    # Not "(hardware trigger)": a software-start profile is tested too.
    assert tab.stim_test_button.text() == "Test stim"


def test_run_pulse_is_internal_even_when_the_channel_has_a_trigger_route(qapp, app_model):
    # Defaulting to external armed Run pulse for a board STIM pulse nothing on
    # this tab sends; on christielab10 every attempt ended in a DAQmx timeout.
    draft = LaserPulseProfile("builder-draft", 1, 1.0, 1.0)
    tab, _started, _statuses = make_tab(
        app_model, trigger_source="/Dev1/PXI_Trig0", draft_provider=lambda: draft)

    assert tab._trigger_mode.currentText() == "internal"
    assert tab._trigger_source.text() == "/Dev1/PXI_Trig0"
    assert tab._build_pulse_train().trigger_source is None

    tab._trigger_mode.setCurrentText("external")
    assert tab._build_pulse_train().trigger_source == "/Dev1/PXI_Trig0"


def test_the_board_route_is_unavailable_without_the_lasers_board_wiring(qapp, app_model):
    tab, _started, _statuses = make_tab(app_model, trigger_source=None)

    board = tab._stim_route.findData("hardware_stim3")

    assert not tab._stim_route.model().item(board).isEnabled()
    assert tab._stim_route.currentData() == "direct_ni_software"


def test_running_a_stim_test_with_no_profile_selected_reports_rather_than_fires(
    channel_tab,
):
    tab, started, statuses = channel_tab
    tab.stim_profile_selector.clear()

    tab._run_stim_test()

    assert started == []
    assert statuses
    assert statuses[-1][1] is True


def test_running_a_stim_test_starts_a_background_operation(app_model, qapp, monkeypatch):
    tab, started, _statuses = make_tab(
        app_model, trigger_source="/Dev1/PXI_Trig0", board_stim_line=3)
    tab.stim_profile_selector.clear()
    tab.stim_profile_selector.addItem("stim-a", "stim-a")
    tab.stim_profile_selector.setCurrentIndex(0)
    monkeypatch.setattr(
        type(app_model), "laser_profile",
        lambda _self, profile_id: a_profile(profile_id="stim-a"))
    calls = []
    app_model.run_stim_bench_test = (
        lambda profile, channel_id, route: calls.append(
            (profile.profile_id, channel_id, route)))

    tab._run_stim_test()

    assert len(started) == 1
    started[0][1]()
    assert calls == [("stim-a", 1, "hardware_stim3")]


def with_laser_profiles(monkeypatch, app_model, profiles):
    """trial_protocol_state is a read-only property, so patch it on the class."""
    monkeypatch.setattr(
        type(app_model),
        "trial_protocol_state",
        property(lambda _self: {"laser_profiles": profiles}),
    )


def test_every_saved_profile_is_offered_on_every_laser(qapp, app_model, monkeypatch):
    with_laser_profiles(monkeypatch, app_model, (
        {"profile_id": "one", "revision": 1, "summary": ""},
        {"profile_id": "two", "revision": 1, "summary": ""},
    ))
    tab, _started, _statuses = make_tab(app_model, channel_id=LaserChannelId.LASER_2)

    listed = [tab.stim_profile_selector.itemData(index)
              for index in range(tab.stim_profile_selector.count())]

    assert listed == ["builder-draft", "one", "two"]


def a_profile(**overrides):
    values = dict(
        profile_id="burst",
        revision=1,
        amplitude_volts=1.25,
        pulse_duration_ms=3.0,
        pulse_count=100,
        frequency_hz=20.0,
        baseline_ms=15.0,
        post_stim_ms=7.0,
    )
    values.update(overrides)
    return LaserPulseProfile(**values)


def with_saved_profile(monkeypatch, app_model, profile):
    monkeypatch.setattr(
        type(app_model),
        "laser_profile",
        lambda _self, profile_id: (
            profile if profile_id == profile.profile_id else None
        ),
    )


def with_one_listed_profile(monkeypatch, app_model):
    with_laser_profiles(
        monkeypatch,
        app_model,
        ({"profile_id": "burst", "revision": 1, "summary": "1 V · 1 × 3 ms · 0.00 s"},),
    )


def test_run_pulse_fires_the_picked_profile_on_this_laser(qapp, app_model, monkeypatch):
    profile = a_profile()
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig0")

    tab.stim_profile_selector.setCurrentIndex(tab.stim_profile_selector.findData("burst"))
    train = tab._build_pulse_train()

    assert train.amplitude_volts == pytest.approx(1.25)
    assert train.pulse_count == 100
    assert train.channel_id.value == 1
    assert train.trigger_source is None
    assert tab._profile_summary.text() == profile.summary()


def test_run_pulse_can_fire_the_builder_draft(qapp, app_model):
    draft = LaserPulseProfile("builder-draft", 1, 0.5, 2.0)
    tab, _started, _statuses = make_tab(app_model, draft_provider=lambda: draft)

    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData("builder-draft"))

    assert tab._build_pulse_train().amplitude_volts == pytest.approx(0.5)


def test_run_pulse_says_why_it_is_unavailable(channel_tab):
    """A greyed button with no reason reads as a fault."""
    tab, _started, _statuses = channel_tab

    assert "press Run" in tab.run_pulse_refusal()


def test_stim_test_is_unavailable_until_run_pulse_is(channel_tab):
    # christielab10, 2026-09-24: pressed before the system was running, it
    # failed with "Laser controller is not configured".
    tab, _started, _statuses = channel_tab

    tab.set_controls_enabled(True, False, False)
    assert not tab.stim_test_button.isEnabled()
    assert "press Run" in tab.stim_test_button.toolTip()

    tab.set_controls_enabled(True, True, True)
    assert tab.stim_test_button.isEnabled()


def test_the_pulse_builder_is_the_first_laser_control_tab(qapp, app_model):
    from tools.acquisition.view.laser_control_content import LaserControlContent
    content = LaserControlContent(app_model)
    try:
        assert content._tabs.tabText(0) == "Pulse Builder"
        assert content._tabs.tabText(1) == "Laser 1"
        content._refresh_from_model()
        assert content._tabs.widget(0) is content._builder
    finally:
        content.on_close()


def test_a_laser_keeps_its_profile_when_laser_control_rebuilds(qapp, app_model, monkeypatch):
    # Every system Run/Stop and DAQ Ports save rebuilds the laser tabs. Each
    # new tab fell back to the builder draft, so a laser set to a saved
    # profile silently fired whatever was on the builder instead.
    from tools.acquisition.view.laser_control_content import LaserControlContent
    profile = a_profile()
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    content = LaserControlContent(app_model)
    try:
        laser_2 = content._channel_tabs[1]
        laser_2.stim_profile_selector.setCurrentIndex(
            laser_2.stim_profile_selector.findData("burst"))
        content._builder._amplitude.setValue(1.5)

        content._refresh_from_model()

        rebuilt = content._channel_tabs[1]
        assert rebuilt is not laser_2
        assert rebuilt.stim_profile_selector.currentData() == "burst"
        assert rebuilt._profile_summary.text() == profile.summary()
        assert content._channel_tabs[0].stim_profile_selector.currentData() == "builder-draft"
        assert content._builder._amplitude.value() == pytest.approx(1.5)
        assert content._tabs.count() == 1 + len(content._channel_tabs)
        assert content._tabs.tabText(1) == "Laser 1"
    finally:
        content.on_close()


