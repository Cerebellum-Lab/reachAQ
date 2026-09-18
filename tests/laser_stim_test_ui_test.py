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
    LaserTriggerRoute,
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


def make_tab(app_model, channel_id=LaserChannelId.LASER_1):
    channel = LaserChannelConfiguration(
        channel_id=channel_id,
        analog_output="/Dev1/ao0",
        diode_input="/Dev1/ai0",
        shutter_output="/Dev1/port0/line0",
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
    )
    return tab, started, statuses


@pytest.fixture
def channel_tab(qapp, app_model):
    return make_tab(app_model)


def test_the_channel_tab_offers_a_stim_test_control(channel_tab):
    tab, _started, _statuses = channel_tab

    assert tab.stim_test_button.text() == "Test stim (hardware trigger)"


def test_running_a_stim_test_with_no_profile_selected_reports_rather_than_fires(
    channel_tab,
):
    tab, started, statuses = channel_tab
    tab.stim_profile_selector.clear()

    tab._run_stim_test()

    assert started == []
    assert statuses
    assert statuses[-1][1] is True


def test_running_a_stim_test_starts_a_background_operation(channel_tab, app_model):
    tab, started, _statuses = channel_tab
    tab.stim_profile_selector.clear()
    tab.stim_profile_selector.addItem("stim-a", "stim-a")
    tab.stim_profile_selector.setCurrentIndex(0)
    calls = []
    app_model.run_stim_bench_test = lambda profile_id: calls.append(profile_id)

    tab._run_stim_test()

    assert len(started) == 1
    started[0][1]()
    assert calls == ["stim-a"]


def with_laser_profiles(monkeypatch, app_model, profiles):
    """trial_protocol_state is a read-only property, so patch it on the class."""
    monkeypatch.setattr(
        type(app_model),
        "trial_protocol_state",
        property(lambda _self: {"laser_profiles": profiles}),
    )


def test_the_selector_lists_only_profiles_for_this_channel(
    qapp, app_model, monkeypatch
):
    with_laser_profiles(
        monkeypatch,
        app_model,
        (
            {"profile_id": "one", "revision": 1, "summary": "channel 1, 2 V, 5 ms"},
            {"profile_id": "two", "revision": 1, "summary": "channel 2, 2 V, 5 ms"},
        ),
    )
    tab, _started, _statuses = make_tab(app_model)

    listed = [
        tab.stim_profile_selector.itemData(index)
        for index in range(tab.stim_profile_selector.count())
    ]

    # The first entry is the way back to a train built by hand.
    assert listed == [None, "one"]


def test_the_selector_is_empty_when_no_profile_targets_this_channel(
    qapp, app_model, monkeypatch
):
    with_laser_profiles(
        monkeypatch,
        app_model,
        (
            {"profile_id": "two", "revision": 1, "summary": "channel 2, 2 V, 5 ms"},
        ),
    )
    tab, _started, _statuses = make_tab(app_model)

    listed = [
        tab.stim_profile_selector.itemData(index)
        for index in range(tab.stim_profile_selector.count())
    ]
    assert listed == [None]


def a_profile(**overrides):
    values = dict(
        profile_id="burst",
        revision=1,
        channel_id=1,
        amplitude_volts=1.25,
        pulse_duration_ms=3.0,
        pulse_count=100,
        frequency_hz=20.0,
        baseline_ms=15.0,
        post_stim_ms=7.0,
        trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
        trigger_terminal="/Dev1/PFI0",
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
        ({"profile_id": "burst", "revision": 1, "summary": "channel 1, 1 V, 3 ms"},),
    )


def test_selecting_a_profile_rebuilds_the_pulse_train_it_describes(
    qapp, app_model, monkeypatch
):
    """The build graph kept showing typed values with a profile selected."""
    profile = a_profile()
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    tab, _started, _statuses = make_tab(app_model)

    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData("burst")
    )

    rebuilt = tab._build_pulse_train()
    assert rebuilt.amplitude_volts == pytest.approx(1.25)
    assert rebuilt.duration_ms == pytest.approx(3.0)
    assert rebuilt.pulse_count == 100
    assert rebuilt.frequency_hz == pytest.approx(20.0)
    assert rebuilt.baseline_ms == pytest.approx(15.0)
    assert rebuilt.post_stim_ms == pytest.approx(7.0)
    assert rebuilt.trigger_source == "/Dev1/PFI0"
    # The preview is drawn from the same train: 100 pulses at 20 Hz run for
    # five seconds, plus the baseline and post-stim margins.
    x_values, _y_values = tab._build_preview_points(rebuilt)
    assert x_values[-1] == pytest.approx(5.022, abs=0.01)


def test_choosing_new_profile_leaves_the_built_train_alone(
    qapp, app_model, monkeypatch
):
    profile = a_profile()
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    tab, _started, _statuses = make_tab(app_model)
    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData("burst")
    )

    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData(None)
    )

    # Nothing is reset: what was loaded stays as the starting point to edit.
    assert tab._build_pulse_train().pulse_count == 100


def test_run_pulse_says_why_it_is_unavailable(channel_tab):
    """A greyed button with no reason reads as a fault."""
    tab, _started, _statuses = channel_tab

    assert "press Run" in tab.run_pulse_refusal()
