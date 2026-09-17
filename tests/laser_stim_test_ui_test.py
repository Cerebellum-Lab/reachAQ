import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
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

    assert listed == ["one"]


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

    assert tab.stim_profile_selector.count() == 0
