import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

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
    _ProfileSaveChoice,
    _SaveProfileDialog,
)


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_tab(app_model, channel_id=LaserChannelId.LASER_1, trigger_source=None):
    channel = LaserChannelConfiguration(
        channel_id=channel_id,
        analog_output="/Dev1/ao0",
        diode_input="/Dev1/ai0",
        shutter_output="/Dev1/port0/line0",
        trigger_source=trigger_source,
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

    # Not "(hardware trigger)": a software-start profile is tested too.
    assert tab.stim_test_button.text() == "Test stim"


def test_run_pulse_is_internal_even_when_the_channel_has_a_trigger_route(qapp, app_model):
    # Defaulting to external armed Run pulse for a board STIM pulse nothing on
    # this tab sends; on christielab10 every attempt ended in a DAQmx timeout.
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig0")

    assert tab._trigger_mode.currentText() == "internal"
    assert tab._trigger_source.text() == "/Dev1/PXI_Trig0"
    assert tab._build_pulse_train().trigger_source is None

    tab._trigger_mode.setCurrentText("external")
    assert tab._build_pulse_train().trigger_source == "/Dev1/PXI_Trig0"


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
    # The waveform only: the trigger stays the tab's own.
    assert rebuilt.trigger_source is None
    # The preview is drawn from the same train: 15 ms of baseline, then the
    # last of 100 pulses at 20 Hz starts at 4.95 s and runs 3 ms, then 7 ms
    # of post-stim.
    x_values, _y_values = tab._build_preview_points(rebuilt)
    assert x_values[-1] == pytest.approx(0.015 + 99 * 0.05 + 0.003 + 0.007)


def test_selecting_a_profile_leaves_run_pulse_on_the_tabs_trigger(
    qapp, app_model, monkeypatch
):
    # christielab10, 2026-09-24: every saved profile names a board trigger
    # terminal, so picking one for Test stim switched Run Pulse to external,
    # and Run Pulse then waited for a board STIM pulse it never sends.
    profile = a_profile(
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PXI_Trig0",
    )
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig2")

    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData("burst")
    )

    assert tab._trigger_mode.currentText() == "internal"
    assert tab._trigger_source.text() == "/Dev1/PXI_Trig2"
    assert tab._build_pulse_train().trigger_source is None


def test_selecting_a_profile_keeps_a_deliberately_chosen_external_trigger(
    qapp, app_model, monkeypatch
):
    profile = a_profile(
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PXI_Trig0",
    )
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PFI3")
    tab._trigger_mode.setCurrentText("external")

    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData("burst")
    )

    assert tab._build_pulse_train().trigger_source == "/Dev1/PFI3"


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


def test_stim_test_is_unavailable_until_run_pulse_is(channel_tab):
    # christielab10, 2026-09-24: pressed before the system was running, it
    # failed with "Laser controller is not configured".
    tab, _started, _statuses = channel_tab

    tab.set_controls_enabled(True, False, False)
    assert not tab.stim_test_button.isEnabled()
    assert "press Run" in tab.stim_test_button.toolTip()

    tab.set_controls_enabled(True, True, True)
    assert tab.stim_test_button.isEnabled()


def capture_saves(monkeypatch, app_model):
    saved = []

    def save(_self, **values):
        saved.append(dict(values))
        return LaserPulseProfile(revision=1, **values)

    monkeypatch.setattr(type(app_model), "save_laser_profile", save)
    return saved


def choose(tab, route, stim_line=None, name="new"):
    tab._ask_profile_to_save = lambda _defaults: _ProfileSaveChoice(name, route, stim_line)


def test_saving_a_board_profile_uses_the_channel_terminal_and_the_chosen_line(
    qapp, app_model, monkeypatch
):
    # On internal, the default, this saved hardware_stim3 with no terminal,
    # which the profile refuses, and every profile got STIM3.
    saved = capture_saves(monkeypatch, app_model)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig2")
    choose(tab, LaserTriggerRoute.HARDWARE_STIM3, stim_line=2)

    tab._save_as_profile()

    assert saved[0]["trigger_route"] is LaserTriggerRoute.HARDWARE_STIM3
    assert saved[0]["trigger_terminal"] == "/Dev1/PXI_Trig2"
    assert saved[0]["stim_line"] == 2


def test_saving_a_software_start_profile(qapp, app_model, monkeypatch):
    saved = capture_saves(monkeypatch, app_model)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig0")
    choose(tab, LaserTriggerRoute.DIRECT_NI_SOFTWARE)

    tab._save_as_profile()

    assert saved[0]["trigger_route"] is LaserTriggerRoute.DIRECT_NI_SOFTWARE
    assert saved[0]["trigger_terminal"] == ""


def test_the_bench_trigger_does_not_become_the_profiles_route(
    qapp, app_model, monkeypatch
):
    # External on the bench with an outside terminal typed in: the profile is
    # still a board STIM profile on the channel's own terminal.
    saved = capture_saves(monkeypatch, app_model)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig0")
    tab._trigger_mode.setCurrentText("external")
    tab._trigger_source.setText("/Dev1/PFI3")
    choose(tab, LaserTriggerRoute.HARDWARE_STIM3, stim_line=3)

    tab._save_as_profile()

    assert saved[0]["trigger_route"] is LaserTriggerRoute.HARDWARE_STIM3
    assert saved[0]["trigger_terminal"] == "/Dev1/PXI_Trig0"


def test_a_board_profile_is_refused_without_a_channel_terminal(
    qapp, app_model, monkeypatch
):
    saved = capture_saves(monkeypatch, app_model)
    tab, _started, statuses = make_tab(app_model, trigger_source=None)
    choose(tab, LaserTriggerRoute.HARDWARE_STIM3, stim_line=3)

    tab._save_as_profile()

    assert saved == []
    assert statuses and statuses[-1][1] is True
    assert "terminal" in statuses[-1][0]


def test_save_defaults_follow_the_loaded_profile(qapp, app_model, monkeypatch):
    profile = a_profile(
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PXI_Trig2",
        stim_line=2,
    )
    with_one_listed_profile(monkeypatch, app_model)
    with_saved_profile(monkeypatch, app_model, profile)
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig2")
    tab.stim_profile_selector.setCurrentIndex(
        tab.stim_profile_selector.findData("burst"))

    defaults = tab._profile_save_defaults()

    assert defaults.profile_id == "burst"
    assert defaults.trigger_route is LaserTriggerRoute.HARDWARE_STIM3
    assert defaults.stim_line == 2


def test_with_nothing_loaded_the_line_comes_from_the_channels_profiles(
    qapp, app_model, monkeypatch
):
    profiles = {
        "a": a_profile(profile_id="a", trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
                       trigger_terminal="/Dev1/PXI_Trig2", stim_line=2),
        "b": a_profile(profile_id="b", trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
                       trigger_terminal="/Dev1/PXI_Trig2", stim_line=2),
    }
    with_laser_profiles(monkeypatch, app_model, tuple(
        {"profile_id": pid, "revision": 1, "summary": "channel 1, 1 V, 3 ms"}
        for pid in profiles))
    monkeypatch.setattr(type(app_model), "laser_profile",
                        lambda _self, pid: profiles.get(pid))
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig2")

    defaults = tab._profile_save_defaults()

    assert defaults.trigger_route is LaserTriggerRoute.HARDWARE_STIM3
    assert defaults.stim_line == 2


def test_with_no_profiles_the_line_is_left_to_choose(qapp, app_model):
    tab, _started, _statuses = make_tab(app_model, trigger_source="/Dev1/PXI_Trig0")

    assert tab._profile_save_defaults().stim_line is None


def a_channel(trigger_source):
    return LaserChannelConfiguration(
        channel_id=LaserChannelId.LASER_1,
        analog_output="/Dev1/ao0",
        diode_input="/Dev1/ai0",
        shutter_output="/Dev1/port0/line0",
        trigger_source=trigger_source,
    )


def test_the_save_dialog_needs_a_line_for_a_board_profile(qapp):
    dialog = _SaveProfileDialog(
        _ProfileSaveChoice("p", LaserTriggerRoute.HARDWARE_STIM3, None),
        a_channel("/Dev1/PXI_Trig0"))
    ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)

    assert not ok.isEnabled()
    dialog._line.setCurrentIndex(dialog._line.findData(2))
    assert ok.isEnabled()
    assert dialog.choice() == _ProfileSaveChoice("p", LaserTriggerRoute.HARDWARE_STIM3, 2)

    dialog._route.setCurrentIndex(
        dialog._route.findData(LaserTriggerRoute.DIRECT_NI_SOFTWARE))
    assert not dialog._line.isEnabled()
    assert dialog.choice().stim_line is None


def test_the_save_dialog_offers_no_board_route_without_a_terminal(qapp):
    dialog = _SaveProfileDialog(
        _ProfileSaveChoice("p", LaserTriggerRoute.DIRECT_NI_SOFTWARE, None),
        a_channel(None))

    assert not dialog._route.model().item(0).isEnabled()
    assert dialog.choice().trigger_route is LaserTriggerRoute.DIRECT_NI_SOFTWARE
