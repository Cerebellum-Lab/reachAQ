import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QSizePolicy  # noqa: E402

from tools.acquisition.model.trial_action import LaserPulseProfile  # noqa: E402
from tools.acquisition.view.pulse_builder_tab import (  # noqa: E402
    DRAFT_PROFILE_ID,
    PulseBuilderTab,
    preview_points,
    pulse_shape_refusal,
)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def builder(app_model):
    statuses = []
    tab = PulseBuilderTab(app_model, lambda message, is_error: statuses.append((message, is_error)))
    tab.set_amplitude_range(0.0, 5.0)
    return tab, statuses


def with_one_saved_profile(monkeypatch, app_model, profile):
    monkeypatch.setattr(type(app_model), "trial_protocol_state", property(
        lambda _self: {"laser_profiles": ({"profile_id": profile.profile_id,
                                           "revision": profile.revision,
                                           "summary": profile.summary()},)}))
    monkeypatch.setattr(type(app_model), "laser_profile",
                        lambda _self, pid: profile if pid == profile.profile_id else None)


def test_the_draft_is_the_train_on_the_controls(qapp, app_model):
    tab, _statuses = builder(app_model)
    tab._amplitude.setValue(1.5)
    tab._duration_ms.setValue(2.0)
    tab._pulse_count.setValue(10)
    tab._frequency_hz.setValue(50.0)

    draft = tab.draft_profile()

    assert draft == LaserPulseProfile(
        DRAFT_PROFILE_ID, 1, 1.5, 2.0, pulse_count=10, frequency_hz=50.0)


def test_a_draft_whose_pulse_outlasts_its_period_is_not_offered(qapp, app_model):
    tab, _statuses = builder(app_model)
    tab._duration_ms.setValue(30.0)
    tab._pulse_count.setValue(2)
    tab._frequency_hz.setValue(50.0)

    assert tab.draft_profile() is None


def test_picking_a_saved_profile_loads_it(qapp, app_model, monkeypatch):
    saved = LaserPulseProfile("burst", 2, 1.0, 1.0, pulse_count=500, frequency_hz=100.0,
                              pmt_open_lead_ms=3.0)
    with_one_saved_profile(monkeypatch, app_model, saved)
    tab, _statuses = builder(app_model)

    tab._profile_selector.setCurrentIndex(tab._profile_selector.findData("burst"))

    assert tab.draft_profile() == LaserPulseProfile(
        DRAFT_PROFILE_ID, 1, 1.0, 1.0, pulse_count=500, frequency_hz=100.0,
        pmt_open_lead_ms=3.0)


def test_loading_a_profile_the_builder_cannot_show_says_what_changed(
    qapp, app_model, monkeypatch,
):
    # The amplitude spinbox clamped it silently, so saving again would have
    # stored 5 V under the name of a 6 V profile.
    saved = LaserPulseProfile("hot", 1, 6.0, 1.0)
    with_one_saved_profile(monkeypatch, app_model, saved)
    tab, statuses = builder(app_model)

    tab._profile_selector.setCurrentIndex(tab._profile_selector.findData("hot"))

    assert statuses == [(
        "Profile 'hot' is 6 V; the builder allows 0..5 V, so it now shows 5 V", False)]


def test_loading_a_profile_within_range_reports_nothing(qapp, app_model, monkeypatch):
    with_one_saved_profile(monkeypatch, app_model, LaserPulseProfile("ok", 1, 1.2345, 1.0))
    tab, statuses = builder(app_model)

    tab._profile_selector.setCurrentIndex(tab._profile_selector.findData("ok"))

    assert statuses == []


def test_the_draft_name_cannot_be_saved(qapp, app_model, monkeypatch):
    # A saved profile named "builder-draft" is indistinguishable from the
    # draft in a laser tab's picker, which fires the draft for that name.
    saved = []
    monkeypatch.setattr(type(app_model), "save_laser_profile",
                        lambda _self, **values: saved.append(values))
    tab, statuses = builder(app_model)
    tab._ask_profile_name = lambda _suggested: DRAFT_PROFILE_ID

    tab._save()

    assert saved == []
    assert statuses == [(
        "'builder-draft' is reserved for the unsaved builder draft; "
        "choose another name", True)]


def test_saving_stores_the_train_under_the_given_name(qapp, app_model, monkeypatch):
    saved = []

    def save(_self, **values):
        saved.append(values)
        return LaserPulseProfile(revision=1, **values)

    monkeypatch.setattr(type(app_model), "save_laser_profile", save)
    tab, _statuses = builder(app_model)
    tab._ask_profile_name = lambda _suggested: "fresh"
    emitted = []
    tab.profiles_changed.connect(lambda: emitted.append(True))

    tab._save()

    assert saved[0]["profile_id"] == "fresh"
    assert "channel_id" not in saved[0]
    assert emitted == [True]


def test_deleting_asks_first_and_removes_the_profile(qapp, app_model, monkeypatch):
    deleted = []
    monkeypatch.setattr(type(app_model), "delete_stimulus_profile",
                        lambda _self, kind, pid: deleted.append((kind, pid)))
    with_one_saved_profile(monkeypatch, app_model, LaserPulseProfile("old", 1, 1.0, 1.0))
    tab, _statuses = builder(app_model)
    tab._profile_selector.setCurrentIndex(tab._profile_selector.findData("old"))

    tab._confirm_delete = lambda _pid: False
    tab._delete()
    assert deleted == []

    tab._confirm_delete = lambda _pid: True
    tab._delete()
    assert deleted == [("laser", "old")]


def test_delete_needs_a_saved_profile_and_editing_allowed(qapp, app_model, monkeypatch):
    # Refreshing the list re-enabled Delete even while the tab was locked.
    with_one_saved_profile(monkeypatch, app_model, LaserPulseProfile("old", 1, 1.0, 1.0))
    tab, _statuses = builder(app_model)
    assert not tab._delete_button.isEnabled()

    tab._profile_selector.setCurrentIndex(tab._profile_selector.findData("old"))
    assert tab._delete_button.isEnabled()

    tab.set_controls_enabled(False)
    tab.refresh_profiles()
    assert not tab._delete_button.isEnabled()

    tab.set_controls_enabled(True)
    assert tab._delete_button.isEnabled()


def test_a_profile_just_saved_is_selected_and_can_be_deleted(qapp, app_model, monkeypatch):
    listed = []
    monkeypatch.setattr(type(app_model), "trial_protocol_state", property(
        lambda _self: {"laser_profiles": tuple(listed)}))

    def save(_self, **values):
        listed.append({"profile_id": values["profile_id"], "revision": 1, "summary": ""})
        return LaserPulseProfile(revision=1, **values)

    monkeypatch.setattr(type(app_model), "save_laser_profile", save)
    tab, _statuses = builder(app_model)
    tab._ask_profile_name = lambda _suggested: "fresh"

    tab._save()

    assert tab._profile_selector.currentData() == "fresh"
    assert tab._delete_button.isEnabled()
    # Selected with its signals blocked; the hover text follows it anyway.
    assert tab._profile_selector.toolTip().splitlines()[0] == (
        tab._profile_selector.currentText())


def test_the_preview_draws_every_pulse():
    profile = LaserPulseProfile("p", 1, 1.25, 3.0, pulse_count=100, frequency_hz=20.0,
                                baseline_ms=15.0, post_stim_ms=7.0)

    x_values, _y_values = preview_points(profile)

    # 15 ms of baseline, the last of 100 pulses at 20 Hz starting at 4.95 s
    # and running 3 ms, then 7 ms of post-stim.
    assert x_values[-1] == pytest.approx(0.015 + 99 * 0.05 + 0.003 + 0.007)


def test_the_shape_check_names_the_period():
    profile = LaserPulseProfile("p", 1, 1.0, 30.0, pulse_count=2, frequency_hz=50.0)

    assert "20 ms period" in pulse_shape_refusal(profile)


def test_the_preview_plot_follows_its_panel(qapp, app_model):
    tab, _statuses = builder(app_model)

    assert tab._preview_plot.minimumSize().isEmpty()
    assert tab._preview_plot.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Ignored
