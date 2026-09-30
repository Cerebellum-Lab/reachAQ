"""Laser Control opens as it was left: its folds and each laser's Command output.

Both were kept only while reachAQ ran, so every start opened every section
again and showed every laser's command trace (Ben, 2026-09-30). They are now
saved in the user's preferences as they change, under ui/ beside the main
splitter's position, and read back when the panel is made.

Every test here saves to the settings file the app_model fixture gives its
preferences, settings_ini_path under tmp_path; none reaches the user's own
~/.config/Colorado/Auto Trainer.conf.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QByteArray  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import (  # noqa: E402
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
)
from tools.acquisition.model.user_preferences import UserPreferences  # noqa: E402
from tools.acquisition.view.laser_control_content import (  # noqa: E402
    LaserControlContent,
)

#: Every laser tab's sections, each as it starts with nothing saved.
LASER_SECTIONS = {
    "run_pulse": True,
    "test_stim": True,
    "calibration_ramp": True,
    "output_stream": True,
    "signals": False,
    "board_trigger": True,
}
#: The Pulse Builder's, likewise.
BUILDER_SECTIONS = {"pulse_train": True, "pmt_margins": False}


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _lasers():
    return LaserSystemConfiguration.from_channels(
        tuple(
            LaserChannelConfiguration(
                channel_id=channel_id,
                analog_output=f"Dev1/ao{index}",
                diode_input=f"Dev1/ai{index}",
                shutter_output=f"Dev1/port0/line{index}",
            )
            for index, channel_id in enumerate((LaserChannelId.LASER_1, LaserChannelId.LASER_2))
        ),
        backend="null",
        sample_rate_hz=10000.0,
    )


@pytest.fixture
def make_panel(qapp, app_model):
    """Make a Laser Control panel, as each start of reachAQ does."""
    app_model.laser.set_configuration_offline(_lasers())
    made = []

    def make():
        content = LaserControlContent(app_model)
        made.append(content)
        return content

    try:
        yield make
    finally:
        for content in made:
            content.on_close()
            content.deleteLater()


def _restart(app_model, monkeypatch, settings_ini_path):
    """The preferences reachAQ's next start reads: the settings file, loaded again."""
    app_model.preferences.save()
    reloaded = UserPreferences(settings_file_path=settings_ini_path)
    monkeypatch.setattr(app_model, "_preferences", reloaded)
    return reloaded


def _sections(owner):
    return {name: section.is_expanded for name, section in owner.sections.items()}


def _laser_control_keys(preferences):
    return [key for key in preferences._settings.allKeys() if "laser_control" in key]


def _ui_lines(settings_ini_path):
    """The Laser Control lines of the settings file's [ui] group, as written."""
    lines, group = [], None
    for line in settings_ini_path.read_text().splitlines():
        if line.startswith("["):
            group = line
        elif group == "[ui]" and line.startswith("laser_control"):
            lines.append(line)
    return lines


def test_a_folded_section_stays_folded_after_a_restart(
    make_panel, app_model, monkeypatch, settings_ini_path,
):
    first = make_panel()
    # Read when the panel is made, written only when something changes.
    assert _laser_control_keys(app_model.preferences) == []
    tab = first._channel_tabs[0]
    tab.sections["board_trigger"].set_expanded(False)
    tab.sections["signals"].set_expanded(True)
    first._builder.sections["pulse_train"].set_expanded(False)
    first._builder.sections["pmt_margins"].set_expanded(True)

    reloaded = _restart(app_model, monkeypatch, settings_ini_path)
    second = make_panel()

    expected = dict(LASER_SECTIONS, board_trigger=False, signals=True)
    for rebuilt in second._channel_tabs:
        assert _sections(rebuilt) == expected, rebuilt.channel_id_value
    assert _sections(second._builder) == {"pulse_train": False, "pmt_margins": True}
    # Saved as the main splitter is, under [ui] in the same file, as
    # splitters\main_horizontal is; only what was changed.
    assert reloaded._settings.fileName() == settings_ini_path.as_posix()
    lines = _ui_lines(settings_ini_path)
    for line in (
        r"laser_control\sections\board_trigger=false",
        r"laser_control\sections\signals=true",
        r"laser_control\sections\pulse_train=false",
        r"laser_control\sections\pmt_margins=true",
    ):
        assert line in lines, lines
    assert not [line for line in lines if "run_pulse" in line]


def test_a_laser_s_command_output_stays_off_after_a_restart(
    make_panel, app_model, monkeypatch, settings_ini_path,
):
    first = make_panel()
    first._channel_tabs[1]._trace_command_checkbox.setChecked(False)

    reloaded = _restart(app_model, monkeypatch, settings_ini_path)
    second = make_panel()

    assert [tab.command_trace_visible for tab in second._channel_tabs] == [
        True, False, True, True]
    assert not second._channel_tabs[1]._trace_curves["command"].isVisible()
    assert _ui_lines(settings_ini_path) == [r"laser_control\command_output\laser2=false"]
    assert reloaded.laser_command_output_shown(2) is False


@pytest.mark.parametrize("damaged", (
    "banana", "", "1", 7, ["true", "false"], QByteArray(b"\x00\xff"),
), ids=("word", "empty", "digit", "number", "list", "bytes"))
def test_a_damaged_saved_value_falls_back_to_the_default(
    make_panel, app_model, monkeypatch, settings_ini_path, damaged,
):
    settings = app_model.preferences._settings
    # One good value, so the panel is shown to read the file at all.
    settings.setValue("ui/laser_control/sections/signals", True)
    settings.setValue("ui/laser_control/sections/board_trigger", damaged)
    settings.setValue("ui/laser_control/sections/pmt_margins", damaged)
    settings.setValue("ui/laser_control/command_output/laser1", damaged)
    _restart(app_model, monkeypatch, settings_ini_path)

    panel = make_panel()

    for tab in panel._channel_tabs:
        assert _sections(tab) == dict(LASER_SECTIONS, signals=True), tab.channel_id_value
    assert _sections(panel._builder) == BUILDER_SECTIONS
    assert all(tab.command_trace_visible for tab in panel._channel_tabs)


class _UnreadableSettings:
    """A settings store whose every read fails."""

    def value(self, *_args, **_kwargs):
        raise OSError("the settings file could not be read")

    def setValue(self, *_args, **_kwargs):
        pass


def test_unreadable_preferences_open_the_panel_as_before(make_panel, app_model, monkeypatch):
    monkeypatch.setattr(app_model.preferences, "_settings", _UnreadableSettings())

    panel = make_panel()

    for tab in panel._channel_tabs:
        assert _sections(tab) == LASER_SECTIONS
        assert tab.command_trace_visible
    assert _sections(panel._builder) == BUILDER_SECTIONS


def test_a_panel_given_no_preferences_keeps_its_state_in_memory_only(
    make_panel, app_model, monkeypatch,
):
    # As the signal-stream tests' application stubs have none; those build
    # the panel with no preferences attribute at all. Nothing is saved, and
    # nothing falls back to a QSettings of its own, which would be the
    # user's real file.
    user_preferences = app_model.preferences
    monkeypatch.setattr(type(app_model), "preferences", property(lambda _self: None))
    panel = make_panel()
    panel._channel_tabs[0].sections["board_trigger"].set_expanded(False)
    panel._channel_tabs[1]._trace_command_checkbox.setChecked(False)

    panel._refresh_from_model()

    assert not panel._channel_tabs[0].sections["board_trigger"].is_expanded
    assert not panel._channel_tabs[1].command_trace_visible
    assert _laser_control_keys(user_preferences) == []
