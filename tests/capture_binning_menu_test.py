import inspect
from types import SimpleNamespace

import pytest

from tools.acquisition.view.main_window import MainWindow


class _Action:
    def __init__(self):
        self.checked = None
        self.enabled = None
        self.text = None

    def blockSignals(self, _blocked):
        pass

    def setChecked(self, checked):
        self.checked = checked

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setText(self, text):
        self.text = text


def _window(current, available=(4, 2, 1)):
    app_model = SimpleNamespace(
        reach_capture_binning=current,
        reach_capture_binning_available=lambda binning: binning in available,
        left_camera=SimpleNamespace(shape=(256, 256), base_binning=4),
    )
    return SimpleNamespace(
        _app_model=app_model,
        capture_binning_actions={binning: _Action() for binning in (4, 2, 1)},
    )


def test_each_entry_names_the_size_it_captures_and_the_current_one_is_checked():
    window = _window(current=2)
    MainWindow._sync_capture_binning_actions(window)
    actions = window.capture_binning_actions
    assert [actions[b].text for b in (4, 2, 1)] == [
        "256 x 256 (bin 4)", "512 x 512 (bin 2)", "1024 x 1024 (bin 1, encoder at limit)"]
    assert [actions[b].checked for b in (4, 2, 1)] == [False, True, False]


def test_nothing_is_checked_when_left_and_right_disagree():
    window = _window(current=None)
    MainWindow._sync_capture_binning_actions(window)
    assert not any(action.checked for action in window.capture_binning_actions.values())


def test_entries_are_offered_only_while_idle_and_only_where_the_base_divides():
    window = _window(current=4, available=(4, 2))
    MainWindow._set_capture_binning_actions_enabled(window, True)
    assert {b: a.enabled for b, a in window.capture_binning_actions.items()} == {4: True, 2: True, 1: False}
    MainWindow._set_capture_binning_actions_enabled(window, False)
    assert not any(action.enabled for action in window.capture_binning_actions.values())


class _MalformedBinningModel:
    """An app model whose camera params hold a value the binning helpers cannot read."""

    def __init__(self, fails_in):
        self._fails_in = fails_in

    @property
    def reach_capture_binning(self):
        if self._fails_in == "current":
            raise ValueError("capture_binning='two' is not a number")
        return 4

    @property
    def left_camera(self):
        if self._fails_in == "base":
            raise ValueError("hbin='4x' is not a number")
        return SimpleNamespace(shape=(256, 256), base_binning=4)

    def reach_capture_binning_available(self, _binning):
        raise ValueError("hbin='4x' is not a number")


@pytest.mark.parametrize("fails_in", ["current", "base"])
def test_a_malformed_config_value_leaves_every_entry_unavailable_and_unchecked(fails_in, caplog):
    # This runs from the configuration-load handler: raising would skip every
    # enablement after it. The model still refuses Run for the same value.
    window = SimpleNamespace(
        _app_model=_MalformedBinningModel(fails_in),
        capture_binning_actions={binning: _Action() for binning in (4, 2, 1)},
    )
    with caplog.at_level("WARNING"):
        MainWindow._sync_capture_binning_actions(window)
    assert {b: (a.text, a.checked) for b, a in window.capture_binning_actions.items()} == {
        4: ("bin 4 (unavailable)", False),
        2: ("bin 2 (unavailable)", False),
        1: ("bin 1 (unavailable)", False),
    }
    assert "is not a number" in caplog.text


def test_a_malformed_config_value_leaves_every_entry_disabled(caplog):
    window = SimpleNamespace(
        _app_model=_MalformedBinningModel("available"),
        capture_binning_actions={binning: _Action() for binning in (4, 2, 1)},
    )
    for action in window.capture_binning_actions.values():
        action.enabled = True  # enabled by an earlier, healthy refresh
    with caplog.at_level("WARNING"):
        MainWindow._set_capture_binning_actions_enabled(window, True)
    assert [a.enabled for a in window.capture_binning_actions.values()] == [False, False, False]
    assert "is not a number" in caplog.text


def test_a_selection_goes_to_the_app_model_and_the_menu_resyncs():
    calls, messages = [], []
    window = _window(current=4)
    window._app_model.set_reach_capture_binning = calls.append
    window._sync_capture_binning_actions = lambda: calls.append("synced")
    window.statusBar = lambda: SimpleNamespace(showMessage=lambda text, _ms: messages.append(text))
    MainWindow._set_capture_binning_from_menu(window, 2, True)
    assert calls == [2, "synced"]
    assert "binning 2" in messages[0]


def test_a_refused_selection_reports_why_and_resyncs():
    calls, messages = [], []
    window = _window(current=4)

    def refuse(_binning):
        raise RuntimeError("Changing the camera resolution is unavailable while acquisition is running")

    window._app_model.set_reach_capture_binning = refuse
    window._sync_capture_binning_actions = lambda: calls.append("synced")
    window.statusBar = lambda: SimpleNamespace(showMessage=lambda text, _ms: messages.append(text))
    MainWindow._set_capture_binning_from_menu(window, 2, True)
    assert calls == ["synced"]
    assert "not changed" in messages[0] and "acquisition is running" in messages[0]


def test_the_menu_is_wired_to_idle_availability_and_configuration_loads():
    assert 'addMenu("Camera resolution")' in inspect.getsource(MainWindow._configure_menubar)
    assert "_set_capture_binning_actions_enabled(state.idle_configuration)" in inspect.getsource(
        MainWindow._refresh_ui_availability)
    assert "_sync_capture_binning_actions()" in inspect.getsource(
        MainWindow._on_app_model_configuration_loaded)
