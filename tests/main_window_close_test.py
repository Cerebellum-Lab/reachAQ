"""Closing the window over a session asks first and never deletes unasked.

Closing stopped acquisition, and a stop during a recording aborted it: the
session directory was deleted with no prompt (pre-release audit 2026-11-23).
"""

import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QMessageBox

from reachaq.app.model.app_model_status import SessionRecordingStatus
from reachaq.app.view import main_window as main_window_module
from reachaq.app.view.main_window import MainWindow


class _ScriptedBox:
    """A QMessageBox that clicks the button whose text it is told."""

    Icon = QMessageBox.Icon
    ButtonRole = QMessageBox.ButtonRole
    StandardButton = QMessageBox.StandardButton
    click = None
    shown = []

    def __init__(self, *_args):
        self._buttons = {}
        self._text = ""
        _ScriptedBox.shown.append(self)

    def setWindowTitle(self, _title):
        pass

    def setIcon(self, _icon):
        pass

    def setText(self, text):
        self._text = text

    def addButton(self, button, _role=None):
        label = "Cancel" if button == QMessageBox.StandardButton.Cancel else button
        token = object()
        self._buttons[label] = token
        return token

    def exec(self):
        return 0

    def clickedButton(self):
        return self._buttons[_ScriptedBox.click]


@pytest.fixture
def box(monkeypatch):
    _ScriptedBox.shown = []
    monkeypatch.setattr(main_window_module, "QMessageBox", _ScriptedBox)
    return _ScriptedBox


def _window(status, abort_result=True):
    app_model = SimpleNamespace(
        session_recording_status=status,
        project=SimpleNamespace(short_id="20261123_session003"),
        abort_recording=mock.Mock(return_value=abort_result),
    )
    return SimpleNamespace(_app_model=app_model, _show_msg_box=mock.Mock())


def test_no_session_closes_without_asking(box):
    window = _window(SessionRecordingStatus.READY)

    assert MainWindow._confirm_close_with_session(window) is True
    assert box.shown == []


@pytest.mark.parametrize("status", [SessionRecordingStatus.ARMING, SessionRecordingStatus.RECORDING])
def test_a_recording_is_saved_unless_abort_is_chosen(box, status):
    window = _window(status)
    box.click = "Stop, save and close"

    assert MainWindow._confirm_close_with_session(window) is True
    window._app_model.abort_recording.assert_not_called()
    assert set(box.shown[-1]._buttons) == {
        "Stop, save and close", "Abort (delete session) and close", "Cancel"}
    assert "20261123_session003 is recording" in box.shown[-1]._text


def test_abort_and_close_aborts_first(box):
    window = _window(SessionRecordingStatus.RECORDING)
    box.click = "Abort (delete session) and close"

    assert MainWindow._confirm_close_with_session(window) is True
    window._app_model.abort_recording.assert_called_once_with()


def test_a_refused_abort_does_not_close(box):
    window = _window(SessionRecordingStatus.RECORDING, abort_result=False)
    box.click = "Abort (delete session) and close"

    assert MainWindow._confirm_close_with_session(window) is False
    window._show_msg_box.assert_called_once()


def test_cancel_keeps_the_window_open(box):
    window = _window(SessionRecordingStatus.RECORDING)
    box.click = "Cancel"

    assert MainWindow._confirm_close_with_session(window) is False
    window._app_model.abort_recording.assert_not_called()


@pytest.mark.parametrize("status", [SessionRecordingStatus.STOPPING, SessionRecordingStatus.ANALYZING])
def test_a_session_being_saved_is_waited_for(box, status):
    window = _window(status)
    box.click = "Close when saved"

    assert MainWindow._confirm_close_with_session(window) is True
    assert set(box.shown[-1]._buttons) == {"Close when saved", "Cancel"}


def test_a_cancelled_close_ignores_the_close_event(box):
    window = _window(SessionRecordingStatus.RECORDING)
    window._app_model.app_lock = mock.MagicMock()
    window._closing = False
    event = mock.Mock()
    window._close_event = event
    window._confirm_close_with_session = lambda: MainWindow._confirm_close_with_session(window)
    box.click = "Cancel"

    MainWindow.close(window)

    event.ignore.assert_called_once_with()
    assert window._closing is False
    assert window._close_event is None


def test_a_close_arriving_while_the_prompt_is_open_waits_for_it(box):
    window = _window(SessionRecordingStatus.RECORDING)
    window._app_model.app_lock = mock.MagicMock()
    window._closing = False
    outer, inner = mock.Mock(), mock.Mock()

    def confirm():
        # A second close (Ctrl-C, another click) lands in the prompt's loop.
        window._close_event = inner
        MainWindow.close(window)
        return False

    window._confirm_close_with_session = confirm
    window._close_event = outer

    MainWindow.close(window)

    inner.ignore.assert_called_once_with()
    outer.ignore.assert_called_once_with()
    outer.accept.assert_not_called()
    assert window._closing is False


def test_an_abort_in_progress_is_waited_for(box):
    window = _window(SessionRecordingStatus.ABORTING)
    box.click = "Close when done"

    assert MainWindow._confirm_close_with_session(window) is True
    assert "being aborted" in box.shown[-1]._text


def test_a_second_close_while_saving_keeps_the_window_until_the_close_finishes(box):
    window = _window(SessionRecordingStatus.RECORDING)
    window._app_model.app_lock = mock.MagicMock()
    window._closing = True
    early, final = mock.Mock(), mock.Mock()

    window._close_event = early
    MainWindow.close(window)
    window._close_finishing = True
    window._close_event = final
    MainWindow.close(window)

    early.ignore.assert_called_once_with()
    early.accept.assert_not_called()
    final.accept.assert_called_once_with()
