"""Installing a new 3D calibration: its quality report decides, an override is explicit and recorded.

A pass or warn report installs at once; a fail keeps the previous calibration unless the
operator confirms an override twice, and the override is written into the calibration
before it is installed. The previous calibration is renamed to a dated backup, and the new
one is copied in beside it and swapped into place, so the location never holds a half copy.
camera_offsets.pkl is never carried over: offsets come only from DCS, redone against the
new calibration.
"""

import getpass
import logging
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.event.api_event_kind import ApiEventKind
from autotrainer.core.project.project_info import DATE_TIME_FORMAT
from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from autotrainer.inference.multires_calibration import store
from autotrainer.inference.multires_calibration.fit import BoardSpec
from autotrainer.inference.multires_calibration.geometry import presets_from_base
from autotrainer.inference.multires_calibration.report import QualityReport
from multires_calibration_fakes import BASE, FIT
from tools.acquisition.model.app_model_status import AppModelStatus
from tools.acquisition.model.calibration_install import decide_install, install_calibration, override_record
from tools.acquisition.model.dcs_state import DcsState
from tools.acquisition.view import main_window as main_window_module
from tools.acquisition.view.main_window import MainWindow


NOW = datetime(2026, 10, 23, 14, 5, 6)
STALE_DCS_TEXT = "Redo Tools → Calibrate Coordinate System before running sessions."


def _folder(path: Path, files: dict) -> Path:
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(content)
    return path


def _contents(path: Path) -> dict:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_install_backs_up_and_never_copies_offsets(tmp_path):
    location = tmp_path / "config"
    old = {"calibration.yaml": b"old", DEFAULT_CAM_OFFSET_FILE_NAME: b"old offsets", "presets/a/frame.yaml": b"a"}
    _folder(location / DEFAULT_3D_CALIB_DIR_NAME, old)
    # An offsets file in the new folder is skipped too: none is ever installed with a calibration.
    new = {"calibration.yaml": b"new", "presets/b/frame.yaml": b"b", DEFAULT_CAM_OFFSET_FILE_NAME: b"stray"}
    new_dir = _folder(tmp_path / "session" / f"{DEFAULT_3D_CALIB_DIR_NAME}.new", new)

    installed, backup = install_calibration(new_dir, location, now=NOW)

    assert installed == location / DEFAULT_3D_CALIB_DIR_NAME
    assert backup == location / f"{DEFAULT_3D_CALIB_DIR_NAME}_{NOW.strftime(DATE_TIME_FORMAT)}"
    assert _contents(backup) == old
    assert _contents(installed) == {"calibration.yaml": b"new", "presets/b/frame.yaml": b"b"}
    assert _contents(new_dir) == new, "the session's folder is copied, not moved"
    assert sorted(p.name for p in location.iterdir()) == sorted([installed.name, backup.name])


def test_install_into_empty_location(tmp_path):
    location = tmp_path / "config"
    location.mkdir()
    new_dir = _folder(tmp_path / "new", {"calibration.yaml": b"new"})

    installed, backup = install_calibration(new_dir, location, now=NOW)

    assert (installed, backup) == (location / DEFAULT_3D_CALIB_DIR_NAME, None)
    assert _contents(installed) == {"calibration.yaml": b"new"}
    assert [p.name for p in location.iterdir()] == [DEFAULT_3D_CALIB_DIR_NAME]


def test_install_clears_a_partial_copy_left_by_an_earlier_crash(tmp_path):
    location = tmp_path / "config"
    _folder(location / f"{DEFAULT_3D_CALIB_DIR_NAME}.partial", {"calibration.yaml": b"half", "junk": b"x"})
    new_dir = _folder(tmp_path / "new", {"calibration.yaml": b"new"})

    installed, _ = install_calibration(new_dir, location, now=NOW)

    assert _contents(installed) == {"calibration.yaml": b"new"}
    assert [p.name for p in location.iterdir()] == [DEFAULT_3D_CALIB_DIR_NAME]


def test_a_failed_copy_leaves_the_previous_calibration_in_place(tmp_path):
    location = tmp_path / "config"
    old = {"calibration.yaml": b"old", DEFAULT_CAM_OFFSET_FILE_NAME: b"old offsets"}
    _folder(location / DEFAULT_3D_CALIB_DIR_NAME, old)

    with pytest.raises(FileNotFoundError):
        install_calibration(tmp_path / "missing", location, now=NOW)

    assert [p.name for p in location.iterdir()] == [DEFAULT_3D_CALIB_DIR_NAME]
    assert _contents(location / DEFAULT_3D_CALIB_DIR_NAME) == old


def test_a_failed_swap_puts_the_previous_calibration_back(tmp_path, monkeypatch):
    location = tmp_path / "config"
    old = {"calibration.yaml": b"old", DEFAULT_CAM_OFFSET_FILE_NAME: b"old offsets"}
    _folder(location / DEFAULT_3D_CALIB_DIR_NAME, old)
    new_dir = _folder(tmp_path / "new", {"calibration.yaml": b"new"})

    def refuse(src, dst):
        raise PermissionError(f"cannot replace {dst}")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        install_calibration(new_dir, location, now=NOW)
    monkeypatch.undo()

    assert _contents(location / DEFAULT_3D_CALIB_DIR_NAME) == old
    backup = location / f"{DEFAULT_3D_CALIB_DIR_NAME}_{NOW.strftime(DATE_TIME_FORMAT)}"
    assert not backup.exists()


def test_install_refuses_to_overwrite_a_backup_of_the_same_second(tmp_path):
    location = tmp_path / "config"
    old = {"calibration.yaml": b"old"}
    _folder(location / DEFAULT_3D_CALIB_DIR_NAME, old)
    earlier = {"calibration.yaml": b"earlier"}
    backup = _folder(location / f"{DEFAULT_3D_CALIB_DIR_NAME}_{NOW.strftime(DATE_TIME_FORMAT)}", earlier)
    new_dir = _folder(tmp_path / "new", {"calibration.yaml": b"new"})

    with pytest.raises(FileExistsError):
        install_calibration(new_dir, location, now=NOW)

    assert _contents(location / DEFAULT_3D_CALIB_DIR_NAME) == old
    assert _contents(backup) == earlier


def test_override_record_fields():
    assert override_record("ben", NOW) == {"confirmed": True, "by": "ben", "at": "2026-10-23T14:05:06"}


def _never_asked():
    raise AssertionError("asked to override a report that did not fail")


def test_fail_verdict_keeps_previous_without_confirmation():
    assert decide_install("fail", lambda: False) == (False, False)
    assert decide_install("fail", lambda: True) == (True, True)
    assert decide_install("warn", _never_asked) == (True, False)
    assert decide_install("pass", _never_asked) == (True, False)


def test_an_unknown_verdict_installs_nothing():
    with pytest.raises(ValueError, match="maybe"):
        decide_install("maybe", _never_asked)


# The install flow behind Tools > Make 3D calibration, driven through MainWindow with fakes
# for the window, the model and the dialogs.

def _produced(session: Path, verdict: str):
    """What make_3d_calib returns, with its calibration written where it says."""
    capture_dir = session / DEFAULT_3D_CALIB_DIR_NAME
    capture_dir.mkdir(parents=True)
    output_dir = session / f"{DEFAULT_3D_CALIB_DIR_NAME}.new"
    report = QualityReport(created="2026-10-23T09:00:00-06:00", verdict=verdict, metrics=[], thresholds={}, notes=[])
    index = store.write_calibration(
        output_dir, fit=FIT, presets=presets_from_base(BASE), board=BoardSpec(), report=report, capture={},
    )
    return capture_dir, output_dir, index, report


class _Action:
    def __init__(self, *, enabled: bool, checked: bool = False):
        self.enabled = enabled
        self.checked = checked

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setChecked(self, checked):
        self.checked = checked


class _Label:
    """The status bar's calibration-due label, as the monthly reminder left it: shown."""

    def __init__(self):
        self.text = "3D calibration due (last: 2026-09-01)"
        self.visible = True

    def setText(self, text):
        self.text = text

    def setVisible(self, visible):
        self.visible = visible


def _given_back(window) -> bool:
    """Run, the 3D action and the app status are the operator's again."""
    return (window.run_action.enabled and window.make_3d_calib_action.enabled
            and not window.make_3d_calib_action.checked and window._app_model.status is AppModelStatus.IDLE)


def _window(location: Path, monkeypatch, *, override_answer=None, dcs_after=DcsState.VALID, reload_error=None):
    """A window, mid-calibration, whose dialogs, events and model calls are recorded.

    override_answer is what the fail dialog and its confirmation answer together; None
    fails the test if they are shown. reload_calib sets dcs_state to dcs_after, as the
    real one recomputes it against the calibration it loads, and then raises reload_error
    if one is given. given_back records, at each call of the dialog and of reload_calib,
    whether the app had already been given back.
    """
    seen = SimpleNamespace(boxes=[], errors=[], events=[], reloaded=[], given_back=[])

    def reload_calib(path):
        seen.given_back.append(("reload", _given_back(window)))
        seen.reloaded.append(Path(path))
        app_model.dcs_state = dcs_after
        if reload_error is not None:
            raise reload_error

    def ask(report):
        seen.given_back.append(("ask", _given_back(window)))
        if override_answer is None:
            raise AssertionError(f"asked to override a {report.verdict} report")
        return override_answer

    app_model = SimpleNamespace(reload_calib=reload_calib, dcs_state=DcsState.VALID,
                                status=AppModelStatus.CALIBRATION_3D)
    window = SimpleNamespace(
        _app_model=app_model,
        _preferences=SimpleNamespace(configuration_location=str(location)),
        _post_api_event=lambda kind, *args: seen.events.append((kind, *args)),
        _show_msg_box=lambda title, text, icon: seen.boxes.append((title, text)),
        _ask_install_failed_3d_calib=ask,
        run_action=_Action(enabled=False),
        make_3d_calib_action=_Action(enabled=False, checked=True),
        _calibration_due_label=_Label(),
    )
    window._install_3d_calib = partial(MainWindow._install_3d_calib, window)
    window._show_3d_calib_installed = partial(MainWindow._show_3d_calib_installed, window)
    window._end_3d_calib = partial(MainWindow._end_3d_calib, window)
    window._set_calibration_due_label = partial(MainWindow._set_calibration_due_label, window)
    monkeypatch.setattr(main_window_module, "QMessageBox", SimpleNamespace(
        critical=lambda parent, title, text, *args: seen.errors.append((title, text, _given_back(window))),
        Icon=QMessageBox.Icon,
        StandardButton=QMessageBox.StandardButton,
    ))
    return window, seen


def _finish(window, error, produced) -> None:
    """The part of show_result after the capture, for a calibration started from IDLE."""
    MainWindow._finish_3d_calib(window, error, produced, AppModelStatus.IDLE)


@pytest.fixture
def location(tmp_path) -> Path:
    """The configuration location, holding the previous calibration and its DCS offsets."""
    location = tmp_path / "config"
    _folder(location / DEFAULT_3D_CALIB_DIR_NAME, {"calibration.yaml": b"old", DEFAULT_CAM_OFFSET_FILE_NAME: b"old"})
    return location


def _calibration_logs(caplog, calibration_id: str):
    """What main_window logged about calibration `calibration_id`, after its "<id>: " prefix."""
    prefix = f"3D calibration {calibration_id}: "
    messages = [r.getMessage() for r in caplog.records if r.name == main_window_module.__name__]
    return [m[len(prefix):] for m in messages if m.startswith(prefix)]


def test_a_confirmed_override_is_recorded_in_the_installed_calibration(tmp_path, location, monkeypatch, caplog):
    produced = _produced(tmp_path / "session", "fail")
    window, seen = _window(location, monkeypatch, override_answer=True, dcs_after=DcsState.STALE)

    with caplog.at_level(logging.DEBUG):
        _finish(window, None, produced)

    installed = location / DEFAULT_3D_CALIB_DIR_NAME
    index = store.load_index(installed)
    assert index.id == produced[2].id
    assert index.override["confirmed"] is True and index.override["by"] == getpass.getuser()
    assert datetime.fromisoformat(index.override["at"]) <= datetime.now()
    assert not (installed / DEFAULT_CAM_OFFSET_FILE_NAME).exists()
    assert seen.reloaded == [installed]
    assert [title for title, _ in seen.boxes] == ["3D calibration installed", "Coordinate system out of date"]
    assert "Override recorded: confirmed True, by" in seen.boxes[0][1]
    assert seen.boxes[1][1] == STALE_DCS_TEXT
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert seen.errors == []
    logs = _calibration_logs(caplog, index.id)
    assert [m.split(":")[0] for m in logs] == ["override recorded (install pending)", "installed"]


@pytest.mark.parametrize("override_answer", [True, False], ids=["install_anyway", "keep_previous"])
def test_the_app_is_given_back_only_after_the_install_decision_and_the_reload(
        tmp_path, location, monkeypatch, override_answer):
    # Until then the app stays in CALIBRATION_3D, so nothing starts a session under the
    # fail dialog or while the new calibration loads.
    produced = _produced(tmp_path / "session", "fail")
    window, seen = _window(location, monkeypatch, override_answer=override_answer)

    _finish(window, None, produced)

    expected = [("ask", False), ("reload", False)] if override_answer else [("ask", False)]
    assert seen.given_back == expected
    assert _given_back(window)


class _InlineThread:
    """A thread that runs its target as soon as it starts, so on_3d_calibrate runs start to end."""

    def __init__(self, target, name, args=(), daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)

    def join(self):
        pass


def test_on_3d_calibrate_hands_the_calibration_to_the_install_flow_still_held(tmp_path, location, monkeypatch):
    # The whole handler: make_3d_calib's result reaches the install flow, and show_result
    # leaves giving the app back to it.
    produced = _produced(tmp_path / "session", "fail")
    window, seen = _window(location, monkeypatch, override_answer=False)
    window._app_model.status = AppModelStatus.IDLE
    window._finish_3d_calib = partial(MainWindow._finish_3d_calib, window)
    monkeypatch.setattr(main_window_module, "make_3d_calib", lambda app_model: produced)
    monkeypatch.setattr(main_window_module, "threading", SimpleNamespace(Thread=_InlineThread))
    # show_result runs where it is called, instead of being posted to a Qt event loop.
    monkeypatch.setattr(main_window_module, "invoke_method", lambda method: method)

    MainWindow.on_3d_calibrate(window, True)

    assert seen.given_back == [("ask", False)]
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert _given_back(window)


def test_keep_previous_installs_nothing(tmp_path, location, monkeypatch):
    produced = _produced(tmp_path / "session", "fail")
    before = _contents(location)
    window, seen = _window(location, monkeypatch, override_answer=False)

    _finish(window, None, produced)

    assert _contents(location) == before
    assert store.load_index(produced[1]).override is None
    assert seen.reloaded == [] and seen.boxes == [] and seen.errors == []
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert _given_back(window)


@pytest.mark.parametrize("verdict", ["pass", "warn"])
@pytest.mark.parametrize("dcs_after", [DcsState.VALID, DcsState.MISSING])
def test_a_passing_report_installs_at_once_and_only_a_stale_dcs_gets_the_notice(
        tmp_path, location, monkeypatch, verdict, dcs_after):
    produced = _produced(tmp_path / "session", verdict)
    window, seen = _window(location, monkeypatch, dcs_after=dcs_after)

    _finish(window, None, produced)

    installed = location / DEFAULT_3D_CALIB_DIR_NAME
    assert store.load_index(installed).id == produced[2].id
    assert store.load_index(installed).override is None
    assert seen.reloaded == [installed]
    assert seen.given_back == [("reload", False)]
    assert [title for title, _ in seen.boxes] == ["3D calibration installed"]
    assert f"3D calibration quality: {verdict.upper()}" in seen.boxes[0][1]
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert _given_back(window)


def test_a_calibration_error_is_shown_with_its_reason(tmp_path, location, monkeypatch):
    before = _contents(location)
    window, seen = _window(location, monkeypatch)
    err = ValueError("the live camera geometry is not a valid base: width 250 is not a multiple of 4")

    _finish(window, err, None)

    assert seen.errors == [("3D calibration failed", str(err), True)]
    assert seen.events == [(ApiEventKind.calibration3dFailed, dict(reason=str(err)))]
    assert _contents(location) == before and seen.reloaded == []
    assert _given_back(window)


def _refuse_replace(src, dst):
    raise PermissionError(f"cannot replace {dst}")


def test_an_install_error_is_shown_and_reported_as_a_failure(tmp_path, location, monkeypatch):
    produced = _produced(tmp_path / "session", "pass")
    before = _contents(location)
    window, seen = _window(location, monkeypatch)

    monkeypatch.setattr(os, "replace", _refuse_replace)
    _finish(window, None, produced)
    monkeypatch.undo()

    assert seen.errors == [("3D calibration failed", f"cannot replace {location / DEFAULT_3D_CALIB_DIR_NAME}", True)]
    assert seen.events == [(ApiEventKind.calibration3dFailed, dict(reason=seen.errors[0][1]))]
    assert seen.reloaded == [] and seen.boxes == []
    assert _given_back(window)
    # The previous calibration is back in place; the copy is left for the next install to clear.
    partial = f"{DEFAULT_3D_CALIB_DIR_NAME}.partial/"
    assert {k: v for k, v in _contents(location).items() if not k.startswith(partial)} == before


def test_an_override_whose_install_fails_is_logged_as_failed(tmp_path, location, monkeypatch, caplog):
    # The record is in the session's folder by then, but the calibration never went live.
    produced = _produced(tmp_path / "session", "fail")
    window, seen = _window(location, monkeypatch, override_answer=True)

    def refuse_install(new_dir, configuration_location, *, now):
        raise PermissionError(f"cannot install {new_dir}")

    # Not os.replace: record_override replaces its files with it too.
    monkeypatch.setattr(main_window_module, "install_calibration", refuse_install)
    with caplog.at_level(logging.DEBUG):
        _finish(window, None, produced)

    assert store.load_index(produced[1]).override["confirmed"] is True
    logs = _calibration_logs(caplog, produced[2].id)
    assert [m.split(":")[0] for m in logs] == ["override recorded (install pending)", "install failed"]
    assert [kind for kind, *_ in seen.events] == [ApiEventKind.calibration3dFailed]
    assert _given_back(window)


# Once the new calibration is installed, the calibration has completed: a reload that fails
# after it, or a dialog that cannot be shown, must neither call it failed nor lose the event.

def test_a_reload_that_fails_after_the_install_says_installed_but_not_loaded(tmp_path, location, monkeypatch):
    produced = _produced(tmp_path / "session", "pass")
    window, seen = _window(location, monkeypatch, reload_error=OSError("the preset file is gone"))

    _finish(window, None, produced)

    installed = location / DEFAULT_3D_CALIB_DIR_NAME
    assert store.load_index(installed).id == produced[2].id
    assert seen.reloaded == [installed]
    assert seen.errors == [], "not reported as a failed calibration"
    assert ("3D calibration not loaded",
            "3D calibration installed, but it could not be loaded: the preset file is gone. Restart the app."
            ) in seen.boxes
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert window._calibration_due_label.visible is False
    assert _given_back(window)


def test_a_dialog_that_fails_after_the_install_still_reports_completed(tmp_path, location, monkeypatch, caplog):
    produced = _produced(tmp_path / "session", "pass")
    window, seen = _window(location, monkeypatch)

    def no_display(title, text, icon):
        raise RuntimeError("no display")

    window._show_msg_box = no_display
    with caplog.at_level(logging.ERROR, logger=main_window_module.__name__):
        _finish(window, None, produced)

    assert store.load_index(location / DEFAULT_3D_CALIB_DIR_NAME).id == produced[2].id
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert _given_back(window)
    assert any(r.exc_info and "no display" in str(r.exc_info[1]) for r in caplog.records)


def test_a_status_restore_that_fails_still_shows_the_install_and_reports_completed(
        tmp_path, location, monkeypatch, caplog):
    produced = _produced(tmp_path / "session", "pass")
    window, seen = _window(location, monkeypatch)

    def refused(prev_status):
        raise RuntimeError("status refused")

    window._end_3d_calib = refused
    with caplog.at_level(logging.ERROR, logger=main_window_module.__name__):
        _finish(window, None, produced)

    assert [title for title, _ in seen.boxes] == ["3D calibration installed"]
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert any(r.exc_info and "status refused" in str(r.exc_info[1]) for r in caplog.records)


# The status bar's "3D calibration due" label (shown by the monthly reminder) goes away when a
# new calibration is installed, and stays while the old one is still the active one.

@pytest.mark.parametrize("verdict, override_answer", [("pass", None), ("warn", None), ("fail", True)])
def test_installing_a_calibration_hides_the_due_label(tmp_path, location, monkeypatch, verdict, override_answer):
    produced = _produced(tmp_path / "session", verdict)
    window, seen = _window(location, monkeypatch, override_answer=override_answer)

    _finish(window, None, produced)

    assert seen.reloaded == [location / DEFAULT_3D_CALIB_DIR_NAME]
    assert window._calibration_due_label.visible is False


def test_keeping_the_previous_calibration_leaves_the_due_label_shown(tmp_path, location, monkeypatch):
    produced = _produced(tmp_path / "session", "fail")
    window, seen = _window(location, monkeypatch, override_answer=False)

    _finish(window, None, produced)

    assert seen.reloaded == []
    assert window._calibration_due_label.visible is True
    assert window._calibration_due_label.text == "3D calibration due (last: 2026-09-01)"


def test_a_calibration_or_an_install_that_fails_leaves_the_due_label_shown(tmp_path, location, monkeypatch):
    window, seen = _window(location, monkeypatch)
    _finish(window, ValueError("the board was not seen"), None)
    assert window._calibration_due_label.visible is True

    def refuse_install(new_dir, configuration_location, *, now):
        raise PermissionError(f"cannot install {new_dir}")

    monkeypatch.setattr(main_window_module, "install_calibration", refuse_install)
    window, seen = _window(location, monkeypatch)
    _finish(window, None, _produced(tmp_path / "session", "pass"))
    assert [kind for kind, *_ in seen.events] == [ApiEventKind.calibration3dFailed]
    assert window._calibration_due_label.visible is True
