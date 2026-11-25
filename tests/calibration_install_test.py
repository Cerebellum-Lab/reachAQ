"""Installing a new 3D calibration: its quality report decides, an override is explicit and recorded.

A pass or warn report offers Set as active (the default) or Keep, don't activate, which adds
the calibration to the library without moving the pointer; a fail keeps the previous calibration unless the
operator confirms an override twice, and the override is recorded as a decision in the
calibration library, never written into the calibration. An install adds the calibration to
the library, <location>/calibrations/<id>, and makes it active; the 4mm_6r_8c_4x folder and
its backups from before the library are left as they are.
camera_offsets.pkl is never carried over: offsets come only from DCS, redone against the
new calibration.
"""

import getpass
import importlib
import json
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.core.event.api_event_kind import ApiEventKind
from reachaq.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from reachaq.inference.multires_calibration import library, store
from reachaq.inference.multires_calibration.fit import BoardSpec
from reachaq.inference.multires_calibration.geometry import presets_from_base
from reachaq.inference.multires_calibration.report import QualityReport
from multires_calibration_fakes import BASE, FIT
from multires_synthetic import synthetic_rig
from reachaq.app.model.app_model_status import AppModelStatus
from reachaq.app.model.calibration_install import decide_install, install_calibration, override_record
from reachaq.app.model.dcs_state import DcsState
from reachaq.app.view import main_window as main_window_module
from reachaq.app.view.calibration_dialog import CalibrationRequest
from reachaq.app.view.main_window import MainWindow


calibration_install_module = importlib.import_module("reachaq.app.model.calibration_install")

NOW = datetime(2026, 10, 23, 14, 5, 6)
RIG = synthetic_rig(noise_px=0.0)
_REPORT = QualityReport(created="2026-10-23T09:00:00-06:00", verdict="pass", metrics=[], thresholds={}, notes=[])
STALE_DCS_TEXT = "Redo Tools → Calibrate Coordinate System before running sessions."


def _folder(path: Path, files: dict) -> Path:
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(content)
    return path


def _contents(path: Path) -> dict:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def _calibration(path: Path, rig=RIG) -> str:
    """A real calibration folder in `path`; its id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return store.write_calibration(
        path, fit=rig.truth, presets=presets_from_base(BASE), board=BoardSpec(), report=_REPORT, capture={},
    ).id


def _decisions(root: Path) -> list:
    return [json.loads(line) for line in (root / "decisions.jsonl").read_text().splitlines()]


def test_install_adds_an_active_entry_and_leaves_the_old_folders(tmp_path):
    location = tmp_path / "config"
    old_id = _calibration(location / DEFAULT_3D_CALIB_DIR_NAME)
    (location / DEFAULT_3D_CALIB_DIR_NAME / DEFAULT_CAM_OFFSET_FILE_NAME).write_bytes(b"old offsets")
    old = _contents(location / DEFAULT_3D_CALIB_DIR_NAME)
    new_dir = tmp_path / "session" / f"{DEFAULT_3D_CALIB_DIR_NAME}.new"
    new_id = _calibration(new_dir)
    new = _contents(new_dir)

    installed = install_calibration(new_dir, location, now=NOW, by="ben")

    root = library.library_root(location)
    assert installed == root / new_id
    assert _contents(installed) == new
    assert _contents(new_dir) == new, "the session's folder is copied, not moved"
    # The folder from before the library is only read: no swap, no dated backup.
    assert _contents(location / DEFAULT_3D_CALIB_DIR_NAME) == old
    assert sorted(p.name for p in location.iterdir()) == sorted([DEFAULT_3D_CALIB_DIR_NAME, "calibrations"])
    assert library.active_id(root) == new_id != old_id
    assert [e.id for e in library.list_entries(root)] == [new_id]
    assert [(d["event"], d["id"], d["by"]) for d in _decisions(root)] == [("set_active", new_id, "ben")]


def test_install_into_empty_location(tmp_path):
    location = tmp_path / "config"
    location.mkdir()
    new_dir = tmp_path / "new"
    new_id = _calibration(new_dir)

    installed = install_calibration(new_dir, location, now=NOW)

    assert installed == library.library_root(location) / new_id
    assert _contents(installed) == _contents(new_dir)
    assert [p.name for p in location.iterdir()] == ["calibrations"]
    assert library.active_id(library.library_root(location)) == new_id


def test_install_never_installs_offsets(tmp_path):
    # Offsets come only from DCS, measured against the calibration they belong to.
    location = tmp_path / "config"
    location.mkdir()
    new_dir = tmp_path / "new"
    _calibration(new_dir)
    (new_dir / DEFAULT_CAM_OFFSET_FILE_NAME).write_bytes(b"stray")

    with pytest.raises(ValueError, match=DEFAULT_CAM_OFFSET_FILE_NAME):
        install_calibration(new_dir, location, now=NOW)

    assert list(location.iterdir()) == []


def test_a_failed_copy_leaves_the_previous_calibration_in_place(tmp_path):
    location = tmp_path / "config"
    _calibration(location / DEFAULT_3D_CALIB_DIR_NAME)
    (location / DEFAULT_3D_CALIB_DIR_NAME / DEFAULT_CAM_OFFSET_FILE_NAME).write_bytes(b"old offsets")
    old = _contents(location / DEFAULT_3D_CALIB_DIR_NAME)

    with pytest.raises(ValueError, match="not a multi-resolution calibration"):
        install_calibration(tmp_path / "missing", location, now=NOW)

    assert [p.name for p in location.iterdir()] == [DEFAULT_3D_CALIB_DIR_NAME]
    assert _contents(location / DEFAULT_3D_CALIB_DIR_NAME) == old


def test_a_failed_pointer_write_leaves_no_entry_and_the_previous_one_active(tmp_path, monkeypatch):
    location = tmp_path / "config"
    location.mkdir()
    first = tmp_path / "first"
    first_id = _calibration(first)
    install_calibration(first, location, now=NOW)
    new_dir = tmp_path / "new"
    _calibration(new_dir)

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(calibration_install_module, "set_active", refuse)
    with pytest.raises(OSError, match="disk full"):
        install_calibration(new_dir, location, now=NOW)
    monkeypatch.undo()

    root = library.library_root(location)
    assert [e.id for e in library.list_entries(root)] == [first_id]
    assert library.active_id(root) == first_id
    # Retrying works: the discarded entry does not block its own id.
    install_calibration(new_dir, location, now=NOW)
    assert library.active_id(root) == store.load_index(new_dir).id


def test_a_failed_activation_decision_keeps_the_pointer_and_reports_its_own_error(tmp_path, monkeypatch):
    location = tmp_path / "config"
    location.mkdir()
    first = tmp_path / "first"
    first_id = _calibration(first)
    install_calibration(first, location, now=NOW)
    root = library.library_root(location)
    pointer = (root / library.ACTIVE_FILE).read_bytes()
    new_dir = tmp_path / "new"
    _calibration(new_dir)

    def refuse(root, record):
        raise OSError("decisions.jsonl: no space left on device")

    monkeypatch.setattr(library, "_append_decision", refuse)
    with pytest.raises(OSError, match="no space left on device"):
        install_calibration(new_dir, location, now=NOW)
    monkeypatch.undo()

    assert (root / library.ACTIVE_FILE).read_bytes() == pointer
    assert library.active_id(root) == first_id
    assert [e.id for e in library.list_entries(root)] == [first_id], "the new entry is removed again"


def test_a_retry_after_a_crash_before_activation_installs_the_same_entry(tmp_path):
    location = tmp_path / "config"
    location.mkdir()
    new_dir = tmp_path / "new"
    new_id = _calibration(new_dir)
    root = library.library_root(location)
    library.add_entry(root, new_dir)  # the crash came before set_active

    installed = install_calibration(new_dir, location, now=NOW, by="ben")

    assert installed == root / new_id and library.active_id(root) == new_id
    assert [e.id for e in library.list_entries(root)] == [new_id]


def test_an_override_is_a_decision_and_the_calibration_is_not_edited(tmp_path):
    location = tmp_path / "config"
    new_dir = tmp_path / "new"
    new_id = _calibration(new_dir)
    before = _contents(new_dir)
    record = override_record("ben", NOW)

    install_calibration(new_dir, location, now=NOW, by="ben", override=record)

    root = library.library_root(location)
    assert _contents(root / new_id) == before
    assert store.load_index(root / new_id).override is None
    events = _decisions(root)
    assert [(d["event"], d["id"]) for d in events] == [("set_active", new_id), ("override", new_id)]
    assert events[1]["detail"] == record


def test_override_record_fields():
    assert override_record("ben", NOW) == {"confirmed": True, "by": "ben", "at": "2026-10-23T14:05:06"}


def _never_asked():
    raise AssertionError("asked to override a report that did not fail")


def _never_asked_to_activate():
    raise AssertionError("asked Set as active for a report that failed")


def test_fail_verdict_keeps_previous_without_confirmation():
    assert decide_install("fail", lambda: False, _never_asked_to_activate) == (False, False, False)
    assert decide_install("fail", lambda: True, _never_asked_to_activate) == (True, True, True)
    assert decide_install("warn", _never_asked) == (True, False, True)
    assert decide_install("pass", _never_asked) == (True, False, True)


@pytest.mark.parametrize("verdict", ["pass", "warn"])
def test_a_passing_report_asks_set_as_active_or_keep(verdict):
    assert decide_install(verdict, _never_asked, lambda: True) == (True, False, True)
    assert decide_install(verdict, _never_asked, lambda: False) == (True, False, False)


def test_keep_adds_the_entry_and_leaves_the_pointer(tmp_path):
    location = tmp_path / "config"
    first = tmp_path / "first"
    first_id = _calibration(first)
    install_calibration(first, location, now=NOW, by="ben")
    root = library.library_root(location)
    pointer = (root / library.ACTIVE_FILE).read_bytes()
    new_dir = tmp_path / "new"
    new_id = _calibration(new_dir, synthetic_rig(noise_px=0.0, frame_rotation_deg=5.0))

    kept = install_calibration(new_dir, location, now=NOW, by="ben", activate=False)

    assert kept == root / new_id and _contents(kept) == _contents(new_dir)
    assert (root / library.ACTIVE_FILE).read_bytes() == pointer
    assert library.active_id(root) == first_id
    assert sorted(e.id for e in library.list_entries(root)) == sorted([first_id, new_id])
    assert [(d["event"], d["id"]) for d in _decisions(root)] == [("set_active", first_id), ("kept", new_id)]
    assert _decisions(root)[-1]["detail"] == {"by": "ben"}
    # It can be set active later, from the library.
    library.set_active(root, new_id, by="ben")
    assert library.active_id(root) == new_id


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


def _window(location: Path, monkeypatch, *, override_answer=None, dcs_after=DcsState.VALID, reload_error=None,
            activate_answer=True):
    """A window, mid-calibration, whose dialogs, events and model calls are recorded.

    override_answer is what the fail dialog and its confirmation answer together; None
    fails the test if they are shown. reload_calib sets dcs_state to dcs_after, as the
    real one recomputes it against the calibration it loads, and then raises reload_error
    if one is given. given_back records, at each call of the dialog and of reload_calib,
    whether the app had already been given back.
    """
    seen = SimpleNamespace(boxes=[], errors=[], events=[], reloaded=[], given_back=[], activate_asked=[])

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

    def ask_activate(report):
        seen.activate_asked.append(report.verdict)
        return activate_answer

    app_model = SimpleNamespace(reload_calib=reload_calib, dcs_state=DcsState.VALID,
                                status=AppModelStatus.CALIBRATION_3D, left_camera="left", right_camera="right")
    window = SimpleNamespace(
        _app_model=app_model,
        _preferences=SimpleNamespace(configuration_location=str(location)),
        _post_api_event=lambda kind, *args: seen.events.append((kind, *args)),
        _show_msg_box=lambda title, text, icon: seen.boxes.append((title, text)),
        _ask_install_failed_3d_calib=ask,
        _ask_activate_3d_calib=ask_activate,
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

    installed = library.library_root(location) / produced[2].id
    index = store.load_index(installed)
    assert index.id == produced[2].id
    # The override is a decision about the calibration; the calibration is not edited.
    assert index.override is None
    root = library.library_root(location)
    assert library.active_id(root) == index.id
    override, = [d for d in _decisions(root) if d["event"] == "override"]
    assert override["id"] == index.id
    assert override["detail"]["confirmed"] is True and override["detail"]["by"] == getpass.getuser()
    assert datetime.fromisoformat(override["detail"]["at"]) <= datetime.now()
    assert not (installed / DEFAULT_CAM_OFFSET_FILE_NAME).exists()
    assert (location / DEFAULT_3D_CALIB_DIR_NAME / DEFAULT_CAM_OFFSET_FILE_NAME).read_bytes() == b"old"
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
    left, right = SimpleNamespace(name="left"), SimpleNamespace(name="right")
    window._app_model.cameras = (left, right)
    window._ask_calibration_cameras = lambda: CalibrationRequest(["left", "right"], "handwritten", False)
    asked = []
    monkeypatch.setattr(main_window_module, "make_3d_calib",
                        lambda app_model, cameras, move_list=None, **kwargs: asked.append(cameras) or produced)
    monkeypatch.setattr(main_window_module, "threading", SimpleNamespace(Thread=_InlineThread))
    # show_result runs where it is called, instead of being posted to a Qt event loop.
    monkeypatch.setattr(main_window_module, "invoke_method", lambda method: method)

    MainWindow.on_3d_calibrate(window, True)

    assert asked == [[left, right]], "the cameras the dialog chose"
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

    installed = library.library_root(location) / produced[2].id
    assert store.load_index(installed).id == produced[2].id
    assert store.load_index(installed).override is None
    assert seen.reloaded == [installed]
    assert seen.given_back == [("reload", False)]
    assert seen.activate_asked == [verdict]
    assert library.active_id(library.library_root(location)) == produced[2].id
    assert [title for title, _ in seen.boxes] == ["3D calibration installed"]
    assert f"3D calibration quality: {verdict.upper()}" in seen.boxes[0][1]
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert _given_back(window)


@pytest.mark.parametrize("verdict", ["pass", "warn"])
def test_keep_dont_activate_adds_the_entry_and_leaves_the_previous_one_active(
        tmp_path, location, monkeypatch, verdict):
    first = tmp_path / "first"
    first_id = _calibration(first)
    install_calibration(first, location, now=NOW, by="ben")
    root = library.library_root(location)
    pointer = (root / library.ACTIVE_FILE).read_bytes()
    produced = _produced(tmp_path / "session", verdict)
    window, seen = _window(location, monkeypatch, activate_answer=False)

    _finish(window, None, produced)

    new_id = produced[2].id
    assert (root / library.ACTIVE_FILE).read_bytes() == pointer and library.active_id(root) == first_id
    assert new_id in [e.id for e in library.list_entries(root)]
    assert (_decisions(root)[-1]["event"], _decisions(root)[-1]["id"]) == ("kept", new_id)
    assert seen.reloaded == [], "the active calibration is unchanged, so nothing reloads"
    assert seen.boxes == [("3D calibration kept",
                           f"3D calibration {new_id[:8]} is in the calibration library, not active. Set it active"
                           " from Tools → Calibration library when you want to use it.")]
    assert seen.events == [(ApiEventKind.calibration3dCompleted,)]
    assert window._calibration_due_label.visible is True, "the active calibration is still the old one"
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


def test_an_install_error_is_shown_and_reported_as_a_failure(tmp_path, location, monkeypatch):
    produced = _produced(tmp_path / "session", "pass")
    before = _contents(location)
    window, seen = _window(location, monkeypatch)

    def refuse(*args, **kwargs):
        raise PermissionError("cannot write active.yaml")

    monkeypatch.setattr(calibration_install_module, "set_active", refuse)
    _finish(window, None, produced)
    monkeypatch.undo()

    assert seen.errors == [("3D calibration failed", "cannot write active.yaml", True)]
    assert seen.events == [(ApiEventKind.calibration3dFailed, dict(reason=seen.errors[0][1]))]
    assert seen.reloaded == [] and seen.boxes == []
    assert _given_back(window)
    # The new entry is discarded again; the folders from before are untouched.
    assert library.list_entries(library.library_root(location)) == []
    assert {k: v for k, v in _contents(location).items() if not k.startswith("calibrations/")} == before


def test_an_override_whose_install_fails_is_logged_as_failed(tmp_path, location, monkeypatch, caplog):
    # The record is in the session's folder by then, but the calibration never went live.
    produced = _produced(tmp_path / "session", "fail")
    window, seen = _window(location, monkeypatch, override_answer=True)

    def refuse_install(new_dir, configuration_location, **kwargs):
        raise PermissionError(f"cannot install {new_dir}")

    monkeypatch.setattr(main_window_module, "install_calibration", refuse_install)
    with caplog.at_level(logging.DEBUG):
        _finish(window, None, produced)

    assert store.load_index(produced[1]).override is None, "the session's calibration is never edited"
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

    installed = library.library_root(location) / produced[2].id
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

    assert library.active_id(library.library_root(location)) == produced[2].id
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

    assert seen.reloaded == [library.library_root(location) / produced[2].id]
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

    def refuse_install(new_dir, configuration_location, **kwargs):
        raise PermissionError(f"cannot install {new_dir}")

    monkeypatch.setattr(main_window_module, "install_calibration", refuse_install)
    window, seen = _window(location, monkeypatch)
    _finish(window, None, _produced(tmp_path / "session", "pass"))
    assert [kind for kind, *_ in seen.events] == [ApiEventKind.calibration3dFailed]
    assert window._calibration_due_label.visible is True
