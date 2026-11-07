"""No test writes into the operator's data folder, or into ~/.config/Colorado.

The full runs on christielab10 on 2026-10-01 left 1,533 logs, two hourly event
files and two empty session folders in ~/Documents/rawdatalocal/20261001/
christielab10, the folder its real sessions are recorded into. AppModel opened
them under its default data folder while it was constructed, before any
configuration's outputLocation was loaded. The same runs created and deleted
the operator's ~/.config/Colorado/autotrainer_running_status.env.

These tests move HOME under tmp_path to stand in for the operator's, and the
guard's tests give it a folder there to protect, so even against code that
leaks they write nothing outside pytest's tmp.
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QFile, QIODevice, QSettings

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core import PersistenceConfiguration
from tools.acquisition.model.app_model import AppModel

import top_fixtures


@pytest.fixture
def operator_home(tmp_path, monkeypatch) -> Path:
    """HOME, moved under tmp_path before the app is built."""
    home = tmp_path.joinpath("operator-home")
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    return home


def _data_folder_under(home: Path) -> Path:
    # Where PersistenceConfiguration's default, ~/Documents/rawdatalocal, lands.
    return home.joinpath("Documents", "rawdatalocal")


def _is_under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _event_files(root: Path, timeout: float = 5.0) -> list:
    # The event manager writes from its own thread.
    deadline = time.monotonic() + timeout
    while True:
        found = sorted(root.rglob("*_events.csv"))
        if found or time.monotonic() > deadline:
            return found
        time.sleep(0.05)


def _assert_writes_only_under_tmp(app: AppModel, home: Path, basetemp: Path) -> None:
    project_root = Path(app.project.root)
    session = Path(app.project.get_session_path().location)  # this creates it
    events = _event_files(project_root)
    home_data = _data_folder_under(home)
    assert not home_data.exists(), (
        "the app wrote under HOME's data folder: "
        f"{sorted(str(path) for path in home_data.rglob('*'))[:5]}"
    )
    assert _is_under(project_root, basetemp), project_root
    log = app._log_file_path
    assert log is not None and log.is_file() and _is_under(log, project_root), log
    assert events and all(_is_under(path, project_root) for path in events), events
    assert session.is_dir() and _is_under(session, project_root), session


def test_each_test_has_a_default_data_folder_of_its_own(tmp_path_factory):
    default = PersistenceConfiguration.get_default_output_path()
    assert _is_under(default, tmp_path_factory.getbasetemp()), default
    assert default.is_dir() and not any(default.iterdir()), default


def test_each_test_has_a_running_status_file_of_its_own(tmp_path_factory):
    path = AppModel.status_file_path.expanduser()
    assert _is_under(path, tmp_path_factory.getbasetemp()), path
    assert path.parent.is_dir() and not path.exists(), path


def test_a_tests_monkeypatch_undo_leaves_both_in_place(monkeypatch, tmp_path_factory):
    # As test_saving_resumes_once_a_load_completes once did, mid-test.
    monkeypatch.undo()
    basetemp = tmp_path_factory.getbasetemp()
    for path in (PersistenceConfiguration.get_default_output_path(),
                 AppModel.status_file_path.expanduser()):
        assert _is_under(path, basetemp), path


def test_the_app_fixture_keeps_its_log_events_and_sessions_under_tmp(
    operator_home, app_model, tmp_path_factory,
):
    _assert_writes_only_under_tmp(app_model, operator_home, tmp_path_factory.getbasetemp())


def test_an_app_built_without_the_fixtures_keeps_them_under_tmp(
    operator_home, user_pref, calib_dir, monkeypatch, tmp_path_factory,
):
    # As headless.py and the main window build it.
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    app = AppModel(user_pref, calib_dir=calib_dir)
    try:
        _assert_writes_only_under_tmp(app, operator_home, tmp_path_factory.getbasetemp())
    finally:
        app.capture_stop(force=True)
        app.on_close()
        top_fixtures.release_multiprocessing_resources(app)


@pytest.fixture
def guarded_folder(tmp_path, monkeypatch) -> Path:
    """A stand-in for the operator's data folder, holding one recorded session.

    The guard protects it as well as the real one.
    """
    folder = _data_folder_under(tmp_path.joinpath("operator-home"))
    session = folder.joinpath("20261001", "christielab10", "session001")
    session.mkdir(parents=True)
    session.joinpath("left.mp4").write_bytes(b"recorded")
    guard = top_fixtures.data_root_guard
    monkeypatch.setattr(guard, "roots", (*guard.roots, folder))
    return folder


def _listing(folder: Path) -> list:
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


def test_the_guard_protects_the_production_default_data_folder():
    assert top_fixtures.data_root_guard.roots[0] == (
        top_fixtures.PRODUCTION_DEFAULT_OUTPUT_PATH.expanduser())


def test_the_guard_protects_and_watches_the_operators_preferences_folder():
    production = top_fixtures.PRODUCTION_STATUS_FILE_PATH
    assert production == Path("~/.config/Colorado/autotrainer_running_status.env")
    folder = production.parent.expanduser()
    # The folder, so the preferences file in it too; and that file, which Qt
    # writes from C++, is watched.
    assert folder in top_fixtures.data_root_guard.roots
    assert top_fixtures.data_root_guard.watched == (folder / "Auto Trainer.conf",)


@pytest.fixture
def preferences_folder(tmp_path, monkeypatch) -> Path:
    """A stand-in for ~/.config/Colorado, protected and watched beside the real one."""
    folder = tmp_path.joinpath("operator-home", ".config", "Colorado")
    folder.mkdir(parents=True)
    folder.joinpath("Auto Trainer.conf").write_text("[system]\nserial_number=christielab10\n")
    guard = top_fixtures.data_root_guard
    monkeypatch.setattr(guard, "roots", (*guard.roots, folder))
    monkeypatch.setattr(guard, "watched", (*guard.watched, folder / "Auto Trainer.conf"))
    return folder


def test_the_guard_refuses_the_status_file_there(preferences_folder):
    status = preferences_folder.joinpath("autotrainer_running_status.env")
    with pytest.raises(top_fixtures.DataRootWriteRefused):
        status.write_text("status='running'\n")
    with pytest.raises(top_fixtures.DataRootWriteRefused):
        preferences_folder.joinpath("Auto Trainer.conf").unlink()
    assert sorted(path.name for path in preferences_folder.iterdir()) == ["Auto Trainer.conf"]
    assert len(top_fixtures.data_root_guard.take_refused()) == 2


def test_the_guard_sees_a_preferences_write_the_hook_cannot(preferences_folder):
    guard = top_fixtures.data_root_guard
    before = guard.snapshot()
    assert guard.changes_since(before) == []
    # QSettings writes from C++, out of the audit hook's sight.
    conf = preferences_folder.joinpath("Auto Trainer.conf")
    settings = QSettings(conf.as_posix(), QSettings.Format.IniFormat)
    settings.setValue("system/serial_number", "pytest")
    settings.sync()
    assert settings.status() == QSettings.Status.NoError
    assert guard.take_refused() == []
    assert guard.changes_since(before) == [f"{conf} changed"]
    with pytest.raises(pytest.fail.Exception,
                       match=r"(?s)changed during this test \(by this test, or by another"
                             r" process such as a running reachAQ\):.*Auto Trainer\.conf changed"):
        guard.fail_on_refusals(before)


def _write_from_another_process(path: Path, content: bytes) -> None:
    # Written in C++ with QFile, which the hook does not see, as a running
    # reachAQ's writes are not in this process at all.
    file = QFile(path.as_posix())
    assert file.open(QIODevice.OpenModeFlag.WriteOnly)
    file.write(content)
    file.close()


def test_the_guard_ignores_the_operators_app_writing_its_status_file(preferences_folder):
    # A running reachAQ rewrites or removes it at every mode change.
    guard = top_fixtures.data_root_guard
    before = guard.snapshot()
    time.sleep(0.05)  # past the kernel's clock tick, so the folder's mtime moves
    status = preferences_folder.joinpath("autotrainer_running_status.env")
    _write_from_another_process(status, b"status='running'\n")
    assert QFile.remove(status.as_posix())
    _write_from_another_process(status, b"status='calibration_3d'\n")
    assert guard.take_refused() == []
    assert guard.changes_since(before) == []


def test_the_guard_sees_a_lock_left_beside_the_preferences(preferences_folder):
    guard = top_fixtures.data_root_guard
    before = guard.snapshot()
    lock = preferences_folder.joinpath("Auto Trainer.conf.lock")
    _write_from_another_process(lock, b"12345\n")
    assert guard.changes_since(before) == [f"{lock} created"]


def test_a_refusal_after_the_last_tests_check_fails_the_run(guarded_folder):
    # As from a session fixture's teardown: nothing but the session's end sees it.
    with pytest.raises(top_fixtures.DataRootWriteRefused):
        guarded_folder.joinpath("20261002").mkdir()
    lines = []
    reporter = SimpleNamespace(write_line=lambda line, **_markup: lines.append(line))
    session = SimpleNamespace(
        exitstatus=pytest.ExitCode.OK,
        config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter)),
    )
    top_fixtures.pytest_sessionfinish(session, pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("20261002" in line for line in lines), lines
    assert top_fixtures.data_root_guard.take_refused() == []


def test_the_guard_refuses_each_write_under_the_data_folder_and_records_it(
    guarded_folder, tmp_path,
):
    session = guarded_folder.joinpath("20261001", "christielab10", "session001")
    before = _listing(guarded_folder)
    writes = {
        "a new day": lambda: guarded_folder.joinpath("20261002").mkdir(),
        "an existing session": lambda: session.mkdir(parents=True, exist_ok=True),
        "a log": lambda: open(session.joinpath("x.log"), "a"),
        "an event file": lambda: session.joinpath("x_events.csv").write_text("x"),
        "a write probe": lambda: tempfile.mkstemp(dir=session),
        "a recording moved out": lambda: os.replace(
            session.joinpath("left.mp4"), tmp_path.joinpath("left.mp4")),
        "a recording removed": lambda: session.joinpath("left.mp4").unlink(),
        "an aborted session removed": lambda: shutil.rmtree(session),
    }
    not_refused = []
    for name, write in writes.items():
        try:
            write()
        except top_fixtures.DataRootWriteRefused:
            continue
        not_refused.append(name)
    assert not_refused == []
    assert _listing(guarded_folder) == before
    assert session.joinpath("left.mp4").read_bytes() == b"recorded"
    refused = top_fixtures.data_root_guard.take_refused()
    assert len(refused) == len(writes), refused
    assert all(str(guarded_folder) in what for what in refused), refused


def test_the_guard_lets_reads_there_and_writes_elsewhere_through(guarded_folder, tmp_path):
    session = guarded_folder.joinpath("20261001", "christielab10", "session001")
    assert session.joinpath("left.mp4").read_bytes() == b"recorded"
    assert [path.name for path in session.iterdir()] == ["left.mp4"]
    elsewhere = tmp_path.joinpath("elsewhere")
    elsewhere.mkdir()
    elsewhere.joinpath("x.log").write_text("x")
    shutil.rmtree(elsewhere)
    # A neighbour whose name starts with the folder's is not under it.
    neighbour = guarded_folder.with_name(guarded_folder.name + "-copy")
    neighbour.mkdir()
    neighbour.rmdir()
    assert top_fixtures.data_root_guard.take_refused() == []


def test_the_guard_refuses_to_start_reachaq_with_this_home(tmp_path):
    # Nothing here can start: neither program exists, so without the guard
    # every Popen raises FileNotFoundError instead.
    python = tmp_path.joinpath("missing", "python").as_posix()
    no_home = {name: value for name, value in os.environ.items() if name != "HOME"}
    refused = (
        ([python, "-m", "tools.acquisition.headless"], None),
        ([python, "/opt/reachAQ/tools/acquisition/gui.py"], None),
        ([python, "-m", "reachAQ.app"], dict(os.environ)),
        ([tmp_path.joinpath("missing", "reachaq").as_posix()], no_home),
    )
    allowed = (
        ([python, "-m", "tools.acquisition.headless"],
         dict(os.environ, HOME=tmp_path.joinpath("home").as_posix())),
        ([python, "-c", "from tools.acquisition.instance_lock import acquire_instance_lock"],
         None),
    )
    for command, env in refused:
        with pytest.raises(top_fixtures.DataRootWriteRefused):
            subprocess.Popen(command, env=env)
    for command, env in allowed:
        with pytest.raises(FileNotFoundError):
            subprocess.Popen(command, env=env)
    assert len(top_fixtures.data_root_guard.take_refused()) == len(refused)


def test_a_refused_write_fails_the_test_that_made_it(guarded_folder):
    with pytest.raises(top_fixtures.DataRootWriteRefused):
        guarded_folder.joinpath("20261002").mkdir()
    with pytest.raises(pytest.fail.Exception, match="20261002"):
        top_fixtures.data_root_guard.fail_on_refusals()
    # The failure consumed the refusal, so this test's own check passes.
    assert top_fixtures.data_root_guard.take_refused() == []
