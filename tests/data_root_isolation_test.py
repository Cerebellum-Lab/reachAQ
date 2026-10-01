"""No test writes into the operator's data folder.

The full runs on christielab10 on 2026-10-01 left 1,533 logs, two hourly event
files and two empty session folders in ~/Documents/rawdatalocal/20261001/
christielab10, the folder its real sessions are recorded into. AppModel opened
them under its default data folder while it was constructed, before any
configuration's outputLocation was loaded.

These tests move HOME under tmp_path to stand in for the operator's, so even
against code that leaks they write nothing outside pytest's tmp.
"""
import time
from pathlib import Path

import pytest

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
