import importlib
import os
import sys
from datetime import datetime, timezone, timedelta
from unittest import mock

import pytest

from autotrainer.core.project import ProjectInfo
from autotrainer.core.project.project_info import REACH_EVENT_SUFFIX

device_id = "A1357"


@pytest.fixture
def root(tmp_path) -> str:
    return tmp_path.joinpath("output").as_posix()


def test_today(root):
    when = datetime(2023, 6, 8)
    info = ProjectInfo(root=root, device_id=device_id, when=when)
    (location, today) = info.get_day_path()
    assert today == "20230608"
    assert location == os.path.join(root, "20230608", "A1357")


def test_hourly(root):
    when = datetime(2023, 6, 8, 8, 45, 23)
    info = ProjectInfo(root=root, device_id=device_id, when=when)
    hourly_source = info.get_interval_path("camera-1")
    assert hourly_source.location == os.path.join(root, "20230608", "A1357")
    assert hourly_source.prefix == f"20230608_A1357_h08_camera-1"
    assert hourly_source.interval == 8


def test_implicit_session(root):
    when = datetime(2024, 6, 8, 8, 45, 23)
    info = ProjectInfo(root=root, device_id=device_id, when=when)
    session_source = info.get_session_path("camera-1")
    assert session_source.location == os.path.join(root, "20240608", "A1357", "session001")
    assert session_source.prefix == f"20240608_A1357_session001_camera-1"
    assert session_source.session_index == 1
    # Would normally happen through calculate_next_session_index, but this uses the filesystem
    info.session = 9
    session_source = info.get_session_path("camera-1")
    assert session_source.location == os.path.join(root, "20240608", "A1357", "session009")
    assert session_source.prefix == f"20240608_A1357_session009_camera-1"
    assert session_source.session_index == 9


def test_explicit_session(root):
    when = datetime(2023, 6, 8, 8, 45, 23)
    info = ProjectInfo(root=root, device_id=device_id, when=when)
    session_source = info.get_session_path("camera-1", session=12)
    assert session_source.location == os.path.join(root, "20230608", "A1357", "session012")
    assert session_source.prefix == f"20230608_A1357_session012_camera-1"
    assert session_source.session_index == 12


def test_camera_names_preserve_legacy_camera_fields(root):
    info = ProjectInfo(root=root, camera_names=("left", "right", "camera3"))

    assert info.camera_1 == "left"
    assert info.camera_2 == "right"
    assert info.camera_names == ("left", "right", "camera3")
    assert info.to_local_value().camera_names == info.camera_names


def test_without_session_and_when_are_shared(root):
    module = importlib.import_module(ProjectInfo.__module__)
    unix_start_as_local = datetime(2001, 1, 1, tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    with mock.patch.object(module, "_get_datetime_now") as m_get_datetime:
        m_get_datetime.return_value = unix_start_as_local
        info = ProjectInfo(root=root, device_id=device_id)
    assert info.session == 1
    assert info.when == unix_start_as_local
    info.calculate_next_session_index()
    assert info.when != unix_start_as_local
    assert info.session == 1
    assert hasattr(info._when, "acquire")  # noqa
    assert hasattr(info._session, "acquire")  # noqa


def test_get_path_at_different_day_does_not_change_result(root):
    info = ProjectInfo(root=root, device_id=device_id)
    some_day = datetime(2025, 1, 1)
    other_day = datetime(2025, 1, 2)
    module = importlib.import_module(ProjectInfo.__module__)
    with mock.patch.object(module, "_get_datetime_now") as m_datetime_now:
        m_datetime_now.return_value = some_day
        info.calculate_next_session_index()
        loc1, d1 = info.get_day_path()
        assert d1 == "20250101"
        # now change the day:
        m_datetime_now.return_value = other_day
        # and get path again:
        loc2, d2 = info.get_day_path()
        assert loc1 == loc2 and d1 == d2, "must be same still"
        # but with a new calculate_next_session_index:
        info.calculate_next_session_index()
        loc3, d3 = info.get_day_path()
        assert loc3 != loc2 and d3 != d2, "must be different *AFTER* calculate_next_session_index"
        assert d3 == "20250102"


@pytest.mark.skipif(os.name != "posix", reason="disabled on non-posix")
def test_attached_and_detached_can_be_compared(root):
    prj = ProjectInfo(root=root)
    detached = prj.to_local_value()
    assert prj == detached
    detached.session += 1
    assert prj != detached
    prj.session = detached.session
    assert prj == detached


@pytest.mark.skipif(os.name != "posix", reason="disabled on non-posix")
def test_can_use_with_with_statement(root):
    prj = ProjectInfo(root=root)
    before = prj.to_local_value()
    with prj:
        pass
    after = prj.to_local_value()
    assert after == before


@pytest.mark.skipif(os.name != "posix", reason="disabled on non-posix")
def test_fast_path(root):
    prj = ProjectInfo(root=root)
    prj.calculate_next_session_index()
    assert prj.session == 1
    prj.calculate_next_session_index()
    assert prj.session == 2
    local_prj = prj.to_local_value()
    local_prj.calculate_next_session_index()
    assert local_prj.session == 3
    assert prj.session == 2  # still
    prj.calculate_next_session_index()
    assert prj.session == 4, "slow path taken"


def test_new_day_if_output_dir_exists(root, monkeypatch):
    now = datetime.now()
    prj = ProjectInfo(root=root, when=now, session=100)
    m = mock.MagicMock()
    from autotrainer.core.project.project_info import _get_datetime_now
    monkeypatch.setattr(f"{_get_datetime_now.__module__}."
                        f"{_get_datetime_now.__qualname__}", m)
    tomorrow = now + timedelta(days=1)
    m.return_value = tomorrow
    prj.calculate_next_session_index()
    assert prj.when == tomorrow
    assert prj.session == 1
    # now retry again, should get 2:
    prj = ProjectInfo(root=root, when=now, session=100)
    prj.calculate_next_session_index()
    assert prj.when == tomorrow
    assert prj.session == 2


def test_reach_event(project_info):
    p = project_info.get_reach_event_path()
    assert str(p).endswith(REACH_EVENT_SUFFIX)
