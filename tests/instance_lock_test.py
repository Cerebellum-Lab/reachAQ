import os
import subprocess
import sys
import textwrap

import pytest

from tools.acquisition import instance_lock
from tools.acquisition.instance_lock import AlreadyRunning, acquire_instance_lock

pytestmark = pytest.mark.skipif(instance_lock.fcntl is None, reason="flock is POSIX only")


def test_second_holder_is_refused_and_named(tmp_path):
    path = tmp_path / "reachaq" / "app.lock"
    first = acquire_instance_lock(path)
    try:
        with pytest.raises(AlreadyRunning) as refused:
            acquire_instance_lock(path)
        assert refused.value.holder == str(os.getpid())
        assert str(os.getpid()) in str(refused.value)
    finally:
        first.close()


def test_lock_is_free_again_once_released(tmp_path):
    path = tmp_path / "reachaq" / "app.lock"
    acquire_instance_lock(path).close()
    again = acquire_instance_lock(path)
    assert again is not None
    again.close()


def test_a_holder_that_dies_does_not_leave_a_stale_lock(tmp_path):
    path = tmp_path / "reachaq" / "app.lock"
    holder = textwrap.dedent(f"""
        import os
        from tools.acquisition.instance_lock import acquire_instance_lock
        lock = acquire_instance_lock({str(path)!r})
        os._exit(0)  # no clean shutdown; the kernel must still release it
    """)
    subprocess.run([sys.executable, "-c", holder], check=True, cwd=os.getcwd())
    assert path.read_text().strip()  # it did hold the lock
    acquire_instance_lock(path).close()


def test_default_path_uses_an_owned_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert instance_lock.default_lock_path() == tmp_path / "reachaq" / "app.lock"


def test_default_path_falls_back_without_a_runtime_dir(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    path = instance_lock.default_lock_path()
    assert path.parent.parent.name == f"reachaq-runtime-{os.getuid()}"
    assert path.name == "app.lock"


def test_symlinked_lock_directory_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "reachaq").symlink_to(target)
    with pytest.raises(RuntimeError, match="symbolic link"):
        acquire_instance_lock(tmp_path / "runtime" / "reachaq" / "app.lock")
