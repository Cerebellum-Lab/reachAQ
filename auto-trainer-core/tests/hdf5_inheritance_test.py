import logging
import os
import subprocess
from types import SimpleNamespace

import h5py
import pytest

from autotrainer.core.hdf5_inheritance import mark_close_on_exec

posix_only = pytest.mark.skipif(os.name != "posix", reason="flock and exec inheritance are POSIX")


def _write_and_start_a_child(path, mark):
    """Open `path` for writing, mark it or not, start an exec'd child that inherits every
    inheritable descriptor (as NI-DAQmx's CEIP launcher does), then close the file."""
    store = h5py.File(path, "w")
    if mark:
        assert mark_close_on_exec(store) is True
    store["values"] = [1.0, 2.0]
    child = subprocess.Popen(["sleep", "60"], close_fds=False)
    store.close()
    return child


@posix_only
def test_a_child_holds_the_lock_of_an_unmarked_file_but_not_of_a_marked_one(tmp_path, monkeypatch):
    # christielab10, 2026-11-13: HDF5 opens without O_CLOEXEC and takes flock on the open file
    # description, which a forked child shares and HDF5's close does not unlock.
    monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)
    children = []
    try:
        children.append(_write_and_start_a_child(tmp_path / "unmarked.h5", mark=False))
        with pytest.raises(BlockingIOError):
            h5py.File(tmp_path / "unmarked.h5", "r")

        children.append(_write_and_start_a_child(tmp_path / "marked.h5", mark=True))
        with h5py.File(tmp_path / "marked.h5", "r") as store:
            assert store["values"][()].tolist() == [1.0, 2.0]
    finally:
        for child in children:
            child.kill()
            child.wait()


@posix_only
def test_a_marked_file_keeps_working(tmp_path):
    with h5py.File(tmp_path / "store.h5", "w") as store:
        assert mark_close_on_exec(store) is True
        assert not os.get_inheritable(store.id.get_vfd_handle())
        store["values"] = [3.0]
    with h5py.File(tmp_path / "store.h5", "r") as store:
        assert store["values"][()].tolist() == [3.0]


def test_a_store_without_a_descriptor_is_logged_and_left_usable(caplog):
    def no_handle():
        raise ValueError("this driver has no single descriptor")

    store = SimpleNamespace(id=SimpleNamespace(get_vfd_handle=no_handle), filename="core.h5")
    with caplog.at_level(logging.WARNING, logger="autotrainer.core.hdf5_inheritance"):
        assert mark_close_on_exec(store) is False
    if os.name == "posix":
        assert "core.h5" in caplog.text
