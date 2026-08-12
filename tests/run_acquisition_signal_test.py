from unittest import mock

from tools.acquisition import run_acquisition


def test_owned_child_termination_never_signals_a_process_group(monkeypatch):
    alive = mock.Mock(pid=1234)
    alive.is_alive.return_value = True
    stopped = mock.Mock(pid=4321)
    stopped.is_alive.return_value = False
    monkeypatch.setattr(
        run_acquisition.multiprocessing,
        "active_children",
        lambda: [alive, stopped],
    )

    assert run_acquisition._terminate_owned_children() == (1234,)
    alive.terminate.assert_called_once_with()
    alive.kill.assert_not_called()
    stopped.terminate.assert_not_called()

    assert run_acquisition._terminate_owned_children(force=True) == (1234,)
    alive.kill.assert_called_once_with()
