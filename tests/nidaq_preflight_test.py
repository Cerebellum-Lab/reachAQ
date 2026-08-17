from types import SimpleNamespace

from tools.acquisition.model.nidaq_preflight import (
    NidaqPreflightResult,
    _preflight_worker,
)


class _Queue:
    def __init__(self):
        self.value = None

    def put(self, value):
        self.value = value


def test_exact_preflight_verifies_commits_and_closes(monkeypatch):
    calls = []

    class Controller:
        def __init__(self, configuration, *, timing_plan):
            calls.append(("create", configuration, timing_plan))

        def verify_tasks(self, *, commit):
            calls.append(("verify", commit))
            return ("Dev1.ai",)

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        "tools.acquisition.model.nidaq_preflight.NidaqSignalStreamController",
        Controller,
    )
    result_queue = _Queue()
    _preflight_worker("configuration", "plan", result_queue)

    assert result_queue.value == NidaqPreflightResult(
        "verified", verified_tasks=("Dev1.ai",)
    )
    assert calls == [
        ("create", "configuration", "plan"),
        ("verify", True),
        ("close",),
    ]


def test_exact_preflight_preserves_first_create_error(monkeypatch):
    class DaqError(RuntimeError):
        error_code = -200077

    class Controller:
        def __init__(self, *_args, **_kwargs):
            raise DaqError("route unavailable")

    monkeypatch.setattr(
        "tools.acquisition.model.nidaq_preflight.NidaqSignalStreamController",
        Controller,
    )
    result_queue = _Queue()
    _preflight_worker(None, None, result_queue)
    result = result_queue.value
    assert not result.is_valid
    assert result.error == "route unavailable"
    assert result.daqmx_code == -200077
