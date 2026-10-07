"""The live stim p99 belongs to one session, and its failures are its own.

The app-lifetime StimLatencyBudget rolls over the last 512 triggers across every
session since launch. Feeding the panel and the session metadata from it put an
earlier session's slow triggers into the next session's figure. These drive the
real handler on a bare AppModel, with real budgets and a real SessionTelemetry,
because what is under test is which budget feeds which readout and when it is
cleared, not the session.
"""

import math
from types import SimpleNamespace
from unittest import mock

import pytest

from tools.acquisition.model import app_model as app_model_module
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.session_telemetry import SessionTelemetry
from tools.acquisition.model.stim_latency_budget import StimLatencyBudget


def _trigger(total_seconds, base=1000.0):
    """One accepted direct-route result spanning total_seconds, frame to DAQmx return."""
    return {
        "accepted": True, "stim_frame_id": 1,
        "frame_perf_time": base,
        "decision_perf_time": base + total_seconds * 0.1,
        "ipc_send_perf_time": base + total_seconds * 0.2,
        "ipc_receive_perf_time": base + total_seconds * 0.5,
        "daqmx_start_entry_perf_time": base + total_seconds * 0.9,
        "daqmx_start_return_perf_time": base + total_seconds,
    }


@pytest.fixture
def model():
    item = object.__new__(AppModel)
    item.captured = []
    item._session_telemetry = SessionTelemetry()
    item._stim_latency_budget = StimLatencyBudget()
    item._session_stim_latency_budget = StimLatencyBudget()
    item._session_data_recorder = SimpleNamespace(
        latency_events=SimpleNamespace(record_stim_dispatch=lambda payload: None),
        capture_external_event=lambda *args, **kwargs: item.captured.append(args) or True,
    )
    item._begin_session_telemetry(0.0)
    return item


def test_a_new_session_starts_its_stim_p99_from_nothing(model):
    for _ in range(3):
        model._on_direct_stim_trigger_result(_trigger(0.020))
    assert model._session_telemetry.stim_p99_ms == pytest.approx(20.0)

    model._begin_session_telemetry(2000.0)
    assert math.isnan(model._session_telemetry.stim_p99_ms)

    # Read off the app-lifetime window (2, 20, 20, 20 ms) this would say 20 ms.
    model._on_direct_stim_trigger_result(_trigger(0.002))
    assert model._session_telemetry.stim_p99_ms == pytest.approx(2.0)
    assert model._session_telemetry.summary()["stimP99Ms"] == pytest.approx(2.0)

    # The session boundary leaves the app-lifetime budget, which the capture-stop
    # report reads, exactly as it was.
    assert model._stim_latency_budget.summary().accepted == 4


def test_a_failing_p99_feed_is_logged_once_per_session_and_never_as_a_dispatch_row(
        model, monkeypatch):
    log = mock.MagicMock()
    monkeypatch.setattr(app_model_module, "logger", log)

    def refuse():
        raise RuntimeError("summary failed")

    model._session_stim_latency_budget = SimpleNamespace(
        observe=lambda payload: None, summary=refuse, reset=lambda: None)

    for _ in range(3):
        model._on_direct_stim_trigger_result(_trigger(0.002))

    # Every trigger still reaches the event ledger, and the failure is said once.
    assert len(model.captured) == 3
    messages = [call.args[0] for call in log.exception.call_args_list]
    assert len(messages) == 1
    assert "stim p99" in messages[0].lower()
    assert "dispatch latency row" not in messages[0]

    # A new session gets its own chance to report it.
    model._begin_session_telemetry(2000.0)
    model._on_direct_stim_trigger_result(_trigger(0.002))
    assert len(log.exception.call_args_list) == 2
