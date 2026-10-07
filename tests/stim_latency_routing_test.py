"""Which stim route records its latency row where, and that each trigger gets one.

A direct-NI trigger reaches the GUI twice: as the STIM_CAMERA_TRIGGER message and
as the receiver's result. A STIM3 trigger reaches it only as the message. The
dispatch row is taken from the result on the first route and from the message on
the second, so a handler that records on both would count every direct trigger
twice, and one that records on neither would lose a route's rows silently.

The handlers are driven on a bare AppModel with a real LatencyEventLog: what is
under test is which handler writes which stamp into which column, not the session.
"""

import math
import time
from types import SimpleNamespace

import pytest

from autotrainer.core.latency import LATENCY_RECORD_ENV_VAR
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.latency_event_log import LatencyEventLog
from tools.acquisition.model.stim_latency_budget import StimLatencyBudget


@pytest.fixture
def harness(monkeypatch):
    # LatencyEventLog.begin() reads this switch, and the rig's shell may set it.
    monkeypatch.delenv(LATENCY_RECORD_ENV_VAR, raising=False)
    model = object.__new__(AppModel)
    log = LatencyEventLog()
    log.begin()
    captured = []
    model._session_data_recorder = SimpleNamespace(
        latency_events=log,
        capture_external_event=lambda *args, **kwargs: captured.append((args, kwargs)) or True,
    )
    # What _on_direct_stim_trigger_result reaches after the dispatch row.
    model._stim_latency_budget = StimLatencyBudget()
    model._session_telemetry = SimpleNamespace(record_stim_p99=lambda p99_ms: None)
    # With no camera, _on_stim_camera_trigger returns at its unowned-trigger
    # guard, after the dispatch row and before any session state is touched.
    model._stim_camera = None
    model._recording_session = SimpleNamespace(token=lambda: None, status=None)
    model._trial_action_executor = SimpleNamespace(operation=None)
    return SimpleNamespace(model=model, log=log, captured=captured)


def _direct_result(base):
    """One accepted direct-route result, every stage at its own offset from base."""
    return {
        "operation_id": "op-1", "trigger_route": "direct_ni_software",
        "stim_frame_id": 9, "accepted": True,
        "frame_perf_time": base,
        "decision_perf_time": base + 0.001,
        "evidence_done_perf_time": base + 0.002,
        "clip_done_perf_time": base + 0.003,
        "ipc_send_perf_time": base + 0.004,
        "ipc_receive_perf_time": base + 0.005,
        "validated_perf_time": base + 0.006,
        "daqmx_start_entry_perf_time": base + 0.007,
        "daqmx_start_return_perf_time": base + 0.008,
    }


def _stim3_message(base):
    """The STIM_CAMERA_TRIGGER payload the capture process sends for STIM3."""
    return {
        "operation_id": "op-2", "trigger_route": "hardware_stim3",
        "stim_frame_id": 5, "frame_perf_time": base,
        "decision_perf_time": base + 0.001,
        "evidence_done_perf_time": base + 0.002,
        "clip_done_perf_time": math.nan,
        "msg_send_perf_time": base + 0.003,
    }


def test_a_direct_result_records_one_row_with_every_stamp_in_its_column(harness):
    base = 1000.0
    harness.model._on_direct_stim_trigger_result(_direct_result(base))

    rows = harness.log.end()["stim_dispatch"]
    assert len(rows) == 1
    row = rows[0]
    assert row["operation_id"] == b"op-1"
    assert row["route"] == b"direct_ni_software"
    assert row["stim_frame_id"] == 9
    # Each stamp sits at a distinct offset, so a swapped or dropped column shows.
    assert [row[name] for name in (
        "frame_perf", "decision_perf", "evidence_done_perf", "clip_done_perf",
        "send_perf", "gui_recv_perf", "validated_perf", "start_entry_perf",
        "start_return_perf",
    )] == [base + offset / 1000 for offset in range(9)]
    assert row["accepted"] == 1
    assert len(harness.captured) == 1


def test_a_failing_dispatch_row_does_not_stop_the_direct_handler(harness):
    def refuse(*args, **kwargs):
        raise RuntimeError("row could not be built")

    harness.model._session_data_recorder.latency_events = SimpleNamespace(
        record_stim_dispatch=refuse)

    harness.model._on_direct_stim_trigger_result(_direct_result(1000.0))

    # The event ledger entry is the trigger's own record; losing the latency row
    # must not cost it.
    assert len(harness.captured) == 1
    assert harness.captured[0][0][0] == "stimCameraDirectNiStart"


def test_a_stim3_message_records_one_row_stamped_on_gui_receive(harness):
    base = 1000.0
    before = time.perf_counter()
    harness.model._on_stim_camera_trigger(2, _stim3_message(base))
    after = time.perf_counter()

    rows = harness.log.end()["stim_dispatch"]
    assert len(rows) == 1
    row = rows[0]
    assert row["operation_id"] == b"op-2"
    assert row["route"] == b"hardware_stim3"
    assert row["send_perf"] == base + 0.003  # the capture process's msg_send_perf_time
    assert before <= row["gui_recv_perf"] <= after  # taken by the handler, not the message
    assert row["evidence_done_perf"] == base + 0.002
    assert math.isnan(row["clip_done_perf"])
    # The STIM3 route has no NI start to time, and no accepted flag to report.
    assert math.isnan(row["validated_perf"])
    assert math.isnan(row["start_entry_perf"])
    assert math.isnan(row["start_return_perf"])
    assert row["accepted"] == -1


def test_a_direct_route_message_records_no_row(harness):
    message = {**_stim3_message(1000.0), "trigger_route": "direct_ni_software"}

    harness.model._on_stim_camera_trigger(2, message)

    assert len(harness.log.end()["stim_dispatch"]) == 0


def test_one_direct_trigger_is_one_row_however_its_two_hops_arrive(harness):
    # Both hops reach the GUI for a direct trigger, in either order.
    message = {**_direct_result(1000.0), "msg_send_perf_time": 1000.004}
    for first, second in (
        (lambda: harness.model._on_stim_camera_trigger(2, message),
         lambda: harness.model._on_direct_stim_trigger_result(_direct_result(1000.0))),
        (lambda: harness.model._on_direct_stim_trigger_result(_direct_result(1000.0)),
         lambda: harness.model._on_stim_camera_trigger(2, message)),
    ):
        harness.log.begin()
        first()
        second()
        rows = harness.log.end()["stim_dispatch"]
        assert len(rows) == 1
        assert rows[0]["route"] == b"direct_ni_software"
        assert rows[0]["accepted"] == 1  # the row is the result's, with the NI start


def test_the_stim3_pulse_is_timed_around_the_board_call(harness):
    seen = {}

    def pulse_stim(pulse_us, stim_line):
        seen["inside"] = time.perf_counter()
        seen["args"] = (pulse_us, stim_line)
        return "tok-1"

    harness.model._hardware = SimpleNamespace(
        pulse_stim=pulse_stim,
        wait_pending_command_acked=lambda token, timeout: None,
    )
    recipe = SimpleNamespace(
        laser_firing=SimpleNamespace(trigger_pulse_us=1000, stim_line=3),
        operation_id="op-3",
    )

    before = time.perf_counter()
    harness.model._trigger_protocol_stim3(None, recipe, "stim-camera frame 5")
    after = time.perf_counter()

    rows = harness.log.end()["stim3_pulses"]
    assert len(rows) == 1
    row = rows[0]
    assert row["operation_id"] == b"op-3"
    assert row["token"] == b"tok-1"
    assert seen["args"] == (1000, 3)
    # The call stamp precedes the board call and the return stamp follows it.
    assert before <= row["pulse_call_perf"] <= seen["inside"] <= row["pulse_return_perf"] <= after
