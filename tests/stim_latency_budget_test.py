import math

import pytest

from tools.acquisition.model.stim_latency_budget import (
    STIM_LATENCY_STAGES,
    StimLatencyBudget,
    percentile,
)


def _record(
    *,
    frame=1.000000,
    decision=1.000200,
    send=1.000250,
    receive=1.000900,
    daq_entry=1.001000,
    daq_return=1.001400,
    accepted=True,
    camera_timestamp_ns=500_000_000,
):
    """One complete direct stim-trigger record, shaped like laser_model emits."""
    return {
        "accepted": accepted,
        "camera_timestamp_ns": camera_timestamp_ns,
        "frame_perf_time": frame,
        "decision_perf_time": decision,
        "ipc_send_perf_time": send,
        "ipc_receive_perf_time": receive,
        "daqmx_start_entry_perf_time": daq_entry,
        "daqmx_start_return_perf_time": daq_return,
        "stim_frame_id": 42,
    }


def test_percentile_handles_empty_single_and_interpolated():
    assert math.isnan(percentile([], 0.5))
    assert percentile([0.25], 0.99) == pytest.approx(0.25)
    # p50 of 1..4 interpolates between the two middle samples.
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == pytest.approx(1.0)
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == pytest.approx(4.0)
    # Ordering is applied internally, so unsorted input is fine.
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)


def test_percentile_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 1.5)


def test_complete_record_produces_every_stage_and_total():
    budget = StimLatencyBudget()
    budget.observe(_record())
    summary = budget.summary()

    assert summary.accepted == 1
    assert summary.rejected == 0
    assert summary.incomplete == 0
    assert {stage.key for stage in summary.stages} == {
        stage.key for stage in STIM_LATENCY_STAGES
    }

    stages = {stage.key: stage for stage in summary.stages}
    assert stages["capture_to_decision"].p50 == pytest.approx(0.000200)
    assert stages["queue_put"].p50 == pytest.approx(0.000050)
    assert stages["ipc"].p50 == pytest.approx(0.000650)
    assert stages["daq_dispatch"].p50 == pytest.approx(0.000100)
    assert stages["daq_start"].p50 == pytest.approx(0.000400)

    assert summary.total is not None
    assert summary.total.p50 == pytest.approx(0.001400)
    # 1.4 ms sits inside the 5 ms default budget.
    assert summary.meets_budget is True
    assert summary.over_budget == 0


def test_rejected_triggers_are_counted_but_excluded_from_latency():
    budget = StimLatencyBudget()
    # A rejected trigger never reached DAQmx, so it must not look like a fast loop.
    budget.observe(_record(accepted=False, daq_return=1.000300))
    summary = budget.summary()

    assert summary.rejected == 1
    assert summary.accepted == 0
    assert summary.total is None
    assert summary.meets_budget is False
    assert all(stage.count == 0 for stage in summary.stages)


def test_over_budget_total_is_counted_and_fails_the_verdict():
    budget = StimLatencyBudget(budget_seconds=0.005)
    budget.observe(_record(daq_return=1.020000))
    summary = budget.summary()

    assert summary.over_budget == 1
    assert summary.total is not None
    assert summary.total.p99 == pytest.approx(0.020000)
    assert summary.meets_budget is False


def test_missing_stamp_marks_record_incomplete_but_keeps_other_stages():
    budget = StimLatencyBudget()
    record = _record()
    del record["daqmx_start_return_perf_time"]
    budget.observe(record)
    summary = budget.summary()

    assert summary.accepted == 1
    assert summary.incomplete == 1
    assert summary.total is None
    assert summary.meets_budget is False

    stages = {stage.key: stage for stage in summary.stages}
    # The stages that do not depend on the missing stamp still recorded.
    assert stages["capture_to_decision"].count == 1
    assert stages["ipc"].count == 1
    assert stages["daq_start"].count == 0


def test_non_monotonic_span_is_discarded_rather_than_reported():
    budget = StimLatencyBudget()
    # decision before frame arrival means the stamps are not one clock domain.
    budget.observe(_record(decision=0.999900))
    summary = budget.summary()

    stages = {stage.key: stage for stage in summary.stages}
    assert stages["capture_to_decision"].count == 0
    assert summary.incomplete == 1


def test_window_is_bounded():
    budget = StimLatencyBudget(window=3)
    for index in range(10):
        offset = index * 0.001
        budget.observe(_record(
            frame=1.0 + offset,
            decision=1.000200 + offset,
            send=1.000250 + offset,
            receive=1.000900 + offset,
            daq_entry=1.001000 + offset,
            daq_return=1.001400 + offset,
        ))
    summary = budget.summary()

    assert summary.accepted == 10
    assert summary.total is not None
    # Counters keep the session total; the percentile window stays bounded.
    assert summary.total.count == 3
    assert all(stage.count == 3 for stage in summary.stages)


def test_observe_never_raises_on_malformed_records():
    budget = StimLatencyBudget()
    malformed = (
        {},                                              # no accepted key -> rejected
        {"accepted": False},                             # explicit reject
        {"accepted": True},                              # accepted, no stamps
        {"accepted": True, "frame_perf_time": "x"},      # accepted, unparseable stamp
    )
    for record in malformed:
        budget.observe(record)
    summary = budget.summary()

    assert summary.total is None
    # Records without a truthy accepted flag are rejects, not fast loops.
    assert summary.rejected == 2
    # Accepted records with unusable stamps are accepted-but-incomplete.
    assert summary.accepted == 2
    assert summary.incomplete == 2


def test_camera_clock_offset_is_tracked_but_not_latency():
    budget = StimLatencyBudget()
    budget.observe(_record(camera_timestamp_ns=500_000_000))
    budget.observe(_record(camera_timestamp_ns=500_000_000 + 2_000_000))
    summary = budget.summary()

    # Offsets are a drift indicator only; they never enter a stage or the total.
    assert summary.camera_clock_offset_span_seconds == pytest.approx(0.002, abs=1e-6)
    assert summary.total is not None
    assert summary.total.p50 == pytest.approx(0.001400)


def test_reset_clears_window_and_counters():
    budget = StimLatencyBudget()
    budget.observe(_record())
    budget.reset()
    summary = budget.summary()

    assert summary.accepted == 0
    assert summary.rejected == 0
    assert summary.incomplete == 0
    assert summary.over_budget == 0
    assert summary.total is None


def test_invalid_construction_is_rejected():
    with pytest.raises(ValueError):
        StimLatencyBudget(budget_seconds=0)
    with pytest.raises(ValueError):
        StimLatencyBudget(budget_seconds=math.inf)
    with pytest.raises(ValueError):
        StimLatencyBudget(window=0)


def test_format_report_includes_verdict_and_stage_rows():
    budget = StimLatencyBudget()
    budget.observe(_record())
    report = budget.format_report()

    assert "Tier 1 stim loop latency" in report
    for stage in STIM_LATENCY_STAGES:
        assert stage.key in report
    assert "within budget" in report

    over = StimLatencyBudget(budget_seconds=0.001)
    over.observe(_record())
    assert "OVER BUDGET at p99" in over.format_report()
