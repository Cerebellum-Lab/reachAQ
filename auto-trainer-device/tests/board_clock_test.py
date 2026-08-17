import pytest

from autotrainer.device.board_clock import BoardClockModel, BoardSequenceTracker


def test_board_clock_maps_offset_and_drift_with_bounded_uncertainty():
    model = BoardClockModel()
    for request_id in range(8):
        board_us = 10_000_000 + request_id * 5_000_000
        host_mid = 100.0 + (board_us / 1e6) * 1.00001
        model.observe(
            request_id=request_id,
            host_send_perf_ns=int((host_mid - 0.001) * 1e9),
            host_receive_perf_ns=int((host_mid + 0.001) * 1e9),
            board_receive_time_us=board_us - 100,
            board_send_time_us=board_us + 100,
            boot_id=7,
        )

    aligned = model.align(30_000_000)
    assert aligned["perf_time"] == pytest.approx(130.0003, abs=1e-6)
    assert aligned["uncertainty_seconds"] <= 0.001
    assert model.estimate.observation_count == 8


def test_board_reboot_invalidates_old_model_and_far_extrapolation():
    model = BoardClockModel(max_extrapolation_seconds=1)
    first = model.observe(
        request_id=1, host_send_perf_ns=1_000_000_000,
        host_receive_perf_ns=1_002_000_000, board_receive_time_us=100,
        board_send_time_us=200, boot_id=1,
    )
    second = model.observe(
        request_id=2, host_send_perf_ns=2_000_000_000,
        host_receive_perf_ns=2_002_000_000, board_receive_time_us=100,
        board_send_time_us=200, boot_id=2,
    )
    assert second.generation == first.generation + 1
    assert second.observation_count == 1
    assert model.align(10_000_000) is None


def test_board_sequence_reports_gaps_duplicates_and_reboot():
    tracker = BoardSequenceTracker()
    assert tracker.observe(1, 10)["gap"] == 0
    assert tracker.observe(1, 12)["gap"] == 1
    assert tracker.observe(1, 12)["duplicate"]
    assert tracker.observe(1, 11)["backward"]
    assert tracker.observe(2, 1)["reboot"]
