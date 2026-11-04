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


def _exchange(model, request_id, board_us, host_mid, *, rtt=0.0006, boot_id=7):
    return model.observe(
        request_id=request_id,
        host_send_perf_ns=int((host_mid - rtt / 2) * 1e9),
        host_receive_perf_ns=int((host_mid + rtt / 2) * 1e9),
        board_receive_time_us=board_us - 50,
        board_send_time_us=board_us + 50,
        boot_id=boot_id,
    )


def test_closely_spaced_noisy_exchanges_fit_an_offset_and_never_raise():
    # reachAQ's open() sends 4 exchanges a few ms apart; host receive times are
    # quantized by 5 ms read windows, so a slope fitted to them is noise.
    model = BoardClockModel()
    noise = (0.0, 0.0009, -0.0007, 0.0012)
    for request_id, jitter in enumerate(noise):
        board_us = 2_000_000 + request_id * 5_000
        estimate = _exchange(model, request_id, board_us, 50.0 + board_us / 1e6 + jitter)

    assert estimate.slope == 1.0
    assert not estimate.slope_fitted
    assert model.align(2_010_000)["perf_time"] == pytest.approx(52.01, abs=0.002)


def test_slope_is_fitted_once_the_board_span_is_long_enough():
    model = BoardClockModel()
    for request_id in range(4):
        board_us = 1_000_000 + request_id * 4_000_000  # 12 s span
        estimate = _exchange(model, request_id, board_us, 10.0 + (board_us / 1e6) * 1.00002)

    assert estimate.slope_fitted
    assert estimate.slope == pytest.approx(1.00002, abs=2e-6)


def test_implausible_slope_falls_back_to_an_offset_instead_of_raising():
    model = BoardClockModel()
    for request_id in range(4):
        board_us = 1_000_000 + request_id * 5_000_000
        estimate = _exchange(model, request_id, board_us, 10.0 + (board_us / 1e6) * 1.01)

    assert estimate.slope == 1.0
    assert not estimate.slope_fitted
