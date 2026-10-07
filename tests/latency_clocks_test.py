import numpy as np
import pytest

from tools.latency import clocks


def test_fit_line_recovers_offset_and_drift_and_rejects_an_outlier():
    rng = np.random.default_rng(1)
    x = np.linspace(1000.0, 2000.0, 500)
    y = 5.0 + (1 + 20e-6) * x + rng.normal(0, 1e-6, x.size)
    y[250] += 0.01

    fit = clocks.fit_line(x, y, max_residual=1e-4)

    assert fit.valid, fit.reason
    assert fit.drift == pytest.approx(1 + 20e-6, rel=1e-7)
    assert fit.map(1500.0) == pytest.approx(5.0 + (1 + 20e-6) * 1500.0, abs=1e-5)
    assert fit.points == 499


def test_too_few_points_is_invalid():
    fit = clocks.fit_line(np.arange(5.0), np.arange(5.0), max_residual=1.0, min_points=10)
    assert not fit.valid
    assert "need 10" in fit.reason


def test_transitions_count_both_edges_after_the_initial_state():
    assert clocks.transitions([0, 0, 1, 1, 0, 1]).tolist() == [2, 4, 5]


def test_rising_edges_find_pulses_on_a_quiet_baseline():
    rng = np.random.default_rng(2)
    values = rng.normal(0, 0.01, 1000)
    values[100:110] += 5.0
    values[600:620] += 5.0

    result = clocks.rising_edges(values, min_swing=0.1)

    assert result.usable
    assert result.positions.tolist() == [100, 600]


def test_a_floating_input_is_unusable():
    rng = np.random.default_rng(3)
    result = clocks.rising_edges(rng.normal(0, 1.0, 1000), min_swing=0.1)
    assert not result.usable
    assert "noise" in result.reason


def test_each_read_stamp_bounds_the_last_sample_of_the_read_before_it():
    # A block's stamp is taken before the blocking read that returns it: 2.0 is
    # when the read of samples 4-6 began, after samples 0-3 were in hand.
    ends, seen = clocks.ni_block_ends(np.arange(10), np.array([1.0] * 4 + [2.0] * 3 + [3.0] * 3))
    assert ends.tolist() == [3, 6]
    assert seen.tolist() == [2.0, 3.0]


def _back_to_back_reads(rate, block, blocks, seed=4):
    """Per-sample NI arrays as the reader stores them, and the true NI-to-host relation.

    Each read returns 0.2-3.2 ms after its block's last sample, the next read
    starts at once, and every block carries the stamp taken before its read.
    """
    rng = np.random.default_rng(seed)

    def truth(t):
        return 100.0 + (1 + 11e-6) * np.asarray(t)

    last = np.arange(1, blocks + 1) * block - 1
    returned = truth(last / rate) + 0.0002 + rng.uniform(0, 0.003, blocks)
    started = np.concatenate([[truth(0.0) - 0.001], returned[:-1]])
    return np.arange(blocks * block), np.repeat(started, block), last / rate, truth


def test_the_envelope_maps_late_by_about_the_minimum_delay_and_never_early():
    index, stamps, t, truth = _back_to_back_reads(1000.0, 50, 2000)

    fit = clocks.fit_ni_to_host(*clocks.ni_block_ends(index, stamps), 1000.0)

    assert fit.valid, fit.reason
    bias = fit.map(t) - truth(t)
    # Pairing a stamp with its own block maps a whole block (50 ms) early.
    assert bias.min() > 0, f"bias.min() = {bias.min()}"
    assert bias.max() < 0.002


def test_a_restarted_ni_task_is_not_fitted():
    # A restarted NI task has backward step in sample index, not increasing.
    # Create 120 reads: first 60 with increasing indices, then restart (jump back), then 60 more.
    end_index = np.concatenate([np.arange(100, 160), np.arange(50, 110)])
    seen = np.arange(120.0)
    fit = clocks.fit_ni_to_host(end_index, seen, 1000.0)
    assert not fit.valid
    assert "not increasing" in fit.reason


def test_ni_fit_rejects_too_few_host_reads():
    # Fewer than 100 host reads are rejected before any processing.
    end_index = np.arange(30)
    seen = np.arange(30.0)
    fit = clocks.fit_ni_to_host(end_index, seen, 1000.0)
    assert not fit.valid
    assert "30 read pairs, need 100" in fit.reason


def test_pairing_picks_the_transition_before_arrival():
    period = 1 / 150
    ids = np.arange(1000, 1300)
    exposure = 50.0 + (ids - 1000) * period
    transition_perf = 50.0 + np.arange(-5, 300) * period  # the window starts 5 exposures early

    pairing = clocks.choose_pairing(ids, exposure + 0.002, transition_perf, period)

    assert pairing.shift == 995  # frame 1000 is transition number 5
    assert pairing.median_lag == pytest.approx(0.002, abs=1e-9)
    assert not pairing.ambiguous


def test_a_lag_too_close_to_zero_is_ambiguous():
    period = 1 / 150
    ids = np.arange(100)
    exposure = 10.0 + ids * period
    pairing = clocks.choose_pairing(ids, exposure + 0.00005, exposure, period)
    assert pairing.ambiguous


def test_camera_to_ni_fit_uses_the_pairing():
    rate = 10_000.0
    period = 1 / 150
    ids = np.arange(200)
    exposure_ni = 0.05 + ids * period
    transition_index = np.round(exposure_ni * rate).astype(np.int64)
    camera_ts_ns = np.round((exposure_ni + 5.0) * 1e9).astype(np.int64)
    pairing = clocks.Pairing(shift=0, median_lag=0.002, agreement=1.0, ambiguous=False)

    fit = clocks.fit_camera_to_ni(ids, camera_ts_ns, transition_index, rate, pairing, period)

    assert fit.valid, fit.reason
    assert fit.map(5.5) == pytest.approx(0.5, abs=1e-4)


def test_wall_map_splits_at_a_clock_step():
    wall = np.arange(0.0, 100.0)
    perf = wall + 1000.0
    perf[50:] += 0.5  # an NTP step

    wall_map = clocks.fit_wall_to_host(wall, perf)

    assert len(wall_map.fits) == 2
    assert wall_map.map(np.array([10.0, 80.0])) == pytest.approx([1010.0, 1080.5])
    # Issue 1: measured data should always win over padded ranges
    assert wall_map.map(np.array([48.5])) == pytest.approx([1048.5])


def test_wall_map_handles_backward_steps_with_nans():
    # Issue 2: real backward wall step causes segment overlap; overlaps are NaN.
    # Segment A: wall 0-49 at perf 1000-1049
    # Segment B: wall 20-70 at perf 1049-1099 (wall goes backward from 49 to 20)
    wall = np.concatenate([np.arange(0.0, 50.0), np.arange(20.0, 70.0)])
    perf = np.concatenate([np.arange(1000.0, 1050.0), np.arange(1049.0, 1099.0)])

    wall_map = clocks.fit_wall_to_host(wall, perf)

    # Walls 20-49 are in both segments' ranges: they should be NaN (ambiguous)
    assert len(wall_map.fits) == 2
    # Value in segment A only (10.0): should map through A
    assert wall_map.map(np.array([10.0]))[0] == pytest.approx(1010.0)
    # Value in overlap (25.0): should be NaN
    assert np.isnan(wall_map.map(np.array([25.0]))[0])
    # 20.5 and 48.5 sit inside the overlap but near its two ends. Sorting by wall
    # time interleaves the segments there into singleton pieces that get
    # dropped, leaving fits A=[0, 20] and B=[49, 69]; with the 2 s padding, 25.0
    # lands in the gap between them and reads NaN by accident, while these two
    # points fall in the padding and map to a value from the wrong segment.
    assert np.isnan(wall_map.map(np.array([20.5, 48.5]))).all()
    # Value in segment B only (65.0): should map through B (slope=1, so 1049+(65-20)=1094)
    assert wall_map.map(np.array([65.0]))[0] == pytest.approx(1094.0)


def test_the_envelope_fit_is_tighter_than_a_plain_fit():
    # Issue 5: bias.max() should be tighter than 0.002
    index, stamps, t, truth = _back_to_back_reads(1000.0, 50, 2000)

    fit = clocks.fit_ni_to_host(*clocks.ni_block_ends(index, stamps), 1000.0)

    assert fit.valid, fit.reason
    bias = fit.map(t) - truth(t)
    assert bias.min() > 0, f"bias.min() = {bias.min()}"
    assert bias.max() < 0.0005, f"bias.max() = {bias.max()}"


def test_pairing_marks_slips_as_ambiguous():
    # Issue 4: a lost transition mid-session drops agreement and marks pairing ambiguous
    period = 1 / 150
    ids = np.arange(200)
    exposure = 50.0 + ids * period
    # Create transitions: continuous for first 100, then drop one mid-session
    transition_perf = 50.0 + np.arange(200) * period
    # Drop transition 100: frames after it are now off by one
    transition_perf = np.delete(transition_perf, 100)

    pairing = clocks.choose_pairing(ids, exposure + 0.002, transition_perf, period)

    assert pairing.ambiguous, f"dropped transition should mark pairing ambiguous: {pairing.reason}"
    # Should be marked ambiguous due to low agreement from the slip
    assert pairing.agreement < 0.95, f"agreement {pairing.agreement} should be < 0.95"


def test_camera_fit_with_nonzero_shift_and_chosen_pairing():
    # Issue 6: test camera fit with nonzero shift, and feed choose_pairing result
    rate = 10_000.0
    period = 1 / 150
    ids = np.arange(1000, 1200)
    # Exposure time in NI seconds for each frame
    exposure_ni = 0.05 + (ids - 1000) * period
    # Camera timestamps (5 seconds ahead of exposure_ni for this test)
    camera_ts_ns = np.round((exposure_ni + 5.0) * 1e9).astype(np.int64)
    # Transitions occur every frame period, starting 5 frames before first exposure
    # transition_index[i] is the NI sample index of the i-th transition
    transition_ni = 0.05 + np.arange(-5, 200) * period
    transition_index = np.round(transition_ni * rate).astype(np.int64)
    # transition_perf: host time when each transition was detected
    transition_perf = transition_ni
    # Arrival perf: when frame arrived at host, 2ms after its exposure transition
    arrival_perf = exposure_ni + 0.002

    # Choose pairing
    pairing = clocks.choose_pairing(ids, arrival_perf, transition_perf, period)

    # Verify pairing is correct and not ambiguous
    assert not pairing.ambiguous, f"pairing should not be ambiguous: {pairing.reason}"
    assert pairing.shift == 995  # frame 1000 is transition 5

    # Fit camera to NI using the pairing
    fit = clocks.fit_camera_to_ni(ids, camera_ts_ns, transition_index, rate, pairing, period)

    assert fit.valid, fit.reason
    # Camera timestamp 5.05 should map to NI time 0.05
    assert fit.map(5.05) == pytest.approx(0.05, abs=1e-4)


LATCH_DTYPE = np.dtype([("perf_before", "<f8"), ("camera_ns", "<i8"), ("perf_after", "<f8")])
PERIOD_150 = 1 / 150
CAMERA_BEHIND_HOST = 40.0  # camera clock = host perf_counter - 40 s


def _camera_clock(host):
    return np.asarray(host, dtype=np.float64) - CAMERA_BEHIND_HOST


def _latches(host_times, *, camera_of=_camera_clock, bracket=100e-6, error=0.0):
    host = np.asarray(host_times, dtype=np.float64)
    rows = np.zeros(host.size, dtype=LATCH_DTYPE)
    rows["perf_before"] = host - bracket / 2
    rows["perf_after"] = host + bracket / 2
    rows["camera_ns"] = np.round((camera_of(host) + error) * 1e9).astype(np.int64)
    return rows


def _exposures(count, *, early=5):
    """Frame ids from 1000, their exposure host times, and NI transitions from ``early`` frames before."""
    ids = np.arange(1000, 1000 + count)
    exposure = 50.0 + (ids - 1000 + early) * PERIOD_150
    transition_perf = 50.0 + np.arange(count + early + 5) * PERIOD_150
    return ids, exposure, transition_perf


def test_host_arrival_pairing_aliases_a_lag_over_one_period_and_the_camera_clock_does_not():
    # christielab10, Task 15: exposure to arrival measured 18.9 ms, at 150 fps 2.8 frame periods.
    lag = 0.0189
    ids, exposure, transition_perf = _exposures(9000)
    camera_ts_ns = np.round(_camera_clock(exposure) * 1e9).astype(np.int64)
    latches = _latches([exposure[0] + 0.001, exposure[-1] + 0.001])

    by_arrival = clocks.choose_pairing(ids, exposure + lag, transition_perf, PERIOD_150)
    by_clock = clocks.choose_pairing_by_camera_clock(ids, camera_ts_ns, latches, transition_perf,
                                                     PERIOD_150)

    # The latest transition before each arrival is two frames on, and nothing says so.
    assert by_arrival.method == "host_arrival"
    assert by_arrival.shift == 993 and not by_arrival.ambiguous
    assert by_arrival.median_lag == pytest.approx(lag - 2 * PERIOD_150, abs=1e-6)
    assert by_clock.method == "camera_latch"
    assert by_clock.shift == 995  # frame 1000 is transition 5
    assert not by_clock.ambiguous, by_clock.reason
    assert by_clock.agreement == 1.0
    assert abs(by_clock.median_residual) < 1e-5


def test_a_latch_off_by_more_than_half_a_period_is_ambiguous():
    ids, exposure, transition_perf = _exposures(3000)
    camera_ts_ns = np.round(_camera_clock(exposure) * 1e9).astype(np.int64)
    latches = _latches([exposure[0] + 0.001], error=0.6 * PERIOD_150)

    pairing = clocks.choose_pairing_by_camera_clock(ids, camera_ts_ns, latches, transition_perf,
                                                    PERIOD_150)

    assert pairing.ambiguous
    assert "residual" in pairing.reason
    assert abs(pairing.median_residual) > PERIOD_150 / 4


def test_an_hour_of_camera_drift_is_fitted_through_two_latches_or_windowed_around_one():
    drift = 20e-6

    def drifting(host):
        return _camera_clock(host) * (1 + drift)

    ids, exposure, transition_perf = _exposures(150 * 3600)
    camera_ts_ns = np.round(drifting(exposure) * 1e9).astype(np.int64)
    start, end = exposure[0] + 0.001, exposure[-1] + 0.001

    both = clocks.choose_pairing_by_camera_clock(
        ids, camera_ts_ns, _latches([start, end], camera_of=drifting), transition_perf, PERIOD_150)
    start_only = clocks.choose_pairing_by_camera_clock(
        ids, camera_ts_ns, _latches([start], camera_of=drifting), transition_perf, PERIOD_150)

    # 20 ppm over an hour is 72 ms, ten frame periods: one offset cannot pair the whole session.
    for pairing in (both, start_only):
        assert pairing.shift == 995 and not pairing.ambiguous, pairing.reason
        assert abs(pairing.median_residual) < PERIOD_150 / 4
    assert abs(both.median_residual) < 1e-5


@pytest.mark.parametrize("latches, why", [
    (np.zeros(0, dtype=LATCH_DTYPE), "0 rows"),
    (np.concatenate([_latches([50.1], bracket=0.005),
                     _latches([60.0], bracket=float("nan"))]), "2 rows"),
], ids=["empty", "too_wide"])
def test_unusable_latches_give_an_ambiguous_pairing_with_a_reason(latches, why):
    ids, exposure, transition_perf = _exposures(300)
    camera_ts_ns = np.round(_camera_clock(exposure) * 1e9).astype(np.int64)

    pairing = clocks.choose_pairing_by_camera_clock(ids, camera_ts_ns, latches, transition_perf,
                                                    PERIOD_150)

    assert pairing.ambiguous
    assert pairing.method == "camera_latch"
    assert "no usable clock latch" in pairing.reason and why in pairing.reason


def test_a_pairing_one_frame_off_at_50_fps_is_ambiguous():
    # christielab10 with NI blocks paired to their own stamps: the mapped edges
    # sat 16.6 ms before the camera stamps, so at 50 fps the nearest edge was
    # the next frame's, 3.39 ms on: under P/4, and wrong.
    period = 1 / 50
    ids = np.arange(1000, 4000)
    exposure = 50.0 + (ids - 1000 + 5) * period
    camera_ts_ns = np.round(_camera_clock(exposure) * 1e9).astype(np.int64)
    transition_perf = 50.0 + np.arange(ids.size + 10) * period - 0.01661
    latches = _latches([exposure[0] + 0.001, exposure[-1] + 0.001])

    pairing = clocks.choose_pairing_by_camera_clock(ids, camera_ts_ns, latches, transition_perf,
                                                    period)

    assert pairing.shift == 994  # frame 1000 is transition 5
    assert pairing.median_residual == pytest.approx(0.00339, abs=1e-5)
    assert pairing.ambiguous
    assert "residual" in pairing.reason


def test_a_correct_pairing_with_a_small_residual_is_accepted():
    ids, exposure, transition_perf = _exposures(9000)
    camera_ts_ns = np.round(_camera_clock(exposure) * 1e9).astype(np.int64)
    latches = _latches([exposure[0] + 0.001, exposure[-1] + 0.001])

    pairing = clocks.choose_pairing_by_camera_clock(ids, camera_ts_ns, latches,
                                                    transition_perf + 0.0001, PERIOD_150)

    assert pairing.shift == 995 and not pairing.ambiguous, pairing.reason
    assert pairing.median_residual == pytest.approx(0.0001, abs=1e-6)


@pytest.mark.parametrize("frame_period, tolerance", [
    (1 / 150, 1 / 600), (1 / 50, 0.002), (1 / 25, 0.002), (float("nan"), float("nan")),
], ids=["150fps", "50fps", "25fps", "unknown"])
def test_the_latch_tolerance_is_a_quarter_period_capped_at_2_ms(frame_period, tolerance):
    assert clocks.latch_tolerance(frame_period) == pytest.approx(tolerance, nan_ok=True)


def test_an_unknown_frame_period_is_ambiguous_without_printing_nan():
    ids, exposure, transition_perf = _exposures(300)
    camera_ts_ns = np.round(_camera_clock(exposure) * 1e9).astype(np.int64)

    pairing = clocks.choose_pairing_by_camera_clock(ids, camera_ts_ns, _latches([exposure[0]]),
                                                    transition_perf, float("nan"))

    assert pairing.ambiguous
    assert "frame period is unknown" in pairing.reason
    assert "nan" not in pairing.reason
