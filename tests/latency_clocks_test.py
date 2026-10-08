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


_TAU = 512.0
_POLL = 2048.0
_PLL_GAIN = 1024 / 2 ** 22


def _slewed_host(drift_ppm, polls, host0=1000.0):
    """perf_counter as systemd-timesyncd leaves it (kernel/time/ntp.c, pll_sim.py):
    each poll's offset theta is slewed in with tau = 512 s and leaves a permanent
    rate step of theta * 1024 / 2**22; the NI clock runs drift_ppm against it."""
    def host(t):
        t = np.asarray(t, dtype=np.float64)
        out = host0 + (1 + drift_ppm * 1e-6) * t
        for when, theta in polls:
            after = np.clip(t - when, 0.0, None)
            out = out + theta * (1 - np.exp(-after / _TAU)) + theta * _PLL_GAIN * after
        return out
    return host


def _stamped_reads(duration, host, rng, rate=10_000.0, read=167):
    """(last sample of the previous read, stamp before this read), as ni_block_ends gives
    them: reads of `read` samples back to back, each returning 0.2-3.2 ms after its last
    sample, the next read stamped at once."""
    reads = int(duration * rate // read)
    last = np.arange(1, reads + 1) * read - 1
    returned = host(last / rate) + 0.0002 + rng.uniform(0, 0.003, reads)
    return last[:-1], returned[:-1]


def test_the_piecewise_envelope_follows_ntp_slews_that_one_line_cannot():
    rng = np.random.default_rng(11)
    duration = 4 * 3600.0
    polls = [(when, rng.normal(0, 0.002)) for when in np.arange(900.0, duration, _POLL)]
    host = _slewed_host(-10.0, polls)
    ends, seen = _stamped_reads(duration, host, rng)

    piecewise = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)
    line = clocks.fit_ni_to_host(ends, seen, 10_000.0)

    assert piecewise.valid, piecewise.reason
    assert piecewise.method == clocks.PIECEWISE_ENVELOPE
    assert 100 <= piecewise.knots.size <= 125  # 4 h / 120 s, less a merged tail
    probe = np.linspace(ends[0] / 10_000.0, ends[-1] / 10_000.0, 100_000)
    truth = host(probe)
    error = piecewise.map(probe) - truth
    error -= np.median(error)  # the envelope's delivery floor, not its shape
    assert np.percentile(np.abs(error), 95) < 0.0003
    assert np.abs(error).max() < 0.0005
    line_error = line.map(probe) - truth
    line_error -= np.median(line_error)
    assert np.percentile(np.abs(line_error), 95) > 0.001


def test_a_short_session_is_one_window_of_twenty_points_mapped_late_never_early():
    rng = np.random.default_rng(12)
    host = _slewed_host(-10.0, [])
    ends, seen = _stamped_reads(60.0, host, rng)

    fit = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)

    assert fit.valid, fit.reason
    assert fit.knots.size == 2 and fit.points == 20
    probe = np.linspace(ends[0] / 10_000.0, ends[-1] / 10_000.0, 5_000)
    bias = fit.map(probe) - host(probe)
    # Late by the 0.2 ms delivery floor and little more: a fit through the mean
    # delay (1.7 ms) would pass a 2 ms bound, so the bound is tight.
    assert bias.min() > 0 and bias.max() < 0.0005
    # 20 envelope points over 60 s pin the drift to a few 1e-7 (seeds 12-21: within 3.2e-7).
    assert fit.drift == pytest.approx(1 - 10e-6, abs=6e-7)
    assert fit.segment_drifts().shape == (1,)


def test_a_sparse_tail_window_merges_into_the_window_before_it():
    rng = np.random.default_rng(13)
    ends, seen = _stamped_reads(125.0, _slewed_host(0.0, []), rng)
    fit = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)
    assert fit.valid and fit.knots.size == 2
    ends, seen = _stamped_reads(700.0, _slewed_host(0.0, []), rng)
    fit = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)
    assert fit.valid and fit.knots.size == 7  # 0,120,...,600,700: the 100 s tail keeps its own window


def test_map_and_inverse_round_trip_and_extrapolate_past_the_knots():
    fit = clocks.PiecewiseClockFit(knots=np.array([0.0, 120.0, 240.0]),
                                   knot_host=np.array([1000.0, 1120.001, 1239.999]),
                                   residual_rms=1e-5, points=40, valid=True)
    x = np.array([-5.0, 0.0, 60.0, 120.0, 200.0, 240.0, 300.0])
    assert np.allclose(fit.inverse(fit.map(x)), x, atol=1e-9)
    assert fit.map(-5.0) == pytest.approx(1000.0 - 5.0 * (120.001 / 120.0), abs=1e-9)
    assert fit.map(300.0) == pytest.approx(1239.999 + 60.0 * (119.998 / 120.0), abs=1e-9)
    assert fit.segment_drifts().tolist() == pytest.approx([120.001 / 120, 119.998 / 120])


def test_an_invalid_piecewise_fit_maps_to_nan_and_says_why():
    fit = clocks.fit_piecewise_envelope(np.arange(30), np.arange(30.0), 1000.0)
    assert not fit.valid
    assert "30 read pairs, need 100" in fit.reason
    assert np.isnan(fit.map([1.0, 2.0])).all()
    restarted = clocks.fit_piecewise_envelope(
        np.concatenate([np.arange(100, 160), np.arange(50, 110)]), np.arange(120.0), 1000.0)
    assert "not increasing" in restarted.reason


def test_fit_from_record_rebuilds_the_table_and_accepts_a_straight_line_record():
    rng = np.random.default_rng(14)
    ends, seen = _stamped_reads(300.0, _slewed_host(5.0, [(100.0, 0.003)]), rng)
    fit = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)

    record = fit.as_record()
    rebuilt = clocks.fit_from_record(record)

    assert set(record) >= {"offset", "drift", "residual_rms", "points", "valid", "reason",
                           "x_min", "x_max", "method", "knotCount", "knotSeconds",
                           "segmentDriftPpmMin", "segmentDriftPpmMax", "knots", "knotHost"}
    assert "segmentDrifts" not in record  # derived from the table, not stored twice
    assert "knots" not in fit.as_record(include_knots=False)
    assert rebuilt is not None and rebuilt.method == clocks.PIECEWISE_ENVELOPE
    probe = np.linspace(0.0, 300.0, 1000)
    assert np.allclose(rebuilt.map(probe), fit.map(probe), atol=1e-9)
    line = clocks.fit_line(np.linspace(0.0, 100.0, 200), 7.0 + 1.00001 * np.linspace(0.0, 100.0, 200),
                           max_residual=1e-6)
    from_line = clocks.fit_from_record(line.as_record())
    assert from_line.method == clocks.STRAIGHT_LINE
    assert from_line.map(50.0) == pytest.approx(line.map(50.0), abs=1e-9)
    assert clocks.fit_from_record(None) is None
    assert clocks.fit_from_record({"valid": False, "reason": "x"}) is None
    # With the summary keys present, so it is the table check that refuses it, not a missing key.
    assert clocks.fit_from_record({"valid": True, "knots": [1.0], "knotHost": [2.0], "offset": 1.0,
                                   "drift": 1.0, "x_min": 0.0, "x_max": 10.0}) is None


def test_ni_block_stamps_carry_raw_beside_perf():
    index = np.arange(10)
    perf = np.array([1.0] * 4 + [2.0] * 3 + [3.0] * 3)
    raw = perf + 10.0

    stamps = clocks.ni_block_stamps(index, perf, raw)
    without = clocks.ni_block_stamps(index, perf)

    assert stamps.end_index.tolist() == [3, 6]
    assert stamps.perf.tolist() == [2.0, 3.0]
    assert stamps.raw.tolist() == [12.0, 13.0]
    assert np.isnan(without.raw).all() and without.raw.shape == (2,)
    assert clocks.run_ends(perf).tolist() == [3, 6, 9]


def test_a_span_of_whole_windows_leaves_no_sliver_window_at_the_end():
    # numpy.arange(first, last, 120) can reach `last` itself, or a rounding step past it,
    # when the span is a whole number of windows. That knot must not survive as a zero or
    # negative window: np.histogram rejects it (ValueError), and finalization must not raise.
    rate = 1000.0
    total = 7 * 120 * 1000
    for first in range(16336800, 16336830):
        ends = np.unique(np.append(np.arange(first, first + total, total // 400), first + total))
        fit = clocks.fit_piecewise_envelope(ends, 1000.0 + ends / rate + 0.001, rate)
        assert fit.valid, (first, fit.reason)
        assert fit.knots.size == 8 and np.all(np.diff(fit.knots) > 100.0), first


def _raw_stamped_reads(duration, host, raw_clock, rng, rate=10_000.0, read=167):
    """As _stamped_reads, with CLOCK_MONOTONIC_RAW read at the same instants."""
    reads = int(duration * rate // read)
    last = np.arange(1, reads + 1) * read - 1
    delay = 0.0002 + rng.uniform(0, 0.003, reads)
    perf = host(last / rate) + delay
    raw = raw_clock(last / rate) + delay
    return clocks.BlockStamps(last[:-1], perf[:-1], raw[:-1])


def test_the_raw_bridge_follows_a_slew_step_the_piecewise_fit_cannot():
    rng = np.random.default_rng(21)
    host = _slewed_host(-10.0, [(150.0, 0.010)])      # one 10 ms correction mid-session
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)  # the same clock without it
    stamps = _raw_stamped_reads(300.0, host, raw_clock, rng)

    bridged = clocks.fit_ni_to_host_timeline(stamps, 10_000.0)
    piecewise = clocks.fit_piecewise_envelope(stamps.end_index, stamps.perf, 10_000.0)

    assert bridged.valid, bridged.reason
    assert bridged.method == clocks.RAW_BRIDGED
    assert 29 <= bridged.knots.size <= 32  # tabulated every 10 s over 300 s
    probe = np.linspace(stamps.end_index[0] / 10_000.0, stamps.end_index[-1] / 10_000.0, 30_000)
    error = bridged.map(probe) - host(probe)
    error -= np.median(error)
    assert np.abs(error).max() < 0.00005
    worse = piecewise.map(probe) - host(probe)
    worse -= np.median(worse)
    assert np.abs(worse).max() > 0.00015
    assert np.allclose(bridged.inverse(bridged.map(probe)), probe, atol=1e-9)
    record = bridged.as_record()
    assert record["method"] == "raw_bridged" and record["knotSeconds"] == pytest.approx(10.0, abs=0.5)
    assert clocks.fit_from_record(record).map(123.4) == pytest.approx(bridged.map(123.4), abs=1e-9)


def test_without_raw_stamps_the_selector_fits_the_piecewise_envelope():
    rng = np.random.default_rng(22)
    host = _slewed_host(-10.0, [])
    ends, seen = _stamped_reads(200.0, host, rng)
    no_raw = clocks.BlockStamps(ends, seen, np.full(seen.shape, np.nan))
    few_raw = clocks.BlockStamps(ends, seen, np.where(np.arange(seen.size) < 50, seen + 1.0, np.nan))

    for stamps in (no_raw, few_raw):
        fit = clocks.fit_ni_to_host_timeline(stamps, 10_000.0)
        assert fit.valid, fit.reason
        assert fit.method == clocks.PIECEWISE_ENVELOPE
        assert fit.knots.size == 3  # 0, 120, 200


def test_an_outlier_stamp_pair_does_not_move_a_bridge_knot():
    # A reader preempted between perf_counter() and the RAW read leaves one pair
    # whose perf is late; a knot built by interpolating between pairs would follow it.
    rng = np.random.default_rng(23)
    host = _slewed_host(-10.0, [])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    stamps = _raw_stamped_reads(100.0, host, raw_clock, rng)
    perf = stamps.perf.copy()
    perf[stamps.perf.size // 2] += 0.050
    spoiled = clocks.BlockStamps(stamps.end_index, perf, stamps.raw)

    fit = clocks.fit_ni_to_host_timeline(spoiled, 10_000.0)

    assert fit.valid and fit.method == clocks.RAW_BRIDGED
    probe = np.linspace(stamps.end_index[0] / 10_000.0, stamps.end_index[-1] / 10_000.0, 10_000)
    error = fit.map(probe) - host(probe)
    assert error.max() - error.min() < 0.0001  # the 50 ms pair left no mark
    assert fit.knots[-1] - fit.knots[-2] >= 5.0  # no millisecond-long end segment


def test_the_selector_reports_both_reasons_when_neither_path_fits():
    stamps = clocks.BlockStamps(np.arange(30), np.arange(30.0), np.arange(30.0) + 1.0)
    fit = clocks.fit_ni_to_host_timeline(stamps, 1000.0)
    assert not fit.valid
    assert "30 read pairs, need 100" in fit.reason


def test_late_pairs_sitting_on_a_knot_do_not_move_it():
    # The pair spoiled above sits 17 ms from the nearest knot, so a knot taken from
    # its nearest pair, or a line between its two nearest, would pass that test too.
    # Spoil the pairs the knot itself falls between.
    rng = np.random.default_rng(24)
    host = _slewed_host(-10.0, [])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    stamps = _raw_stamped_reads(100.0, host, raw_clock, rng)
    knot = clocks.fit_ni_to_host_timeline(stamps, 10_000.0).knots[5]
    at_knot = int(np.searchsorted(stamps.end_index / 10_000.0, knot))
    perf = stamps.perf.copy()
    perf[at_knot - 1:at_knot + 2] += 0.050
    spoiled = clocks.BlockStamps(stamps.end_index, perf, stamps.raw)

    fit = clocks.fit_ni_to_host_timeline(spoiled, 10_000.0)

    # A fit that fell back to the perf stamps would be spoiled too, but less visibly.
    assert fit.valid and fit.method == clocks.RAW_BRIDGED, fit.reason
    probe = np.linspace(stamps.end_index[0] / 10_000.0, stamps.end_index[-1] / 10_000.0, 10_000)
    error = fit.map(probe) - host(probe)
    assert error.max() - error.min() < 0.0001


def test_the_selector_prefixes_the_raw_reason_when_both_paths_fail():
    restarted = np.concatenate([np.arange(100, 160), np.arange(50, 110)])
    stamps = clocks.BlockStamps(restarted, np.arange(120.0), np.arange(120.0) + 1.0)

    fit = clocks.fit_ni_to_host_timeline(stamps, 1000.0)

    assert not fit.valid
    assert fit.reason.startswith("NI-to-RAW: ") and fit.reason.count("not increasing") == 2


def _piecewise_record(duration=400.0, seed=31):
    ends, seen = _stamped_reads(duration, _slewed_host(5.0, [(100.0, 0.003)]), np.random.default_rng(seed))
    fit = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)
    assert fit.valid, fit.reason
    return fit, fit.as_record()


def test_fit_from_record_refuses_a_knot_table_it_cannot_rebuild():
    # The line through a long record's summary keys is milliseconds off after an hour, so an
    # unusable table is None, never a valid straight line that nothing flags.
    fit, record = _piecewise_record()
    assert record["knotCount"] > 2
    host = list(record["knotHost"])
    cases = {
        "a truncated knotHost": {**record, "knotHost": host[:-1]},
        "a None inside knotHost": {**record, "knotHost": host[:3] + [None] + host[4:]},
        "both columns cut, knotCount unchanged": {**record, "knots": record["knots"][:-1],
                                                  "knotHost": host[:-1]},
        "host time running backwards": {**record, "knotHost": host[:3] + [host[2] - 1.0] + host[4:]},
        "an absent table (as_record(include_knots=False))": fit.as_record(include_knots=False),
        "knotCount above 2 and an empty table": {**record, "knots": [], "knotHost": []},
    }
    for why, bad in cases.items():
        assert clocks.fit_from_record(bad) is None, why


def test_a_two_knot_record_stored_without_its_table_still_rebuilds_exactly():
    ends, seen = _stamped_reads(60.0, _slewed_host(-10.0, []), np.random.default_rng(32))
    fit = clocks.fit_piecewise_envelope(ends, seen, 10_000.0)
    assert fit.knots.size == 2

    rebuilt = clocks.fit_from_record(fit.as_record(include_knots=False))

    assert rebuilt is not None
    probe = np.linspace(fit.x_min - 5.0, fit.x_max + 5.0, 500)
    assert np.allclose(rebuilt.map(probe), fit.map(probe), atol=1e-9)


def test_stamps_and_fits_compare_and_hash_by_identity_not_by_array_contents():
    def stamps():
        return clocks.BlockStamps(np.arange(3), np.arange(3.0), np.arange(3.0))

    def fit():
        return clocks.PiecewiseClockFit(np.array([0.0, 1.0]), np.array([5.0, 6.0]), 0.0, 2, True)

    # A field-wise == on ndarray fields raises ("truth value of an array is ambiguous"), and a
    # field-wise hash raises TypeError.
    assert stamps() != stamps() and fit() != fit()
    assert len({stamps(), stamps(), fit(), fit()}) == 4


@pytest.mark.parametrize("shape", ["falling", "rising_then_falling"])
def test_a_fit_whose_host_time_runs_backwards_is_invalid(shape):
    # With a segment going backwards, inverse() would not be a function.
    ends = np.arange(0, 600_000, 1_000)
    t = ends / 1000.0
    if shape == "falling":
        seen = 1000.0 - 0.5 * t
    else:
        seen = np.where(t < 240.0, 100.0 + t, 340.0 - (t - 240.0))
    forward = clocks.fit_piecewise_envelope(ends, 100.0 + t, 1000.0)
    assert forward.valid, forward.reason

    fit = clocks.fit_piecewise_envelope(ends, seen, 1000.0)

    assert not fit.valid
    assert "drift" in fit.reason
    assert np.isnan(fit.map([5.0])).all() and np.isnan(fit.inverse([105.0])).all()


@pytest.mark.parametrize("rate", [0.0, -10_000.0, float("nan"), float("inf"), None],
                         ids=["zero", "negative", "nan", "inf", "none"])
def test_an_unusable_sample_rate_is_an_invalid_fit_not_an_exception(rate):
    ends, seen = _stamped_reads(60.0, _slewed_host(-10.0, []), np.random.default_rng(41))

    for fit in (clocks.fit_piecewise_envelope(ends, seen, rate),
                clocks.fit_ni_to_host(ends, seen, rate),
                clocks.fit_ni_to_host_timeline(clocks.BlockStamps(ends, seen, seen + 1.0), rate)):
        assert not fit.valid
        assert fit.reason.startswith("sample rate") and "not a positive number" in fit.reason


def test_a_non_finite_read_pair_is_left_out_not_raised_on():
    ends, seen = _stamped_reads(60.0, _slewed_host(-10.0, []), np.random.default_rng(42))
    holed_ends, holed_seen = ends.astype(np.float64), seen.copy()
    holed_ends[[0, 50, -1]] = np.nan   # NaN at either end of the array is the case that raised
    holed_seen[7], holed_seen[-2] = np.nan, np.inf

    for fit in (clocks.fit_piecewise_envelope(holed_ends, holed_seen, 10_000.0),
                clocks.fit_ni_to_host(holed_ends, holed_seen, 10_000.0)):
        assert fit.valid, fit.reason

    nothing = clocks.fit_piecewise_envelope(np.full(seen.shape, np.nan), seen, 10_000.0)
    assert not nothing.valid and "0 read pairs, need 100" in nothing.reason
    mismatched = clocks.fit_piecewise_envelope(ends[:-1], seen, 10_000.0)
    assert not mismatched.valid and "against" in mismatched.reason


def test_the_selector_says_why_it_fell_back_when_raw_cannot_be_used():
    rng = np.random.default_rng(31)
    host = _slewed_host(-10.0, [])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    stamps = _raw_stamped_reads(200.0, host, raw_clock, rng)
    noisy = clocks.BlockStamps(stamps.end_index, stamps.perf,
                               stamps.raw + rng.normal(0.0, 0.005, stamps.raw.size))
    few = clocks.BlockStamps(stamps.end_index, stamps.perf,
                             np.where(np.arange(stamps.perf.size) < 50, stamps.raw, np.nan))
    absent = clocks.BlockStamps(stamps.end_index, stamps.perf, np.full(stamps.perf.shape, np.nan))

    fit = clocks.fit_ni_to_host_timeline(noisy, 10_000.0)
    assert fit.valid and fit.method == clocks.PIECEWISE_ENVELOPE
    assert fit.reason.startswith("NI-to-RAW: ") and "residual rms" in fit.reason
    few_fit = clocks.fit_ni_to_host_timeline(few, 10_000.0)
    assert few_fit.valid and few_fit.method == clocks.PIECEWISE_ENVELOPE
    assert few_fit.reason == "RAW on 50 stamps, need 100"
    # No RAW at all is an older session or a host without the clock, not a failure.
    assert clocks.fit_ni_to_host_timeline(absent, 10_000.0).reason == ""


def test_a_bridge_whose_host_time_runs_backwards_falls_back_to_the_perf_stamps():
    # A burst of corrupt perf stamps (20 s late) moves the knots inside it and none outside,
    # so the bridged table steps back at the burst's far edge. RAW itself is clean, and the
    # perf fit's robust refit survives the burst, so the bridge is what must be refused.
    host = _slewed_host(-10.0, [])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    stamps = _raw_stamped_reads(200.0, host, raw_clock, np.random.default_rng(41))
    t = stamps.end_index / 10_000.0
    perf = np.where((t >= 80.0) & (t < 105.0), stamps.perf + 20.0, stamps.perf)

    fit = clocks.fit_ni_to_host_timeline(clocks.BlockStamps(stamps.end_index, perf, stamps.raw), 10_000.0)

    assert fit.valid and fit.method == clocks.PIECEWISE_ENVELOPE
    assert fit.reason.startswith("RAW bridge is not increasing")
    assert np.all(np.diff(fit.knot_host) > 0)


def test_raw_on_only_part_of_the_session_is_not_bridged_over_the_rest():
    # A 10 ms slew at 50 s is invisible to RAW stamps that stop at 15 s: a bridge would
    # extrapolate the offset it last saw over the rest of the session.
    host = _slewed_host(-10.0, [(50.0, 0.010)])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    stamps = _raw_stamped_reads(100.0, host, raw_clock, np.random.default_rng(42))
    t = stamps.end_index / 10_000.0

    def timeline(raw_until):
        raw = np.where(t < raw_until, stamps.raw, np.nan)
        return clocks.fit_ni_to_host_timeline(clocks.BlockStamps(stamps.end_index, stamps.perf, raw),
                                              10_000.0)

    for until in (15.0, 85.0):
        fit = timeline(until)
        assert fit.valid and fit.method == clocks.PIECEWISE_ENVELOPE, until
        assert "RAW covers" in fit.reason and "need 90%" in fit.reason, fit.reason
    assert timeline(95.0).method == clocks.RAW_BRIDGED


@pytest.mark.parametrize("before, after", [(1.2, 3.0), (3.0, 1.2)],
                         ids=["previous_pair_closer", "next_pair_closer"])
def test_a_knot_with_no_pair_within_a_second_takes_the_closer_neighbour(before, after):
    # A half-second NTP step at 40 s leaves perf minus RAW changing about 1 ms every second,
    # so a neighbour 3 s from the knot is 3 ms off and one 1.2 s away about 1.2 ms.
    host = _slewed_host(-10.0, [(40.0, 0.5)])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    stamps = _raw_stamped_reads(100.0, host, raw_clock, np.random.default_rng(43))
    whole = clocks.fit_ni_to_host_timeline(stamps, 10_000.0)
    knot = 5
    target = raw_clock(whole.knots[knot])
    gap = (stamps.raw > target - before) & (stamps.raw < target + after)
    holed = clocks.BlockStamps(stamps.end_index[~gap], stamps.perf[~gap], stamps.raw[~gap])

    fit = clocks.fit_ni_to_host_timeline(holed, 10_000.0)

    assert fit.method == clocks.RAW_BRIDGED and fit.knots.size == whole.knots.size
    assert abs(fit.knot_host[knot] - host(fit.knots[knot])) < 0.0025


def test_a_knot_closer_than_five_seconds_to_the_end_is_dropped_for_the_end_knot():
    host = _slewed_host(-10.0, [])
    raw_clock = _slewed_host(-10.0, [], host0=4000.0)
    # 100 s: the last 10 s knot (90.0) is 10 s from the end, so it stays. 103 s: the one at
    # 100.0 is 3 s from the end and goes, leaving a 12.96 s segment rather than a 2.96 s one.
    for duration, last_segment in ((100.0, 9.97), (103.0, 12.96)):
        stamps = _raw_stamped_reads(duration, host, raw_clock, np.random.default_rng(32))

        fit = clocks.fit_ni_to_host_timeline(stamps, 10_000.0)

        assert fit.method == clocks.RAW_BRIDGED, fit.reason
        assert np.diff(fit.knots)[-1] == pytest.approx(last_segment, abs=0.05), duration
