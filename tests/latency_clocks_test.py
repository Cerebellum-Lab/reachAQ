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


def test_block_ends_are_the_last_sample_each_read_delivered():
    ends, seen = clocks.ni_block_ends(np.arange(10), np.array([1.0] * 4 + [2.0] * 3 + [3.0] * 3))
    assert ends.tolist() == [3, 6, 9]
    assert seen.tolist() == [1.0, 2.0, 3.0]


def test_the_envelope_maps_late_by_about_the_minimum_delay_and_never_early():
    rng = np.random.default_rng(4)
    rate = 1000.0
    ends = np.arange(1, 2001) * 50 - 1
    t = ends / rate
    truth = 100.0 + (1 + 11e-6) * t
    seen = truth + 0.0002 + rng.uniform(0, 0.003, t.size)

    fit = clocks.fit_ni_to_host(ends, seen, rate)

    assert fit.valid, fit.reason
    bias = fit.map(t) - truth
    assert bias.min() > 0
    assert bias.max() < 0.002


def test_a_restarted_ni_task_is_not_fitted():
    fit = clocks.fit_ni_to_host(np.array([10, 20, 5] * 10), np.arange(30.0), 1000.0)
    assert not fit.valid


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
    # Issue 2: backward wall steps can interleave segments; overlaps are NaN
    wall = np.concatenate([np.arange(0.0, 50.0), np.arange(99.5, 150.0)])
    perf = np.concatenate([np.arange(1000.0, 1050.0), np.arange(1050.5, 1101.0)])

    wall_map = clocks.fit_wall_to_host(wall, perf)

    # Map values in non-overlapping parts of the segments
    result = wall_map.map(np.array([10.0, 130.0]))
    assert result[0] == pytest.approx(1010.0)
    assert result[1] == pytest.approx(1081.0)


def test_the_envelope_fit_is_tighter_than_a_plain_fit():
    # Issue 5: bias.max() should be tighter than 0.002
    rng = np.random.default_rng(4)
    rate = 1000.0
    ends = np.arange(1, 2001) * 50 - 1
    t = ends / rate
    truth = 100.0 + (1 + 11e-6) * t
    seen = truth + 0.0002 + rng.uniform(0, 0.003, t.size)

    fit = clocks.fit_ni_to_host(ends, seen, rate)

    assert fit.valid, fit.reason
    bias = fit.map(t) - truth
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
