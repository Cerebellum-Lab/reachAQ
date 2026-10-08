"""Self-tests for the synthetic rig the frame-alignment tests are built on.

They check the fixture's own physics, so each one is pinned to a number worked
out by hand (or to a closed form written out here) and not to the helper's own
formulas: dropping or sign-flipping the epoch offset, the NI drift, the NTP
slew, the pre-read stamping or the latch bracket fails at least one of them.
"""

import csv
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from autotrainer.core import ProjectInfo
from autotrainer.core.latency.schema import CLOCK_LATCH_DTYPE
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.latency import clocks
from frame_alignment_synthetic import (
    ARRIVAL, CAMERA_OFFSET, DRIFT, EPOCH_EARLY, EXPOSURE, FIRST_ID, HOST0, PERIOD, RATE, READ,
    SyntheticRig,
)


@pytest.fixture
def project(tmp_path):
    return ProjectInfo(root=str(tmp_path), device_id="test",
                       when=datetime(2026, 1, 2, 3, 4, 5), session=1)


def test_the_epoch_timeline_is_early_and_the_ni_clock_drifts_against_the_host():
    rig = SyntheticRig()
    assert float(rig.epoch_perf(0.0)) == pytest.approx(HOST0 - 0.0017, abs=1e-12)
    assert float(rig.host(0.0)) == HOST0

    def gap(t):
        return float(rig.epoch_perf(t) - rig.host(t))

    # epoch - host = -EPOCH_EARLY - DRIFT * t, DRIFT = -10 ppm: the epoch timeline starts
    # 1.7 ms early and gains on the host as the NI clock runs against it.
    assert gap(0.0) == pytest.approx(-0.0017, abs=1e-12)
    assert gap(5.0) == pytest.approx(-0.00165, abs=1e-9)
    assert gap(700.0) == pytest.approx(0.0053, abs=1e-9)


def test_a_timesyncd_poll_slews_the_host_clock_by_a_decay_and_a_rate_step():
    plain = SyntheticRig()
    rig = SyntheticRig(polls=((10.0, 0.004),))
    # Hand-worked: theta * (1 - exp(-(t - T) / 512)) + theta * 1024 / 2**22 * (t - T).
    assert float(rig.slew(5.0)) == 0.0
    assert float(rig.slew(20.0)) == pytest.approx(8.713262844184566e-05, abs=1e-12)
    assert float(rig.slew(522.0)) == pytest.approx(0.0030284822353142307, abs=1e-12)
    for t in (5.0, 20.0, 522.0):
        assert float(rig.host(t) - plain.host(t)) == pytest.approx(float(rig.slew(t)), abs=1e-9)
        # CLOCK_MONOTONIC_RAW and the epoch timeline never see the slew.
        assert float(rig.raw(t)) == pytest.approx(HOST0 + 3.0 + (1 - 10e-6) * t, abs=1e-9)
        assert float(rig.epoch_perf(t)) == pytest.approx(HOST0 - 0.0017 + t, abs=1e-9)
    # A later poll in the other direction adds, with its own sign.
    both = SyntheticRig(polls=((10.0, 0.004), (100.0, -0.002)))
    assert float(both.slew(612.0)) == pytest.approx(0.0018393382710241886, abs=1e-12)


def test_edges_frame_ids_and_the_camera_clock_follow_the_model():
    rig = SyntheticRig()
    assert rig.edge_sample(0) == 123
    assert PERIOD == pytest.approx(1 / 150)
    assert float(rig.edge_time(11) - rig.edge_time(10)) == pytest.approx(PERIOD)
    n0 = rig.first_edge_at_or_after(5.0)
    assert n0 == 749
    assert float(rig.edge_time(n0)) >= 5.0 > float(rig.edge_time(n0 - 1))
    assert rig.frame_id(n0) == FIRST_ID + 749 == 1749
    # The camera clock is NI seconds plus seven.
    assert rig.camera_ns(0.0) == 7_000_000_000
    assert rig.camera_ns(1.5) == 8_500_000_000
    assert rig.frame_perf_c(n0, 0.001) == pytest.approx(
        1000.0 + (1 - 10e-6) * (0.0123 + 749 / 150) + 0.001, abs=1e-9)


def test_the_spool_toggles_cam_frames_at_every_exposure_edge_and_stamps_each_read(tmp_path):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=20.0)

    names, indices, perf, values, rate = SessionDataRecorder._nidaq_arrays(
        spool, requested_names=("cam_frames",))
    transitions = indices[SessionDataRecorder._transition_positions(values[0])]
    expected = np.array([rig.edge_sample(n) for n in range(0, 3000) if rig.edge_sample(n) < 200_000])
    assert names == ("cam_frames",)
    assert transitions.tolist() == expected.tolist()
    assert rate == RATE and indices[0] == 0 and indices.size == 200_000
    # The epoch timeline is 1.7 ms early at sample 0, and the NI clock runs -10 ppm against
    # the host, so epoch - host = -EPOCH_EARLY - DRIFT * t: -1.65 ms at 5 s, +5.3 ms at 700 s.
    assert perf[50_000] - float(rig.host(5.0)) == pytest.approx(-EPOCH_EARLY - DRIFT * 5.0, abs=1e-9)
    assert perf[50_000] - float(rig.host(5.0)) == pytest.approx(-0.00165, abs=1e-9)
    stamps = SessionDataRecorder._nidaq_block_stamps(spool)
    assert stamps.end_index.size >= 1100  # 20 s of 167-sample reads
    assert stamps.end_index[0] % 167 == 166
    assert (np.diff(stamps.end_index) == READ).all()
    assert np.isnan(stamps.raw).all()
    delay = stamps.perf - rig.host(stamps.end_index / RATE)
    assert 0.0002 <= delay.min() and delay.max() <= 0.0032


def test_every_sample_of_a_read_carries_the_stamp_taken_before_that_read(tmp_path):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=5.0)
    with h5py.File(spool.path, "r") as store:
        assert set(store) == {
            "sample_index", "perf_time", "wall_time", "epoch", "values",
            "block_observation_perf_time", "block_observation_wall_time",
            "block_observation_raw_time",
        }
        assert tuple(store.attrs["channel_names"]) == ("cam_frames",)
        assert store.attrs["sample_rate_hz"] == RATE
        assert store["sample_index"].dtype == np.int64 and store["epoch"].dtype == np.uint64
        assert store["values"].shape == (1, 50_000) and store["values"].dtype == np.float32
        seen = store["block_observation_perf_time"][()]
        wall = store["block_observation_wall_time"][()]
        raw = store["block_observation_raw_time"][()]
    # One run of equal stamps per 167-sample read, the stamp changing at each read boundary.
    assert np.flatnonzero(np.diff(seen)).tolist() == list(range(READ - 1, seen.size - 1, READ))
    # Read k carries what was taken before its blocking call, when read k-1 had just returned:
    # 0.2-3.2 ms after read k-1's last sample. A stamp taken after its own read would sit a
    # whole read (16.7 ms) later, and one carried by the wrong read a read earlier.
    read = np.arange(seen.size) // READ
    delay = seen - rig.host((read * READ - 1) / RATE)
    assert 0.0002 <= delay.min() and delay.max() <= 0.0032
    assert delay.max() - delay.min() > 0.0025
    assert np.isnan(raw).all()
    assert np.allclose(wall - seen, 1e9, rtol=0, atol=1e-6)


def test_the_raw_column_is_the_unslewed_clock_and_the_stamps_can_be_blank(tmp_path):
    rig = SyntheticRig(polls=((10.0, 0.004),))
    with_raw = SessionDataRecorder._nidaq_block_stamps(
        rig.write_spool(tmp_path / "raw.h5", t_from=0.0, t_to=20.0, raw=True))
    assert np.isfinite(with_raw.raw).all()
    gap = with_raw.perf - with_raw.raw
    # perf_counter and RAW differ by the 3 s offset and, from the poll at 10 s on, by the slew
    # (87.04 us at the last stamp, NI time 19.9898 s): the stamping delay cancels.
    assert gap[0] == pytest.approx(-3.0, abs=1e-9)
    assert gap[-1] == pytest.approx(-3.0 + 8.70445205217029e-05, abs=1e-9)
    assert gap[-1] - gap[0] > 8e-5
    assert (np.diff(gap) >= -1e-9).all()
    blank_spool = rig.write_spool(tmp_path / "blank.h5", t_from=0.0, t_to=20.0, stamps=False)
    assert SessionDataRecorder._nidaq_block_stamps(blank_spool).perf.size == 0
    # Only the stamps are blank: the samples and the edges are all still there.
    _, indices, _, values, _ = SessionDataRecorder._nidaq_arrays(
        blank_spool, requested_names=("cam_frames",))
    assert indices.size == 200_000
    assert SessionDataRecorder._transition_positions(values[0]).size == 2999


def test_a_spool_that_starts_mid_stream_keeps_absolute_indices_and_the_edge_parity(tmp_path):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "mid.h5", t_from=5.2, t_to=8.0, tone_pulses=((5.1, 0.15),))
    names, indices, perf, values, _ = SessionDataRecorder._nidaq_arrays(
        spool, requested_names=("cam_frames", "tone1"))
    assert names == ("cam_frames", "tone1")
    assert indices[0] == 52_000 and indices[-1] == 79_999 and indices.size == 28_000
    assert perf[0] == pytest.approx(1000.0 - 0.0017 + 5.2, abs=1e-9)
    # 779 exposure edges (n = 0..778) precede sample 52 000, so cam_frames starts high.
    assert values[0][0] == 1.0
    inside = [rig.edge_sample(n) for n in range(0, 1300) if 52_000 < rig.edge_sample(n) < 80_000]
    assert indices[SessionDataRecorder._transition_positions(values[0])].tolist() == inside
    # A pulse that began before the window is cut at its start and still ends on time.
    assert values[1][:500].tolist() == [1.0] * 500
    assert not values[1][500:].any()
    # The first read is partial: read 311 spans samples 51 937-52 103, so the first complete
    # pairing stamps the copy after it with the end of that read.
    stamps = SessionDataRecorder._nidaq_block_stamps(spool)
    assert stamps.end_index[0] == 52_103
    assert (np.diff(stamps.end_index) == READ).all()


def test_tone_pulses_are_a_second_channel_stored_after_cam_frames(tmp_path):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "tone.h5", t_from=0.0, t_to=5.0,
                            tone_pulses=((2.0, 0.1), (3.5, 0.05)))
    names, indices, _, values, _ = SessionDataRecorder._nidaq_arrays(spool)
    assert names == ("cam_frames", "tone1")
    assert indices[SessionDataRecorder._rising_edge_positions(values[1])].tolist() == [20_000, 35_000]
    assert indices[SessionDataRecorder._transition_positions(values[1])].tolist() == [
        20_000, 21_000, 35_000, 35_500]
    assert values[1].sum() == 1_500
    # Asked for in the other order, the rows come back in that order.
    names, _, _, swapped, _ = SessionDataRecorder._nidaq_arrays(spool, ("tone1", "cam_frames"))
    assert names == ("tone1", "cam_frames") and (swapped[0] == values[1]).all()


def test_the_same_seed_gives_the_same_stamps_and_another_seed_different_ones(tmp_path):
    def stamps(seed, name):
        spool = SyntheticRig(seed=seed).write_spool(tmp_path / name, t_from=0.0, t_to=2.0)
        return SessionDataRecorder._nidaq_block_stamps(spool).perf

    assert (stamps(5, "a.h5") == stamps(5, "b.h5")).all()
    assert not (stamps(5, "a.h5") == stamps(6, "c.h5")).any()


def test_the_ledger_and_the_latency_stream_describe_the_recorded_frames(tmp_path, project):
    rig = SyntheticRig()
    n0 = rig.first_edge_at_or_after(5.0)
    start_perf, start_wall, boundary = rig.start(project, n0, 0.001)
    rig.write_frame_ledger(project, n0, 10, 0.001)
    path = rig.write_latency_stream(project, "left", n0, 10, 0.001)

    assert boundary.primary_frame_id == FIRST_ID + n0
    assert boundary.session_id == project.short_id and boundary.primary_camera == "left"
    assert start_perf == pytest.approx(float(rig.host(rig.edge_time(n0))) + 0.001, abs=1e-9)
    assert boundary.start_perf_time == start_perf
    assert boundary.start_wall_time == start_wall == pytest.approx(1.7e9 + start_perf, abs=1e-6)
    assert boundary.camera_when == rig.camera_ns(rig.edge_time(n0) + EXPOSURE)
    ledger = Path(project.get_frame_timing_path()).read_text().splitlines()
    assert ledger[0] == "frame_id,frame_when,frame_present_primary,frame_present_secondary,utc_when"
    assert ledger[1].startswith(f"{FIRST_ID + n0},")
    with h5py.File(path, "r") as store:
        frames, latches = store["frames"][()], store["clock_latches"][()]
    assert frames["frame_id"].tolist() == list(range(FIRST_ID + n0, FIRST_ID + n0 + 10))
    camera, host, _ = clocks.usable_latches(latches)
    assert camera.size == 6
    fit = clocks.fit_latches(latches)
    assert fit.valid and fit.map(camera[0]) == pytest.approx(host[0], abs=1e-9)
    assert rig.edge_time(n0 + 1) - rig.edge_time(n0) == pytest.approx(PERIOD)


def test_the_ledger_rows_carry_camera_stamps_and_wall_times_for_every_frame(tmp_path, project):
    rig = SyntheticRig()
    n0 = rig.first_edge_at_or_after(5.0)
    lag = 0.0018
    _, _, boundary = rig.start(project, n0, lag)
    rig.write_frame_ledger(project, n0, 10, lag)
    path = rig.write_latency_stream(project, "left", n0, 10, lag)
    with Path(project.get_frame_timing_path()).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    n = np.arange(n0, n0 + 10)
    assert [int(row["frame_id"]) for row in rows] == list(range(FIRST_ID + n0, FIRST_ID + n0 + 10))
    stamped = [rig.camera_ns(float(rig.edge_time(k)) + EXPOSURE) for k in n]
    assert [int(row["frame_when"]) for row in rows] == stamped
    assert {row["frame_present_primary"] for row in rows} == {"1"}
    assert {row["frame_present_secondary"] for row in rows} == {"1"}
    # utc_when is the wall anchor plus each frame's perf time, so it carries the same lag.
    utc = np.array([float(row["utc_when"]) for row in rows])
    assert np.allclose(utc - 1.7e9, rig.host(rig.edge_time(n)) + lag, rtol=0, atol=1e-6)
    assert utc[0] == pytest.approx(boundary.start_wall_time, abs=1e-6)
    assert int(rows[0]["frame_when"]) == boundary.camera_when
    # The capture process stamped the same frames with the same camera clock.
    with h5py.File(path, "r") as store:
        assert store["frames"]["camera_ts_ns"].tolist() == stamped


def test_the_latency_stream_rows_follow_the_frame_clock(tmp_path, project):
    rig = SyntheticRig()
    n0 = rig.first_edge_at_or_after(5.0)
    path = rig.write_latency_stream(project, "left", n0, 10, 0.001)
    edge = rig.edge_time(np.arange(n0, n0 + 10))
    with h5py.File(path, "r") as store:
        frames, pairs = store["frames"][()], store["clock_pairs"][()]
        assert store.attrs["schema_version"] == 1 and store.attrs["camera"] == "left"
        assert bool(store.attrs["is_primary"])
    assert np.allclose(frames["arrival_perf"] - rig.host(edge), ARRIVAL, rtol=0, atol=1e-9)
    assert (frames["poll_perf"] < frames["arrival_perf"]).all()
    assert (frames["capture_return_perf"] > frames["arrival_perf"]).all()
    assert np.allclose(frames["camera_ts_ns"] / 1e9, edge + EXPOSURE + CAMERA_OFFSET,
                       rtol=0, atol=1e-9)
    assert np.allclose(pairs["wall"] - pairs["perf"], 1.7e9, rtol=0, atol=1e-6)
    assert pairs["perf"].tolist() == [frames["arrival_perf"][0], frames["arrival_perf"][-1]]


def test_each_latch_brackets_the_host_time_of_the_camera_clock_reading(tmp_path, project):
    rig = SyntheticRig(polls=((1.0, 0.002),))
    n0 = rig.first_edge_at_or_after(5.0)
    path = rig.write_latency_stream(project, "left", n0, 10, 0.001)
    with h5py.File(path, "r") as store:
        frames, latches = store["frames"][()], store["clock_latches"][()]
    # Five fields, the RAW pair blank: this rig does not model a camera RAW latch.
    assert latches.dtype == CLOCK_LATCH_DTYPE and len(latches) == 6
    assert np.isnan(latches["raw_before"]).all() and np.isnan(latches["raw_after"]).all()
    # Three latches 1 ms after the first frame's exposure start, three after the last's.
    when = np.repeat([float(rig.edge_time(n0)) + 0.001, float(rig.edge_time(n0 + 9)) + 0.001], 3)
    camera, host, uncertainty = clocks.usable_latches(latches)
    assert np.allclose(camera, when + CAMERA_OFFSET, rtol=0, atol=1e-9)
    # The bracket is centred on the host time of that instant, slew included, and 0.2 ms wide.
    assert np.allclose(host, rig.host(when), rtol=0, atol=1e-9)
    # The poll at 1 s has put 17 us of slew on the host by then; the latch reads the host.
    assert (host - SyntheticRig().host(when) > 1.5e-5).all()
    assert np.allclose(uncertainty, 1e-4, rtol=0, atol=1e-12)
    assert (latches["perf_before"] < latches["perf_after"]).all()
    # Placed on the host through the latches, each frame's exposure end lands on its own host
    # time to within the drift across the 60 ms the stream spans (0.6 us).
    fit = clocks.fit_latches(latches)
    placed = fit.map(frames["camera_ts_ns"] / 1e9)
    assert np.allclose(placed, rig.host(rig.edge_time(np.arange(n0, n0 + 10)) + EXPOSURE),
                       rtol=0, atol=2e-6)


def test_a_stream_written_without_latches_has_an_empty_latch_table(tmp_path, project):
    rig = SyntheticRig()
    n0 = rig.first_edge_at_or_after(5.0)
    path = rig.write_latency_stream(project, "left", n0, 10, 0.001, latches=False)
    with h5py.File(path, "r") as store:
        latches = store["clock_latches"][()]
        assert store["frames"].shape == (10,)
    assert latches.shape == (0,) and latches.dtype == CLOCK_LATCH_DTYPE
    assert not clocks.fit_latches(latches).valid
