"""The recorder's frame anchor and timeline on the drift-corrected NI-to-host fit
(docs/superpowers/specs/2026-10-07-frame-alignment-design.md, Verification)."""

import copy
import csv
import dataclasses
import json
import tracemalloc
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from autotrainer.core import ProjectInfo
from autotrainer.core.latency.schema import CAMERA_FRAME_DTYPE
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.latency import clocks
from tools.latency.finalize import finalize_session_latency, read_raw_stream
from frame_alignment_synthetic import (
    ARRIVAL, EXPOSURE, FPS, LATCH_HALF_BRACKET, RATE, SyntheticRig,
)


def _project(tmp_path, session=1):
    return ProjectInfo(root=str(tmp_path), device_id="test",
                       when=datetime(2026, 1, 2, 3, 4, 5), session=session)


def _record(tmp_path, *, t0, lag, seconds=20.0, latches=True, stream=True, exposure=EXPOSURE,
            polls=(), raw=False, stamps=True, tone_pulses=(), device_rows=(), laser_rows=(),
            trial_records=(), arrival=ARRIVAL, latch_error=0.0, stream_tail_seconds=0.0,
            raw_latches=False, drop_edge_at=None, latch_half_bracket=LATCH_HALF_BRACKET,
            spool_tail_seconds=0.0):
    """One finalized session: the spool holds 5 s before the recording and `seconds` of it.

    The latency stream runs stream_tail_seconds past the end of the spool. raw puts RAW
    stamps on the NI reads and raw_latches a RAW bracket on the latches; drop_edge_at
    (NI seconds) loses one cam_frames transition. The spool runs spool_tail_seconds past
    the recording, as it does through Stop."""
    rig = SyntheticRig(polls=polls)
    project = _project(tmp_path)
    session_dir = Path(project.get_session_path().location)
    n0 = rig.first_edge_at_or_after(t0)
    count = int(seconds * FPS) - 2
    start_perf, start_wall, boundary = rig.start(project, n0, lag)
    end_perf = rig.frame_perf_c(n0 + count - 1, lag)
    spool = rig.write_spool(session_dir / "nidaq_capture.h5", t_from=t0 - 5.0,
                            t_to=t0 + seconds + spool_tail_seconds, tone_pulses=tone_pulses, raw=raw, stamps=stamps,
                            drop_edge_at=drop_edge_at)
    rig.write_frame_ledger(project, n0, count, lag)
    if stream:
        rig.write_latency_stream(
            project, "left", n0, count + int(stream_tail_seconds * FPS), lag, latches=latches,
            arrival=arrival, latch_error=latch_error, raw_latches=raw_latches,
            latch_half_bracket=latch_half_bracket)
    result = SessionDataRecorder._write_session(
        project, start_perf, start_wall, end_perf, device_rows, laser_rows, (), spool,
        boundary=boundary, trial_records=trial_records,
        trial_summary={"pending_analysis_attempts": len(trial_records)},
        camera_exposure_seconds={"left": exposure},
    )
    return rig, project, n0, result


def _alignment(project):
    path = Path(project.get_session_path().location) / "streams" / "alignment.json"
    return json.loads(path.read_text())


def test_the_anchor_is_the_first_frames_own_edge_where_the_epoch_method_took_the_next(tmp_path):
    # t0 = 5 s, L = 1.75 ms: the epoch timeline is 1.65 ms early here (1.7 ms less 10 ppm
    # over 5 s), so the first frame's edge sits 3.4 ms from start_perf, more than half the
    # 6.67 ms period, and the next frame's edge wins on the epoch method (the design's F+1
    # case; 14 of 36 current sessions).
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.00175)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["status"] == "matched" and camera["confidence"] == "hardware_edge"
    assert camera["anchorMethod"] == "camera_latch" and camera["anchorsAgree"] is True
    assert camera["arrivalSampleIndex"] == rig.edge_sample(n0)
    assert camera["arrival"]["status"] == "paired"
    # The implied lag is the true 1.75 ms less the fit's lateness, the synthetic's
    # documented 0.2-0.3 ms bias. A 1.5 ms bound left 8-68 us over the seeds, inside that
    # bias; 1.45 ms leaves 58-118 us.
    assert 0.00145 <= camera["impliedLagSeconds"] <= 0.0021
    assert camera["niToHost"]["valid"] and camera["niToHost"]["method"] == "piecewise_envelope"
    assert _alignment(project)["canonicalBoundary"]["nidaqSampleIndex"] == rig.edge_sample(n0)


def test_after_twelve_minutes_of_drift_without_a_latency_stream_the_first_frame_stamp_finds_the_frame(
    tmp_path,
):
    # t0 = 700 s, L = 1.0 ms, no latency stream: the epoch timeline is now 5.3 ms late and the
    # epoch method takes the previous frame's edge (the design's F-1 case). The first frame's
    # own host stamp, on the fitted timeline, still picks the latest transition before it.
    rig, project, n0, result = _record(tmp_path, t0=700.0, lag=0.001, stream=False)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorMethod"] == "first_frame_stamp"
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert camera["latch"]["status"] == camera["arrival"]["status"] == "unavailable"
    assert "camera-clock match not available" in camera["reason"]
    assert "arrival match not available" in camera["reason"]


def test_a_host_event_one_millisecond_before_an_exposure_maps_to_that_frame(tmp_path):
    # t0 = 5 s, L = 1.0 ms: no slip on the epoch method, so this isolates its 1.65 ms
    # timeline error, which put frame F+3's edge before the event and answered F+4.
    model = SyntheticRig()
    n0 = model.first_edge_at_or_after(5.0)
    send = float(model.host(model.edge_time(n0 + 3))) - 0.001
    record = {"session_id": "s", "operation_id": "send-1", "trial_id": 1, "attempt_id": 1,
              "attempt_label": "1.1", "send_perf_time": send, "outcome": "pending_analysis"}
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, trial_records=(record,))

    trials = Path(project.get_session_path().location) / "streams" / "trials.jsonl"
    frame = json.loads(trials.read_text().strip())["event_alignment"]["send"]["frameAssociation"]
    assert frame["frameId"] == rig.frame_id(n0 + 3)
    assert frame["method"].endswith("nidaq_camera_exposure_timeline_verified_by_writer")
    # The fitted edge is late by the delivery floor (0.2-0.3 ms), never early.
    assert 0.0010 <= frame["eventToFrameStartSeconds"] <= 0.0016


def test_a_tone_just_before_an_exposure_edge_maps_to_that_frame(tmp_path):
    model = SyntheticRig()
    n0 = model.first_edge_at_or_after(5.0)
    t_tone = float(model.edge_time(n0 + 3)) - 0.0004
    device_rows = ((float(model.host(t_tone)) + 0.002, 1.7e9, "inbound", "STIMULUS_INPUTS",
                    "PELLET_DEVICE", "status-1", None, None, json.dumps({"tone1": True})),)
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.00175,
                                       tone_pulses=((t_tone, 0.005),), device_rows=device_rows)

    tone = result["toneConfirmation"]
    assert tone["status"] == "complete" and len(tone["matched"]) == 1
    assert tone["matched"][0]["sampleIndex"] == int(round(t_tone * RATE))
    assert tone["matched"][0]["frameAssociation"]["frameId"] == rig.frame_id(n0 + 3)


def test_without_block_stamps_the_epoch_method_runs_at_lower_confidence_and_nothing_raises(tmp_path):
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, stamps=False)

    camera = result["cameraNidaqAlignment"]
    assert camera["status"] == "matched"
    assert camera["anchorMethod"] == "epoch_nearest_edge"
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert "read pairs" in camera["reason"]
    assert camera["niToHost"]["valid"] is False
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)  # no slip at t0 = 5 s, L = 1.0 ms
    assert _alignment(project)["schemaVersion"] == 4


def test_a_camera_clock_a_frame_off_disagrees_with_arrival_and_is_kept_but_flagged(tmp_path):
    # The latches place every camera stamp a frame period late, so the camera-clock match names
    # the next transition; the host arrival, which needs no camera clock, still says F.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, latch_error=1 / FPS)

    camera = result["cameraNidaqAlignment"]
    assert camera["anchorMethod"] == "camera_latch"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0 + 1)
    assert camera["arrivalSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorsAgree"] is False
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert "disagree" in camera["reason"]


def test_a_1024_px_frame_lag_anchors_at_hardware_edge_without_any_lag_value(tmp_path):
    # 1024 px frames reach the host about 2.2 ms later than 256 px ones; there is no lag to
    # expect, so the measured match is what confirms the camera-clock one.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.0035, arrival=0.0045)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["confidence"] == "hardware_edge" and camera["anchorsAgree"] is True
    assert 0.0040 <= camera["arrival"]["medianLagSeconds"] <= 0.0046  # less the fit's 0.2-0.3 ms lateness
    assert not {"expectedLagSeconds", "lagGateSeconds", "expectedLagSampleIndex"} & set(camera)


def test_an_arrival_lag_longer_than_a_frame_is_flagged_not_paired_wrong(tmp_path):
    # The host arrival only holds for a lag under one frame period. A longer one pairs every
    # frame with the next exposure, and the disagreement with the camera clock shows it.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, arrival=1 / FPS + 0.002)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)            # the latch is right
    assert camera["arrivalSampleIndex"] == rig.edge_sample(n0 + 1)
    assert camera["anchorsAgree"] is False and camera["confidence"] == "ambiguous_hardware_edge"


def test_without_usable_latches_the_arrival_match_anchors_at_lower_confidence(tmp_path):
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, latches=False)

    camera = result["cameraNidaqAlignment"]
    assert camera["anchorMethod"] == "host_arrival"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorsAgree"] is None
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert "camera-clock match not available" in camera["reason"]


def test_latency_frames_beyond_the_ni_spool_do_not_cost_the_arrival_match(tmp_path):
    # The camera runs on after the spool ends: those frames arrive past the last transition
    # and have none of their own, so they must not count against either pairing: counted,
    # each would vote a shift of its own and the agreement would fall below 1.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, stream_tail_seconds=2.0)

    camera = result["cameraNidaqAlignment"]
    assert camera["arrival"]["status"] == "paired" and camera["arrival"]["agreement"] == 1.0
    # 20 s of the 150 fps stream lies inside the spool; the 2 s after it is left out of both.
    assert camera["arrival"]["frames"] == camera["latch"]["frames"] == int(20.0 * FPS)
    assert camera["confidence"] == "hardware_edge"


def test_a_delivery_backlog_leaves_the_arrival_anchor_paired():
    # Three of 9000 frames held a period late in delivery (a host stall): a run by the slip
    # rule, but an interior one with the majority on both sides, voting below it, which is
    # what a backlog looks like. The anchor stays paired, names the first frame's own
    # transition, and records the run.
    period, early = 1 / FPS, 5
    transition_host = 100.0 + np.arange(9000 + 2 * early) * period
    frames = np.zeros(9000, dtype=CAMERA_FRAME_DTYPE)
    frames["frame_id"] = 5000 + np.arange(9000)
    frames["arrival_perf"] = transition_host[early:early + 9000] + 0.002
    frames["arrival_perf"][4000:4003] += period
    boundary = SimpleNamespace(primary_frame_id=5000)

    evidence, number = SessionDataRecorder._arrival_anchor(
        boundary, {"frames": frames}, "", transition_host, period, ni_residual_rms=0.00002)

    assert evidence["status"] == "paired", evidence["reason"]
    assert evidence["ambiguous"] is False and evidence["reason"] == ""
    assert evidence["slip"]["reverts"] is True and evidence["slip"]["frames"] == 3
    assert number == early


def test_the_first_frame_stamp_anchor_is_the_latest_transition_at_or_before_it():
    edges = np.array([10.0, 10.01, 10.02])
    stamp = SessionDataRecorder._first_frame_stamp_anchor
    assert stamp(10.015, edges, 0.01) == 1
    assert stamp(10.02, edges, 0.01) == 2          # at the transition counts
    assert stamp(9.999, edges, 0.01) is None       # before every transition
    assert stamp(10.031, edges, 0.01) is None      # over a period past the last
    assert stamp(10.031, edges, None) == 2         # unknown period: no upper bound


def test_a_fitted_anchor_that_raises_says_so_and_does_not_blame_the_fit(tmp_path, monkeypatch):
    def broken(*arguments, **keywords):
        raise RuntimeError("boom")

    monkeypatch.setattr(SessionDataRecorder, "_match_camera_nidaq_edge_on_fit",
                        staticmethod(broken))
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001)

    camera = result["cameraNidaqAlignment"]
    assert camera["status"] == "matched" and camera["anchorMethod"] == "epoch_nearest_edge"
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert camera["reason"].startswith("fitted anchor failed (RuntimeError: boom); ")
    assert "fit unavailable" not in camera["reason"]
    # The fit is fine and stays in the record: the frame timeline maps through it.
    assert camera["niToHost"]["valid"] is True
    # The no-fit reason is unchanged.
    rig, project, n0, result = _record(tmp_path / "no_stamps", t0=5.0, lag=0.001, stamps=False)
    assert result["cameraNidaqAlignment"]["reason"].startswith("NI-to-host fit unavailable (")


def test_with_no_anchor_on_the_fit_the_session_is_unmatched_and_says_why(tmp_path, monkeypatch):
    # No latency stream and no first-frame-stamp anchor: nothing names the first frame's
    # transition. A fault in the unmatched branch is swallowed into the epoch fallback by
    # the outer try, so this also pins that the session did not quietly take that way out.
    monkeypatch.setattr(SessionDataRecorder, "_first_frame_stamp_anchor",
                        staticmethod(lambda *arguments, **keywords: None))
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, stream=False)

    camera = result["cameraNidaqAlignment"]
    assert camera["status"] == "unmatched" and camera["confidence"] == "none"
    assert camera["anchorMethod"] is None and camera["anchorMethod"] != "epoch_nearest_edge"
    assert camera["latch"]["status"] == camera["arrival"]["status"] == "unavailable"
    for fragment in ("camera-clock match not available", "arrival match not available",
                     "no transition lies at or before the first frame's host stamp"):
        assert fragment in camera["reason"]
    assert "fitted anchor failed" not in camera["reason"]


def test_a_latency_stream_that_cannot_be_looked_for_costs_the_latch_and_never_raises(
    tmp_path, monkeypatch,
):
    # An existence check can fail too (PermissionError on a locked share), and finalization
    # must go on with the first-frame stamp anchor.
    def denied(self):
        if self.suffix == ".h5":
            raise PermissionError("access denied")
        return original_is_file(self)

    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", denied)

    stream, reason = SessionDataRecorder._read_primary_latency_stream(
        _project(tmp_path), SimpleNamespace(primary_camera="left"))

    assert stream is None
    assert "camera_left.h5" in reason and "PermissionError" in reason


def test_the_latency_stream_is_read_only_when_the_fitted_anchor_can_use_it(tmp_path, monkeypatch):
    reads = []
    original = SessionDataRecorder._read_primary_latency_stream

    def counting(project, boundary):
        reads.append(boundary.primary_camera)
        return original(project, boundary)

    monkeypatch.setattr(SessionDataRecorder, "_read_primary_latency_stream",
                        staticmethod(counting))

    # No fit: the epoch method needs no latches, so a locked stream would cost its retry
    # budget for nothing.
    _record(tmp_path / "no_fit", t0=5.0, lag=0.001, stamps=False)
    assert reads == []
    rig, project, n0, result = _record(tmp_path / "fit", t0=5.0, lag=0.001)
    assert reads == ["left"]
    assert result["cameraNidaqAlignment"]["anchorMethod"] == "camera_latch"


@pytest.mark.parametrize("exposure", (None, 0.0055, 6.0, -0.005))
def test_the_exposure_is_recorded_and_never_moves_the_anchor(tmp_path, exposure):
    # Unknown, as long as the exposure of a long frame, a units mix-up or a sign slip: none of
    # it matters to the camera-clock and arrival matches, which take no exposure.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, exposure=exposure)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["confidence"] == "hardware_edge"
    assert camera["exposureSeconds"] == (None if exposure is None else pytest.approx(exposure))
    assert not any(note in camera["reason"]
                   for note in ("assumed 0", "is negative", "not shorter than the frame period"))


@pytest.mark.parametrize("damage", ["shorter", "longer"])
def test_a_raw_column_of_the_wrong_size_costs_the_bridge_and_not_the_fit(tmp_path, damage):
    # A shorter column raised inside the run-end lookup and cost the whole fit (and with it the
    # fitted anchor); a longer one lined up with nothing and was bridged through as if it did.
    rig = SyntheticRig(polls=((10.0, 0.004),))
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=25.0, raw=True)
    whole = SessionDataRecorder._fit_nidaq_to_host(spool)
    assert whole.valid and whole.method == clocks.RAW_BRIDGED
    with h5py.File(spool.path, "r+") as store:
        column = store["block_observation_raw_time"][()]
        del store["block_observation_raw_time"]
        damaged = column[:-1] if damage == "shorter" else np.append(column, 1.0)
        store.create_dataset("block_observation_raw_time", data=damaged)

    fit = SessionDataRecorder._fit_nidaq_to_host(spool)

    assert fit.valid, fit.reason
    assert fit.method == clocks.PIECEWISE_ENVELOPE
    assert fit.reason == (f"RAW stamps unusable ({damaged.size} values against {column.size} "
                          "block_observation_perf_time stamps)")
    # The perf stamps alone still map within the delivery floor of the host clock.
    probe = np.linspace(fit.x_min, fit.x_max, 5_000)
    late = fit.map(probe) - rig.host(probe)
    assert late.min() > -0.0001 and late.max() < 0.0005


def test_raw_stamps_bridge_the_fit_through_an_ntp_step(tmp_path):
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.00175, raw=True,
                                       polls=((10.0, 0.004),))

    camera = result["cameraNidaqAlignment"]
    assert camera["niToHost"]["method"] == "raw_bridged"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["confidence"] == "hardware_edge"
    fit = clocks.fit_from_record(camera["niToHost"])
    probe = np.linspace(5.0, 24.0, 2000)
    error = fit.map(probe) - rig.host(probe)
    assert 0.0 < error.min() and error.max() < 0.0006  # late by the delivery floor, through the step


def _copies_of_a_long_session_then_one_late_at_stop():
    """(chunks, end_perf): 45 minutes of recorder copies, one sample each so the test is
    light, then a copy after the last frame whose stamp is alone in its envelope bin."""
    rng = np.random.default_rng(7)

    def copy_of(sample, stamp):
        index = np.array([sample], dtype=np.int64)
        return (index, index / RATE, index / RATE, np.zeros((1, 1), dtype=np.float32), ("barcode",),
                RATE, 1, 0, 0, stamp, stamp + 90.0, stamp - 7.0)

    samples = np.arange(1, 27_000) * 1_000  # one every 0.1 s
    stamps = samples / RATE + 0.0002 + rng.uniform(0, 0.001, samples.size)
    chunks = [copy_of(int(sample), float(stamp)) for sample, stamp in zip(samples, stamps)]
    end_perf = float(stamps[-1])
    # 2700.2 s opens a new 6 s bin; the next stamp is the only one to bound that sample.
    chunks.append(copy_of(27_002_000, 2700.2005))
    chunks.append(copy_of(27_003_000, 2701.3))
    return tuple(chunks), end_perf


def test_a_late_copy_after_the_last_frame_does_not_cost_a_long_session_its_fit():
    # 2026-11-05: a 45-minute session's fit came back "a knot window has no envelope point",
    # and its anchor fell back to the epoch, while the nidaq.h5 written from the same spool
    # fitted cleanly: that file ends at the last frame, and the spool runs on through Stop.
    chunks, end_perf = _copies_of_a_long_session_then_one_late_at_stop()
    assert not SessionDataRecorder._fit_nidaq_to_host(chunks).valid  # what the whole spool gives

    fit = SessionDataRecorder._fit_nidaq_to_host(chunks, end_perf=end_perf)

    assert fit.valid, fit.reason
    assert fit.method == clocks.RAW_BRIDGED
    assert fit.x_max < end_perf


def test_the_session_is_fitted_on_its_own_stamps_and_not_the_spools_tail(tmp_path):
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, raw=True, spool_tail_seconds=3.0)

    camera = result["cameraNidaqAlignment"]
    assert camera["niToHost"]["valid"]
    assert 24.0 < camera["niToHost"]["x_max"] < 25.0
    assert camera["confidence"] == "hardware_edge"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)


def test_with_raw_latches_a_perf_slew_does_not_cost_the_anchor(tmp_path):
    # A 5 ms timesyncd correction at 12 s slews perf_counter under the latches and the
    # transitions; both are placed on CLOCK_MONOTONIC_RAW, which it does not touch.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, raw=True, raw_latches=True,
                                       polls=((12.0, 0.005),))

    camera = result["cameraNidaqAlignment"]
    latch, arrival = camera["latch"], camera["arrival"]
    assert camera["confidence"] == "hardware_edge"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert latch["clock"] == "raw" and latch["slip"] is None
    assert arrival["slip"] is None
    # Six identical +-0.1 ms brackets, three at each end: every line through them lies within
    # the half-bracket (plus the clock's resolution) of the map, and none is left out.
    assert latch["uncertaintySeconds"] == pytest.approx(1e-4, abs=1e-6)
    assert latch["dropped"] == []
    # The residual tolerance is a quarter of the measured period (under the 2 ms cap here).
    assert latch["limitSeconds"] == pytest.approx(camera["framePeriodSeconds"] / 4)


@pytest.mark.parametrize("raw, raw_latches", [(False, False), (True, False), (False, True)],
                         ids=["neither", "ni_only", "latches_only"])
def test_without_raw_latches_the_anchor_says_perf(tmp_path, raw, raw_latches):
    # RAW on one side only is no RAW map: the transitions and the latches must share a clock.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, raw=raw,
                                       raw_latches=raw_latches)

    camera = result["cameraNidaqAlignment"]
    latch = camera["latch"]
    assert latch["clock"] == "perf"
    assert camera["confidence"] == "hardware_edge" and latch["slip"] is None
    assert latch["uncertaintySeconds"] == pytest.approx(1e-4, abs=1e-6)


@pytest.mark.parametrize("share", [0.3, 0.7])
def test_a_lost_ni_transition_is_a_slip_in_both_pairings(tmp_path, share):
    # One cam_frames toggle missing: every frame after it pairs one transition earlier, so the
    # votes change shift there. Before mid-session the minority run opens the session, after
    # it the run closes it; either way the slip names where the votes change.
    t0, seconds = 5.0, 20.0
    drop = t0 + share * seconds
    rig, project, n0, result = _record(tmp_path, t0=t0, lag=0.001, seconds=seconds,
                                       drop_edge_at=drop)

    camera = result["cameraNidaqAlignment"]
    lost = rig.frame_id(rig.first_edge_at_or_after(drop))
    assert camera["confidence"] == "ambiguous_hardware_edge"
    slips = [camera["latch"]["slip"], camera["arrival"]["slip"]]
    assert any(slip is not None and abs(slip["frameId"] - lost) <= 1 for slip in slips), slips
    for name in ("latch", "arrival"):
        evidence = camera[name]
        assert evidence["status"] == "ambiguous", (name, evidence["reason"])
        assert evidence["slip"] is not None and abs(evidence["slip"]["frameId"] - lost) <= 1, name
        assert f"slip at frame {evidence['slip']['frameId']}" in evidence["reason"], name


def test_wide_latch_brackets_leave_the_anchor_ambiguous(tmp_path):
    # 4 ms brackets: the latch map alone is 2 ms uncertain, over a quarter of the 6.7 ms
    # period, so the camera clock cannot choose a transition and only the arrival match is left.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, latch_half_bracket=0.002)

    camera = result["cameraNidaqAlignment"]
    latch = camera["latch"]
    assert "latch mapping uncertainty 2.000 ms" in latch["reason"]
    assert "exceeds a quarter frame period" in latch["reason"]
    assert latch["status"] == "ambiguous" and latch["uncertaintySeconds"] == pytest.approx(
        0.002, abs=1e-6)
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert camera["anchorMethod"] == "host_arrival"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)


def test_a_latch_left_out_of_the_map_is_listed_in_the_anchor_evidence(tmp_path):
    # One latch's host stamp 3 ms off: no line through the other five passes through its
    # bracket, so the map is made without it and the evidence says which one and why.
    rig = SyntheticRig()
    n0 = rig.first_edge_at_or_after(5.0)
    count = 3000
    stream = read_raw_stream(rig.write_latency_stream(_project(tmp_path), "left", n0, count, 0.001))
    stream["clock_latches"]["perf_before"][4] += 0.003
    stream["clock_latches"]["perf_after"][4] += 0.003
    # The true host time of every edge from ten before the first frame's.
    transitions = rig.host(rig.edge_time(np.arange(n0 - 10, n0 + count + 10)))

    evidence, number = SessionDataRecorder._latch_anchor(
        SimpleNamespace(primary_frame_id=rig.frame_id(n0)), stream, "", transitions, 1 / FPS,
        ni_residual_rms=4e-5, transition_raw=None)

    assert evidence["status"] == "paired" and number == 10
    assert evidence["dropped"] == [4]
    assert "latch 4 left out" in evidence["reason"]
    assert evidence["clock"] == "perf" and evidence["usableLatches"] == 6
    # A first frame whose transition is outside the spool still says which latch was left out.
    evidence, number = SessionDataRecorder._latch_anchor(
        SimpleNamespace(primary_frame_id=rig.frame_id(n0) - 20), stream, "", transitions, 1 / FPS,
        ni_residual_rms=4e-5, transition_raw=None)
    assert evidence["status"] == "outside" and number is None
    assert "(-10) is outside the spool" in evidence["reason"]
    assert "latch 4 left out" in evidence["reason"]
    assert evidence["dropped"] == [4]


def test_a_3_ms_slew_needs_the_raw_map_to_pair(tmp_path):
    # perf_counter takes a 3 ms bump mid-session, gone again at both ends where the latches are:
    # the perf latch line cannot follow it, and RAW, which no slew touches, is not in it. The
    # `clock` label alone cannot tell the two apart; this outcome rests on the numbers. By
    # hand: half the frames sit under 3 ms x sin(pi/4) = 2.12 ms of bump, and less the 0.175 ms
    # exposure the perf residual's median is about 1.95 ms, over the 1.67 ms quarter period.
    rig = SyntheticRig()
    n0 = rig.first_edge_at_or_after(5.0)
    stream = read_raw_stream(rig.write_latency_stream(_project(tmp_path), "left", n0, 3000, 0.001,
                                                      raw_latches=True))
    t = rig.edge_time(np.arange(n0 - 10, n0 + 3010))
    perf = rig.host(t) + 0.003 * np.sin(np.pi * (t - t[0]) / (t[-1] - t[0]))
    boundary = SimpleNamespace(primary_frame_id=rig.frame_id(n0))

    def anchor(transition_raw):
        return SessionDataRecorder._latch_anchor(
            boundary, stream, "", perf, 1 / FPS, ni_residual_rms=4e-5,
            transition_raw=transition_raw)

    on_perf, perf_number = anchor(None)
    on_raw, raw_number = anchor(rig.raw(t))

    assert on_perf["status"] == "ambiguous" and perf_number is None
    assert on_perf["clock"] == "perf" and "over the latch tolerance" in on_perf["reason"]
    assert on_perf["medianResidualSeconds"] == pytest.approx(0.00195, abs=5e-5)
    assert on_raw["status"] == "paired" and raw_number == 10
    assert on_raw["clock"] == "raw" and on_raw["slip"] is None
    assert on_raw["medianResidualSeconds"] == pytest.approx(-EXPOSURE, abs=1e-6)


def test_raw_on_one_latch_burst_only_anchors_on_perf(tmp_path):
    # clocks.latch_clock through the recorder's own wiring, on a real RAW-bridged NI fit.
    rig = SyntheticRig()
    project = _project(tmp_path)
    n0 = rig.first_edge_at_or_after(5.0)
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=25.0, raw=True)
    fit = SessionDataRecorder._fit_nidaq_to_host(spool)
    assert fit.has_raw, fit.reason
    start_perf, _, boundary = rig.start(project, n0, 0.001)
    stream = read_raw_stream(rig.write_latency_stream(project, "left", n0, 2000, 0.001,
                                                      raw_latches=True))

    def anchored():
        return SessionDataRecorder._match_camera_nidaq_edge(
            boundary, start_perf, spool, ni_fit=fit, exposure_seconds=EXPOSURE,
            read_latency_stream=lambda: (stream, ""))

    assert anchored()["latch"]["clock"] == "raw"
    stream["clock_latches"]["raw_before"][3:] = np.nan
    stream["clock_latches"]["raw_after"][3:] = np.nan

    camera = anchored()

    assert camera["latch"]["clock"] == "perf" and camera["latch"]["usableLatches"] == 6
    assert camera["confidence"] == "hardware_edge"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)


def test_alignment_schema_four_keeps_the_old_fields_and_records_both_matches(tmp_path):
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001)

    alignment = _alignment(project)
    assert alignment["schemaVersion"] == 4
    camera = alignment["cameraNidaqAlignment"]
    for key in ("cameraFrameId", "cameraPerfTime", "cameraTimestamp", "channel",
                "matchedSampleIndex", "matchedNidaqPerfTime", "signedOffsetSeconds",
                "resolutionSeconds", "ambiguitySeconds", "edgePolarity", "status",
                "confidence", "reason"):
        assert key in camera, key
    for key in ("anchorMethod", "niToHost", "exposureSeconds", "impliedLagSeconds",
                "framePeriodSeconds", "latch", "anchorsAgree", "arrival", "arrivalSampleIndex"):
        assert key in camera, key
    for key in ("expectedLagSeconds", "lagGateSeconds", "expectedLagSampleIndex"):
        assert key not in camera, key
    assert camera["signedOffsetSeconds"] == pytest.approx(-camera["impliedLagSeconds"])
    assert camera["exposureSeconds"] == pytest.approx(EXPOSURE)
    # At 10 kHz, 150 fps transitions are 67, 67, 66 samples apart: the median is
    # 67 samples, 6.7 ms, 33 us over 1/150.
    assert camera["framePeriodSeconds"] == pytest.approx(1 / FPS, abs=5e-5)
    assert len(camera["niToHost"]["knotHost"]) == len(camera["niToHost"]["knots"]) >= 2
    assert camera["latch"]["status"] == "paired" and camera["latch"]["usableLatches"] == 6
    assert camera["arrival"]["status"] == "paired" and camera["arrival"]["agreement"] == 1.0
    # The measured limits, additive inside the evidence (still schemaVersion 4).
    for key in ("clock", "uncertaintySeconds", "limitSeconds", "slip", "dropped"):
        assert key in camera["latch"], key
    for key in ("limitSeconds", "slip"):
        assert key in camera["arrival"], key
    lag, period = camera["arrival"]["medianLagSeconds"], camera["framePeriodSeconds"]
    # The arrival margin is measured, and the lag clears it on both sides.
    assert 0.0 < camera["arrival"]["limitSeconds"] < min(lag, period - lag)
    assert camera["ambiguitySeconds"] == pytest.approx(min(lag, period - lag))
    assert camera["edgePolarity"] in ("rising", "falling")


def test_the_post_analysis_ledger_reuses_the_stored_fit_and_the_anchor_before_the_file(tmp_path):
    # RED today: the anchor edge is not in the trimmed nidaq.h5, so the ledger update
    # falls back to the writer timeline and disagrees with finalization. An event 0.8 ms
    # after F+3's edge is F+4 on the NI timeline, and F+3 on the writer timeline, where
    # frame_perf_c lags the edge by 1.75 ms.
    model = SyntheticRig()
    n0 = model.first_edge_at_or_after(5.0)
    send = float(model.host(model.edge_time(n0 + 3))) + 0.0008
    record = {"session_id": "s", "operation_id": "send-1", "trial_id": 1, "attempt_id": 1,
              "attempt_label": "1.1", "send_perf_time": send, "outcome": "pending_analysis"}
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.00175, trial_records=(record,))
    streams = Path(project.get_session_path().location) / "streams"

    def send_frame():
        trial = json.loads((streams / "trials.jsonl").read_text().strip())
        return trial["event_alignment"]["send"]["frameAssociation"]

    before = send_frame()
    SessionDataRecorder.update_persisted_trial_ledger(
        project, ({**record, "outcome": "success"},), {"pending_analysis_attempts": 0})
    after = send_frame()

    assert before["frameId"] == after["frameId"] == rig.frame_id(n0 + 4)
    assert after["method"].endswith("nidaq_camera_exposure_timeline_verified_by_writer")
    assert after["frameStartPerfTime"] == pytest.approx(before["frameStartPerfTime"], abs=1e-9)
    with h5py.File(streams / "nidaq.h5", "r") as store:
        first_saved = int(store["sample_index"][0])
    assert result["cameraNidaqAlignment"]["matchedSampleIndex"] < first_saved


_SCHEMA_TWO_KEYS = ("cameraFrameId", "cameraPerfTime", "cameraTimestamp", "channel",
                    "matchedSampleIndex", "matchedNidaqPerfTime", "signedOffsetSeconds",
                    "resolutionSeconds", "ambiguitySeconds", "edgePolarity", "status",
                    "confidence", "reason")


def test_a_schema_two_alignment_and_a_nidaq_without_raw_still_read(tmp_path):
    model = SyntheticRig()
    n0 = model.first_edge_at_or_after(5.0)
    send = float(model.host(model.edge_time(n0 + 3))) - 0.001
    record = {"session_id": "s", "operation_id": "send-1", "trial_id": 1, "attempt_id": 1,
              "attempt_label": "1.1", "send_perf_time": send, "outcome": "pending_analysis"}
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, trial_records=(record,))
    session_dir = Path(project.get_session_path().location)
    streams = session_dir / "streams"
    # What a session finalized before this change left behind: no fit, no evidence,
    # schemaVersion 2, and a nidaq.h5 without the RAW column.
    alignment = _alignment(project)
    alignment["schemaVersion"] = 2
    alignment["cameraNidaqAlignment"] = {
        key: value for key, value in alignment["cameraNidaqAlignment"].items()
        if key in _SCHEMA_TWO_KEYS
    }
    (streams / "alignment.json").write_text(json.dumps(alignment))
    with h5py.File(streams / "nidaq.h5", "r") as source:
        datasets = {name: source[name][()] for name in source
                    if name != "block_observation_raw_time"}
        attrs = dict(source.attrs)
    with h5py.File(streams / "nidaq.h5", "w") as target:
        for name, data in datasets.items():
            target.create_dataset(name, data=data)
        for key, value in attrs.items():
            target.attrs[key] = value

    SessionDataRecorder.update_persisted_trial_ledger(
        project, ({**record, "outcome": "success"},), {"pending_analysis_attempts": 0})
    status = finalize_session_latency(session_dir, primary_camera="left")

    trial = json.loads((streams / "trials.jsonl").read_text().strip())
    frame = trial["event_alignment"]["send"]["frameAssociation"]
    # Without a stored fit the timeline is the epoch one, as it was: the edge reads
    # 1.65 ms early, so an event 1 ms before it is already "after" F+3.
    assert frame["frameId"] == rig.frame_id(n0 + 4)
    assert frame["method"].endswith("nidaq_camera_exposure_timeline_verified_by_writer")
    assert status["clocks"]["niToHost"]["method"] == "piecewise_envelope"
    assert status["clocks"]["niToHost"]["valid"] is True


def test_a_tone_just_after_an_exposure_edge_maps_to_the_next_frame_without_mapping_the_spool(
    tmp_path, monkeypatch,
):
    # Tones on the epoch timeline sit 1.65 ms early beside frames on the fit, which
    # would put a tone 0.8 ms after F+3's edge before it. Both go through the fit, and
    # only the samples in use do: the spool here is 250 000 samples, its transitions
    # about 3 750.
    sizes = []
    original_map = clocks.PiecewiseClockFit.map

    def counting_map(self, x):
        sizes.append(int(np.asarray(x).size))
        return original_map(self, x)

    monkeypatch.setattr(clocks.PiecewiseClockFit, "map", counting_map)
    model = SyntheticRig()
    n0 = model.first_edge_at_or_after(5.0)
    t_tone = float(model.edge_time(n0 + 3)) + 0.0008
    device_rows = ((float(model.host(t_tone)) + 0.002, 1.7e9, "inbound", "STIMULUS_INPUTS",
                    "PELLET_DEVICE", "status-1", None, None, json.dumps({"tone1": True})),)
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001,
                                       tone_pulses=((t_tone, 0.005),), device_rows=device_rows)

    matched = result["toneConfirmation"]["matched"]
    assert len(matched) == 1
    assert matched[0]["frameAssociation"]["frameId"] == rig.frame_id(n0 + 4)
    # The pulse's host time is the fitted one: late by the delivery floor, never early.
    assert 0.0 < matched[0]["physicalEventPerfTime"] - float(rig.host(t_tone)) < 0.0006
    assert matched[0]["latencySeconds"] == pytest.approx(-0.002, abs=0.0006)
    assert sizes and max(sizes) < 5_000, max(sizes)


def test_host_times_map_to_ni_sample_indices_through_the_inverse_fit(tmp_path):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=20.0)
    fit = SessionDataRecorder._fit_nidaq_to_host(spool)
    assert fit.valid, fit.reason
    host = np.array([float(rig.host(5.0)), float(rig.host(12.3456))])

    indices = SessionDataRecorder._pressure_nidaq_indices(host, spool, ni_fit=fit)
    legacy = SessionDataRecorder._pressure_nidaq_indices(host, spool)
    beyond = SessionDataRecorder._pressure_nidaq_indices(
        np.array([float(rig.host(25.0))]), spool, ni_fit=fit)
    # Before the first stored sample (the spool starts at NI second 0): no match, as at the
    # other edge.
    before = SessionDataRecorder._pressure_nidaq_indices(
        np.array([float(rig.host(-1.0))]), spool, ni_fit=fit)

    # The fit maps late by the delivery floor, so the inverse lands 2-3 samples early.
    assert abs(indices[0] - 50_000) <= 4 and abs(indices[1] - 123_456) <= 4
    assert abs(legacy[0] - 50_000) >= 15  # the epoch timeline: 1.65 ms is 16.5 samples off
    assert beyond.tolist() == [-1]
    assert before.tolist() == [-1]


def test_a_host_time_inside_a_sample_index_gap_takes_the_sample_before_the_gap(tmp_path):
    rig = SyntheticRig()
    whole = rig.write_spool(tmp_path / "whole.h5", t_from=0.0, t_to=20.0)
    fit = SessionDataRecorder._fit_nidaq_to_host(whole)
    assert fit.valid, fit.reason
    # Samples 60 000-60 999 (NI seconds 6.0-6.1) were never stored, as when the task restarts.
    with h5py.File(whole.path, "r") as source, h5py.File(tmp_path / "gapped.h5", "w") as target:
        keep = (source["sample_index"][()] < 60_000) | (source["sample_index"][()] >= 61_000)
        for name in source:
            data = source[name][()]
            target.create_dataset(name, data=data[:, keep] if data.ndim == 2 else data[keep])
        for key, value in source.attrs.items():
            target.attrs[key] = value
    gapped = dataclasses.replace(whole, path=tmp_path / "gapped.h5")
    host = np.array([float(rig.host(6.05)), float(rig.host(5.0)), float(rig.host(6.5))])

    indices = SessionDataRecorder._pressure_nidaq_indices(host, gapped, ni_fit=fit)
    legacy = SessionDataRecorder._pressure_nidaq_indices(host, gapped)

    # Inside the window a gap is not looked for: a time in one takes the sample before it.
    assert indices[0] == legacy[0] == 59_999
    # And the times either side of it still map by their sample, 2-3 early as above.
    assert abs(indices[1] - 50_000) <= 4 and abs(indices[2] - 65_000) <= 4


def test_the_host_to_ni_lookup_maps_only_the_events_and_builds_nothing_the_length_of_the_spool(
    tmp_path, monkeypatch,
):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=20.0)
    fit = SessionDataRecorder._fit_nidaq_to_host(spool)
    host = np.array([float(rig.host(5.0)), float(rig.host(12.3456))])
    sizes = []
    original_inverse = clocks.PiecewiseClockFit.inverse

    def counting_inverse(self, x):
        sizes.append(int(np.asarray(x).size))
        return original_inverse(self, x)

    def peak_bytes(**keywords):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            SessionDataRecorder._pressure_nidaq_indices(host, spool, **keywords)
            return tracemalloc.get_traced_memory()[1] - before
        finally:
            tracemalloc.stop()

    peak_bytes(), peak_bytes(ni_fit=fit)  # warm-up: first-use allocations belong to neither
    monkeypatch.setattr(clocks.PiecewiseClockFit, "inverse", counting_inverse)
    epoch_peak = peak_bytes()
    fitted_peak = peak_bytes(ni_fit=fit)

    assert sizes == [2]
    # Both read the spool's index and time (16 bytes a sample) and nothing else of its
    # length. A float key would make numpy promote the whole stored index to float64
    # for the search, 8 bytes a sample more: 1.6 MB here, against the 0.4 MB allowed.
    assert fitted_peak - epoch_peak < 0.25 * 8 * 200_000, (epoch_peak, fitted_peak)


class _FitThatFails:
    """A valid fit whose mapping runs out of memory: for every input, or only arrays."""
    valid = True
    method = "failing"

    def __init__(self, real, *, only_arrays=False):
        self._real, self._only_arrays = real, only_arrays

    def _check(self, x):
        if not self._only_arrays or np.ndim(x):
            raise MemoryError("out of memory")

    def map(self, x):
        self._check(x)
        return self._real.map(x)

    def inverse(self, host):
        self._check(host)
        return self._real.inverse(host)


class _FitReturningNaN:
    valid = True
    method = "nan"

    def map(self, x):
        return np.full(np.shape(x), np.nan)

    inverse = map


def test_host_to_ni_indices_fall_back_to_the_epoch_lookup_when_the_fit_cannot_be_used(
    tmp_path, caplog,
):
    rig = SyntheticRig()
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=20.0)
    fit = SessionDataRecorder._fit_nidaq_to_host(spool)
    host = np.array([float(rig.host(5.0)), float(rig.host(12.3456))])
    legacy = SessionDataRecorder._pressure_nidaq_indices(host, spool)
    # A sample rate of 1e-320 is positive and finite, and its period is infinite.
    useless_rate = dataclasses.replace(spool, sample_rate_hz=1e-320)

    # A usable-looking fit that fails says why in the log; one known to be invalid already
    # did, in alignment.json's niToHost.
    for name, unusable, logged in (
        ("no fit", None, ""),
        ("an invalid fit", clocks.invalid_piecewise_fit("no read pairs"), ""),
        ("a fit that runs out of memory", _FitThatFails(fit), "MemoryError: out of memory"),
        ("a fit that maps to NaN", _FitReturningNaN(), "ArithmeticError: the NI-to-host inverse"),
    ):
        caplog.clear()
        result = SessionDataRecorder._pressure_nidaq_indices(host, spool, ni_fit=unusable)
        assert result.tolist() == legacy.tolist(), name
        assert (logged in caplog.text) and bool(caplog.records) == bool(logged), name
    assert (SessionDataRecorder._pressure_nidaq_indices(host, useless_rate, ni_fit=fit).tolist()
            == SessionDataRecorder._pressure_nidaq_indices(host, useless_rate).tolist())


def test_laser_rows_name_their_ni_sample_on_the_fitted_timeline(tmp_path):
    rig = SyntheticRig()
    event = 10.0
    # perf, wall, event, channel, source, command, diode, command copy
    row = (float(rig.host(event)), 1.7e9 + 10.0, "pulse", "laser1", "protocol", 0.0, 0.0, 0.0)
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, laser_rows=(row,))

    streams = Path(project.get_session_path().location) / "streams"
    with (streams / "laser.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    # The inverse fit lands 2-3 samples early (the delivery floor); the epoch lookup it
    # replaces was 16-17 samples late here.
    assert abs(int(rows[0]["nidaq_sample_index"]) - int(event * RATE)) <= 4


def _tone_inputs(tmp_path):
    """A spool with a tone 0.8 ms after F+3's exposure edge, the recording boundary
    at frame n0, and the call finalization makes, minus the frame ledger: without a
    ready frame timeline the tone's frame comes from the spool's own cam_frames."""
    rig = SyntheticRig()
    project = _project(tmp_path)
    n0 = rig.first_edge_at_or_after(5.0)
    t_tone = float(rig.edge_time(n0 + 3)) + 0.0008
    spool = rig.write_spool(tmp_path / "spool.h5", t_from=0.0, t_to=25.0,
                            tone_pulses=((t_tone, 0.005),))
    fit = SessionDataRecorder._fit_nidaq_to_host(spool)
    assert fit.valid, fit.reason
    start_perf, start_wall, boundary = rig.start(project, n0, 0.001)
    # hardware_edge needs both measured matches, so the anchor is given the latency stream.
    rig.write_latency_stream(project, "left", n0, 2000, 0.001)
    camera = SessionDataRecorder._match_camera_nidaq_edge(
        boundary, start_perf, spool, ni_fit=fit, exposure_seconds=EXPOSURE,
        read_latency_stream=lambda: SessionDataRecorder._read_primary_latency_stream(
            project, boundary))
    assert camera["status"] == "matched", camera["reason"]
    rows = ((float(rig.host(t_tone)) + 0.002, 1.7e9, "inbound", "STIMULUS_INPUTS",
             "PELLET_DEVICE", "status-1", None, None, json.dumps({"tone1": True})),)

    def correlate(ni_fit, chunks=spool):
        return SessionDataRecorder._correlate_tone_confirmations(
            rows, chunks, start_perf=start_perf, start_wall=start_wall, boundary=boundary,
            camera_alignment=camera, ni_fit=ni_fit)

    return rig, n0, t_tone, fit, correlate, spool


def test_without_a_frame_ledger_a_tone_names_its_frame_from_the_fitted_transitions(tmp_path):
    rig, n0, t_tone, fit, correlate, _ = _tone_inputs(tmp_path)

    tone = correlate(fit)
    legacy = correlate(None)

    match, frame = tone["matched"][0], tone["matched"][0]["frameAssociation"]
    assert frame["status"] == "matched" and frame["frameId"] == rig.frame_id(n0 + 4)
    # F+4's edge on the fitted timeline: late by the delivery floor, where the epoch
    # timeline (the legacy answer below) has it 1.65 ms early.
    assert 0.0 < frame["frameStartPerfTime"] - float(rig.host(rig.edge_time(n0 + 4))) < 0.0006
    assert frame["eventToFrameStartSeconds"] == pytest.approx(1 / FPS - 0.0008, abs=0.0006)
    assert match["physicalEventPerfTime"] - legacy["matched"][0]["physicalEventPerfTime"] > 0.0015
    assert legacy["matched"][0]["frameAssociation"]["frameId"] == rig.frame_id(n0 + 4)
    assert "hostTimeFallbackReason" not in tone and "hostTimeFallbackReason" not in legacy


def test_tones_fall_back_whole_to_the_epoch_timeline_when_the_fit_cannot_map_them(
    tmp_path, caplog,
):
    rig, n0, t_tone, fit, correlate, spool = _tone_inputs(tmp_path)
    legacy = correlate(None)

    tone = correlate(_FitThatFails(fit))

    # Everything is the epoch answer, with the one difference that the camera anchor was on
    # the fit and this frame association no longer is: its confidence says so.
    lowered = copy.deepcopy(legacy["matched"])
    for match in lowered:
        assert match["frameAssociation"]["confidence"] == "hardware_edge"
        match["frameAssociation"]["confidence"] = "ambiguous_hardware_edge"
    assert tone["matched"] == lowered
    assert tone["hostTimeFallbackReason"] == "MemoryError: out of memory"
    assert "MemoryError" in caplog.text
    # An invalid fit, or none, is no failure here: the epoch timeline as before, and the
    # reason is the fit's own, in alignment.json.
    for quiet in (clocks.invalid_piecewise_fit("no read pairs"), None):
        assert correlate(quiet)["matched"] == legacy["matched"]
        assert "hostTimeFallbackReason" not in correlate(quiet)
    # 1e-320 samples a second is a finite rate, yet no sample has a finite host time.
    useless_rate = dataclasses.replace(spool, sample_rate_hz=1e-320)
    assert "no finite host time" in correlate(fit, useless_rate)["hostTimeFallbackReason"]


def test_a_fit_that_fails_only_on_the_transitions_puts_tone_and_frame_whole_on_the_epoch(
    tmp_path, caplog,
):
    # The onset maps (one scalar) and the camera transitions (an array) do not: the tone is
    # recomputed whole on the epoch timeline rather than left on the fit beside epoch
    # transitions, and it keeps its frame, which the events.tones rule requires.
    rig, n0, t_tone, fit, correlate, _ = _tone_inputs(tmp_path)
    legacy = correlate(None)

    tone = correlate(_FitThatFails(fit, only_arrays=True))

    match, frame = tone["matched"][0], tone["matched"][0]["frameAssociation"]
    assert match["physicalEventPerfTime"] == legacy["matched"][0]["physicalEventPerfTime"]
    assert frame["status"] == "matched" and frame["frameId"] == rig.frame_id(n0 + 4)
    assert frame["frameStartPerfTime"] == legacy["matched"][0]["frameAssociation"]["frameStartPerfTime"]
    assert legacy["matched"][0]["frameAssociation"]["confidence"] == "hardware_edge"
    assert frame["confidence"] == "ambiguous_hardware_edge"
    assert tone["hostTimeFallbackReason"] == "MemoryError: out of memory"
    assert "MemoryError" in caplog.text


def test_a_tone_fallback_beside_a_ready_frame_timeline_keeps_its_frame_at_a_lower_confidence(
    tmp_path, monkeypatch,
):
    # The frame timeline is built from the stored fit, so with one ready the tone's frame is
    # compared against fitted frames and cannot be moved to the epoch with its onset. It keeps
    # a frame and says what happened: the epoch onset beside fitted frames is flagged, not
    # presented at the camera anchor's hardware_edge.
    original_map = clocks.PiecewiseClockFit.map

    def refuse_scalars(self, x):
        if not np.ndim(x):  # a tone onset; the camera edges and the frame timeline map arrays
            raise MemoryError("out of memory")
        return original_map(self, x)

    monkeypatch.setattr(clocks.PiecewiseClockFit, "map", refuse_scalars)
    model = SyntheticRig()
    n0 = model.first_edge_at_or_after(5.0)
    t_tone = float(model.edge_time(n0 + 3)) + 0.0008
    device_rows = ((float(model.host(t_tone)) + 0.002, 1.7e9, "inbound", "STIMULUS_INPUTS",
                    "PELLET_DEVICE", "status-1", None, None, json.dumps({"tone1": True})),)
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001,
                                       tone_pulses=((t_tone, 0.005),), device_rows=device_rows)

    tone, camera = result["toneConfirmation"], result["cameraNidaqAlignment"]
    match, frame = tone["matched"][0], tone["matched"][0]["frameAssociation"]
    assert tone["status"] == "complete" and len(tone["matched"]) == 1
    assert tone["hostTimeFallbackReason"] == "MemoryError: out of memory"
    # The onset is the epoch's: the pulse's sample on the epoch clock, to a sample.
    assert abs(match["physicalEventPerfTime"] - float(rig.epoch_perf(t_tone))) <= 1 / RATE
    assert frame["status"] == "matched" and frame["frameId"] is not None
    assert frame["method"].endswith("nidaq_camera_exposure_timeline_verified_by_writer")
    assert camera["confidence"] == "hardware_edge"
    assert frame["confidence"] == "ambiguous_hardware_edge"
