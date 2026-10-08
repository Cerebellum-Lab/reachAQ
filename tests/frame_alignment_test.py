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
from tools.acquisition.model.session_data_recorder import SessionDataRecorder
from tools.latency import clocks
from tools.latency.finalize import finalize_session_latency
from frame_alignment_synthetic import EXPOSURE, FPS, RATE, SyntheticRig


def _project(tmp_path, session=1):
    return ProjectInfo(root=str(tmp_path), device_id="test",
                       when=datetime(2026, 1, 2, 3, 4, 5), session=session)


def _record(tmp_path, *, t0, lag, seconds=20.0, latches=True, stream=True, exposure=EXPOSURE,
            polls=(), raw=False, stamps=True, tone_pulses=(), device_rows=(), laser_rows=(),
            trial_records=()):
    """One finalized session: the spool holds 5 s before the recording and `seconds` of it."""
    rig = SyntheticRig(polls=polls)
    project = _project(tmp_path)
    session_dir = Path(project.get_session_path().location)
    n0 = rig.first_edge_at_or_after(t0)
    count = int(seconds * FPS) - 2
    start_perf, start_wall, boundary = rig.start(project, n0, lag)
    end_perf = rig.frame_perf_c(n0 + count - 1, lag)
    spool = rig.write_spool(session_dir / "nidaq_capture.h5", t_from=t0 - 5.0, t_to=t0 + seconds,
                            tone_pulses=tone_pulses, raw=raw, stamps=stamps)
    rig.write_frame_ledger(project, n0, count, lag)
    if stream:
        rig.write_latency_stream(project, "left", n0, count, lag, latches=latches)
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
    # The implied lag is the true 1.75 ms less the fit's lateness, the synthetic's
    # documented 0.2-0.3 ms bias. A 1.5 ms bound left 8-68 us over the seeds, inside that
    # bias; 1.45 ms leaves 58-118 us.
    assert 0.00145 <= camera["impliedLagSeconds"] <= 0.0021
    assert camera["niToHost"]["valid"] and camera["niToHost"]["method"] == "piecewise_envelope"
    assert _alignment(project)["canonicalBoundary"]["nidaqSampleIndex"] == rig.edge_sample(n0)


def test_after_twelve_minutes_of_drift_the_expected_lag_anchor_finds_the_frame(tmp_path):
    # t0 = 700 s, L = 1.0 ms, no latency stream: the epoch timeline is now 5.3 ms late and the
    # epoch method takes the previous frame's edge (the design's F-1 case).
    rig, project, n0, result = _record(tmp_path, t0=700.0, lag=0.001, stream=False)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorMethod"] == "expected_lag" and camera["confidence"] == "hardware_edge"
    assert camera["latch"]["status"] == "unavailable"
    assert camera["expectedLagSeconds"] == pytest.approx(EXPOSURE + 0.0011)
    assert camera["expectedLagSampleIndex"] == rig.edge_sample(n0)


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
    assert _alignment(project)["schemaVersion"] == 3


def test_a_latch_anchor_that_disagrees_with_the_expected_lag_is_kept_but_flagged(tmp_path):
    # An exposure of 5.5 ms puts the expected edge one frame early; the latch still says F.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, exposure=0.0055)

    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorMethod"] == "camera_latch"
    assert camera["expectedLagSampleIndex"] == rig.edge_sample(n0 - 1)
    assert camera["anchorsAgree"] is False
    assert camera["confidence"] == "ambiguous_hardware_edge"
    assert "disagree" in camera["reason"] and "outside the gate" in camera["reason"]


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


def test_a_latency_stream_that_cannot_be_looked_for_costs_the_latch_and_never_raises(
    tmp_path, monkeypatch,
):
    # An existence check can fail too (PermissionError on a locked share), and finalization
    # must go on with the expected-lag anchor.
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


def test_an_exposure_that_is_not_known_is_said_so_in_the_anchor_reason(tmp_path):
    rig, project, n0, result = _record(tmp_path / "unknown", t0=5.0, lag=0.001, exposure=None)
    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["exposureSeconds"] is None
    assert camera["expectedLagSeconds"] == pytest.approx(0.0011)
    assert camera["reason"].endswith("exposure unknown, assumed 0")

    rig, project, n0, result = _record(tmp_path / "known", t0=5.0, lag=0.001)
    assert "exposure" not in result["cameraNidaqAlignment"]["reason"]


def test_an_exposure_as_long_as_the_frame_period_is_treated_as_unknown(tmp_path):
    # 6 "seconds" is a units mix-up, and any exposure of at least a frame is impossible. It
    # would have put the expected edge 6 s from every transition: no match at all.
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001, exposure=6.0)

    camera = result["cameraNidaqAlignment"]
    assert camera["status"] == "matched"
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["exposureSeconds"] is None
    assert camera["expectedLagSeconds"] == pytest.approx(0.0011)
    assert "exposure 6000.000 ms is not shorter than the frame period 6.700 ms" in camera["reason"]
    assert camera["reason"].endswith("exposure unknown, assumed 0")


def test_a_negative_exposure_is_treated_as_unknown_and_cannot_move_the_anchor(tmp_path):
    # -5 ms would put the expected edge, and the lag gate with it, 5 ms later than it is. With a
    # true lag of 2.1 ms the next frame's edge is then the nearest to it, and its implied lag
    # (about -4.8 ms) sits inside the moved gate (-4.9..-2.9 ms): the wrong frame, anchored at
    # full confidence. Unknown, it is the gate [0.1, 2.1] ms around 1.1 ms again.
    rig, project, n0, result = _record(tmp_path / "expected", t0=5.0, lag=0.0021, stream=False,
                                       exposure=-0.005)
    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorMethod"] == "expected_lag" and camera["confidence"] == "hardware_edge"
    assert camera["exposureSeconds"] is None
    assert camera["expectedLagSeconds"] == pytest.approx(0.0011)
    assert camera["lagGateSeconds"] == pytest.approx([0.0001, 0.0021])
    assert "exposure -5.000 ms is negative" in camera["reason"]
    assert camera["reason"].endswith("exposure unknown, assumed 0")

    # With the camera latches the anchor was right all along, but the moved expected edge
    # disagreed with it and took the session's confidence.
    rig, project, n0, result = _record(tmp_path / "latched", t0=5.0, lag=0.001, exposure=-0.005)
    camera = result["cameraNidaqAlignment"]
    assert camera["matchedSampleIndex"] == rig.edge_sample(n0)
    assert camera["anchorMethod"] == "camera_latch" and camera["anchorsAgree"] is True
    assert camera["confidence"] == "hardware_edge"
    assert "negative" in camera["reason"] and "disagree" not in camera["reason"]


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


def test_alignment_schema_three_keeps_every_old_field_and_adds_the_evidence(tmp_path):
    rig, project, n0, result = _record(tmp_path, t0=5.0, lag=0.001)

    alignment = _alignment(project)
    assert alignment["schemaVersion"] == 3
    camera = alignment["cameraNidaqAlignment"]
    for key in ("cameraFrameId", "cameraPerfTime", "cameraTimestamp", "channel",
                "matchedSampleIndex", "matchedNidaqPerfTime", "signedOffsetSeconds",
                "resolutionSeconds", "ambiguitySeconds", "edgePolarity", "status",
                "confidence", "reason"):
        assert key in camera, key
    for key in ("anchorMethod", "niToHost", "exposureSeconds", "expectedLagSeconds",
                "impliedLagSeconds", "lagGateSeconds", "framePeriodSeconds", "latch",
                "expectedLagSampleIndex", "anchorsAgree"):
        assert key in camera, key
    assert camera["signedOffsetSeconds"] == pytest.approx(-camera["impliedLagSeconds"])
    assert camera["exposureSeconds"] == pytest.approx(EXPOSURE)
    # At 10 kHz, 150 fps transitions are 67, 67, 66 samples apart: the median is
    # 67 samples, 6.7 ms, 33 us over 1/150.
    assert camera["framePeriodSeconds"] == pytest.approx(1 / FPS, abs=5e-5)
    assert camera["lagGateSeconds"] == pytest.approx([EXPOSURE + 0.0001, EXPOSURE + 0.0021])
    assert len(camera["niToHost"]["knotHost"]) == len(camera["niToHost"]["knots"]) >= 2
    assert camera["latch"]["status"] == "paired" and camera["latch"]["usableLatches"] == 6
    assert camera["edgePolarity"] in ("rising", "falling")
    assert camera["ambiguitySeconds"] > 0.004  # the runner-up is a whole period further


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
    camera = SessionDataRecorder._match_camera_nidaq_edge(
        boundary, start_perf, spool, ni_fit=fit, exposure_seconds=EXPOSURE)
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
