"""The NI delivery calibration's analysis, plan and configuration handling, without hardware."""

import math

import numpy as np
import pytest

from autotrainer.core import CameraConfiguration, CameraId, SystemConfiguration
from autotrainer.core.configuration import (
    NidaqDeliveryCalibration,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from tools.hardware import calibrate_ni_delivery as calibrate

RATE = 10_000.0


def test_the_plan_covers_every_cell_once_per_pass_in_a_seeded_order():
    plan = calibrate.plan_sessions((60, 120), (50, 150), (175,), passes=2, seconds=120.0, seed=7)

    assert len(plan) == 8
    for number in (1, 2):
        cells = {(s.refresh_hz, s.fps) for s in plan if s.name.endswith(f"_p{number}")}
        assert cells == {(60.0, 50), (60.0, 150), (120.0, 50), (120.0, 150)}
    assert plan == calibrate.plan_sessions((60, 120), (50, 150), (175,), passes=2,
                                           seconds=120.0, seed=7)
    assert {s.seconds for s in plan} == {120.0} and {s.exposure_us for s in plan} == {175}


def _square(levels, samples_each):
    return np.repeat(np.asarray(levels, dtype=float), samples_each)


def test_loopback_edges_are_the_first_sample_at_each_new_level():
    values = _square([0, 1, 0, 1], 1000)  # 100 ms levels at 10 kHz
    index = np.arange(values.size)

    seconds, rising = calibrate.loopback_edges(values, index, RATE)

    assert seconds.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert rising.tolist() == [True, False, True]


def test_an_edge_beside_a_short_level_is_left_out():
    values = _square([0, 1, 0, 1, 0], 1000)
    values[1500:1600] = 0.0  # a 10 ms dip, shorter than any commanded level
    index = np.arange(values.size)
    index[3500:] += 5        # a 5-sample gap inside a level only moves the later edges

    seconds, _ = calibrate.loopback_edges(values, index, RATE)

    assert seconds.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4005])


def test_an_edge_at_a_gap_in_the_samples_is_left_out():
    index = np.arange(3000)
    index[1000:] += 3        # three samples lost exactly at the 0.1 s edge

    seconds, _ = calibrate.loopback_edges(_square([0, 1, 0], 1000), index, RATE)

    assert seconds.tolist() == pytest.approx([0.2003])


def test_commands_are_matched_by_level_and_within_the_window():
    commands = [(1.0, 10.000, 10.0001), (0.0, 10.100, 10.1001), (1.0, 10.200, 10.2001)]

    matched = calibrate.match_commands([10.0003, 10.1003, 10.25], [True, False, True], commands)

    # The third edge is 50 ms from its command, past MATCH_SECONDS.
    assert matched.tolist() == [0, 1, -1]
    assert calibrate.match_commands([10.0003], [False], commands).tolist() == [-1]


def test_the_delivery_floor_undoes_the_scan_quantization_and_convert_offset():
    # Steps at random host times; the input is converted 56 us after each sample clock tick;
    # the first scan to see a step is the first tick whose convert follows it; the fit maps
    # NI time 0.25 ms late. The difference the tool sees is the mapped tick less the step.
    rng = np.random.default_rng(3)
    late, convert = 0.00025, 0.000056
    steps = np.sort(rng.uniform(1.0, 100.0, 2000))
    ticks = np.ceil((steps - convert) * RATE) / RATE
    differences = ticks + late - steps

    floor = calibrate.delivery_floor(differences, convert, RATE)

    # The quantization is uniform over one period: sd 28.9 us, so the mean of 2000 is ~0.7 us.
    assert floor == pytest.approx(late, abs=3e-6)
    assert np.std(differences) == pytest.approx(0.5 / RATE / math.sqrt(3), rel=0.05)
    assert math.isnan(calibrate.delivery_floor([], convert, RATE))


@pytest.mark.parametrize("inputs, why", [
    ({}, ""),
    ({"floor": 0.0052}, "half an NI read"),                   # 200 Hz: the fit a read late
    ({"residual": 0.002}, "quarter period"),  # the latch map 2 ms from the NI path
    ({"residual": None}, "no latch residual"),
    ({"floor": float("nan")}, "no loopback edge"),
    ({"slip": {"frameId": 4000}}, "slips at frame 4000"),
    ({"gate_reason": "latch mapping uncertainty 2.000 ms ..."}, "not trusted"),
], ids=["usable", "a_read_late", "a_latch_off", "no_residual", "no_edges", "a_slip",
        "an_untrusted_map"])
def test_a_session_calibrates_only_with_a_floor_under_half_a_read_and_a_frame_on(inputs, why):
    values = {"floor": 0.00025, "read_period": 0.0167, "residual": 0.00015,
              "exposure": 0.000175, "frame_period": 1 / 150, **inputs}
    if inputs.get("floor") == 0.0052:
        values["read_period"] = 0.005

    verdict = calibrate.session_verdict(**values)

    assert (verdict == "") if not why else (why in verdict)


def _session(name, offset, floor, *, reason="", chunk=167, fps=150, exposure=0.000175):
    return {"name": name, "reason": reason, "offsetSeconds": offset,
            "deliveryFloorSeconds": floor, "readChunkSamples": chunk, "fps": fps,
            "exposureSeconds": exposure, "cameraSerial": "24152533"}


def test_the_calibration_is_the_mean_and_spread_of_the_usable_sessions():
    sessions = [_session("a", 0.00030, 0.00024), _session("b", 0.00034, 0.00026, chunk=83, fps=50),
                _session("c", None, 0.0052, reason="the fit is a read late", chunk=50)]

    calibration, why = calibrate.aggregate(sessions, measured="2026-11-13T10:00:00-07:00")

    assert why == ""
    assert calibration.offset_seconds == pytest.approx(0.00032)
    assert calibration.spread_seconds == pytest.approx(np.std([0.00030, 0.00034], ddof=1))
    assert calibration.sessions == 2
    assert calibration.read_chunk_samples == (83, 167)
    assert calibration.frame_rates_hz == (50.0, 150.0)
    assert calibration.camera_serials == ("24152533",)
    assert calibration.delivery_floor_min_seconds == pytest.approx(0.00024)
    assert calibration.delivery_floor_max_seconds == pytest.approx(0.00026)


def test_one_usable_session_is_no_calibration():
    calibration, why = calibrate.aggregate([_session("a", 0.0003, 0.00024)], measured="")

    assert calibration is None and "needs two" in why


def test_a_session_the_others_cannot_hold_is_dropped_and_named():
    # Twelve sessions 8 us apart put the others' band (Student's t on 11 degrees of freedom)
    # about 0.09 ms wide; one 0.3 ms off is not the rig's spread.
    offsets = [0.0003 + 0.000008 * math.sin(k) for k in range(12)] + [0.0006]
    sessions = [_session(f"s{k}", offset, 0.00024) for k, offset in enumerate(offsets)]

    calibration, why = calibrate.aggregate(sessions, measured="")

    assert why == "" and calibration.sessions == 12
    assert calibration.offset_seconds == pytest.approx(np.mean(offsets[:12]))
    assert "prediction band" in sessions[-1]["reason"]
    assert all(not s["reason"] for s in sessions[:12])


def test_few_sessions_keep_a_wide_band_and_drop_nothing():
    # With five sessions t is 54 at 1e-6: a 0.3 ms outlier is not yet distinguishable.
    offsets = (0.000300, 0.000310, 0.000290, 0.000305, 0.000295, 0.000600)
    sessions = [_session(f"s{k}", offset, 0.00024) for k, offset in enumerate(offsets)]

    calibration, _ = calibrate.aggregate(sessions, measured="")

    assert calibration.sessions == 6


def test_sessions_that_agree_to_the_last_digit_give_no_spread_to_scale():
    calibration, why = calibrate.aggregate(
        [_session("a", 0.0003, 0.00024), _session("b", 0.0003, 0.00025)], measured="")

    assert calibration is None and "no spread" in why


def _live(tmp_path):
    cameras = [
        CameraConfiguration(id=CameraId.Left, name="left", is_enabled=True, scheme="spinnaker",
                            host="24152533", params={"fps": 150, "exposure": 175,
                                                     "primary": True}),
        CameraConfiguration(id=CameraId.Right, name="right", is_enabled=True, scheme="spinnaker",
                            host="24152534", params={"fps": 150, "exposure": 175,
                                                     "primary": "no"}),
        CameraConfiguration(id=CameraId.Camera3, name="stimCam", is_enabled=True, scheme="spinnaker",
                            host="20142630", params={"fps": 900, "exposure": 800,
                                                     "primary": "no"}),
    ]
    stream = NidaqSignalStreamConfiguration(
        channels=(NidaqSignalChannelConfiguration(name="cam_frames",
                                                  physical_channel="Dev1/port0/line0",
                                                  kind="digital"),),
        is_enabled=True)
    live = SystemConfiguration(cameras=cameras, nidaq_stream=stream)
    path = tmp_path / "system_configuration.yaml"
    live.save_file(path.with_suffix(""), as_yaml=True)
    return SystemConfiguration.load_yaml_file(path), path


def test_the_primary_flag_is_read_as_reachaq_reads_it():
    yes = CameraConfiguration(params={"primary": "Yes"})
    text_no = CameraConfiguration(params={"primary": "no"})
    flag = CameraConfiguration(params={"primary": True})
    assert calibrate.is_primary(yes) and calibrate.is_primary(flag)
    assert not calibrate.is_primary(text_no)
    assert not calibrate.is_primary(CameraConfiguration(params={}))


def test_a_scratch_configuration_isolates_the_session_and_adds_the_loopback(tmp_path):
    live, _ = _live(tmp_path)

    scratch = calibrate.scratch_configuration(live, tmp_path / "run", "Dev1/ai6", 50, 1000)

    assert scratch.hardware.can_enabled is False
    assert scratch.hardware.pellet_controller_enabled is False
    assert scratch.hardware.nidaq_enabled is True
    assert scratch.laser.backend == "disabled"
    assert scratch.persistence.output_location == str(tmp_path / "run" / "data")
    assert scratch.nidaq_stream.channels[-1].name == "loopback"
    assert scratch.nidaq_stream.channels[-1].physical_channel == "Dev1/ai6"
    assert scratch.nidaq_stream.delivery_calibration is None
    by_name = {c.name: c.params for c in scratch.cameras}
    # The cameras at the primary's rate change; the stimulation camera keeps its own.
    assert by_name["left"]["fps"] == by_name["right"]["fps"] == 50
    assert by_name["left"]["exposure"] == by_name["right"]["exposure"] == 1000
    assert by_name["stimCam"] == {"fps": 900, "exposure": 800, "primary": "no"}
    # The live configuration is untouched.
    assert {c.name: c.params["fps"] for c in live.cameras}["left"] == 150


def test_writing_the_calibration_backs_the_file_up_first(tmp_path):
    _, path = _live(tmp_path)
    before = path.read_bytes()
    calibration = NidaqDeliveryCalibration(offset_seconds=0.000303, spread_seconds=0.000035,
                                           sessions=12, measured="2026-11-13T10:00:00-07:00",
                                           read_chunk_samples=(61, 69, 83, 167))

    backup = calibrate.write_calibration(path, calibration)

    assert backup.read_bytes() == before
    assert backup.name.startswith("system_configuration_before_delivery_calibration_")
    assert SystemConfiguration.load_yaml_file(path).nidaq_stream.delivery_calibration == calibration
    # Replaced in one step: nothing staged is left beside it.
    assert sorted(f.name for f in path.parent.iterdir()) == sorted([path.name, backup.name])
