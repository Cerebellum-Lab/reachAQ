import json
import math
import tracemalloc

import numpy as np
import pytest

from board_clock_chain_synthetic import (
    IO_ADDRESS, IO_BOOT, PELLET_ADDRESS, PELLET_BOOT, RATE, barcode_line, clock_rig, linear_clock,
    rig_ni_to_host, synthetic_rig, wandering_clock,
)
from tools.acquisition.model import board_clock_chain as chain
from tools.latency import clocks

NI_TO_HOST = rig_ni_to_host()  # the host clock of every rig below reads NI seconds plus 1000


def test_barcode_edges_follow_the_start_pulse_and_bits():
    assert chain.barcode_edges(0) == ((0.0, True), (0.001, False))
    # Bit 0 set merges with the start pulse; bit 31 set ends at 33 ms.
    edges = chain.barcode_edges(0x80000001)
    assert edges[0] == (0.0, True)
    assert edges[1] == pytest.approx((0.002, False))
    assert edges[-2] == pytest.approx((0.032, True))
    assert edges[-1] == pytest.approx((0.033, False))


@pytest.mark.parametrize("tick", [0, 1, 0x80000001, 0xFFFFFFFF, 0x12345678])
def test_frames_decode_by_position(tick):
    line = barcode_line(RATE, 2.0, [(0.5, tick), (1.5, tick + 1 & 0xFFFFFFFF)])

    frames, rejected = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [tick, (tick + 1) & 0xFFFFFFFF]
    assert rejected == 0
    assert frames[0].start_position == 5000
    assert frames[0].edges[0] == (5000, 0.0)
    assert len(frames[0].edges) == len(chain.barcode_edges(tick))


def test_a_frame_cut_by_the_recording_start_is_not_decoded():
    full = barcode_line(RATE, 2.0, [(0.010, 0x00F0F0F0), (1.0, 7)])
    line = full[int(0.025 * RATE):]  # starts 15 ms into the first frame

    frames, _ = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [7]


def test_a_glitch_inside_a_frame_rejects_it():
    line = barcode_line(RATE, 2.0, [(0.5, 0), (1.5, 9)])
    line[int(0.512 * RATE)] = 1.0  # a one-sample spike in a low bit of the first frame

    frames, rejected = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [9]
    assert rejected == 1


def test_a_truncated_last_frame_is_ignored():
    line = barcode_line(RATE, 1.52, [(0.5, 3), (1.5, 4)])  # second frame cut at 20 ms

    frames, _ = chain.decode_barcode_frames(line, RATE)

    assert [frame.tick for frame in frames] == [3]


def test_rates_below_the_minimum_decode_nothing():
    line = barcode_line(1000.0, 2.0, [(0.5, 3)])
    assert chain.decode_barcode_frames(line, 1000.0) == ((), 0)


def test_five_kilohertz_still_decodes():
    line = barcode_line(5000.0, 2.0, [(0.5, 0xA5A5A5A5)])
    frames, _ = chain.decode_barcode_frames(line, 5000.0)
    assert [frame.tick for frame in frames] == [0xA5A5A5A5]


@pytest.mark.parametrize("ni_fit", [None, NI_TO_HOST], ids=["tick_cluster", "host_time"])
def test_chain_composes_pellet_to_ni_and_tones_land_within_a_sample(ni_fit):
    rows, channels, indices = synthetic_rig()

    record = chain.board_clock_chain(
        rows, sample_indices=indices, sample_rate_hz=RATE, channels=channels, ni_to_host=ni_fit,
    )

    assert record["status"] == "fitted"
    assert record["pairing"] == ("tick_cluster" if ni_fit is None else "host_time")
    timekeeper = record["timekeeper"]
    assert (timekeeper["address"], timekeeper["bootId"]) == (IO_ADDRESS, IO_BOOT)
    assert timekeeper["barcodeFramesMatched"] == 29
    assert timekeeper["toNi"]["drift"] == pytest.approx(1.00002, abs=2e-6)
    (board,) = record["boards"]
    assert (board["address"], board["bootId"]) == (PELLET_ADDRESS, PELLET_BOOT)
    assert board["toNi"]["valid"]
    check = board["toneCrossCheck"]
    assert check["matched"] == 2 and check["unmatched"] == 0
    assert check["maxAbsResidualSeconds"] <= 1.0 / RATE
    json.dumps(record, allow_nan=False)  # alignment.json rejects NaN


def test_no_barcode_reports_means_no_timekeeper():
    record = chain.board_clock_chain((), sample_indices=np.arange(10), sample_rate_hz=RATE, channels={})
    assert record["status"] == "unavailable"
    assert "no timekeeper" in record["reason"]


def test_missing_barcode_channel_is_unavailable():
    rows, channels, indices = synthetic_rig(seconds=5)
    del channels["barcode"]
    record = chain.board_clock_chain(rows, sample_indices=indices, sample_rate_hz=RATE, channels=channels)
    assert record["status"] == "unavailable"
    assert "barcode" in record["reason"]


def test_too_few_barcodes_leave_the_fit_invalid_and_json_safe():
    rows, channels, indices = synthetic_rig(seconds=2)
    record = chain.board_clock_chain(rows, sample_indices=indices, sample_rate_hz=RATE, channels=channels)
    assert record["status"] == "unavailable"
    json.dumps(record, allow_nan=False)


def _chain(rows, channels, indices, rate=RATE, ni_to_host=None):
    return chain.board_clock_chain(
        rows, sample_indices=indices, sample_rate_hz=rate, channels=channels, ni_to_host=ni_to_host,
    )


def _map_error(link_record, board_seconds, true_seconds):
    """How far a recorded link puts `board_seconds` from the NI second it truly is."""
    return float(clocks.fit_from_record(link_record).map(board_seconds)) - true_seconds


IO_CLOCK = linear_clock(20.0, 0.99998)  # NI = (io - 20) / 0.99998
PELLET_CLOCK = linear_clock(500.0, 0.99999)


# -- Spec 9: ticks pair with boot_id, never alone ------------------------------------------------------


@pytest.mark.parametrize("ni_fit", [None, NI_TO_HOST], ids=["tick_cluster", "host_time"])
def test_a_timekeeper_reboot_is_fitted_from_the_larger_boot_and_the_other_is_listed(ni_fit):
    rows, channels, indices = clock_rig(
        io_eras=[(0xA, IO_CLOCK, 1, 100, 10.0), (0xB, linear_clock(-110.0, 1.00003), 1, 30, 113.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=150.0, tones=[(60.30043, 5000), (130.30071, 5000)],
    )

    record = _chain(rows, channels, indices, ni_to_host=ni_fit)

    assert record["status"] == "fitted", record["reason"]
    assert record["pairing"] == ("tick_cluster" if ni_fit is None else "host_time")
    timekeeper = record["timekeeper"]
    assert (timekeeper["address"], timekeeper["bootId"]) == (IO_ADDRESS, 0xA)
    assert timekeeper["barcodeFramesMatched"] == 100  # not 130: boot B's frames repeat ticks 1-30
    assert timekeeper["toNi"]["drift"] == pytest.approx(1 / 0.99998, abs=2e-7)
    assert _map_error(timekeeper["toNi"], IO_CLOCK(60.0), 60.0) == pytest.approx(0.0, abs=80e-6)
    assert record["ignoredTimekeeperBoots"] == [
        {"address": IO_ADDRESS, "bootId": 0xB, "reports": 30, "framesPaired": 30},
    ]
    (board,) = record["boards"]  # the timekeeper's own claps are never a board
    assert (board["address"], board["bootId"]) == (PELLET_ADDRESS, PELLET_BOOT)
    assert board["clapsMatched"] == 100  # boot B's claps repeat ticks but are a minute away
    for true_s in (15.0, 60.30043, 130.30071):
        assert _map_error(board["toNi"], PELLET_CLOCK(true_s), true_s) == pytest.approx(0.0, abs=80e-6)
    assert board["toneCrossCheck"]["matched"] == 2 and board["toneCrossCheck"]["meetsTarget"]
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("ni_fit", [None, NI_TO_HOST], ids=["tick_cluster", "host_time"])
def test_a_timekeeper_boot_that_never_reached_the_nidaq_makes_no_board_and_spoils_nothing(ni_fit):
    # Boot B plays ticks 1-30 after the recording ended: its reports and claps
    # are post-roll rows whose ticks repeat boot A's.
    rows, channels, indices = clock_rig(
        io_eras=[(0xA, IO_CLOCK, 1, 100, 10.0), (0xB, linear_clock(-110.0, 1.00003), 1, 30, 113.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=112.0, tones=[(60.30043, 5000), (114.30071, 5000)],
    )

    record = _chain(rows, channels, indices, ni_to_host=ni_fit)

    assert record["status"] == "fitted", record["reason"]
    assert record["timekeeper"]["bootId"] == 0xA
    assert [entry["bootId"] for entry in record["ignoredTimekeeperBoots"]] == [0xB]
    (board,) = record["boards"]
    assert board["address"] == PELLET_ADDRESS
    assert _map_error(board["toNi"], PELLET_CLOCK(60.30043), 60.30043) == pytest.approx(0.0, abs=80e-6)
    check = board["toneCrossCheck"]
    # The second tone is after the recording: counted apart, not a miss.
    assert (check["matched"], check["unmatched"], check["outsideRecording"]) == (1, 0, 1)
    assert check["meetsTarget"] is True


def test_a_pellet_reboot_gives_each_boot_its_own_board_entry():
    second_clock = linear_clock(30.0, 1.00001)
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 100, 10.0)],
        pellet_eras=[(0x11, PELLET_CLOCK, -20.0, 60.0), (0x22, second_clock, 60.0, 400.0)],
        ni_seconds=120.0, tones=[(40.30043, 5000), (90.30071, 5000)],
    )

    record = _chain(rows, channels, indices)

    assert record["status"] == "fitted", record["reason"]
    first, second = record["boards"]
    assert [(board["address"], board["bootId"]) for board in (first, second)] == [
        (PELLET_ADDRESS, 0x11), (PELLET_ADDRESS, 0x22),
    ]
    assert first["clapsMatched"] >= 49 and second["clapsMatched"] >= 49
    assert _map_error(first["toNi"], PELLET_CLOCK(40.30043), 40.30043) == pytest.approx(0.0, abs=80e-6)
    assert _map_error(second["toNi"], second_clock(90.30071), 90.30071) == pytest.approx(0.0, abs=80e-6)
    assert first["toneCrossCheck"]["matched"] == 1 and second["toneCrossCheck"]["matched"] == 1


def test_claps_pair_only_within_half_a_second_of_host_time():
    rows, channels, indices = synthetic_rig(seconds=30)
    # Received a minute away from the timekeeper's clap of the same tick, no pair is trustworthy.
    moved = tuple(
        row[:9] + (row[9] + 60.0,) + row[10:]
        if row[3] == "SYNC_CLAP_OBSERVATION" and json.loads(row[8])["board_address"] == PELLET_ADDRESS else row
        for row in rows
    )

    record = _chain(moved, channels, indices)

    assert record["status"] == "fitted"
    (board,) = record["boards"]
    assert board["clapsMatched"] == 0 and not board["toTimekeeper"]["valid"] and not board["toNi"]["valid"]
    assert board["toneCrossCheck"] is None


# -- Spec 11: the accuracy metric ----------------------------------------------------------------------


def _hundred_tone_rig(missed=()):
    tones = [(15.30043 + 1.00137 * k, 5000) for k in range(100)]
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 140, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=150.0, tones=tones,
    )
    for number in missed:
        onset = int(math.ceil(tones[number][0] * RATE))
        channels["tone1"][onset:onset + 2000] = 0.0
    return rows, channels, indices


def test_a_hundred_tones_that_all_land_meet_the_target():
    check = _chain(*_hundred_tone_rig())["boards"][0]["toneCrossCheck"]
    assert (check["tonesConsidered"], check["matched"], check["unmatched"]) == (100, 100, 0)
    assert check["enoughTonesForAcceptance"] is True
    assert check["p95AbsResidualSeconds"] < 1e-4
    assert check["meetsTarget"] is True


def test_missed_tones_inside_the_recording_count_against_the_target():
    # 90 tones within microseconds and 10 the NI-DAQ never saw: the p95 of the
    # matched alone would pass, and the spec's metric does not.
    check = _chain(*_hundred_tone_rig(missed=range(0, 100, 10)))["boards"][0]["toneCrossCheck"]
    assert (check["matched"], check["unmatched"]) == (90, 10)
    assert check["meetsTarget"] is False
    assert check["p95AbsResidualSeconds"] is None  # infinite, and JSON has no infinity
    assert check["maxAbsResidualSeconds"] < 1e-4
    json.dumps(check, allow_nan=False)


def test_all_tones_missed_is_a_failure_not_no_evidence():
    rows, channels, indices = synthetic_rig()
    channels["tone1"][:] = 0.0
    check = _chain(rows, channels, indices)["boards"][0]["toneCrossCheck"]
    assert (check["tonesConsidered"], check["matched"], check["unmatched"]) == (2, 0, 2)
    assert check["meetsTarget"] is False
    assert check["enoughTonesForAcceptance"] is False


def test_a_handful_of_tones_cannot_claim_the_acceptance():
    check = _chain(*synthetic_rig())["boards"][0]["toneCrossCheck"]
    assert check["meetsTarget"] is True
    assert check["enoughTonesForAcceptance"] is False and check["tonesConsidered"] == 2


def test_a_six_kilohertz_tone_is_matched_on_tone2():
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 40, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=60.0, tones=[(30.30043, 6000)],
    )
    assert "tone2" in channels and not channels["tone1"].any()

    check = _chain(rows, channels, indices)["boards"][0]["toneCrossCheck"]

    assert (check["matched"], check["unmatched"]) == (1, 0)


def test_a_missing_tone_channel_is_reported_not_raised():
    rows, channels, indices = synthetic_rig()
    del channels["tone1"]

    record = _chain(rows, channels, indices)

    check = record["boards"][0]["toneCrossCheck"]
    assert record["status"] == "fitted"
    assert check["tonesWithoutChannel"] == 2 and check["channelsMissing"] == ["tone1"]
    assert check["tonesConsidered"] == 0 and check["meetsTarget"] is None


def test_a_tone_channel_of_the_wrong_length_is_reported_not_raised():
    rows, channels, indices = synthetic_rig()
    channels["tone1"] = channels["tone1"][:1000]

    check = _chain(rows, channels, indices)["boards"][0]["toneCrossCheck"]

    assert check["tonesWithoutChannel"] == 2 and check["channelsMissing"] == ["tone1"]


def test_without_any_tone_there_is_no_verdict():
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 40, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)], ni_seconds=60.0,
    )
    check = _chain(rows, channels, indices)["boards"][0]["toneCrossCheck"]
    assert check["tonesConsidered"] == 0 and check["meetsTarget"] is None


# -- Spec 9/12: a sliding fit, not one line per link ---------------------------------------------------


def test_a_clock_whose_drift_wanders_stays_valid_for_an_hour_and_tones_stay_within_target():
    hour = 3600.0
    io_clock = wandering_clock(20.0, 0.99998, 1e-6, hour)
    pellet_clock = wandering_clock(500.0, 0.99999, -1e-6, hour)
    tones = [(40.30043 + 35.00173 * k, 5000) for k in range(100)]
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, io_clock, 1, 3660, 10.0)],
        pellet_eras=[(PELLET_BOOT, pellet_clock, -20.0, 5000.0)],
        ni_seconds=3700.0, tones=tones, index_dtype=np.int32,
    )

    record = _chain(rows, channels, indices, ni_to_host=NI_TO_HOST)

    assert record["status"] == "fitted", record["reason"]
    assert record["timekeeper"]["toNi"]["valid"] and record["timekeeper"]["toNi"]["knotCount"] > 20
    (board,) = record["boards"]
    assert board["toTimekeeper"]["valid"] and board["toNi"]["valid"]
    check = board["toneCrossCheck"]
    assert check["tonesConsidered"] == 100 and check["enoughTonesForAcceptance"] is True
    assert check["p95AbsResidualSeconds"] < 1e-4, check
    assert check["meetsTarget"] is True


def test_the_composed_link_is_the_two_links_applied_in_turn():
    pellet_clock = linear_clock(3600.0, 0.99999)  # a boot counter that has run for an hour
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 40, 10.0)],
        pellet_eras=[(PELLET_BOOT, pellet_clock, -20.0, 400.0)], ni_seconds=60.0,
    )

    record = _chain(rows, channels, indices)

    (board,) = record["boards"]
    to_ni = clocks.fit_from_record(record["timekeeper"]["toNi"])
    to_timekeeper = clocks.fit_from_record(board["toTimekeeper"])
    composed = clocks.fit_from_record(board["toNi"])
    ends = np.array([composed.x_min, 0.5 * (composed.x_min + composed.x_max), composed.x_max])
    assert composed.map(ends) == pytest.approx(to_ni.map(to_timekeeper.map(ends)), abs=1e-9)
    # a2 + b2 * a1 and b2 * b1, the two straight links composed by hand
    assert composed.drift == pytest.approx(to_ni.drift * to_timekeeper.drift, rel=1e-12)
    assert composed.offset == pytest.approx(to_ni.offset + to_ni.drift * to_timekeeper.offset, abs=1e-9)
    # and the pellet second is where it truly is on the NI line
    assert composed.map(pellet_clock(30.0)) == pytest.approx(30.0, abs=80e-6)


def test_composition_follows_both_links_kinks():
    inner = clocks.PiecewiseClockFit(
        np.array([0.0, 100.0, 250.0]), np.array([10.0, 110.0, 260.0]), 2e-6, 250, True, "", "piecewise_line")
    outer = clocks.PiecewiseClockFit(
        np.array([20.0, 90.0, 200.0, 330.0]), np.array([5.0, 75.0, 187.0, 320.0]), 3e-6, 330, True, "", "piecewise_line")

    composed = chain.compose(inner, outer)

    x = np.linspace(0.0, 320.0, 3201)
    assert composed.valid
    assert composed.map(x) == pytest.approx(outer.map(inner.map(x)), abs=1e-12)
    # inner's knots, plus where inner reaches each of outer's knots (320 is past inner's end)
    assert composed.knots.tolist() == pytest.approx(sorted([0.0, 100.0, 250.0, 10.0, 80.0, 190.0, 320.0]))
    assert composed.residual_rms == pytest.approx((3e-6 ** 2 + (outer.drift * 2e-6) ** 2) ** 0.5)
    assert not chain.compose(inner, clocks.invalid_piecewise_fit("no")).valid


# -- Gates, stamp kinds, record shape, memory ------------------------------------------------------------


def test_ten_barcode_frames_are_too_few_to_trust():
    # Ticks 1-9 are nine frames; ticks 1-10 are ten but span nine seconds.
    rows, channels, indices = synthetic_rig(seconds=10)
    record = _chain(rows, channels, indices)
    assert record["status"] == "unavailable"
    assert "9 barcode frames" in record["reason"] and "need" in record["reason"]

    rows, channels, indices = synthetic_rig(seconds=11)
    record = _chain(rows, channels, indices)
    assert record["status"] == "unavailable"
    assert "span" in record["reason"]


def test_a_board_with_a_few_claps_has_no_trusted_link():
    rows, channels, indices = synthetic_rig(seconds=30)
    kept = tuple(
        row for row in rows
        if not (row[3] == "SYNC_CLAP_OBSERVATION" and json.loads(row[8])["board_address"] == PELLET_ADDRESS
                and json.loads(row[8])["tick"] > 6)
    )
    record = _chain(kept, channels, indices)
    (board,) = record["boards"]
    assert record["status"] == "fitted"
    assert board["clapsMatched"] == 6 and not board["toTimekeeper"]["valid"] and not board["toNi"]["valid"]


def test_barcode_reports_stamped_other_than_at_the_start_edge_are_not_used():
    rows, channels, indices = synthetic_rig()
    queued = tuple(
        row[:15] + ("tx_queue",) + row[16:] if row[3] == "SYNC_BARCODE_REPORT" else row for row in rows
    )

    record = _chain(queued, channels, indices)

    assert record["status"] == "unavailable"
    assert "physical_start" in record["reason"]
    assert record["rowsRejectedForStampKind"]["SYNC_BARCODE_REPORT"] == 29
    assert not chain.has_barcode_reports(queued)


def test_claps_stamped_other_than_observed_are_not_used():
    rows, channels, indices = synthetic_rig()
    queued = tuple(
        row[:15] + ("tx_queue",) + row[16:]
        if row[3] == "SYNC_CLAP_OBSERVATION" and json.loads(row[8])["board_address"] == PELLET_ADDRESS else row
        for row in rows
    )

    record = _chain(queued, channels, indices)

    assert record["rowsRejectedForStampKind"]["SYNC_CLAP_OBSERVATION"] == 29
    assert record["boards"] == []  # no clap of the pellet was usable, so there is nothing to list


def test_reports_without_a_tick_are_unavailable_with_a_reason_not_an_exception():
    rows, channels, indices = synthetic_rig(seconds=5)
    untick = tuple(
        row[:8] + ('{"board_address": 8}',) + row[9:] if row[3] == "SYNC_BARCODE_REPORT" else row for row in rows
    )
    record = _chain(untick, channels, indices)
    assert record["status"] == "unavailable"
    assert "tick" in record["reason"]


def test_every_record_has_the_same_top_level_keys():
    rows, channels, indices = synthetic_rig()
    short_rows, short_channels, short_indices = synthetic_rig(seconds=2)
    records = [
        _chain(rows, channels, indices),
        _chain(rows, channels, indices, ni_to_host=NI_TO_HOST),
        _chain(short_rows, short_channels, short_indices),
        _chain((), {}, np.arange(10)),
        _chain(rows, {}, indices),
        _chain(rows, channels, indices, rate=1000.0),
        _chain(rows, channels, indices, rate=None),
        chain.unavailable("because"),
    ]
    assert len({tuple(sorted(record)) for record in records}) == 1
    assert records[3]["niTime"] is None and records[3]["timekeeper"] is None and records[3]["boards"] == []
    for record in records:
        json.dumps(record, allow_nan=False, sort_keys=True)


def test_a_missing_rate_and_a_missing_line_are_told_apart():
    rows, channels, indices = synthetic_rig(seconds=5)
    no_rate = _chain(rows, channels, indices, rate=None)
    assert "sample rate" in no_rate["reason"]
    no_line = _chain(rows, {}, indices)
    assert "barcode" in no_line["reason"] and "rate" not in no_line["reason"]


def test_a_barcode_line_with_no_frames_says_so():
    rows, channels, indices = synthetic_rig(seconds=20)
    channels["barcode"][:] = 0.0

    record = _chain(rows, channels, indices)

    assert record["status"] == "unavailable"
    assert "no barcode frames decoded" in record["reason"] and "wired" in record["reason"]


def test_an_index_that_does_not_match_the_line_is_unavailable_not_an_exception():
    rows, channels, indices = synthetic_rig(seconds=20)
    record = _chain(rows, channels, indices[:1000])
    assert record["status"] == "unavailable"
    assert "differ in length" in record["reason"]


def test_has_barcode_reports_is_decided_from_the_rows_alone():
    rows, _, _ = synthetic_rig(seconds=5)
    assert chain.has_barcode_reports(rows)
    assert not chain.has_barcode_reports(tuple(row for row in rows if row[3] != "SYNC_BARCODE_REPORT"))
    assert not chain.has_barcode_reports(())


def test_the_decoder_does_not_copy_a_float32_line_to_float64():
    line = barcode_line(RATE, 240.0, [(float(second) + 0.5, second) for second in range(1, 230)])
    tracemalloc.start()
    try:
        frames, _ = chain.decode_barcode_frames(line, RATE)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(frames) == 229
    assert peak < line.size * 8 // 2, f"peak {peak} bytes for a {line.size * 4} byte line"


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float64])
def test_integer_bool_and_float64_lines_decode_the_same(dtype):
    line = barcode_line(RATE, 3.0, [(0.5, 0x0F0F0F0F), (1.5, 6)])
    frames, rejected = chain.decode_barcode_frames(line.astype(dtype), RATE)
    assert [frame.tick for frame in frames] == [0x0F0F0F0F, 6] and rejected == 0


def test_nan_samples_read_as_low():
    line = barcode_line(RATE, 3.0, [(0.5, 5)])
    line[int(0.9 * RATE)] = np.nan
    frames, _ = chain.decode_barcode_frames(line, RATE)
    assert [frame.tick for frame in frames] == [5]


def test_the_chain_does_not_copy_the_spool_index_or_the_lines_to_float64():
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 280, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=300.0, tones=[(100.30043, 5000)],
    )
    tracemalloc.start()
    try:
        record = _chain(rows, channels, indices)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert record["status"] == "fitted", record["reason"]
    assert peak < indices.size * 8 // 2, f"peak {peak} bytes for {indices.size} samples"


# -- Spec 9, round 2: which boot a frame belongs to is told by host time, and a tie is not a pick -----------


def test_an_older_timekeeper_boot_with_more_reports_does_not_win():
    # Boot B ran before this session: 300 reports left in the rows, none of its barcodes on the line.
    tones = [(40.30043, 5000), (90.30071, 5000), (140.30011, 5000)]
    rows, channels, indices = clock_rig(
        io_eras=[(0xB, linear_clock(5.0, 1.00003), 1, 300, -400.0), (0xA, IO_CLOCK, 1, 150, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -1000.0, 1000.0)], ni_seconds=170.0, tones=tones,
    )

    record = _chain(rows, channels, indices, ni_to_host=NI_TO_HOST)

    assert record["status"] == "fitted", record["reason"]
    assert record["pairing"] == "host_time"
    assert record["timekeeper"]["bootId"] == 0xA
    assert record["ignoredTimekeeperBoots"] == []  # boot B's rows are outside the recording's host span
    assert record["rowsOutsideNiSpan"] >= 300
    (board,) = record["boards"]
    assert board["clapsMatched"] == 150
    for true_s, _ in tones:
        assert _map_error(board["toNi"], PELLET_CLOCK(true_s), true_s) == pytest.approx(0.0, abs=80e-6)
    assert board["toneCrossCheck"]["matched"] == 3 and board["toneCrossCheck"]["meetsTarget"] is True

    # Without host time the two boots tie at 150 pairs, and report counts do not break a tie.
    unknown = _chain(rows, channels, indices)
    assert unknown["status"] == "unavailable" and "tie" in unknown["reason"]
    assert unknown["timekeeper"] is None and unknown["boards"] == [] and unknown["pairing"] == "tick_cluster"
    assert {entry["bootId"] for entry in unknown["ignoredTimekeeperBoots"]} == {0xA, 0xB}


def test_host_time_rules_out_reports_of_a_boot_whose_barcodes_never_reached_the_line():
    # Boot B's reports sit inside the recording's span, but its barcodes were not on the NI-DAQ line;
    # boot A's frames repeat its ticks a few minutes later.
    rows, channels, indices = clock_rig(
        io_eras=[(0xB, linear_clock(5.0, 1.00003), 1, 150, 10.0, False), (0xA, IO_CLOCK, 1, 150, 200.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 600.0)], ni_seconds=380.0, tones=[(250.30043, 5000)],
    )

    record = _chain(rows, channels, indices, ni_to_host=NI_TO_HOST)

    assert record["status"] == "fitted", record["reason"]
    assert record["timekeeper"]["bootId"] == 0xA and record["timekeeper"]["barcodeFramesMatched"] == 150
    assert record["ignoredTimekeeperBoots"] == [
        {"address": IO_ADDRESS, "bootId": 0xB, "reports": 150, "framesPaired": 0},
    ]
    (board,) = record["boards"]
    assert _map_error(board["toNi"], PELLET_CLOCK(250.30043), 250.30043) == pytest.approx(0.0, abs=80e-6)
    assert board["toneCrossCheck"]["matched"] == 1

    unknown = _chain(rows, channels, indices)  # tick numbers alone cannot tell the boots apart
    assert unknown["status"] == "unavailable" and "tie" in unknown["reason"]


def test_two_timekeeper_boots_that_both_drove_the_line_equally_are_a_tie_not_a_pick():
    # Both boots have 100 frames on the NI-DAQ line; boot B has 20 more reports, in post-roll.
    rows, channels, indices = clock_rig(
        io_eras=[(0xA, IO_CLOCK, 1, 100, 10.0), (0xB, linear_clock(-110.0, 1.00003), 1, 120, 113.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=213.0, tones=[(60.30043, 5000), (160.30071, 5000)],
    )

    for ni_fit in (NI_TO_HOST, None):
        record = _chain(rows, channels, indices, ni_to_host=ni_fit)

        assert record["status"] == "unavailable" and "tie" in record["reason"]
        assert record["timekeeper"] is None and record["boards"] == []
        assert {entry["bootId"] for entry in record["ignoredTimekeeperBoots"]} == {0xA, 0xB}
        json.dumps(record, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("ni_fit", [None, NI_TO_HOST], ids=["tick_cluster", "host_time"])
def test_the_boot_with_more_frames_on_the_line_wins_with_or_without_host_time(ni_fit):
    rows, channels, indices = clock_rig(
        io_eras=[(0xA, IO_CLOCK, 1, 100, 10.0), (0xB, linear_clock(-110.0, 1.00003), 1, 80, 113.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)], ni_seconds=213.0, tones=[(60.30043, 5000)],
    )

    record = _chain(rows, channels, indices, ni_to_host=ni_fit)

    assert record["status"] == "fitted", record["reason"]
    assert record["timekeeper"]["bootId"] == 0xA and record["timekeeper"]["barcodeFramesMatched"] == 100
    assert [entry["bootId"] for entry in record["ignoredTimekeeperBoots"]] == [0xB]
    (board,) = record["boards"]
    assert _map_error(board["toNi"], PELLET_CLOCK(60.30043), 60.30043) == pytest.approx(0.0, abs=80e-6)


def test_a_barcode_report_must_arrive_about_half_a_second_after_its_frame():
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 40, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)], ni_seconds=60.0,
    )

    def received_later(seconds):
        return tuple(
            row[:9] + (row[9] + seconds,) + row[10:] if row[3] == "SYNC_BARCODE_REPORT" else row for row in rows
        )

    assert _chain(received_later(0.4), channels, indices, ni_to_host=NI_TO_HOST)["status"] == "fitted"
    late = _chain(received_later(0.6), channels, indices, ni_to_host=NI_TO_HOST)
    assert late["status"] == "unavailable"
    assert "tick and host time" in late["reason"]
    # The same rows, with no host time to check them against, pair by tick alone.
    assert _chain(received_later(0.6), channels, indices)["status"] == "fitted"


def test_a_host_fit_that_is_not_valid_falls_back_to_pairing_by_tick():
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 40, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)], ni_seconds=60.0,
    )
    invalid = clocks.invalid_piecewise_fit("NI sample rate unknown")
    line = clocks.ClockFit(offset=1000.0, drift=1.0, residual_rms=0.0, points=100, valid=True)

    assert _chain(rows, channels, indices, ni_to_host=invalid)["pairing"] == "tick_cluster"
    assert _chain(rows, channels, indices, ni_to_host=None)["pairing"] == "tick_cluster"
    assert _chain(rows, channels, indices, ni_to_host=line)["pairing"] == "host_time"


def test_rows_outside_the_nidaq_recordings_host_span_are_not_used():
    # A pellet boot that only an earlier session heard: its rows are in the ring, far from this recording.
    rows, channels, indices = clock_rig(
        io_eras=[(0xC, linear_clock(7.0, 1.00001), 1, 30, -2000.0), (IO_BOOT, IO_CLOCK, 1, 60, 10.0)],
        pellet_eras=[(0x99, linear_clock(40.0, 0.99997), -2100.0, -1000.0), (PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)],
        ni_seconds=80.0, tones=[(30.30043, 5000)],
    )

    record = _chain(rows, channels, indices, ni_to_host=NI_TO_HOST)

    assert [board["bootId"] for board in record["boards"]] == [PELLET_BOOT]
    assert record["rowsOutsideNiSpan"] > 60 and record["ignoredTimekeeperBoots"] == []
    # With no valid NI-to-host fit every row is kept, and the earlier session's pellet is listed, unpaired.
    everything = _chain(rows, channels, indices)
    assert {board["bootId"] for board in everything["boards"]} == {0x99, PELLET_BOOT}
    assert everything["rowsOutsideNiSpan"] == 0
    old = next(board for board in everything["boards"] if board["bootId"] == 0x99)
    assert old["clapsMatched"] == 0 and not old["toNi"]["valid"]


def test_a_report_just_outside_the_recording_is_still_inside_the_five_second_margin():
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 80, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)], ni_seconds=60.0, tones=[(62.30043, 5000)],
    )
    record = _chain(rows, channels, indices, ni_to_host=NI_TO_HOST)
    # The tone 2 s after the NI-DAQ stopped is counted, apart, rather than dropped with the far rows.
    assert record["boards"][0]["toneCrossCheck"]["outsideRecording"] == 1
    assert record["rowsOutsideNiSpan"] > 0


def test_claps_pair_on_host_receive_time_not_on_the_rows_perf_time():
    # An aligned board time can sit anywhere near the truth; the receive time (column 9) is what pairs.
    rows, channels, indices = clock_rig(
        io_eras=[(IO_BOOT, IO_CLOCK, 1, 40, 10.0)],
        pellet_eras=[(PELLET_BOOT, PELLET_CLOCK, -20.0, 400.0)], ni_seconds=60.0, pellet_event_skew=100.0,
    )
    for ni_fit in (None, NI_TO_HOST):
        (board,) = _chain(rows, channels, indices, ni_to_host=ni_fit)["boards"]
        assert board["clapsMatched"] == 40 and board["toNi"]["valid"]


def test_invalid_links_carry_the_method_of_the_fit_that_made_them():
    rows, channels, indices = synthetic_rig(seconds=10)
    record = _chain(rows, channels, indices)
    assert record["timekeeper"]["toNi"]["method"] == clocks.PIECEWISE_LINE and not record["timekeeper"]["toNi"]["valid"]

    rows, channels, indices = synthetic_rig(seconds=30)
    few = tuple(
        row for row in rows
        if not (row[3] == "SYNC_CLAP_OBSERVATION" and json.loads(row[8])["board_address"] == PELLET_ADDRESS
                and json.loads(row[8])["tick"] > 6)
    )
    (board,) = _chain(few, channels, indices)["boards"]
    assert board["toTimekeeper"]["method"] == board["toNi"]["method"] == clocks.PIECEWISE_LINE
    assert chain.compose(clocks.invalid_piecewise_line_fit("x"), clocks.invalid_piecewise_line_fit("y")).method \
        == clocks.PIECEWISE_LINE


def test_composition_extends_over_every_bend_of_the_outer_link_even_beyond_the_inner_span():
    inner = clocks.PiecewiseClockFit(
        np.array([0.0, 100.0, 250.0]), np.array([10.0, 110.0, 260.0]), 2e-6, 250, True, "", "piecewise_line")
    outer = clocks.PiecewiseClockFit(
        np.array([-40.0, 20.0, 90.0, 200.0, 330.0]), np.array([-45.0, 5.0, 75.0, 187.0, 320.0]),
        3e-6, 330, True, "", "piecewise_line")

    composed = chain.compose(inner, outer)

    # inner's knots and where inner (carried on beyond its ends) reaches every knot of outer
    assert composed.knots.tolist() == pytest.approx(sorted([0.0, 100.0, 250.0, -50.0, 10.0, 80.0, 190.0, 320.0]))
    x = np.linspace(-60.0, 330.0, 3901)
    assert composed.map(x) == pytest.approx(outer.map(inner.map(x)), abs=1e-12)
