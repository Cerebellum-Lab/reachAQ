"""A channel map is a claim; this is the bookkeeping that tracks the evidence.

The failure these guard against is not a channel that reads nothing - that is
obvious in the data - but one that reads something while being wired to
nothing, and a verification record that quietly outlives the wiring it
described.
"""

import json
from types import SimpleNamespace

from tools.acquisition.model.nidaq_wiring_verification import (
    CONFIRMED,
    OPAQUE,
    SILENT,
    UNEXPECTED,
    UNTESTED,
    WiringCheck,
    WiringPoint,
    WiringVerification,
    summarize,
    wiring_points,
)


def _channel(name, physical_channel, kind="analog"):
    return SimpleNamespace(name=name, physical_channel=physical_channel,
                           kind=kind)


def _laser_channel(number, **overrides):
    values = dict(
        channel_id=SimpleNamespace(value=number),
        analog_output=f"PXI1Slot4/ao{number - 1}",
        shutter_output=f"PXI1Slot5/port0/line{number + 3}",
        trigger_route_source=f"/PXI1Slot5/PFI{number - 1}",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _configuration(stream_channels=(), laser_channels=()):
    return SimpleNamespace(
        nidaq_stream=SimpleNamespace(channels=tuple(stream_channels)),
        laser=SimpleNamespace(channels=tuple(laser_channels)),
    )


def test_every_claim_in_the_configuration_becomes_a_point():
    configuration = _configuration(
        stream_channels=[_channel("tone1", "PXI1Slot5/port0/line0", "digital")],
        laser_channels=[_laser_channel(1)],
    )

    names = {point.name for point in wiring_points(configuration)}

    assert names == {"tone1", "laser1_command", "laser1_shutter",
                     "laser1_trigger_in"}


def test_an_output_with_no_readback_is_marked_rather_than_left_unexplained():
    configuration = _configuration(laser_channels=[_laser_channel(1)])

    shutter = next(p for p in wiring_points(configuration)
                   if p.name == "laser1_shutter")

    assert shutter.kind == OPAQUE
    assert shutter.opaque_reason


def test_moving_a_channel_to_another_pin_discards_its_evidence():
    """Otherwise a confirmation outlives the wiring that earned it."""
    before = WiringPoint("laser1_diode", "PXI1Slot5/ai8", "analog")
    after = WiringPoint("laser1_diode", "PXI1Slot5/ai12", "analog")
    verification = WiringVerification(checks=(
        WiringCheck.for_point(before, CONFIRMED, "held DC", "responded"),
    ))

    assert verification.check_for(before) is not None
    assert verification.check_for(after) is None


def test_an_unverified_channel_is_named_rather_than_counted():
    """A count sends someone to a file; a name sends them to a cable."""
    configuration = _configuration(
        stream_channels=[_channel("laser1_diode", "PXI1Slot5/ai8")])

    summary = summarize(configuration, WiringVerification())

    assert summary.unverified == ("laser1_diode",)
    assert "laser1_diode" in summary.warning()
    assert not summary.is_clean


def test_a_channel_answering_to_the_wrong_driver_is_not_treated_as_verified():
    """This is the case that produces plausible data from nothing."""
    point = WiringPoint("laser2_diode", "PXI1Slot5/ai4", "analog")
    configuration = _configuration(
        stream_channels=[_channel("laser2_diode", "PXI1Slot5/ai4")])
    verification = WiringVerification(checks=(
        WiringCheck.for_point(point, UNEXPECTED, "held DC",
                              "responded to laser 1"),
    ))

    summary = summarize(configuration, verification)

    assert summary.confirmed == ()
    assert summary.failed == ("laser2_diode",)
    assert summary.contradictions() == ("laser2_diode",)


def test_silence_under_a_driver_counts_against_a_channel():
    point = WiringPoint("laser1_diode", "PXI1Slot5/ai8", "analog")
    configuration = _configuration(
        stream_channels=[_channel("laser1_diode", "PXI1Slot5/ai8")])
    verification = WiringVerification(checks=(
        WiringCheck.for_point(point, SILENT, "held DC", "did not respond"),
    ))

    assert summarize(configuration, verification).failed == ("laser1_diode",)


def test_a_point_nothing_could_drive_is_unverified_not_failed():
    """No evidence is not the same as evidence of a fault."""
    point = WiringPoint("cam_frames", "PXI1Slot5/port0/line2", "digital")
    configuration = _configuration(
        stream_channels=[_channel("cam_frames", "PXI1Slot5/port0/line2",
                                  "digital")])
    verification = WiringVerification(checks=(
        WiringCheck.for_point(point, UNTESTED, "no driver available",
                              "nothing in this run drives it"),
    ))

    summary = summarize(configuration, verification)

    assert summary.failed == ()
    assert summary.unverified == ("cam_frames",)


def test_a_confirmed_map_reports_clean_and_warns_about_nothing():
    point = WiringPoint("tone1", "PXI1Slot5/port0/line0", "digital")
    configuration = _configuration(
        stream_channels=[_channel("tone1", "PXI1Slot5/port0/line0", "digital")])
    verification = WiringVerification(checks=(
        WiringCheck.for_point(point, CONFIRMED, "static level", "held high"),
    ))

    summary = summarize(configuration, verification)

    assert summary.is_clean
    assert summary.warning() == ""
    assert "1/1 confirmed" in summary.detail()


def test_the_record_survives_a_round_trip(tmp_path):
    point = WiringPoint("tone1", "PXI1Slot5/port0/line0", "digital")
    original = WiringVerification(
        checks=(WiringCheck.for_point(point, CONFIRMED, "static level", "high"),),
        generated_on="2026-09-22T12:00:00+00:00",
    )
    path = tmp_path / "record.json"

    original.save(path)

    assert WiringVerification.load(path) == original


def test_a_missing_or_corrupt_record_means_nothing_is_known(tmp_path):
    """It must not stop the application; unknown is a valid state."""
    missing = tmp_path / "absent.json"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")

    assert WiringVerification.load(missing).checks == ()
    assert WiringVerification.load(corrupt).checks == ()
