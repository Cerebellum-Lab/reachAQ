"""The report both the terminal and the window read from.

The verification could already say what it found. What it could not say is
where to go: "silent on PXI1Slot5/ai4" sends somebody to a configuration
file, and the next question is always which connector that is.
"""

from types import SimpleNamespace

from tools.acquisition.model.nidaq_wiring_report import (
    build_report,
    format_report,
)
from tools.acquisition.model.nidaq_wiring_verification import (
    CONFIRMED,
    OPAQUE,
    SILENT,
    UNEXPECTED,
    UNTESTED,
    WiringCheck,
    WiringVerification,
    wiring_points,
)


def _configuration(breakout="BNC-2090A"):
    identity = SimpleNamespace(logical_name="PXI1Slot5",
                               runtime_name="PXI1Slot5", breakout=breakout)
    return SimpleNamespace(
        nidaq_ports=SimpleNamespace(device_identities=(identity,)),
        nidaq_stream=SimpleNamespace(channels=(
            SimpleNamespace(name="laser1_diode",
                            physical_channel="PXI1Slot5/ai8", kind="analog"),
            SimpleNamespace(name="laser2_diode",
                            physical_channel="PXI1Slot5/ai4", kind="analog"),
            SimpleNamespace(name="tone1",
                            physical_channel="PXI1Slot5/port0/line0",
                            kind="digital"),
        )),
        laser=SimpleNamespace(channels=(
            SimpleNamespace(channel_id=SimpleNamespace(value=1),
                            analog_output="PXI1Slot5/ao0",
                            shutter_output="PXI1Slot5/port0/line4",
                            trigger_route_source=None),)),
    )


def _checked(configuration, statuses):
    points = {point.name: point for point in wiring_points(configuration)}
    return WiringVerification(
        checks=tuple(
            WiringCheck.for_point(points[name], status, "driven", detail,
                                  checked_on="2026-09-22T00:00:00+00:00")
            for name, (status, detail) in statuses.items()),
        generated_on="2026-09-22T00:00:00+00:00",
    )


def test_a_located_line_names_the_connector_not_just_the_terminal():
    """"check ai4" becomes "check the AI 4 BNC", which is a thing on a bench."""
    configuration = _configuration()

    report = build_report(configuration, _checked(
        configuration, {"laser2_diode": (SILENT, "no change under its driver")}))

    line, = [l for l in report.lines if l.name == "laser2_diode"]
    assert line.label == 'BNC-2090A "AI 4" (BNC)'
    assert line.advice() == ('did not respond when driven; check the cable at '
                             'BNC-2090A "AI 4" (BNC)')


def test_a_digital_line_is_located_on_the_strip():
    configuration = _configuration()

    report = build_report(configuration)

    line, = [l for l in report.lines if l.name == "tone1"]
    assert line.label == 'BNC-2090A "P0 0" (spring terminal, position 3)'


def test_without_a_named_block_the_terminal_is_still_the_answer():
    """A device with no breakout loses the label, not the line."""
    configuration = _configuration(breakout=None)

    report = build_report(configuration)

    line, = [l for l in report.lines if l.name == "laser2_diode"]
    assert line.label == ""
    assert line.where == "PXI1Slot5/ai4"
    assert "PXI1Slot5/ai4" in line.advice()


def test_nothing_checked_reads_as_never_checked_on_every_line():
    """An empty run must not render as an empty report."""
    configuration = _configuration()

    report = build_report(configuration)

    assert report.lines
    assert {l.status for l in report.lines} == {UNTESTED, OPAQUE}
    assert not report.is_clean
    assert "no hardware check has been run" in format_report(report)


def test_the_worst_thing_is_listed_first():
    """A channel answering to the wrong driver records plausible nonsense."""
    configuration = _configuration()
    verification = _checked(configuration, {
        "laser1_diode": (CONFIRMED, ""),
        "laser2_diode": (SILENT, ""),
        "tone1": (UNEXPECTED, "moved with laser1_command"),
    })

    report = build_report(configuration, verification)

    assert [l.status for l in report.lines][:3] == [UNEXPECTED, SILENT, UNTESTED]
    assert report.lines[0].name == "tone1"


def test_an_output_with_no_readback_is_opaque_rather_than_failed():
    """The shutter cannot be seen from here; that is not the same as broken."""
    configuration = _configuration()

    report = build_report(configuration)

    line, = [l for l in report.lines if l.name == "laser1_shutter"]
    assert line.status == OPAQUE
    assert not line.needs_attention
    assert "cannot be checked from here" in line.advice()
    # And it is excluded from the denominator, so 4/4 stays reachable.
    assert line not in [l for l in report.lines if l.status != OPAQUE]
    assert report.checkable == len(report.lines) - 1


def test_the_headline_counts_what_was_confirmed_out_of_what_could_be():
    configuration = _configuration()
    verification = _checked(configuration, {
        "laser1_diode": (CONFIRMED, ""), "laser2_diode": (CONFIRMED, ""),
        "tone1": (CONFIRMED, ""), "laser1_command": (SILENT, ""),
    })

    report = build_report(configuration, verification)

    assert report.headline() == "3/4 confirmed, 1 needing attention"


def test_configuration_problems_are_reported_before_any_cable():
    """They were found without touching the rig, so they come first."""
    configuration = _configuration()

    text = format_report(build_report(
        configuration, issues=("laser 1 triggerSource is '/PXI1Slot5/APFI0': "
                               "PXI1Slot5 has no analog trigger circuit",)))

    assert "Configuration problems" in text
    assert text.index("APFI0") < text.index("laser1_diode")


def test_the_text_says_what_a_confirmation_does_not_mean():
    """A confirmed line says the rig answered, not that the value is right."""
    configuration = _configuration()

    text = format_report(build_report(configuration))

    assert "Nothing above is a reading of the signal itself" in text
    assert "not that the value is right" in text


def test_a_clean_run_says_so_without_the_caveat_block():
    configuration = _configuration()
    verification = _checked(configuration, {
        name: (CONFIRMED, "") for name in
        ("laser1_diode", "laser2_diode", "tone1", "laser1_command")})

    report = build_report(configuration, verification)

    assert report.is_clean
    assert "Nothing above is a reading" not in format_report(report)
