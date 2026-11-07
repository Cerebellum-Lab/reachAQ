"""Every line on every card, and which of them can actually be clocked.

Both hardware rules here were learned by a task failing on this rig rather
than reasoned about, and each one took the whole survey down with it.
"""

from types import SimpleNamespace

from tools.acquisition.model.nidaq_monitor_survey import (
    build_survey,
    survey_stream_configuration,
)


def _device(name="PXI1Slot5", analog_inputs=("ai0", "ai1"),
            digital_ports=("port0", "port1", "port2"), lines=2,
            digital_input_max_rate=1_000_000.0, breakout="BNC-2090A"):
    digital = tuple(f"{name}/{port}/line{n}"
                    for port in digital_ports for n in range(lines))
    return SimpleNamespace(
        name=name,
        analog_inputs=tuple(f"{name}/{c}" for c in analog_inputs),
        analog_outputs=(), digital_inputs=digital, digital_outputs=(),
        counter_inputs=(), counter_outputs=(), terminals=(),
        digital_input_max_rate=digital_input_max_rate,
    )


def _configuration(*identities):
    return SimpleNamespace(
        nidaq_ports=SimpleNamespace(
            device_identities=identities or (
                SimpleNamespace(logical_name="PXI1Slot5",
                                runtime_name="PXI1Slot5",
                                breakout="BNC-2090A"),),
            tone1="PXI1Slot5/port0/line0", tone2=None, tone3_r=None,
            tone3_l=None, cam_frames=None, barcode=None),
        nidaq_stream=SimpleNamespace(channels=(
            SimpleNamespace(name="laser1_diode",
                            physical_channel="PXI1Slot5/ai0"),)),
        laser=SimpleNamespace(channels=(
            SimpleNamespace(channel_id=SimpleNamespace(value=1),
                            analog_output="PXI1Slot4/ao0", diode_input=None,
                            shutter_output=None,
                            trigger_route_source="/PXI1Slot5/PFI0"),)),
    )


def test_every_analog_input_is_streamed():
    survey = build_survey([_device(analog_inputs=tuple(f"ai{n}" for n in range(16)))])

    analog = [l for l in survey.lines if l.kind == "analog"]
    assert len(analog) == 16
    assert all(l.acquisition == "stream" for l in analog)


def test_only_port0_is_clocked_and_the_pfi_ports_are_polled():
    """port1 and port2 are the PFI pins: a level, with no sample clock."""
    survey = build_survey([_device()])

    streamed = {l.terminal for l in survey.streamed() if l.kind == "digital"}
    static = {l.terminal for l in survey.static()}

    assert streamed == {"port0/line0", "port0/line1"}
    assert static == {"port1/line0", "port1/line1",
                      "port2/line0", "port2/line1"}


def test_a_board_that_cannot_clock_digital_has_every_line_polled():
    """Measured: a PXI-6713's port0 in a buffered task fails at -200452.

    The driver says so by refusing the DI maximum rate property outright,
    which is what None means here - not "unknown", but "asked and declined".
    """
    survey = build_survey([_device("PXI1Slot4", analog_inputs=(),
                                   digital_input_max_rate=None)])

    assert survey.streamed() == ()
    assert len(survey.static()) == 6
    assert any("no clocked digital input" in note
               for note in survey.notes_for("PXI1Slot4"))


def test_a_board_with_no_analog_input_says_why_it_is_thin():
    survey = build_survey([_device("PXI1Slot4", analog_inputs=())])

    assert any("no analog input" in note
               for note in survey.notes_for("PXI1Slot4"))
    assert not [l for l in survey.lines if l.kind == "analog"]


def test_a_line_is_named_after_where_it_is_not_what_it_does():
    """Nothing has assigned it a role yet; that is the point of surveying."""
    survey = build_survey([_device()])

    line = survey.for_device("PXI1Slot5")[0]
    assert line.name == "PXI1Slot5.ai0"


def test_a_claimed_line_carries_what_the_configuration_calls_it():
    survey = build_survey([_device()], _configuration())

    by_name = {l.terminal: l for l in survey.lines}
    assert by_name["ai0"].assigned_to == "laser1_diode"
    assert by_name["port0/line0"].assigned_to == "tone1"
    assert by_name["ai1"].assigned_to == ""


def test_a_line_claimed_under_the_cards_other_name_is_still_claimed():
    """The laser trigger is configured as PFI0; the survey sees port1/line0.

    Matching the literal string would survey the one line somebody wired a
    stimulus to and report it as spare, which is the worst possible answer.
    """
    survey = build_survey([_device()], _configuration())

    pfi0 = {l.terminal: l for l in survey.lines}["port1/line0"]
    assert pfi0.assigned_to == "laser 1 trigger in"
    assert pfi0.label == 'BNC-2090A "PFI 0" (BNC)'


def test_the_stream_configuration_carries_only_the_clocked_half():
    survey = build_survey([_device(), _device("PXI1Slot4", analog_inputs=(),
                                              digital_input_max_rate=None)])

    stream = survey_stream_configuration(survey)

    names = {c.physical_channel for c in stream.channels}
    assert names == {"PXI1Slot5/ai0", "PXI1Slot5/ai1",
                     "PXI1Slot5/port0/line0", "PXI1Slot5/port0/line1"}
    assert stream.is_enabled


def test_the_referencing_is_passed_through_rather_than_defaulted():
    """Letting DAQmx choose shows one signal on three of a 6221's inputs."""
    survey = build_survey([_device()])

    assert survey_stream_configuration(
        survey, analog_terminal_config="rse").analog_terminal_config == "rse"
    assert survey_stream_configuration(survey).analog_terminal_config == ""


def test_a_survey_with_nothing_to_stream_is_disabled_rather_than_empty():
    survey = build_survey([_device("PXI1Slot4", analog_inputs=(),
                                   digital_input_max_rate=None)])

    assert not survey_stream_configuration(survey).is_enabled
