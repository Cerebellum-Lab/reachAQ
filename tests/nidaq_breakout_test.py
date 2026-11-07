"""The block over the card, and what the card says about each connector.

Two of these are failures that happened on this rig rather than invented
cases. Someone concluded PFI1 was unavailable because the BNC-2090A has no
PFI 1 BNC - it is on the spring strip. And a stimulus line was wired to the
APFI 0 BNC, which is printed on the block whatever card is behind it, on a
PXI-6221 that has no analog trigger circuit at all.
"""

from types import SimpleNamespace

import pytest

from tools.acquisition.model.nidaq_breakout import (
    ASSIGNED,
    NOT_A_TERMINAL,
    REACHABLE,
    UNREACHABLE,
    assignments_for_device,
    available_breakouts,
    breakout_for_device,
    connector_status,
    connector_statuses,
    describe_terminal,
    load_breakout,
    is_panel_terminal,
    unreachable_terminals,
)


def _device(name="PXI1Slot5", analog_inputs=("ai0", "ai3"), terminals=("PFI0",),
            analog_outputs=(), digital_inputs=(), analog_trigger=None):
    qualify = lambda names: tuple(f"{name}/{n}" for n in names)
    return SimpleNamespace(
        name=name,
        analog_inputs=qualify(analog_inputs),
        analog_outputs=qualify(analog_outputs),
        digital_inputs=qualify(digital_inputs),
        digital_outputs=(), counter_inputs=(), counter_outputs=(),
        terminals=tuple(f"/{name}/{t}" for t in terminals),
        analog_trigger_supported=analog_trigger,
    )


# ------------------------------------------------------------------ the data


def test_both_shipped_blocks_load():
    assert set(available_breakouts()) == {"BNC-2090A", "BNC-2110"}


def test_a_block_nobody_has_a_file_for_is_absent_rather_than_fatal():
    """Losing the labels is acceptable; losing the acquisition is not."""
    assert load_breakout("BNC-2115") is None
    assert load_breakout("") is None


@pytest.mark.parametrize("name,bncs,springs", [("BNC-2090A", 22, 29),
                                               ("BNC-2110", 15, 30)])
def test_each_block_carries_the_connector_count_its_manual_declares(
        name, bncs, springs):
    """A dropped row is the failure mode of transcribing a panel by hand."""
    model = load_breakout(name)

    assert len([c for c in model.connectors if c.connector == "bnc"]) == bncs
    assert len([c for c in model.connectors
                if c.connector == "spring"]) == springs


def test_the_2090a_brings_pfi1_out_on_the_strip_not_a_bnc():
    """The conclusion this exists to prevent: no BNC is not no connector."""
    model = load_breakout("BNC-2090A")

    pfi0, = model.connectors_for("PFI0")
    pfi1, = model.connectors_for("PFI1")

    assert pfi0.connector == "bnc"
    assert pfi1.connector == "spring" and pfi1.position == 13
    assert pfi1.printed == "PFI 1"


def test_the_2090a_has_sixteen_analog_input_bncs():
    """The review's own fact list omitted these, which is how ai8 got used."""
    model = load_breakout("BNC-2090A")

    assert model.offers("ai0") and model.offers("ai15")
    assert model.connectors_for("ai15")[0].label == "AI 15"


def test_a_user_defined_bnc_lands_on_nothing_rather_than_a_guess():
    model = load_breakout("BNC-2090A")

    user, = [c for c in model.connectors
             if c.label == "USER 1" and c.connector == "bnc"]

    assert user.terminal == "" and user.role == "user"


def test_one_2110_bnc_carries_two_meanings():
    """AI 2 and AO 7 are the same connector, and the numbering reverses."""
    model = load_breakout("BNC-2110")

    connector, = model.connectors_for("ai2")

    assert connector.label == "AI 2/AO 7"


def test_the_2110_does_not_bring_out_pfi15_at_all():
    assert load_breakout("BNC-2110").connectors_for("PFI15") == tuple()
    assert load_breakout("BNC-2090A").offers("PFI15")


# ------------------------------------------------- the block crossed with it


def test_the_apfi_bnc_is_refused_on_a_card_with_no_analog_trigger():
    """The day this cost: the connector is printed, the circuit is not there."""
    model = load_breakout("BNC-2090A")
    apfi, = model.connectors_for("APFI0")

    status = connector_status(apfi, _device(analog_trigger=False))

    assert status.status == UNREACHABLE
    assert "no analog trigger circuit" in status.reason
    assert not status.is_usable


def test_a_terminal_the_card_does_not_have_is_refused_by_name():
    model = load_breakout("BNC-2090A")
    ai15, = model.connectors_for("ai15")

    status = connector_status(ai15, _device(analog_inputs=("ai0", "ai3")))

    assert status.status == UNREACHABLE
    assert "has no ai15" in status.reason


def test_a_terminal_the_card_has_is_reachable():
    model = load_breakout("BNC-2090A")
    ai3, = model.connectors_for("ai3")

    assert connector_status(ai3, _device()).status == REACHABLE


def test_a_terminal_the_configuration_already_uses_says_what_uses_it():
    model = load_breakout("BNC-2090A")
    ai3, = model.connectors_for("ai3")

    status = connector_status(ai3, _device(), {"ai3": "laser1_diode"})

    assert status.status == ASSIGNED
    assert status.assigned_to == "laser1_diode"
    assert status.is_usable
    assert "in use by laser1_diode" in status.describe()


def test_grounds_and_supplies_are_not_judged_against_the_card():
    model = load_breakout("BNC-2090A")
    ground = [c for c in model.connectors if c.label == "DGND"][0]

    assert connector_status(ground, _device()).status == NOT_A_TERMINAL


def test_every_connector_gets_a_status_and_no_block_means_none():
    model = load_breakout("BNC-2090A")

    statuses = connector_statuses(model, _device())

    assert len(statuses) == len(model.connectors)
    assert connector_statuses(None, _device()) == tuple()
    assert connector_statuses(model, None) == tuple()


# --------------------------------------------------------- the configuration


def test_the_block_is_found_through_the_device_identity():
    ports = SimpleNamespace(device_identities=(
        SimpleNamespace(logical_name="acquire", runtime_name="PXI1Slot5",
                        breakout="BNC-2090A"),
        SimpleNamespace(logical_name="stimulate", runtime_name="PXI1Slot4",
                        breakout=None),
    ))

    assert breakout_for_device(ports, "PXI1Slot5").name == "BNC-2090A"
    # A device with no block behaves as it did before there was a name for it.
    assert breakout_for_device(ports, "PXI1Slot4") is None
    assert breakout_for_device(ports, "PXI1Slot9") is None


def test_assignments_are_collected_per_device_and_keyed_bare():
    stream = SimpleNamespace(channels=(
        SimpleNamespace(name="laser1_diode", physical_channel="PXI1Slot5/ai3"),
        SimpleNamespace(name="tone1", physical_channel="PXI1Slot5/port0/line2"),
    ))
    laser = SimpleNamespace(channels=(
        SimpleNamespace(channel_id=SimpleNamespace(value=1),
                        analog_output="PXI1Slot4/ao0",
                        trigger_source="/PXI1Slot4/PXI_Trig0",
                        trigger_route_source="/PXI1Slot5/PFI0"),))
    ports = SimpleNamespace(tone1=None, timing=SimpleNamespace(
        reference_clock_source="PXI_CLK10",
        sample_clock_source="/PXI1Slot5/ai/SampleClock"))

    found = assignments_for_device("PXI1Slot5", stream=stream, laser=laser,
                                   ports=ports)

    assert found["ai3"] == "laser1_diode"
    assert found["port0/line2"] == "tone1"
    assert found["pfi0"] == "laser 1 trigger route"
    # The other board's output, and a terminal naming no device at all.
    assert "ao0" not in found
    assert not any("clk10" in key for key in found)


def test_a_terminal_is_described_by_the_label_on_the_block():
    model = load_breakout("BNC-2090A")

    assert describe_terminal(model, "ai3") == 'BNC-2090A "AI 3" (BNC)'
    assert describe_terminal(model, "PFI1") == (
        'BNC-2090A "PFI 1" (spring terminal, position 13)')
    assert describe_terminal(model, "ai3") != describe_terminal(model, "PFI1")
    assert describe_terminal(None, "ai3") == ""


def test_a_configured_terminal_the_block_does_not_reach_is_reported():
    """Not an error - a 68-pin cable reaches it - but say so once."""
    model = load_breakout("BNC-2110")

    assert unreachable_terminals(model, ("PFI0", "PFI15", "ai0")) == ("PFI15",)
    assert unreachable_terminals(None, ("PFI15",)) == tuple()


def test_a_pin_is_a_panel_connection_and_a_backplane_line_is_not():
    """PXI_Trig0 is inside the chassis; no front panel has ever printed it."""
    assert is_panel_terminal("ai3")
    assert is_panel_terminal("PFI0")
    assert is_panel_terminal("port0/line2")
    assert is_panel_terminal("APFI0")

    assert not is_panel_terminal("PXI_Trig0")
    assert not is_panel_terminal("/PXI1Slot4/PXI_Trig2")
    assert not is_panel_terminal("PXI_Clk10")
    assert not is_panel_terminal("RTSI0")
    assert not is_panel_terminal("ai/SampleClock")
    assert not is_panel_terminal("ao/StartTrigger")
    assert not is_panel_terminal("Ctr0InternalOutput")
    assert not is_panel_terminal("")


def test_a_backplane_line_is_not_reported_as_missing_from_the_block():
    """Saying no block brings PXI_Trig0 out would send somebody hunting."""
    model = load_breakout("BNC-2090A")

    assert unreachable_terminals(
        model, ("PXI_Trig0", "/PXI1Slot4/PXI_Trig2", "ai/SampleClock")) == ()
    # A real pin the 2090A does not carry is still reported.
    assert unreachable_terminals(load_breakout("BNC-2110"), ("PFI15",)) == (
        "PFI15",)


def test_a_pfi_line_answers_to_the_cards_other_name_for_the_same_pin():
    """PFI 0 and port1/line0 are one pin, and NI prints both on the 2110."""
    model = load_breakout("BNC-2090A")

    by_pfi, = model.connectors_for("PFI0")
    by_port, = model.connectors_for("port1/line0")

    assert by_pfi is by_port
    assert by_pfi.printed == "PFI 0" and by_pfi.connector == "bnc"
    # PFI 8 upwards live on port2, counting from zero again.
    assert model.connectors_for("port2/line0")[0].printed == "PFI 8"


def test_a_line_configured_by_its_port_name_is_not_called_missing():
    """It is the PFI 0 BNC; saying the block lacks it would be false."""
    model = load_breakout("BNC-2090A")

    assert unreachable_terminals(model, ("port1/line0", "port2/line7")) == ()
    assert describe_terminal(model, "port1/line1") == (
        'BNC-2090A "PFI 1" (spring terminal, position 13)')
