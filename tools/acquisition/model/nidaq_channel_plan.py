from __future__ import annotations

import dataclasses
import re
from typing import Dict, Iterable, Tuple

from autotrainer.core import (
    LaserSystemConfiguration,
    NidaqPortConfiguration,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from autotrainer.core.logging import get_verbose_logger

from tools.acquisition.model.nidaq_monitor_survey import BUFFERED_PORT


logger = get_verbose_logger(__name__)


_PORT_INPUT_ROLES = (
    ("cam_frames", "cam_frames"),
    ("barcode", "barcode"),
    ("tone1", "tone1"),
    ("tone2", "tone2"),
    ("tone3_r", "tone3_r"),
    ("tone3_l", "tone3_l"),
)


#: The names the plan gives its laser roles.
_LASER_ROLE_NAME = re.compile(r"^laser\d+_(diode|command_copy|trigger)$")


def _is_role_name(name: str) -> bool:
    """Whether the plan writes this name for a role, rather than an operator."""
    return (
        any(name == role for _attribute, role in _PORT_INPUT_ROLES)
        or bool(_LASER_ROLE_NAME.match(name))
    )


def _claimed_role_names(lasers) -> set:
    """The names roles claim in a plan, whether or not each role is set.

    Only a role ever writes its name, so a stored channel under one is that
    role's: left on an old input when the role moved, or left behind when it
    was cleared. Claimed only while set, a cleared role's channel came back
    as a custom input and was recorded from then on - choosing "(none)" for
    a trigger readback, or for tone1 or camFrames.

    Every port role claims its name always: the ports are planned whatever
    else is configured. A laser's names are claimed for each laser the plan
    acquires, which is only while the laser backend is enabled; disabled, the
    plan adds no laser role, and a stored laser channel stays a custom input,
    as it always has.
    """
    names = {name for _attribute, name in _PORT_INPUT_ROLES}
    for channel in lasers:
        number = int(channel.channel_id)
        names.update(
            f"laser{number}_{suffix}" for suffix in ("diode", "command_copy", "trigger"))
    return names


def nidaq_channel_kind(physical_channel: str) -> str:
    """Analog or digital, from the NI channel name.

    NI names a digital line by its port and line, as in Dev1/port0/line3, and
    an analog input as Dev1/ai3. The stimulus line can be wired back into
    either, so the kind follows the name rather than another setting to keep
    in step with it.
    """
    lowered = str(physical_channel).lower()
    return "digital" if "port" in lowered or "line" in lowered else "analog"


#: One analog input, Dev1/ai3, or one port0 line, Dev1/port0/line3: what the
#: stream can sample. It puts every digital input in one clocked task, and an
#: M Series board clocks port0 only (see nidaq_monitor_survey).
_STREAMABLE_INPUT = re.compile(
    rf"^/?[^/]+/(ai\d+|{BUFFERED_PORT}/line\d+)$", re.IGNORECASE)
#: One port0 line: the only digital line the stream can sample.
_STREAMABLE_LINE = re.compile(rf"^/?[^/]+/{BUFFERED_PORT}/line\d+$", re.IGNORECASE)
#: A digital line on another port: port1 and port2 are the static PFI pins.
_STATIC_LINE = re.compile(r"^/?[^/]+/port\d+/line\d+$", re.IGNORECASE)
_PFI_PIN_LINES = (
    "port1/port2 lines are PFI pins that cannot be sampled with the stream")

#: A port role's field as the configuration file and Edit DAQ Ports name it.
_PORT_ROLE_FIELDS = {"cam_frames": "camFrames", "tone3_r": "tone3R", "tone3_l": "tone3L"}


def port_role_field(attribute: str) -> str:
    """nidaqPorts' field for a port role: camFrames for cam_frames."""
    return _PORT_ROLE_FIELDS.get(attribute, attribute)


def _is_pfi_pin_line(terminal: str) -> bool:
    """A digital line off port0: STIM3's /PXI1Slot5/PFI0 is PXI1Slot5/port1/line0."""
    return bool(_STATIC_LINE.match(terminal)) and not _STREAMABLE_LINE.match(terminal)


def port_role_refusal(attribute: str, terminal: str) -> str:
    """Why the stream cannot sample this port role's line, or an empty string.

    Every port role is a digital input in the stream's one clocked DI task,
    so it has the trigger readback's rule for lines: port0 only. A port1 or
    port2 line, or anything that is not a line, took every NI input down
    (-200452). Edit DAQ Ports refuses the same, from this.
    """
    subject = f"nidaqPorts.{port_role_field(attribute)} {terminal!r}"
    if _is_pfi_pin_line(terminal):
        return f"{subject} is not streamable: {_PFI_PIN_LINES}; use a port0 line"
    if not _STREAMABLE_LINE.match(terminal):
        return f"{subject} must be one port0 line (port0/lineN) on the input card"
    return ""


def laser_output_lines(
    laser: LaserSystemConfiguration, *, shutters: bool = True,
) -> Dict[str, str]:
    """The laser configuration's output lines, each with what drives it."""
    outputs = {}
    if laser.pmt_shutter_output:
        outputs[laser.pmt_shutter_output] = "the PMT shutter output"
    for channel in laser.channels:
        number = int(channel.channel_id)
        for attribute, role in (
            ("shutter_output", "shutter output"),
            ("auxiliary_output", "auxiliary output"),
            ("trigger_output", "trigger output"),
            ("timing_trigger_output", "timing trigger output"),
        ):
            if attribute == "shutter_output" and not shutters:
                continue
            value = getattr(channel, attribute)
            if value:
                outputs.setdefault(value, f"laser {number} {role}")
    return outputs


def trigger_readback_refusal(
    laser_number: int, terminal: str, outputs: Dict[str, str],
) -> str:
    """Why this trigger readback input cannot be acquired, or an empty string."""
    subject = f"Laser {int(laser_number)} trigger readback input {terminal!r}"
    if _is_pfi_pin_line(terminal):
        return (
            f"{subject} is not streamable: {_PFI_PIN_LINES}; use an analog "
            "input (aiN) or a port0 line"
        )
    if not _STREAMABLE_INPUT.match(terminal):
        return (
            f"{subject} must be one analog input (aiN) or one port0 line "
            "(port0/lineN); a PFI terminal, an output or a counter cannot be "
            "streamed"
        )
    if terminal in outputs:
        return f"{subject} is {outputs[terminal]}"
    return ""


def trigger_readback_refusals(laser: LaserSystemConfiguration) -> Tuple[str, ...]:
    """Why each laser's trigger readback input cannot be acquired, if it cannot.

    Edit DAQ Ports refuses the same things before it closes, with the same
    rule, so the dialog lets nothing through that a load would then refuse.
    A PFI terminal was read as an analog input, and the stream task failed
    and took every NI-DAQ input down with it. An output line is not in the
    acquisition plan, so the plan's own duplicate check never compared one.
    """
    outputs = laser_output_lines(laser)
    return tuple(
        refusal
        for channel in laser.channels
        if channel.trigger_monitor_input
        for refusal in (trigger_readback_refusal(
            int(channel.channel_id), channel.trigger_monitor_input, outputs),)
        if refusal
    )


def build_nidaq_acquisition_configuration(
    configured_stream: NidaqSignalStreamConfiguration,
    ports: NidaqPortConfiguration,
    laser: LaserSystemConfiguration,
) -> NidaqSignalStreamConfiguration:
    """Build the persisted input set independently from plot visibility."""

    mapped: Dict[str, NidaqSignalChannelConfiguration] = {}
    mapped_physical_channels = set()

    def add(channel: NidaqSignalChannelConfiguration) -> None:
        existing = mapped.get(channel.name)
        if existing is not None and existing.physical_channel != channel.physical_channel:
            raise ValueError(
                f"NI-DAQ acquisition role {channel.name!r} maps to both "
                f"{existing.physical_channel!r} and {channel.physical_channel!r}"
            )
        if channel.physical_channel in mapped_physical_channels:
            owner = next(
                item.name
                for item in mapped.values()
                if item.physical_channel == channel.physical_channel
            )
            raise ValueError(
                f"NI-DAQ physical channel {channel.physical_channel!r} is assigned "
                f"to both {owner!r} and {channel.name!r}"
            )
        mapped[channel.name] = channel
        mapped_physical_channels.add(channel.physical_channel)

    configured_by_physical = {
        channel.physical_channel: channel
        for channel in configured_stream.channels
    }
    role_physical_channels = set()
    port_refusals = tuple(
        refusal
        for attribute, _name in _PORT_INPUT_ROLES
        if getattr(ports, attribute)
        for refusal in (port_role_refusal(attribute, getattr(ports, attribute)),)
        if refusal
    )
    if port_refusals:
        raise ValueError("; ".join(port_refusals))
    for attribute, name in _PORT_INPUT_ROLES:
        physical_channel = getattr(ports, attribute)
        if not physical_channel:
            continue
        role_physical_channels.add(physical_channel)
        previous = configured_by_physical.get(physical_channel)
        add(
            NidaqSignalChannelConfiguration(
                name=name,
                physical_channel=physical_channel,
                kind="digital",
                unit="logic",
                scale=1.0 if previous is None else previous.scale,
                offset=0.0 if previous is None else previous.offset,
            )
        )

    lasers = laser.channels if laser.backend != "disabled" else ()
    if lasers:
        refusals = trigger_readback_refusals(laser)
        if refusals:
            raise ValueError("; ".join(refusals))
    for channel in lasers:
        laser_index = int(channel.channel_id)
        for suffix, physical_channel, scale in (
            ("diode", channel.diode_input, channel.feedback_scale),
            ("command_copy", channel.command_copy_input, channel.command_copy_scale),
        ):
            if not physical_channel:
                continue
            role_physical_channels.add(physical_channel)
            previous = configured_by_physical.get(physical_channel)
            add(
                NidaqSignalChannelConfiguration(
                    name=f"laser{laser_index}_{suffix}",
                    physical_channel=physical_channel,
                    kind="analog",
                    unit="V" if previous is None else previous.unit,
                    scale=scale,
                    offset=0.0 if previous is None else previous.offset,
                    minimum=None if previous is None else previous.minimum,
                    maximum=None if previous is None else previous.maximum,
                )
            )

    # The board's stimulus line read back, for the laser tab's Board trigger
    # graph. It was never added, so it was neither streamed nor recorded.
    # Claimed here, so a channel already acquired on that input is taken over
    # rather than kept as a custom one; added last, below. It has no scale in
    # the laser configuration. It keeps the settings of a custom channel it
    # takes over, such as christielab10's laser1_trigger_readback, and its
    # own from an earlier load; another role's entry left on its input, an
    # old command copy's scale for one, is not its to keep.
    triggers = []
    for channel in lasers:
        physical_channel = channel.trigger_monitor_input
        if not physical_channel:
            continue
        name = f"laser{int(channel.channel_id)}_trigger"
        role_physical_channels.add(physical_channel)
        previous = configured_by_physical.get(physical_channel)
        if previous is not None and previous.name != name and _is_role_name(previous.name):
            previous = None
        triggers.append(
            NidaqSignalChannelConfiguration(
                name=name,
                physical_channel=physical_channel,
                kind=nidaq_channel_kind(physical_channel),
                # Empty picks the kind's default: V, or logic for a line.
                unit="" if previous is None else previous.unit,
                scale=1.0 if previous is None else previous.scale,
                offset=0.0 if previous is None else previous.offset,
                minimum=None if previous is None else previous.minimum,
                maximum=None if previous is None else previous.maximum,
            )
        )
    role_pins = {channel.name: channel.physical_channel for channel in mapped.values()}
    role_pins.update((trigger.name, trigger.physical_channel) for trigger in triggers)
    role_names = set(role_pins) | _claimed_role_names(lasers)

    # Existing channels not claimed by a named hardware role are explicit
    # custom acquisition inputs and remain enabled. One stored under a role's
    # name is that role's previous input: every load and save stores the
    # plan, so a role moved to a new input left its old one here, and it
    # came back as a custom input under the same name - "maps to both", and
    # the configuration no longer loaded. A role that is not configured
    # claims no name, so such a channel stays a custom input as before.
    for channel in configured_stream.channels:
        if channel.physical_channel in role_physical_channels:
            continue
        if channel.name in role_names:
            logger.warning(
                "NI-DAQ plan drops the stored channel %r on %s: the %s role is %s",
                channel.name,
                channel.physical_channel,
                channel.name,
                (
                    f"now on {role_pins[channel.name]}"
                    if channel.name in role_pins
                    else "not set"
                ),
            )
            continue
        add(channel)

    # Last, so the readbacks follow every other input in the multiplexed
    # scan and the fast STIM edge is not converted just before a diode. A
    # channel already acquired keeps its place when nothing is taken over,
    # or when what is taken over came last, as christielab10's two
    # *_trigger_readback channels did; one taken over from the middle moves
    # the channels after it up by one.
    for trigger in triggers:
        add(trigger)

    channels = tuple(mapped.values())
    selected = tuple(
        name
        for name in configured_stream.display_channels
        if name in mapped
    )
    return dataclasses.replace(
        configured_stream,
        channels=channels,
        display_channels=selected,
        is_enabled=bool(channels),
    )


def with_display_channels(
    configuration: NidaqSignalStreamConfiguration,
    channel_names: Iterable[str],
) -> NidaqSignalStreamConfiguration:
    requested = tuple(dict.fromkeys(str(name) for name in channel_names))
    available = {channel.name for channel in configuration.channels}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(
            "Unknown NI-DAQ display channel(s): " + ", ".join(unknown)
        )
    return dataclasses.replace(configuration, display_channels=requested)
