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


_PORT_INPUT_ROLES = (
    ("cam_frames", "cam_frames"),
    ("barcode", "barcode"),
    ("tone1", "tone1"),
    ("tone2", "tone2"),
    ("tone3_r", "tone3_r"),
    ("tone3_l", "tone3_l"),
)


def nidaq_channel_kind(physical_channel: str) -> str:
    """Analog or digital, from the NI channel name.

    NI names a digital line by its port and line, as in Dev1/port0/line3, and
    an analog input as Dev1/ai3. The stimulus line can be wired back into
    either, so the kind follows the name rather than another setting to keep
    in step with it.
    """
    lowered = str(physical_channel).lower()
    return "digital" if "port" in lowered or "line" in lowered else "analog"


#: One analog input, Dev1/ai3, or one digital line, Dev1/port0/line3.
_SINGLE_INPUT = re.compile(r"^/?[^/]+/(ai\d+|port\d+/line\d+)$", re.IGNORECASE)


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
    if not _SINGLE_INPUT.match(terminal):
        return (
            f"{subject} must be one analog input (aiN) or one digital input "
            "line (portN/lineN); a PFI terminal, an output or a counter "
            "cannot be streamed"
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
    # the laser configuration, so a channel it takes over keeps its own.
    triggers = []
    for channel in lasers:
        physical_channel = channel.trigger_monitor_input
        if not physical_channel:
            continue
        role_physical_channels.add(physical_channel)
        previous = configured_by_physical.get(physical_channel)
        triggers.append(
            NidaqSignalChannelConfiguration(
                name=f"laser{int(channel.channel_id)}_trigger",
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
    role_names = set(mapped) | {trigger.name for trigger in triggers}

    # Existing channels not claimed by a named hardware role are explicit
    # custom acquisition inputs and remain enabled. One stored under a role's
    # name is that role's previous input: every load and save stores the
    # plan, so a role moved to a new input left its old one here, and it
    # came back as a custom input under the same name - "maps to both", and
    # the configuration no longer loaded. A role that is not configured
    # claims no name, so such a channel stays a custom input as before.
    for channel in configured_stream.channels:
        if (
            channel.physical_channel not in role_physical_channels
            and channel.name not in role_names
        ):
            add(channel)

    # Last, so every channel acquired before keeps its place in the
    # multiplexed scan, and the fast STIM edge is not converted just before
    # a diode.
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
