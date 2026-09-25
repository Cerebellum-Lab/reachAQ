from __future__ import annotations

import dataclasses
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

    for channel in laser.channels if laser.backend != "disabled" else ():
        laser_index = int(channel.channel_id)
        trigger_input = channel.trigger_monitor_input
        for suffix, physical_channel, kind, scale in (
            ("diode", channel.diode_input, "analog", channel.feedback_scale),
            ("command_copy", channel.command_copy_input, "analog", channel.command_copy_scale),
            # The board's stimulus line read back, for the laser tab's Board
            # trigger graph. It was never added, so it was neither streamed
            # nor recorded, and the graph could not show anything. It has no
            # scale in the laser configuration; a channel already acquired on
            # that input keeps its own, as the port roles do.
            (
                "trigger",
                trigger_input,
                nidaq_channel_kind(trigger_input) if trigger_input else "analog",
                None,
            ),
        ):
            if not physical_channel:
                continue
            role_physical_channels.add(physical_channel)
            previous = configured_by_physical.get(physical_channel)
            if scale is None:
                scale = 1.0 if previous is None else previous.scale
            add(
                NidaqSignalChannelConfiguration(
                    name=f"laser{laser_index}_{suffix}",
                    physical_channel=physical_channel,
                    kind=kind,
                    # Empty picks the kind's default: V, or logic for a line.
                    unit="" if previous is None else previous.unit,
                    scale=scale,
                    offset=0.0 if previous is None else previous.offset,
                    minimum=None if previous is None else previous.minimum,
                    maximum=None if previous is None else previous.maximum,
                )
            )

    # Existing channels not claimed by a named hardware role are explicit
    # custom acquisition inputs and remain enabled.
    for channel in configured_stream.channels:
        if channel.physical_channel not in role_physical_channels:
            add(channel)

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
