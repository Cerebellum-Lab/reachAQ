"""Which laser fires a pulse profile, and how it is started.

A profile is the pulse train and nothing else. This is resolved from the rig's
laser configuration at the moment the profile is used - by a protocol row, by
Test stim, or by Run Pulse - so the wiring is written down once, beside the
laser, rather than copied into every profile.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


@dataclasses.dataclass(frozen=True)
class LaserFiring:
    channel_id: int
    trigger_route: LaserTriggerRoute
    #: NI terminal the output arms on; empty for a software start.
    trigger_terminal: str = ""
    #: Board line the board pulses; None for a software start.
    stim_line: Optional[int] = None
    trigger_pulse_us: int = 1000

    @property
    def is_board_trigger(self) -> bool:
        return self.trigger_route is LaserTriggerRoute.HARDWARE_STIM3

    def to_record(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "trigger_route": self.trigger_route.value,
            "trigger_terminal": self.trigger_terminal,
            "stim_line": self.stim_line,
            "trigger_pulse_us": self.trigger_pulse_us,
        }


def resolve_laser_firing(laser_configuration, channel_id, route) -> LaserFiring:
    """The firing for this laser and route, or ValueError naming what is missing."""
    route = LaserTriggerRoute(route)
    try:
        channel = laser_configuration.get_channel(int(channel_id))
    except (KeyError, ValueError):
        raise ValueError(f"Laser {int(channel_id)} is not configured") from None
    number = int(channel.channel_id)
    pulse_us = int(channel.board_trigger_pulse_us)
    if route is LaserTriggerRoute.DIRECT_NI_SOFTWARE:
        return LaserFiring(number, route, trigger_pulse_us=pulse_us)
    if route is not LaserTriggerRoute.HARDWARE_STIM3:
        raise ValueError(f"Laser trigger route {route.value!r} cannot fire a profile")
    terminal = (channel.trigger_source or "").strip()
    if not terminal:
        raise ValueError(
            f"Laser {number} has no trigger terminal; set it in Edit DAQ Ports")
    if channel.board_stim_line is None:
        raise ValueError(
            f"Laser {number} has no board STIM line; set boardStimLine for it in "
            "the system configuration")
    return LaserFiring(number, route, terminal, int(channel.board_stim_line), pulse_us)


def amplitude_refusal(profile, channel) -> str:
    """Why this profile's amplitude cannot drive this laser, or empty."""
    low = channel.minimum_command_volts
    high = channel.maximum_command_volts
    if low <= profile.amplitude_volts <= high:
        return ""
    return (
        f"Profile {profile.profile_id} is {profile.amplitude_volts:g} V; laser "
        f"{int(channel.channel_id)} accepts {low:g}..{high:g} V"
    )
