"""Fire one saved laser profile on the bench, off any recording session.

The board emits its firmware-timed STIM3 pulse into an analog output that has
already been armed on that trigger terminal, which is the same route a trial
takes. Nothing here starts a session or touches session evidence.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Iterable, Optional, Sequence

from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


def _bench_operation_id() -> str:
    return "bench-{}".format(uuid.uuid4().hex[:12])


@dataclasses.dataclass(frozen=True)
class BenchRecipe:
    """The shape prepare_pulse_profile reads, identifying itself as a bench run.

    prepare_pulse_profile uses the recipe only to build the operation's
    provenance context, so a recipe that claims no session and no trial is
    honest rather than a stub: this output genuinely belongs to no session.
    """

    operation_id: str = dataclasses.field(default_factory=_bench_operation_id)
    session_id: str = "bench"
    session_generation: int = 0
    protocol_id: str = "stim-bench-test"
    protocol_revision: int = 1
    logical_trial_id: int = 0
    attempt_id: int = 0


@dataclasses.dataclass(frozen=True)
class StimTestResult:
    profile_id: str
    channel_id: int
    trigger_terminal: str
    trigger_pulse_us: int
    arm_to_terminal_ms: Optional[float]
    detail: str = ""
    stim_line: int = 3

    def __str__(self) -> str:
        if self.arm_to_terminal_ms is None:
            return "Stim test {} on laser {} via STIM{}: {}".format(
                self.profile_id,
                self.channel_id,
                self.stim_line,
                self.detail or "completed",
            )
        return (
            "Stim test {} on laser {}: STIM{} pulse {} us started the waveform "
            "on {}, arm to terminal {:.2f} ms".format(
                self.profile_id,
                self.channel_id,
                self.stim_line,
                self.trigger_pulse_us,
                self.trigger_terminal,
                self.arm_to_terminal_ms,
            )
        )


#: Allowed beyond the trigger pulse and the waveform for the output to report
#: that it finished. Also the shortest wait, as before.
_WAIT_MARGIN_SECONDS = 2.0
_MINIMUM_WAIT_SECONDS = 3.0


def bench_wait_seconds(profile) -> float:
    """How long a bench test waits for the profile's waveform to finish.

    This was max(3 s, trigger pulse + 2 s), which ignored the waveform, so a
    5 s burst was reported as never finishing while it was still running
    (christielab10, 2026-09-24).
    """
    return max(
        _MINIMUM_WAIT_SECONDS,
        profile.trigger_pulse_us / 1e6 + profile.waveform_seconds + _WAIT_MARGIN_SECONDS,
    )


def refuse_reason(
    *,
    profile,
    recording_status_value: str,
    trial_operation_active: bool,
    laser_backend: str,
    configured_channel_ids: Sequence[int],
    firmware_capabilities: Iterable[str],
) -> Optional[str]:
    """Return why this bench test must not run, or None when it may.

    Every branch here is a safety gate. A bench test drives a real laser, so it
    refuses rather than guesses whenever the rig is not in a state where an
    unexpected pulse is harmless.
    """
    if profile is None:
        return "Select a saved laser profile to test."
    if str(recording_status_value).lower() == "recording":
        return "Stim test is refused while a session is recording."
    if trial_operation_active:
        return "Stim test is refused while a trial operation is prepared or active."
    if str(laser_backend) != "nidaq":
        return (
            "Stim test needs the nidaq laser backend; this rig is configured "
            "for {!r}.".format(laser_backend)
        )
    if profile.trigger_route is not LaserTriggerRoute.HARDWARE_STIM3:
        return (
            "Profile {} uses the {} route; the bench test drives the hardware "
            "STIM3 route only.".format(
                profile.profile_id, profile.trigger_route.value
            )
        )
    if not profile.trigger_terminal:
        return (
            "Profile {} has no NI trigger terminal, so the board pulse has "
            "nothing to trigger.".format(profile.profile_id)
        )
    if int(profile.channel_id) not in {int(item) for item in configured_channel_ids}:
        return "Laser channel {} has no hardware mapping.".format(
            int(profile.channel_id)
        )
    reported = set(firmware_capabilities)
    # Only a board that reports *some* capabilities can be said to lack this
    # one. No released pellet firmware answers the capability request at all
    # (see config/pellet-firmware-compatibility.yaml on v2.1.0), so an empty
    # set means "did not say", not "cannot". Refusing on it would block the
    # very path the trial route already drives successfully through
    # _trigger_protocol_stim3, which applies no capability gate either. A board
    # that genuinely cannot pulse fails at the call, where pulse_stim3 returns
    # no token.
    if reported and "finite_stim3_pulse" not in reported:
        return (
            "The pellet firmware reports its capabilities and finite_stim3_pulse "
            "is not among them, so it cannot emit a timed board trigger."
        )
    return None
