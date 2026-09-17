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

    def __str__(self) -> str:
        if self.arm_to_terminal_ms is None:
            return "Stim test {} on laser {}: {}".format(
                self.profile_id, self.channel_id, self.detail or "completed"
            )
        return (
            "Stim test {} on laser {}: board pulse {} us on {}, "
            "arm to terminal {:.2f} ms".format(
                self.profile_id,
                self.channel_id,
                self.trigger_pulse_us,
                self.trigger_terminal,
                self.arm_to_terminal_ms,
            )
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
    if "finite_stim3_pulse" not in set(firmware_capabilities):
        return (
            "The pellet firmware does not report finite_stim3_pulse, so it "
            "cannot emit a timed board trigger."
        )
    return None
