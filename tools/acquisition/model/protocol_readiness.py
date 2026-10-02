"""Whether the selected protocol can run as it says, checked before recording.

Ben, 2026-10-02: laser 2's trials did not fire, and nothing said so before
the session. Each future, enabled row is put through the runtime's own
compile (TrialActionCompiler, resolve_laser_firing, amplitude_refusal) without
preparing anything, and the rules the runtime applies only mid-trial are
applied here first. Errors are what would stop a trial; they block Record.
Warnings are what the software cannot see or is easy to misread; Record stays
available.

Pure: it reads the configuration, the profile library and a snapshot of the
app's state (ReadinessLiveState), and touches no hardware.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from autotrainer.core.configuration.laser_configuration import backplane_line_of

from tools.acquisition.model.laser_firing import amplitude_refusal, resolve_laser_firing
from tools.acquisition.model.trial_action import TrialActionCompiler, TrialCompileContext
from tools.acquisition.model.trial_protocol_schedule import (
    CoverPolicy,
    LaserTriggerRoute,
    PelletLane,
    PelletPositionMode,
    StimulusAssignment,
    StimulusTrigger,
)

#: Record's blocker for a First Reach protocol without the stimCam. The check
#: names it too, and recording_blockers lists it once (protocol_blocker_lines).
STIM_CAMERA_REQUIRED = "Selected protocol requires the enabled stimCam"

#: Record's tooltip names this many protocol errors, then how many more.
MAX_BLOCKER_LINES = 5


@dataclasses.dataclass(frozen=True)
class ReadinessFinding:
    severity: Literal["error", "warning"]
    #: None for a finding about the whole protocol.
    trial_id: Optional[int]
    #: Operator text: what is wrong and how to fix it.
    message: str


@dataclasses.dataclass(frozen=True)
class ProtocolRigConfiguration:
    """What the check reads of the rig's configuration.

    A SystemConfiguration has both fields too. `nidaq_stream` is the
    acquisition plan the stream runs (NidaqSignalMonitorModel.configuration),
    which says whether a trigger readback is acquired.
    """

    laser: object
    nidaq_stream: object = None


@dataclasses.dataclass(frozen=True)
class ReadinessLiveState:
    """The app's state the check reads, taken as one snapshot."""

    laser_connected: bool = False
    #: AppModel.laser_controller_close_refusal: why laser work is refused.
    laser_close_refusal: str = ""
    #: NidaqSignalMonitorModel.stream_state.
    nidaq_stream_state: str = "disabled"
    #: The stream's timing plan as it is now.
    timing_plan: object = None
    #: The plan the laser controller was opened with, which its pulses run on
    #: until it closes (NidaqLaserController._resolve_pulse_timing). A
    #: Refresh Hardware that restarts only the stream leaves it as it was.
    laser_timing_plan: object = None
    #: Configured board names to the names NI-DAQmx uses now.
    device_aliases: Mapping[str, str] = dataclasses.field(default_factory=dict)
    stim_camera_enabled: bool = False
    #: The profile picked on each laser's Laser Control tab, by laser number.
    laser_tab_profiles: Mapping[int, Optional[str]] = dataclasses.field(default_factory=dict)
    #: Rows the current session has completed; they cannot change.
    completed_trial_ids: Tuple[int, ...] = ()


def check_protocol_readiness(
    document, configuration, profile_library, live: Optional[ReadinessLiveState] = None,
) -> Tuple[ReadinessFinding, ...]:
    """Every error and warning for the future, enabled rows of `document`.

    `document` is a TrialProtocolDocument, or a TrialProtocolSchedule: the
    app passes its schedule, the rows its runtime compiles.
    """
    live = live or ReadinessLiveState()
    completed = set(live.completed_trial_ids)
    future = tuple(row for row in _rows_of(document) if row.trial_id not in completed)
    rows = tuple(row for row in future if row.enabled)
    laser = configuration.laser
    driven = _driven_backplane_lines(laser, live)
    laser_profiles = {item.profile_id: item for item in profile_library.laser_profiles}
    trigger_profiles = {
        item.profile_id: item for item in profile_library.stimulus_trigger_profiles}
    compiler = TrialActionCompiler(
        tone_profiles={item.profile_id: item for item in profile_library.tone_profiles},
        laser_profiles=laser_profiles,
        cue_interval_profiles={
            item.profile_id: item for item in profile_library.cue_interval_profiles},
        stimulus_trigger_profiles=trigger_profiles,
        laser_configuration=laser,
        dcs_to_motor=tuple,
    )
    policies = {item.policy_id for item in profile_library.automatic_shift_profiles}
    protocol_id = getattr(document, "protocol_id", None) or getattr(
        getattr(document, "document", None), "protocol_id", "protocol-check")
    revision = getattr(document, "revision", None) or getattr(
        getattr(document, "document", None), "revision", 1)

    findings: List[ReadinessFinding] = list(_disabled_row_errors(future))
    firing_rows = []
    first_reach = False
    for row in rows:
        stimulated = row.stimulus_assignment is not StimulusAssignment.DISABLED
        errors = (
            _laser_errors(row, laser, laser_profiles, stimulated, driven)
            + _trigger_errors(row, stimulated)
            + _randomized_errors(row, trigger_profiles)
        )
        if not errors:
            refusal = _compile_refusal(row, compiler, policies, protocol_id, revision)
            if refusal:
                errors.append(refusal)
        findings.extend(ReadinessFinding("error", row.trial_id, message) for message in errors)
        # A row with a profile and no assignment compiles its laser and then
        # drops it: it never fires, and needs nothing below.
        if row.laser_profile_id and stimulated:
            firing_rows.append(row)
        first_reach = first_reach or (
            stimulated and row.stimulus_trigger is StimulusTrigger.FIRST_REACH)
    if first_reach and not live.stim_camera_enabled:
        findings.append(ReadinessFinding("error", None, STIM_CAMERA_REQUIRED))
    if firing_rows:
        findings.extend(_live_errors(firing_rows, laser, live))
        findings.extend(_warnings(firing_rows, configuration, laser_profiles, live))
    return tuple(findings)


def protocol_blocker_lines(
    findings: Iterable[ReadinessFinding], limit: int = MAX_BLOCKER_LINES,
) -> Tuple[str, ...]:
    """Record's blocker lines: one per error, naming its trials, then how many more.

    An error several trials share is one line, "trials 6-8: ...". The stimCam
    finding is left out: recording_blockers already lists it, from the
    camera's state as it is now.
    """
    grouped: Dict[str, List[int]] = {}
    for finding in findings:
        if finding.severity != "error" or finding.message == STIM_CAMERA_REQUIRED:
            continue
        trials = grouped.setdefault(finding.message, [])
        if finding.trial_id is not None:
            trials.append(finding.trial_id)
    lines = [
        "Protocol check: " + (f"{_trials(trials)}: " if trials else "") + message
        for message, trials in grouped.items()
    ]
    if len(lines) > limit:
        lines = lines[:limit] + [
            f"Protocol check: and {len(lines) - limit} more; see Check protocol "
            "on the Protocol tab"]
    return tuple(lines)


def board_trigger_timing_refusal(
    laser, channel, live: ReadinessLiveState,
) -> str:
    """Why a board STIM pulse on `channel` would not be hardware-synchronized.

    prepare_pulse_profile requires hardware_synchronized for a board STIM
    route and fails the trial otherwise: "Protocol laser timing is not ready"
    (laser_model.py). The reasons are NidaqLaserController._resolve_pulse_timing's,
    for a pulse with a trigger terminal, in its order, on the plan the
    controller was opened with; the configuration's first. Then that plan
    has to be the one the stream runs now.

    The plan's reasons wait for a running stream and an open controller,
    whose own errors say it otherwise. A controller keeps its plan until it
    closes: System Mode opens the laser even when the stream's start failed
    (_start_laser_domain), and Refresh Hardware retries only what failed.
    """
    if laser.backend != "nidaq":
        return (f"the laser backend is {laser.backend!r}, and only the NI-DAQ "
                "backend arms a board STIM trigger")
    if not laser.hardware_timed:
        return ("laser hardwareTimed is off, so the laser is given no timing "
                "plan and a board STIM pulse is not synchronized")
    if live.nidaq_stream_state != "running" or not live.laser_connected:
        return ""
    reopen = ("; restart System Mode so that it opens on the stream's plan "
              "(Refresh Hardware does not reopen a laser that opened)")
    plan = live.laser_timing_plan
    if plan is None or not plan.is_valid:
        return ("the laser was opened without a valid NI timing plan, and keeps "
                "it until it closes" + (
                    f" ({plan.reason})" if plan is not None and plan.reason else "")
                + reopen)
    device = channel.analog_output.strip("/").split("/", 1)[0]
    if (device not in plan.hardware_output_devices
            and live.device_aliases.get(device, device) not in plan.hardware_output_devices):
        return (f"Laser output device {device} was not in the resolved timing "
                "topology")
    if not plan.sample_clock_source:
        return "Resolved timing topology has no shared sample clock"
    if plan != live.timing_plan:
        return ("the laser was opened on an earlier NI timing plan than the "
                "stream runs now" + reopen)
    return ""


def _rows_of(document):
    if hasattr(document, "rows"):
        return document.rows
    return tuple(item.row for item in document.resolve())


def _driven_backplane_lines(laser, live) -> set:
    """Backplane lines something drives: a laser's route, or a timing route.

    The controller connects every channel's route when it opens and holds it
    until it closes (NidaqLaserController._connect_trigger_route), so a line
    one laser's route drives triggers any laser that arms on it. A timing
    plan's routes include the configured externalRoutes. Generous by
    design: a line counted driven that is not misses an error, and one
    counted undriven that is driven blocks a protocol that runs.
    """
    lines = {
        backplane_line_of(channel.trigger_source)
        for channel in laser.channels
        if (channel.trigger_source or "").strip()
        and (channel.trigger_route_source or "").strip()
    }
    for plan in (live.timing_plan, live.laser_timing_plan):
        for route in getattr(plan, "routes", None) or ():
            lines.update(backplane_line_of(terminal) for terminal in route.destinations)
    lines.discard(None)
    return lines


def _disabled_row_errors(future) -> List[ReadinessFinding]:
    """A disabled row with an enabled one after it: the session stops there.

    The ledger takes the next trial in order and never skips one
    (PelletTrialLedger), and the compile refuses a disabled row ("Protocol
    row is disabled"), so every SEND from that row on is refused. Disabled
    rows after the last enabled one are where the protocol ends.
    """
    findings = []
    enabled_after = 0
    first_after = None
    for row in reversed(future):
        if row.enabled:
            enabled_after += 1
            first_after = row.trial_id
        elif enabled_after:
            findings.append(ReadinessFinding("error", row.trial_id, (
                "disabled, and a session stops here: trials run in order and a "
                f"disabled one is refused, so the {enabled_after} enabled "
                f"trial{'' if enabled_after == 1 else 's'} after it (from trial "
                f"{first_after}) never run; enable trial {row.trial_id}, or move "
                "or delete it")))
    return findings[::-1]


def _laser_errors(row, laser, laser_profiles, stimulated, driven) -> List[str]:
    if not row.laser_profile_id:
        return []
    errors = []
    profile = laser_profiles.get(row.laser_profile_id)
    if profile is None:
        errors.append(f"laser profile {row.laser_profile_id!r} is not in the profile library")
    # resolve_laser_firing is the runtime's own rule: an unconfigured laser,
    # and on a board STIM route a missing trigger terminal or boardStimLine.
    try:
        resolve_laser_firing(laser, row.laser_channel_id, row.laser_trigger_route)
    except ValueError as error:
        errors.append(str(error))
    try:
        channel = laser.get_channel(row.laser_channel_id)
    except (KeyError, ValueError):
        return errors
    number = int(channel.channel_id)
    if not (channel.analog_output or "").strip():
        errors.append(f"laser {number} has no analog output")
    if profile is not None:
        refusal = amplitude_refusal(profile, channel)
        if refusal:
            errors.append(refusal)
    terminal = (channel.trigger_source or "").strip()
    if (stimulated
            and row.laser_trigger_route is LaserTriggerRoute.HARDWARE_STIM3
            and backplane_line_of(terminal)
            and backplane_line_of(terminal) not in driven):
        # Nothing but a route drives a backplane line: without one the
        # board's pulse never reaches the terminal, and the laser stays armed.
        # Only for a row that fires: the compile drops the laser of one whose
        # assignment is disabled, and its trial runs without it.
        errors.append(
            f"laser {number} arms on {terminal}, a backplane line, and no "
            "triggerRouteSource drives it; set triggerRouteSource to the input "
            f"the board's STIM{channel.board_stim_line} pulse arrives on")
    return errors


def _trigger_errors(row, stimulated) -> List[str]:
    # _configure_protocol_cover's rule, which otherwise fails the trial as it
    # is prepared. A saved row cannot break it (TrialProtocolRow.validate).
    if not (stimulated and row.stimulus_trigger is StimulusTrigger.PRE_REVEAL):
        return []
    errors = []
    if row.laser_trigger_route is not LaserTriggerRoute.HARDWARE_STIM3:
        errors.append("Pre-reveal stimulation requires a board STIM laser row")
    if row.cover_policy is not CoverPolicy.REVEAL:
        errors.append("Pre-reveal stimulation requires the pellet cover policy Reveal")
    return errors


#: Triggers a randomized row arms for and then never starts: execution keys
#: on the row's own stimulus_trigger (_configure_protocol_cover, and
#: TrialActionExecutor.prepare for First Reach), which a randomized row
#: leaves at none. An error until that is fixed.
_UNRUNNABLE_DRAWS = {
    StimulusTrigger.PRE_REVEAL: "Pre-reveal",
    StimulusTrigger.FIRST_REACH: "First Reach",
}


def _randomized_errors(row, trigger_profiles) -> List[str]:
    if row.stimulus_assignment is not StimulusAssignment.RANDOMIZED:
        return []
    profile = trigger_profiles.get(row.stimulus_trigger_profile_id)
    if profile is None:
        return []  # the compile names the missing profile
    drawn = [
        _UNRUNNABLE_DRAWS[StimulusTrigger(category.trigger)]
        for category in profile.categories
        if category.enabled and StimulusTrigger(category.trigger) in _UNRUNNABLE_DRAWS
    ]
    if not drawn:
        return []
    names = " or ".join(dict.fromkeys(drawn))
    return [
        f"randomized stimulus can draw {names}, and a trial that draws it arms "
        "its laser and never starts it (a known defect); give those trials a "
        "fixed trigger"
    ]


def _compile_refusal(row, compiler, policies, protocol_id, revision) -> str:
    """The runtime's own refusal of this row, in its words, or empty.

    TrialActionCompiler.compile, as _compile_protocol_trial calls it, with a
    stand-in subject at the origin: the position only moves the target, which
    nothing here refuses. Of the checks _compile_protocol_trial makes first,
    the automatic shift policy is here too. Its uncalibrated-lane refusal is
    not: Preferences changes a lane offset without saying so, and a refusal
    cached from before the calibration would hold Record back after it.
    """
    if (row.position_mode is PelletPositionMode.REACH_DERIVED_AUTOMATIC
            and row.automatic_shift_policy_id not in policies):
        return f"Unknown automatic shift policy: {row.automatic_shift_policy_id}"
    offsets = {lane.value: (0.0, 0.0, 0.0) for lane in PelletLane}
    try:
        compiler.compile(row, TrialCompileContext(
            session_id="protocol-check",
            session_generation=0,
            protocol_id=str(protocol_id),
            protocol_revision=int(revision),
            logical_trial_id=row.trial_id,
            attempt_id=1,
            session_seed=0,
            animal_base_dcs=(0.0, 0.0, 0.0),
            lane_offsets_dcs=offsets,
        ))
    except Exception as error:  # the compiler's refusal is the finding
        return str(error) or type(error).__name__
    return ""


def _live_errors(firing_rows, laser, live) -> List[ReadinessFinding]:
    trials = _trials(row.trial_id for row in firing_rows)
    errors = []
    if not live.laser_connected:
        errors.append(
            f"the laser controller is not open, and it fires the laser on {trials}; "
            "start System Mode and check the Laser subsystem")
    if live.laser_close_refusal:
        errors.append(f"laser work is refused: {live.laser_close_refusal}")
    if live.nidaq_stream_state != "running":
        errors.append(
            f"the NI-DAQ stream is {live.nidaq_stream_state}; it has to be running "
            f"to fire the laser on {trials}")
    findings = [ReadinessFinding("error", None, message) for message in errors]
    for number, rows in _by_laser(firing_rows, board_only=True).items():
        channel = _channel(laser, number)
        if channel is None:
            continue  # its rows already say it is not configured
        refusal = board_trigger_timing_refusal(laser, channel, live)
        if refusal:
            findings.append(ReadinessFinding("error", None, (
                f"Protocol laser timing is not ready for laser {number} "
                f"({_trials(row.trial_id for row in rows)}): {refusal}")))
    return findings


def _warnings(firing_rows, configuration, laser_profiles, live) -> List[ReadinessFinding]:
    laser = configuration.laser
    messages = []
    if not laser.pmt_shutter_output:
        by_profile: Dict[str, List[int]] = {}
        for row in firing_rows:
            by_profile.setdefault(row.laser_profile_id, []).append(row.trial_id)
        for profile_id, trials in by_profile.items():
            profile = laser_profiles.get(profile_id)
            if profile is None or not (profile.pmt_open_lead_ms > 0 or profile.pmt_close_lag_ms > 0):
                continue
            # LaserModel.pmt_shutter_for_profile drops them at the pulse.
            messages.append(
                f"profile {profile_id!r} has PMT margins (open lead "
                f"{profile.pmt_open_lead_ms:g} ms, close lag {profile.pmt_close_lag_ms:g} ms), "
                "but no PMT shutter line (pmtShutterOutput) is configured: the "
                f"laser fires without them on {_trials(trials)}")
    acquired = {
        channel.physical_channel
        for channel in getattr(configuration.nidaq_stream, "channels", ())
    }
    for number in _by_laser(firing_rows, board_only=True):
        channel = _channel(laser, number)
        if channel is None:
            continue
        readback = channel.trigger_monitor_input
        if not readback:
            # A custom NI-DAQ channel may record the edge all the same, as
            # christielab10's laser1_trigger_readback on ai9 does; nothing
            # ties it to the laser.
            messages.append(
                f"laser {number} has no trigger readback input (triggerMonitorInput), "
                "so the recording does not tie the board trigger that starts it to "
                "the laser: no Board trigger graph, and no readback named for it")
        elif readback not in acquired:
            messages.append(
                f"laser {number}'s trigger readback {readback} is not in the NI-DAQ "
                "acquisition plan, so that input is not recorded")
    used = _by_laser(firing_rows)
    lines = [
        f"laser {number} shutter {channel.shutter_output}, diode {channel.diode_input}"
        for number in used
        for channel in (_channel(laser, number),)
        if channel is not None
    ]
    if lines:
        messages.append(
            "Shutters and diodes cannot be checked from software; check each is "
            "fitted and wired: " + "; ".join(lines))
    for number, rows in used.items():
        if live.laser_tab_profiles.get(number):
            continue
        profiles = ", ".join(dict.fromkeys(row.laser_profile_id for row in rows))
        messages.append(
            f"laser {number} has no profile picked on its Laser Control tab. That "
            "does not affect the protocol: each row fires its own profile "
            f"({_trials(row.trial_id for row in rows)}: {profiles}); the tab's "
            "pick is for Run Pulse and Test stim")
    return [ReadinessFinding("warning", None, message) for message in messages]


def _by_laser(rows, *, board_only=False) -> Dict[int, list]:
    lasers: Dict[int, list] = {}
    for row in rows:
        if board_only and row.laser_trigger_route is not LaserTriggerRoute.HARDWARE_STIM3:
            continue
        lasers.setdefault(int(row.laser_channel_id), []).append(row)
    return dict(sorted(lasers.items()))


def _channel(laser, number):
    try:
        return laser.get_channel(number)
    except (KeyError, ValueError):
        return None


def _trials(trial_ids) -> str:
    """'trial 3', or 'trials 3-5, 8'."""
    ids = sorted(set(int(value) for value in trial_ids))
    if len(ids) == 1:
        return f"trial {ids[0]}"
    spans = []
    for value in ids:
        if spans and value == spans[-1][1] + 1:
            spans[-1][1] = value
        else:
            spans.append([value, value])
    return "trials " + ", ".join(
        str(start) if start == end else f"{start}-{end}" for start, end in spans)
