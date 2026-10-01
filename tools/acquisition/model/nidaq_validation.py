"""Check a channel map against the hardware before anything tries to use it.

Until now the configuration was validated for syntax and nothing else: enum
membership, non-empty strings. Whether a named terminal existed, whether a
board could do what was asked of it, whether a route was possible at all -
all of that was discovered by a task failing, sometimes with an error code
that named neither the configuration line responsible nor the remedy.

Every check here exists because its absence cost real time:

  * `PXI_CLK10` was set on a PXI-6713 that exposes no Clk10 terminal, failing
    with -200452 and taking the signal stream with it;
  * a stimulus line was wired to an APFI connector on a board whose
    `anlg_trig_supported` reads False, which the driver would have said at
    any point;
  * a laser output on one board was clocked from another with the chassis
    unidentified, failing inside a task at -89125.

The driver is the authority throughout. Nothing here hard-codes what a board
model can do, because that would rot; it compares the configuration against
what discovery reported.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional, Sequence, Tuple

from autotrainer.core.configuration.laser_configuration import backplane_line_of
from tools.acquisition.model.nidaq_monitor_survey import clocks_digital_input
from tools.acquisition.model.nidaq_routing import (
    capability_between,
    chassis_is_identified,
)

ERROR = "error"


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
    """One configuration assertion the hardware does not support."""

    subject: str
    value: str
    problem: str
    #: What the driver says is available instead. Empty when the answer is not
    #: a list of alternatives.
    possible: Tuple[str, ...] = tuple()
    remedy: str = ""
    severity: str = ERROR

    def describe(self) -> str:
        text = f"{self.subject} is {self.value!r}: {self.problem}"
        if self.possible:
            shown = ", ".join(self.possible[:8])
            if len(self.possible) > 8:
                shown += f", and {len(self.possible) - 8} more"
            text += f" (available: {shown})"
        if self.remedy:
            text += f". {self.remedy}"
        return text


class NidaqConfigurationInvalid(RuntimeError):
    """Raised instead of letting a task discover the problem later."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__(
            "NI-DAQ configuration does not match the installed hardware:\n  "
            + "\n  ".join(issue.describe() for issue in self.issues)
        )


def _device_of(terminal: Optional[str]) -> str:
    """The device a terminal or channel name belongs to."""
    if not terminal:
        return ""
    return terminal.strip("/").split("/", 1)[0]


def _tail(name: str) -> str:
    return name.strip("/").lower()


def _find_device(devices, name: str):
    for device in devices:
        if getattr(device, "name", "") == name:
            return device
    return None


def _channel_exists(device, physical_channel: str) -> bool:
    wanted = _tail(physical_channel)
    for attribute in ("analog_inputs", "analog_outputs", "digital_inputs",
                      "digital_outputs", "counter_inputs", "counter_outputs"):
        for candidate in getattr(device, attribute, ()) or ():
            if _tail(str(candidate)) == wanted:
                return True
    return False


def _terminal_exists(device, terminal: str) -> bool:
    wanted = _tail(terminal).rsplit("/", 1)[-1]
    for candidate in getattr(device, "terminals", ()) or ():
        if _tail(str(candidate)).rsplit("/", 1)[-1] == wanted:
            return True
    return False


def _terminal_names(device) -> Tuple[str, ...]:
    return tuple(
        str(terminal).strip("/").rsplit("/", 1)[-1]
        for terminal in (getattr(device, "terminals", ()) or ())
    )


def _check_named_terminal(devices, subject, terminal) -> Optional[ValidationIssue]:
    """A terminal has to exist on the board whose name it carries."""
    if not terminal:
        return None
    device_name = _device_of(terminal)
    device = _find_device(devices, device_name)
    if device is None:
        return ValidationIssue(
            subject, terminal,
            f"names device {device_name!r}, which is not installed",
            possible=tuple(getattr(d, "name", "") for d in devices),
        )
    if not _terminal_exists(device, terminal):
        return ValidationIssue(
            subject, terminal,
            f"names a terminal {device_name} does not expose",
            possible=_terminal_names(device),
            remedy="a board's terminal list is what it reports, and differs "
                   "between models in the same chassis",
        )
    return None


def _is_digital_input(device, physical_channel: str) -> bool:
    wanted = _tail(physical_channel)
    return any(_tail(str(line)) == wanted
               for line in getattr(device, "digital_inputs", ()) or ())


def unclockable_digital_input(device, channel, subject_prefix="stream channel"
                              ) -> Optional[ValidationIssue]:
    """The refusal for a digital input its board cannot clock, or None.

    The stream samples every digital input in one clocked task. A PXI-6713
    port0 line passed every check and failed only when the stream started,
    at -200452, naming neither line nor board. This is the one check for it:
    Run and the DAQ Monitor reach it through validate_channels, and every
    stream start, Idle's included, through the timing plan the start builds
    from its own discovery (nidaq_timing._validate_channels_and_rates).
    """
    physical = getattr(channel, "physical_channel", "")
    if not _is_digital_input(device, physical) or clocks_digital_input(device):
        return None
    return ValidationIssue(
        f"{subject_prefix} {getattr(channel, 'name', physical)!r}", physical,
        f"names a line on {getattr(device, 'name', '')}, which cannot clock "
        "digital input: discovery reports no digital-input rate for it",
        remedy="use a port0 line on a board that clocks digital input",
    )


def validate_channels(devices, channels, subject_prefix="stream channel"):
    """Every configured channel names a device and a channel that exist.

    And a digital input names a board that can clock one
    (unclockable_digital_input).
    """
    issues = []
    for channel in channels or ():
        physical = getattr(channel, "physical_channel", "")
        name = getattr(channel, "name", physical)
        device_name = _device_of(physical)
        device = _find_device(devices, device_name)
        if device is None:
            issues.append(ValidationIssue(
                f"{subject_prefix} {name!r}", physical,
                f"names device {device_name!r}, which is not installed",
                possible=tuple(getattr(d, "name", "") for d in devices),
            ))
            continue
        if not _channel_exists(device, physical):
            issues.append(ValidationIssue(
                f"{subject_prefix} {name!r}", physical,
                f"names a channel {device_name} does not have",
            ))
            continue
        issue = unclockable_digital_input(device, channel, subject_prefix)
        if issue is not None:
            issues.append(issue)
    return tuple(issues)


def validate_terminal_configuration(devices, stream) -> Tuple[ValidationIssue, ...]:
    """The analog referencing has to be one each configured input accepts.

    Checked per channel, not per board. They differ within one board: a
    PXI-6221 offers DIFF on ai0-ai7 and only RSE or NRSE above that, so a
    board-wide answer would accept a differential configuration that half the
    configured channels cannot honour - which is the shape of the defect that
    started this work.
    """
    requested = (getattr(stream, "analog_terminal_config", "") or "").upper()
    if not requested:
        return tuple()
    issues = []
    for channel in getattr(stream, "channels", ()) or ():
        if getattr(channel, "kind", "") == "digital":
            continue
        physical = getattr(channel, "physical_channel", "")
        device = _find_device(devices, _device_of(physical))
        if device is None:
            continue
        for candidate, modes in (
                getattr(device, "analog_input_terminal_configs", ()) or ()):
            if _tail(str(candidate)) != _tail(physical):
                continue
            supported = tuple(str(mode).upper() for mode in modes)
            if supported and requested not in supported:
                issues.append(ValidationIssue(
                    f"analogTerminalConfig for {getattr(channel, 'name', physical)!r}",
                    requested.lower(),
                    f"{physical} does not accept that referencing",
                    possible=tuple(mode.lower() for mode in supported),
                    remedy="terminal configurations differ across a board's "
                           "channel range",
                ))
            break
    return tuple(issues)


def validate_output_timing(devices, laser, timing_plan_clock_source
                           ) -> Tuple[ValidationIssue, ...]:
    """A laser clocked from another board needs a route that exists.

    The rule is not "refuse when the chassis is unidentified". That would
    refuse this rig, which runs exactly that way: a PXI_Trig line driven by
    name carries the trigger even when DAQmx will not reserve one. What is
    refused is relying on the automatic form when it is unavailable and no
    explicit route has been given.
    """
    if laser is None or not getattr(laser, "hardware_timed", False):
        return tuple()
    clock_device = _device_of(timing_plan_clock_source)
    if not clock_device:
        return tuple()
    source = _find_device(devices, clock_device)
    if source is None:
        return tuple()

    issues = []
    for channel in getattr(laser, "channels", ()) or ():
        number = getattr(getattr(channel, "channel_id", None), "value", "?")
        output_device = _device_of(getattr(channel, "analog_output", ""))
        if not output_device or output_device == clock_device:
            continue
        destination = _find_device(devices, output_device)
        if destination is None:
            continue
        capability = capability_between(source, destination)
        if not capability.is_possible:
            issues.append(ValidationIssue(
                f"laser channel {number} output", channel.analog_output,
                f"is clocked from {clock_device}, and {capability.reason}",
                remedy=capability.requires or "put the output on the board "
                                              "that produces the clock",
            ))
            continue
        if not capability.is_automatic and not getattr(
                channel, "trigger_route_source", None):
            issues.append(ValidationIssue(
                f"laser channel {number} output", channel.analog_output,
                f"is clocked from {clock_device} and {capability.reason}, "
                "but no explicit route is configured",
                remedy="set triggerRouteSource to the terminal the stimulus "
                       "arrives on, and triggerSource to the far board's view "
                       "of the backplane line",
            ))
    return tuple(issues)


def validate_trigger_capability(devices, laser) -> Tuple[ValidationIssue, ...]:
    """A board asked for a trigger it cannot do says so through the driver."""
    issues = []
    for channel in getattr(laser, "channels", ()) or ():
        terminal = getattr(channel, "trigger_source", None)
        if not terminal:
            continue
        number = getattr(getattr(channel, "channel_id", None), "value", "?")
        device = _find_device(devices, _device_of(terminal))
        if device is None:
            continue
        if "APFI" in terminal.upper() and not getattr(
                device, "analog_trigger_supported", True):
            issues.append(ValidationIssue(
                f"laser channel {number} triggerSource", terminal,
                f"{device.name} has no analog trigger circuit",
                possible=("digital edge on a PFI or PXI_Trig terminal",),
                remedy="APFI connectors exist on the breakout regardless of "
                       "which board is behind them",
            ))
    return tuple(issues)


#: The timing fields the stream's master exports its signals onto, for the
#: input boards it masters (slave_input_timing).
_EXPORT_FIELDS = (
    ("sample_clock_export_terminal", "timing sampleClockExportTerminal",
     "sample clock"),
    ("start_trigger_export_terminal", "timing startTriggerExportTerminal",
     "start trigger"),
)

_CORRUPTED = ("two signals driven onto one backplane line corrupt each other, "
              "and DAQmx does not see it across these boards")


def validate_export_lines(timing, laser) -> Tuple[ValidationIssue, ...]:
    """Each of the stream's export lines is a backplane line of its own.

    The stream's master drives sampleClockExportTerminal and
    startTriggerExportTerminal for as long as it runs, and an input board it
    masters reads them. Neither may be the other, the laser's
    backplaneClockLine or pulseClockLine, or a laser's triggerSource line
    (its trigger route's destination): a second driver on each. Nor a
    PXI_Trig triggerRouteSource, whose route would carry the stream's signal
    into the trigger, nor a triggerListenerInputs line, which would read it
    rather than a trigger. One refusal each, as clock_line_clashes refuses
    the laser's own lines, with a line nothing here takes as the remedy.
    Lines compare by name, whatever the board.

    Here because this is where the timing and the laser configuration are
    both seen, before any task: Run refuses on it, and the DAQ Monitor and
    the wiring check list it.
    """
    exports = tuple(
        (field, signal, value, backplane_line_of(value))
        for attribute, field, signal in _EXPORT_FIELDS
        for value in (getattr(timing, attribute, None),)
        if backplane_line_of(value) is not None
    )
    if not exports:
        return tuple()
    channels = tuple(getattr(laser, "channels", ()) or ())
    listeners = tuple(getattr(laser, "trigger_listener_inputs", ()) or ())
    clock_lines = tuple(
        (name, backplane_line_of(getattr(laser, attribute, None)))
        for attribute, name in (("backplane_clock_line", "backplaneClockLine"),
                                ("pulse_clock_line", "pulseClockLine"))
    )
    taken = {line for _field, _signal, _value, line in exports}
    taken |= {line for _name, line in clock_lines}
    taken |= {backplane_line_of(terminal) for terminal in listeners}
    taken |= {backplane_line_of(getattr(channel, attribute, None))
              for channel in channels
              for attribute in ("trigger_source", "trigger_route_source")}
    free = next((f"PXI_Trig{number}" for number in range(8)
                 if f"pxi_trig{number}" not in taken), None)
    remedy = ("choose another PXI_Trig line for one of them"
              + (f", such as {free}" if free else ""))

    issues = []
    for index, (field, signal, value, line) in enumerate(exports):
        shown = f"PXI_Trig{line[len('pxi_trig'):]}"
        problems = [
            f"uses {shown}, as {other_field} does: the master would drive its "
            "sample clock and its start trigger onto one line"
            for other_field, _signal, _value, other_line in exports[:index]
            if other_line == line
        ]
        problems += [
            f"uses {shown}, the laser {name}: {_CORRUPTED}"
            for name, clock_line in clock_lines if clock_line == line
        ]
        for channel in channels:
            number = getattr(getattr(channel, "channel_id", None), "value", "?")
            trigger = getattr(channel, "trigger_source", None)
            route = getattr(channel, "trigger_route_source", None)
            if backplane_line_of(trigger) == line:
                problems.append(
                    f"uses {shown}, which laser {number} triggerSource "
                    f"{trigger} takes: {_CORRUPTED}")
            if backplane_line_of(route) == line:
                problems.append(
                    f"uses {shown}, which laser {number} triggerRouteSource "
                    f"{route} routes from: the route would carry the stream's "
                    f"{signal} into laser {number}'s trigger, and the laser "
                    "would arm on its first edge")
        problems += [
            f"uses {shown}, which triggerListenerInputs {terminal} reads: it "
            f"would read the stream's {signal} rather than a trigger"
            for terminal in listeners if backplane_line_of(terminal) == line
        ]
        issues.extend(
            ValidationIssue(field, value, problem, remedy=remedy)
            for problem in problems)
    return tuple(issues)


def validate_nidaq_configuration(
    devices: Iterable,
    *,
    stream=None,
    ports=None,
    laser=None,
    timing_plan=None,
    clock_source=None,
) -> Tuple[ValidationIssue, ...]:
    """Every assertion the configuration makes, against the installed boards.

    The laser's route check needs the board the shared clock comes from:
    `clock_source`, else the timing plan's sample clock.
    """
    devices = tuple(devices)
    if not devices:
        return tuple()

    issues = list(validate_channels(devices, getattr(stream, "channels", ())))
    issues.extend(validate_terminal_configuration(devices, stream))

    timing = getattr(ports, "timing", None)
    for attribute, subject in (
        ("reference_clock_source", "timing referenceClockSource"),
        ("start_trigger_source", "timing startTriggerSource"),
        ("sample_clock_source", "timing sampleClockSource"),
        ("sample_clock_export_terminal", "timing sampleClockExportTerminal"),
        ("start_trigger_export_terminal", "timing startTriggerExportTerminal"),
    ):
        value = getattr(timing, attribute, None)
        # A bare name like PXI_CLK10 carries no device, so there is nothing to
        # check it against; the stream resolves those per board at task time.
        if value and "/" in value:
            issue = _check_named_terminal(devices, subject, value)
            if issue is not None:
                issues.append(issue)

    for channel in getattr(laser, "channels", ()) or ():
        number = getattr(getattr(channel, "channel_id", None), "value", "?")
        for attribute, subject in (
            ("trigger_source", f"laser channel {number} triggerSource"),
            ("trigger_route_source",
             f"laser channel {number} triggerRouteSource"),
        ):
            value = getattr(channel, attribute, None)
            if value and "/" in value:
                issue = _check_named_terminal(devices, subject, value)
                if issue is not None:
                    issues.append(issue)

    issues.extend(validate_trigger_capability(devices, laser))
    issues.extend(validate_export_lines(timing, laser))
    issues.extend(validate_output_timing(
        devices, laser,
        clock_source or getattr(timing_plan, "sample_clock_source", None)))
    return tuple(issues)


def refused_plan_clock_source(plan, timing) -> Optional[str]:
    """A terminal on the board a refused plan's shared clock would come from.

    Only its board is read (validate_output_timing). A start the plan refused
    leaves no clock on it, and the laser's route check, which reads the
    clock's board off the plan, went unasked beside the refusal. The plan
    keeps the master it would have had (build_nidaq_timing_plan), and a mode
    that shares a clock puts it there, unless `timing` names the clock
    itself. None otherwise, as for a valid plan with no shared clock: no
    plan, a valid one (its own clock is the one to read), a refused plan with
    no master, or independent boards, which share no clock at all.
    """
    if plan is None or plan.is_valid or plan.requested_mode == "independent":
        return None
    explicit = getattr(timing, "sample_clock_source", None)
    if explicit:
        return explicit
    master = getattr(plan, "master_device", None)
    return f"/{master}/ai/SampleClock" if master else None


def require_valid_nidaq_configuration(devices, **kwargs) -> None:
    """Refuse a configuration the hardware cannot honour."""
    issues = validate_nidaq_configuration(devices, **kwargs)
    if issues:
        raise NidaqConfigurationInvalid(issues)


def chassis_identification_note(devices) -> str:
    """A sentence about unidentified chassis, or empty when all are placed.

    Not an error on its own - explicit routing works without it - but it
    removes automatic cross-board routing, and that is worth saying once
    rather than leaving someone to meet -89125.
    """
    unplaced = sorted(
        getattr(device, "name", "")
        for device in devices
        if "PXI" in str(getattr(device, "bus_type", "")).upper()
        and not chassis_is_identified(device)
    )
    if not unplaced:
        return ""
    return (
        "PXI chassis not identified for " + ", ".join(unplaced)
        + "; DAQmx will not allocate backplane trigger lines by itself, so "
        "cross-board timing needs an explicit route"
    )
