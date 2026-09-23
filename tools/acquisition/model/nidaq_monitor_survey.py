"""Every line on every card, so a configuration can be read off the rig.

The configured stream carries the ten channels this rig was told about. That
is the wrong set for deciding what to tell it: working out which line a
stimulus arrives on means watching the lines nobody has claimed yet, which is
exactly what the configured stream leaves out. Someone with a cable and no
documentation had to guess, patch, restart and look - and the loop that
produced "there is no slot for PFI1" and a stimulus on a dead APFI connector.

So the survey asks the driver what exists and streams all of it, named after
the terminal rather than after a role nothing has assigned yet.

Two limits are the hardware's, and both are why this is not simply "all the
lines in one task":

  * correlated digital input on an M Series board is port0 only. port1 and
    port2 are the PFI pins, which are static digital I/O - they have levels
    but no sample clock, so they are polled beside the stream rather than in
    it;
  * some boards have no correlated digital input at all. A PXI-6713's
    digital lines hold a level and cannot be clocked, and the driver says so
    by refusing the DI maximum rate property outright. Measured here: putting
    its port0 in a buffered task fails at -200452 on DI_DataXferMech and
    takes the whole survey with it. Which port is streamable is therefore
    asked, not assumed;
  * an analog output board has no analog input either, so it contributes
    digital lines and nothing else. Asking it for ai0 is how the availability
    barrier once waited forever.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)

from tools.acquisition.model.nidaq_breakout import (
    breakout_for_device,
    describe_terminal,
)

#: The only port an M Series board can clock. The others are static.
BUFFERED_PORT = "port0"
#: Polled beside the stream, for level rather than waveform.
STATIC_PORTS = ("port1", "port2")

#: Slow enough that sixteen inputs and eight lines cost nothing, fast enough
#: to see a stimulus pulse a human is triggering by hand.
DEFAULT_SURVEY_RATE_HZ = 2_000.0
DEFAULT_SURVEY_CHUNK = 200


def _tail(name: str) -> str:
    text = str(name or "").strip("/")
    return text.split("/", 1)[-1] if "/" in text else text


def _port_of(physical_channel: str) -> str:
    tail = _tail(physical_channel)
    return tail.split("/", 1)[0] if "/" in tail else ""


def _channel_name(physical_channel: str) -> str:
    """A name that says where it is, because nothing else knows yet.

    `PXI1Slot5/port0/line3` becomes `PXI1Slot5.port0.line3`: unique across
    boards, and readable as a location rather than as a role.
    """
    return str(physical_channel or "").strip("/").replace("/", ".")


@dataclasses.dataclass(frozen=True)
class SurveyLine:
    """One line the survey watches, and everything known about it up front."""

    name: str
    physical_channel: str
    device: str
    kind: str
    #: "stream" when it is clocked into the shared ring, "static" when it is
    #: polled for a level.
    acquisition: str
    #: The breakout connector, when the device has a block named.
    label: str = ""
    #: What the configuration already calls this line, when anything does.
    assigned_to: str = ""

    @property
    def terminal(self) -> str:
        return _tail(self.physical_channel)

    def describe(self) -> str:
        text = self.assigned_to or self.terminal
        return f"{text} — {self.label}" if self.label else text


@dataclasses.dataclass(frozen=True)
class MonitorSurvey:
    """What the rig has, split by how it can be watched."""

    lines: Tuple[SurveyLine, ...] = tuple()
    devices: Tuple[str, ...] = tuple()
    #: Why a device contributes nothing of some kind, keyed by device.
    notes: Tuple[Tuple[str, str], ...] = tuple()

    def for_device(self, device: str) -> Tuple[SurveyLine, ...]:
        return tuple(line for line in self.lines if line.device == device)

    def streamed(self) -> Tuple[SurveyLine, ...]:
        return tuple(line for line in self.lines if line.acquisition == "stream")

    def static(self) -> Tuple[SurveyLine, ...]:
        return tuple(line for line in self.lines if line.acquisition == "static")

    def notes_for(self, device: str) -> Tuple[str, ...]:
        """Everything worth saying about this card - often more than one thing.

        A PXI-6713 has neither analog input nor clocked digital input, and
        returning only the first would leave half the explanation out.
        """
        return tuple(note for name, note in self.notes if name == device)


def _assignments(configuration) -> Dict[str, str]:
    """Every terminal the configuration already claims, by full channel name."""
    found: Dict[str, str] = {}

    def record(value, owner):
        text = str(value or "").strip("/").lower()
        if text:
            found.setdefault(text, owner)

    stream = getattr(configuration, "nidaq_stream", None)
    for channel in getattr(stream, "channels", ()) or ():
        record(getattr(channel, "physical_channel", ""),
               getattr(channel, "name", ""))
    laser = getattr(configuration, "laser", None)
    for channel in getattr(laser, "channels", ()) or ():
        number = getattr(getattr(channel, "channel_id", None), "value", "?")
        for attribute, described in (
            ("analog_output", f"laser {number} command"),
            ("diode_input", f"laser {number} diode"),
            ("shutter_output", f"laser {number} shutter"),
            ("trigger_route_source", f"laser {number} trigger in"),
        ):
            record(getattr(channel, attribute, ""), described)
    ports = getattr(configuration, "nidaq_ports", None)
    for attribute in ("tone1", "tone2", "tone3_r", "tone3_l", "cam_frames",
                      "barcode"):
        record(getattr(ports, attribute, ""), attribute)
    return found


def build_survey(devices: Sequence, configuration=None) -> MonitorSurvey:
    """Every watchable line the driver reports, located and cross-referenced."""
    ports = getattr(configuration, "nidaq_ports", None)
    assignments = _assignments(configuration)
    lines, notes, names = [], [], []

    for device in devices:
        device_name = getattr(device, "name", "")
        if not device_name:
            continue
        names.append(device_name)
        model = breakout_for_device(ports, device_name)

        def claimed_by(physical_channel) -> str:
            """What the configuration calls this pin, under any of its names.

            A pin has more than one name: the PFI 0 BNC is port1/line0 to the
            digital subsystem and PFI0 to everything that triggers on it, and
            this rig's laser trigger is configured as the second. Matching
            only the literal string would survey the line somebody wired a
            stimulus to and report it as spare.
            """
            candidates = [str(physical_channel).strip("/").lower()]
            for connector in (model.connectors_for(_tail(physical_channel))
                              if model is not None else ()):
                for name in connector.terminals:
                    candidates.append(f"{device_name}/{name}".lower())
            for candidate in candidates:
                if candidate in assignments:
                    return assignments[candidate]
            return ""

        def add(physical_channel, kind, acquisition):
            lines.append(SurveyLine(
                name=_channel_name(physical_channel),
                physical_channel=physical_channel,
                device=device_name,
                kind=kind,
                acquisition=acquisition,
                label=describe_terminal(model, _tail(physical_channel)),
                assigned_to=claimed_by(physical_channel),
            ))

        analog_inputs = tuple(getattr(device, "analog_inputs", ()) or ())
        for channel in analog_inputs:
            add(channel, "analog", "stream")
        if not analog_inputs:
            notes.append((
                device_name,
                f"{device_name} reports no analog input, so it contributes "
                "digital lines only"))

        # The driver's own answer. A board with no correlated digital input
        # does not report a maximum rate for it, and none of its lines can go
        # in a clocked task however they are named.
        can_clock_digital = getattr(
            device, "digital_input_max_rate", None) is not None
        if not can_clock_digital and getattr(device, "digital_inputs", ()):
            notes.append((
                device_name,
                f"{device_name} reports no clocked digital input, so its "
                "lines are polled for a level rather than streamed"))

        for channel in tuple(getattr(device, "digital_inputs", ()) or ()):
            port = _port_of(channel)
            if port == BUFFERED_PORT and can_clock_digital:
                add(channel, "digital", "stream")
            elif port in STATIC_PORTS or not can_clock_digital:
                # A level, not a waveform. Polled beside the stream.
                add(channel, "digital", "static")

    return MonitorSurvey(lines=tuple(lines), devices=tuple(names),
                         notes=tuple(notes))


def survey_stream_configuration(
    survey: MonitorSurvey,
    *,
    sample_rate_hz: float = DEFAULT_SURVEY_RATE_HZ,
    read_chunk_size: int = DEFAULT_SURVEY_CHUNK,
    analog_terminal_config: Optional[str] = None,
    devices: Optional[Sequence[str]] = None,
) -> NidaqSignalStreamConfiguration:
    """The clocked half of the survey, as a stream configuration.

    `analog_terminal_config` is passed through rather than defaulted. Letting
    DAQmx choose is not neutral - on a PXI-6221 it gives ai0 to ai7
    differential, pairing each with ai8 to ai15, so a survey of the whole
    range would read half its channels against pins it is also reading
    directly, and show one signal in three places.
    """
    streamed = [line for line in survey.streamed()
                if devices is None or line.device in devices]
    return NidaqSignalStreamConfiguration(
        channels=tuple(
            NidaqSignalChannelConfiguration(
                name=line.name,
                physical_channel=line.physical_channel,
                kind=line.kind,
            )
            for line in streamed
        ),
        is_enabled=bool(streamed),
        sample_rate_hz=sample_rate_hz,
        read_chunk_size=read_chunk_size,
        analog_terminal_config=analog_terminal_config or "",
    )
