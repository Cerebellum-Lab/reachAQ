from __future__ import annotations

import dataclasses
import enum
import re
from typing import ClassVar, Iterable, Optional, Tuple, Union

import yaml

from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper, SystemConfigurationLoader


class LaserChannelId(enum.IntEnum):
    LASER_1 = 1
    LASER_2 = 2
    LASER_3 = 3
    LASER_4 = 4


def normalize_laser_channel_id(value: Union[LaserChannelId, int]) -> LaserChannelId:
    return value if isinstance(value, LaserChannelId) else LaserChannelId(int(value))


#: One PXI backplane trigger line, as the last part of a terminal's name.
_BACKPLANE_LINE = re.compile(r"^pxi_trig\d+$")

def backplane_line_of(terminal: Optional[str]) -> Optional[str]:
    """The PXI_Trig line `terminal` names, lower-cased, or None for any other.

    The device is ignored: every board in the chassis names the same bussed
    line as its own, so /PXI1Slot4/PXI_Trig0 and /PXI1Slot5/pxi_trig0 are one
    line.
    """
    tail = str(terminal or "").strip().strip("/").rsplit("/", 1)[-1].lower()
    return tail if _BACKPLANE_LINE.match(tail) else None


def clock_line_clashes(
    backplane_clock_line: str,
    pulse_clock_line: str,
    channels: Iterable["LaserChannelConfiguration"],
    trigger_listener_inputs: Iterable[str],
) -> Tuple[str, ...]:
    """Each clash with a clock line, as one refusal each, with its remedy.

    Two clocks ride the backplane: backplaneClockLine, and pulseClockLine,
    which carries the laser's own sample clock to a clocked digital line on
    another board. Neither may be a line a trigger takes, nor the other one.
    A channel's trigger_source is the line its trigger rides on, and
    NidaqLaserController._connect_trigger_route drives exactly that line,
    named on the route source's board, as its route's destination: on a
    clock's line, a second driver. The route reads its trigger_route_source:
    a clock's line there would carry that clock into the trigger, and the
    laser would arm on the clock's first edge. Nothing drives a trigger
    listener input, so it cannot corrupt a clock; one on a clock's line is
    refused as a defensive check, since it would watch the clock rather
    than a trigger.
    """
    channels = tuple(channels)
    listeners = tuple(trigger_listener_inputs)
    clashes = []
    lines = (("backplaneClockLine", backplane_clock_line),
             ("pulseClockLine", pulse_clock_line))
    for field, value in lines:
        clock = backplane_line_of(value)
        if clock is None:
            continue
        shown = str(value).strip().strip("/").rsplit("/", 1)[-1]
        remedy = f"choose a different trigger line or {field}"
        for channel in channels:
            if backplane_line_of(channel.trigger_source) == clock:
                clashes.append(
                    f"laser {int(channel.channel_id)} triggerSource "
                    f"{channel.trigger_source} uses {shown}, the {field}; {remedy}")
            if backplane_line_of(channel.trigger_route_source) == clock:
                clashes.append(
                    f"laser {int(channel.channel_id)} triggerRouteSource "
                    f"{channel.trigger_route_source} uses {shown}, the {field}: "
                    "its route would carry that clock into the trigger, and the "
                    f"laser would arm on its first edge; choose a different "
                    f"route source or {field}")
        for terminal in listeners:
            if backplane_line_of(terminal) == clock:
                clashes.append(
                    f"triggerListenerInputs {terminal} uses {shown}, the "
                    f"{field}; {remedy}")
    if (backplane_line_of(pulse_clock_line) is not None
            and backplane_line_of(pulse_clock_line) == backplane_line_of(backplane_clock_line)):
        clashes.append(
            f"pulseClockLine {pulse_clock_line} is the backplaneClockLine, "
            f"{backplane_clock_line}; the two clocks need a line each: choose "
            "another pulseClockLine")
    return tuple(clashes)


def _bare_backplane_line(field: str, value) -> str:
    """`value` as the one PXI_Trig line it names, spelt as DAQmx spells it.

    Bare, without a board: the controller names the line on whichever board
    drives or reads it, so a board given here was either taken for the line
    or silently dropped. Empty, or anything but a PXI_Trig line, is refused,
    and so is a number the backplane does not have: PXI has PXI_Trig0 to
    PXI_Trig7.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(
            f"{field} is empty: it must be a PXI_Trig line, such as PXI_Trig1")
    if "/" in text:
        # Not the value's own tail as the example: that may be refused too,
        # as a PFI is.
        raise ValueError(
            f"{field} {text!r} names a board: give a bare PXI_Trig line, such "
            f"as {_EXAMPLE_LINE[field]}, which every board in the chassis sees "
            "as its own")
    line = backplane_line_of(text)
    if line is None:
        raise ValueError(
            f"{field} {text!r} is not a PXI_Trig line, such as PXI_Trig1")
    number = int(line[len("pxi_trig"):])
    if number > 7:
        raise ValueError(
            f"{field} {text!r} is not a line the backplane has: PXI has "
            "PXI_Trig0 to PXI_Trig7")
    return f"PXI_Trig{number}"


#: A clock line a refusal can suggest: each field's own default.
_EXAMPLE_LINE = {"backplaneClockLine": "PXI_Trig1", "pulseClockLine": "PXI_Trig3"}

#: An argument left out, as distinct from one given as None.
_FIELD_DEFAULT = object()


@dataclasses.dataclass(frozen=True)
class LaserChannelConfiguration:
    """NI-DAQ channel assignment for one independently controlled laser."""

    channel_id: LaserChannelId
    analog_output: str
    diode_input: str
    shutter_output: str
    auxiliary_output: Optional[str] = None
    command_copy_input: Optional[str] = None
    #: NI input the board's stimulus line is wired back into, so the trigger
    #: that starts this channel's waveform can be plotted beside the waveform.
    #: This is a readback and is distinct from trigger_source below, which is
    #: the terminal the analog output task arms on.
    trigger_monitor_input: Optional[str] = None
    trigger_source: Optional[str] = None
    #: Terminal to drive trigger_source from, when the two are on different
    #: boards. DAQmx routes a trigger across a PXI backplane by reserving a
    #: line, and it will not reserve one without knowing the chassis - which
    #: this chassis never reports, failing every cross-board arm with -89125.
    #: Driving a PXI_Trig line explicitly needs no such bookkeeping and was
    #: measured carrying the board's stimulus pulse from the PXI-6221 to the
    #: PXI-6713. Set this to the originating terminal, for example
    #: /PXI1Slot5/PFI0, with trigger_source naming the far board's view of the
    #: same line, /PXI1Slot4/PXI_Trig0.
    trigger_route_source: Optional[str] = None
    #: Board line wired to this laser's trigger input, named as the board's
    #: device tree names it: STIM2 or STIM3. None makes the board STIM route
    #: unavailable for this laser. It describes the wiring, so it lives here
    #: and not in a pulse profile, where it once sent laser 2's trigger to
    #: laser 1's input.
    board_stim_line: Optional[int] = None
    #: Width of the board STIM pulse that starts this laser's waveform.
    board_trigger_pulse_us: int = 1000
    trigger_output: Optional[str] = None
    timing_trigger_output: Optional[str] = None
    minimum_command_volts: float = 0.0
    maximum_command_volts: float = 5.0
    feedback_scale: float = 1.0
    command_copy_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "channel_id", normalize_laser_channel_id(self.channel_id))
        for name in ("analog_output", "diode_input", "shutter_output"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be provided")
        if self.maximum_command_volts <= self.minimum_command_volts:
            raise ValueError("maximum_command_volts must be greater than minimum_command_volts")
        if self.feedback_scale <= 0:
            raise ValueError("feedback_scale must be positive")
        if self.command_copy_scale <= 0:
            raise ValueError("command_copy_scale must be positive")
        if self.board_stim_line is not None:
            if int(self.board_stim_line) not in (2, 3):
                raise ValueError(
                    "board_stim_line must be 2 or 3; STIM0 and STIM1 carry the "
                    "tone confirmations"
                )
            object.__setattr__(self, "board_stim_line", int(self.board_stim_line))
        if not 100 <= int(self.board_trigger_pulse_us) <= 5_000_000:
            raise ValueError("board_trigger_pulse_us must be within 100..5000000")
        object.__setattr__(self, "board_trigger_pulse_us", int(self.board_trigger_pulse_us))

    def clamp_command_voltage(self, volts: float) -> float:
        return min(max(volts, self.minimum_command_volts), self.maximum_command_volts)


@dataclasses.dataclass(frozen=True)
class LaserSystemConfiguration:
    """Configuration for up to four laser channels."""

    VALID_BACKENDS: ClassVar[Tuple[str, ...]] = ("disabled", "null", "nidaq")

    channels: Tuple[LaserChannelConfiguration, ...] = tuple()
    hardware_timed: bool = False
    sample_rate_hz: Optional[float] = None
    backend: str = "disabled"
    pmt_shutter_output: Optional[str] = None
    trigger_listener_inputs: Tuple[str, ...] = tuple()
    #: Bare backplane line (PXI_TrigN, no board) the shared sample clock is
    #: driven onto when a laser's output sits on a different board from the
    #: clock producer, and the calibration ramp's clock too. Never a line a
    #: trigger takes (clock_line_clashes): two drivers on one line corrupt
    #: both, and DAQmx does not see it across christielab10's boards.
    backplane_clock_line: str = "PXI_Trig1"
    #: Bare backplane line every pulse train drives its analog output's own
    #: sample clock onto, for a clocked digital line (PMT shutter, trigger or
    #: timing trigger output) on another board. That line cannot wait for the
    #: pulse's start trigger: an M Series board's clocked digital output takes
    #: none (christielab10: do_trig_usage empty, -200452 at verify), so it runs
    #: on the clock that ticks only once the output has triggered.
    #: backplane_clock_line carries the shared clock in a synchronized pulse,
    #: so this is a line of its own, and a free one (clock_line_clashes).
    pulse_clock_line: str = "PXI_Trig3"

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "trigger_listener_inputs", tuple(self.trigger_listener_inputs))
        for attribute, field in (("backplane_clock_line", "backplaneClockLine"),
                                 ("pulse_clock_line", "pulseClockLine")):
            object.__setattr__(self, attribute, _bare_backplane_line(
                field, getattr(self, attribute)))
        channel_ids = tuple(channel.channel_id for channel in self.channels)
        if len(channel_ids) > 4:
            raise ValueError("at most four laser channels are supported")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("laser channel IDs must be unique")
        if self.backend not in self.VALID_BACKENDS:
            raise ValueError(f"laser backend must be one of: {', '.join(self.VALID_BACKENDS)}")
        if self.backend != "disabled" and not channel_ids:
            raise ValueError(f"laser backend '{self.backend}' requires at least one configured channel")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if self.backend != "disabled" and self.hardware_timed and self.sample_rate_hz is None:
            raise ValueError("hardware_timed laser output requires sample_rate_hz")
        if any(not value for value in self.trigger_listener_inputs):
            raise ValueError("trigger_listener_inputs cannot contain empty channel names")
        clashes = clock_line_clashes(
            self.backplane_clock_line, self.pulse_clock_line, self.channels,
            self.trigger_listener_inputs)
        if clashes:
            raise ValueError("; ".join(clashes))

    @classmethod
    def from_channels(
        cls,
        channels: Iterable[LaserChannelConfiguration],
        *,
        hardware_timed: bool = False,
        sample_rate_hz: Optional[float] = None,
        backend: str = "disabled",
        pmt_shutter_output: Optional[str] = None,
        trigger_listener_inputs: Iterable[str] = tuple(),
        backplane_clock_line=_FIELD_DEFAULT,
        pulse_clock_line=_FIELD_DEFAULT,
    ) -> "LaserSystemConfiguration":
        # The clock lines only when given: their defaults are the fields'.
        # None is given, and refused as the constructor refuses it.
        clock_lines = {
            name: value
            for name, value in (("backplane_clock_line", backplane_clock_line),
                                ("pulse_clock_line", pulse_clock_line))
            if value is not _FIELD_DEFAULT
        }
        return cls(
            tuple(channels),
            hardware_timed=hardware_timed,
            sample_rate_hz=sample_rate_hz,
            backend=backend,
            pmt_shutter_output=pmt_shutter_output,
            trigger_listener_inputs=tuple(trigger_listener_inputs),
            **clock_lines,
        )

    def get_channel(self, channel_id: Union[LaserChannelId, int]) -> LaserChannelConfiguration:
        normalized = normalize_laser_channel_id(channel_id)
        for channel in self.channels:
            if channel.channel_id == normalized:
                return channel
        raise KeyError(f"laser channel {normalized.value} is not configured")


def laser_channel_id_representer(dumper: yaml.SafeDumper, obj: LaserChannelId):
    return dumper.represent_data(int(obj))


SystemConfigurationDumper.add_representer(LaserChannelId, laser_channel_id_representer)

for _tag, _cls in (
    ("LaserChannelConfiguration", LaserChannelConfiguration),
    ("LaserSystemConfiguration", LaserSystemConfiguration),
):
    SystemConfigurationDumper.add_representer(_cls, make_camelize_representer(f"!{_tag}"))
    SystemConfigurationLoader.add_constructor(f"!{_tag}", make_decamelize_constructor(_cls))
