"""The block somebody actually touches, over the card terms the software uses.

Card terms stay canonical: the configuration keeps `PXI1Slot5/ai3` and nothing
here translates it, because two spellings of the same thing drift apart. What
a breakout model adds is three things the card cannot say:

  * what is physically reachable, and how. "PFI 0: BNC" against "PFI 1: spring
    terminal, position 13" is the difference between a five-minute job and a
    wrong conclusion, and this rig produced the wrong conclusion - that PFI1
    was unavailable because it has no BNC, when it is on the strip;
  * which connectors are dead ends on the card behind them. The APFI 0 BNC is
    printed on a BNC-2090A whatever is plugged into it, and a PXI-6221 has no
    analog trigger circuit, so a perfectly reasonable cable produces a signal
    that can never be read or triggered on. That cost a day;
  * the label printed on the thing being touched, beside the terminal name
    the software uses.

The block's labels are generic across every card it can be attached to. The
card decides what each label means, and whether it means anything, so nothing
here answers on the card's behalf - the statuses below are the block crossed
with what discovery reported.

The blocks themselves are data, in `data/breakouts/*.yaml`, so another block
is a file rather than a patch.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, Iterable, Optional, Tuple

import yaml

#: A connector the card behind the block can use.
REACHABLE = "reachable"
#: Present on the block, and the card cannot use it. `reason` says why.
UNREACHABLE = "unreachable"
#: Reachable, and the configuration already has something on it.
ASSIGNED = "assigned"
#: Grounds, supplies, and the user-defined BNCs, which land on no terminal.
NOT_A_TERMINAL = "not-a-terminal"

_DATA_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "breakouts",
)

_CACHE: Dict[str, "BreakoutModel"] = {}


@dataclasses.dataclass(frozen=True)
class BreakoutConnector:
    """One labelled place on the block, and the card terminal it lands on."""

    #: Exactly what is printed there. Where a block prints a group heading
    #: above bare numerals, `block` carries the heading and this carries the
    #: numeral, so neither is invented to make a nicer string.
    label: str
    connector: str = "bnc"
    #: The card terminal, with no device prefix - `ai3`, `PFI0`,
    #: `port0/line2`. Empty when the connector lands on nothing, which is the
    #: honest answer for a ground or a user-defined BNC.
    terminal: str = ""
    block: str = ""
    position: Optional[int] = None
    role: str = "signal"
    note: str = ""

    @property
    def printed(self) -> str:
        """The label as it reads on the block, heading included."""
        return f"{self.block} {self.label}".strip() if self.block else self.label

    def describe(self) -> str:
        where = {"bnc": "BNC", "spring": "spring terminal",
                 "screw": "screw terminal"}.get(self.connector, self.connector)
        text = f'"{self.printed}" ({where}'
        if self.position:
            text += f", position {self.position}"
        return text + ")"


@dataclasses.dataclass(frozen=True)
class BreakoutModel:
    """A named accessory and every labelled connection on it."""

    name: str
    title: str = ""
    source: str = ""
    connectors: Tuple[BreakoutConnector, ...] = tuple()

    def connectors_for(self, terminal: str) -> Tuple[BreakoutConnector, ...]:
        """Every connector landing on this terminal - often none, sometimes two.

        A BNC-2110 brings one BNC out as both AI 2 and AO 7, so asking for a
        single connector would have to pick one and be wrong for the other
        card.
        """
        wanted = _normalize(terminal)
        if not wanted:
            return tuple()
        return tuple(c for c in self.connectors
                     if c.terminal and _normalize(c.terminal) == wanted)

    def offers(self, terminal: str) -> bool:
        return bool(self.connectors_for(terminal))


@dataclasses.dataclass(frozen=True)
class ConnectorStatus:
    """One connector, crossed with the card that is behind it."""

    connector: BreakoutConnector
    status: str
    reason: str = ""
    #: What the configuration has on it, when anything does.
    assigned_to: str = ""

    @property
    def is_usable(self) -> bool:
        return self.status in (REACHABLE, ASSIGNED)

    def describe(self) -> str:
        text = self.connector.describe()
        if self.connector.terminal:
            text += f" -> {self.connector.terminal}"
        if self.status == ASSIGNED:
            return f"{text}: in use by {self.assigned_to}"
        if self.status == UNREACHABLE:
            return f"{text}: {self.reason}"
        if self.status == NOT_A_TERMINAL:
            return f"{text}: {self.reason or 'lands on no card terminal'}"
        return text


def _normalize(name: str) -> str:
    return str(name or "").strip("/").strip().lower()


def _tail(name: str, device_name: str) -> str:
    """A channel or terminal name with its device prefix removed."""
    text = _normalize(name)
    prefix = _normalize(device_name) + "/"
    if device_name and text.startswith(prefix):
        return text[len(prefix):]
    return text


def available_breakouts() -> Tuple[str, ...]:
    """Every block with a data file, by name."""
    try:
        entries = sorted(os.listdir(_DATA_DIRECTORY))
    except OSError:
        return tuple()
    names = []
    for entry in entries:
        if not entry.endswith((".yaml", ".yml")):
            continue
        model = load_breakout(entry.rsplit(".", 1)[0])
        if model is not None:
            names.append(model.name)
    return tuple(names)


def load_breakout(name: str) -> Optional[BreakoutModel]:
    """The named block, or None when there is no file for it.

    None rather than an exception: a configuration naming a block nobody has
    added a file for should lose the labels, not the acquisition.
    """
    key = _normalize(name)
    if not key:
        return None
    if key in _CACHE:
        return _CACHE[key]
    for extension in (".yaml", ".yml"):
        path = os.path.join(_DATA_DIRECTORY, key + extension)
        if os.path.exists(path):
            break
    else:
        return None
    with open(path, encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    model = BreakoutModel(
        name=str(content.get("name") or key),
        title=str(content.get("title") or ""),
        source=str(content.get("source") or ""),
        connectors=tuple(
            BreakoutConnector(
                label=str(entry.get("label") or ""),
                connector=str(entry.get("connector") or "bnc"),
                terminal=str(entry.get("terminal") or ""),
                block=str(entry.get("block") or ""),
                position=entry.get("position"),
                role=str(entry.get("role") or "signal"),
                note=str(entry.get("note") or ""),
            )
            for entry in (content.get("connectors") or ())
        ),
    )
    _CACHE[key] = model
    # Both spellings, so a lookup by file name and by printed name agree.
    _CACHE[_normalize(model.name)] = model
    return model


def breakout_for_device(port_configuration, device_name: str
                        ) -> Optional[BreakoutModel]:
    """The block attached to this device, per the configuration."""
    for identity in getattr(port_configuration, "device_identities", ()) or ():
        names = {_normalize(getattr(identity, "logical_name", "")),
                 _normalize(getattr(identity, "runtime_name", ""))}
        if _normalize(device_name) in names - {""}:
            return load_breakout(getattr(identity, "breakout", "") or "")
    return None


def _device_terminals(device) -> Tuple[str, ...]:
    """Every terminal and channel the device reports, device prefix removed."""
    name = getattr(device, "name", "")
    found = []
    for attribute in ("analog_inputs", "analog_outputs", "digital_inputs",
                      "digital_outputs", "counter_inputs", "counter_outputs",
                      "terminals"):
        for candidate in getattr(device, attribute, ()) or ():
            found.append(_tail(str(candidate), name))
    return tuple(found)


def connector_status(connector: BreakoutConnector, device,
                     assignments: Optional[Dict[str, str]] = None
                     ) -> ConnectorStatus:
    """What this connector is, on the card that is actually behind it."""
    assignments = assignments or {}
    device_name = getattr(device, "name", "") or "the card"

    if not connector.terminal:
        reason = connector.note or "lands on no card terminal"
        return ConnectorStatus(connector, NOT_A_TERMINAL, reason=reason)
    if connector.role == "reference":
        return ConnectorStatus(connector, NOT_A_TERMINAL,
                               reason="a ground or supply, not a terminal")

    terminal = _normalize(connector.terminal)

    # Asked first, because the driver answers it outright and the connector
    # being present says nothing: the APFI BNC is printed on the block
    # whatever card is behind it, and a board with no analog trigger circuit
    # can never read it.
    if terminal.startswith("apfi") and getattr(
            device, "analog_trigger_supported", None) is False:
        return ConnectorStatus(
            connector, UNREACHABLE,
            reason=f"{device_name} has no analog trigger circuit, so nothing "
                   "wired here can be read or triggered on")

    if terminal not in _device_terminals(device):
        return ConnectorStatus(
            connector, UNREACHABLE,
            reason=f"{device_name} has no {connector.terminal}")

    owner = assignments.get(terminal, "")
    if owner:
        return ConnectorStatus(connector, ASSIGNED, assigned_to=owner)
    return ConnectorStatus(connector, REACHABLE)


def connector_statuses(model: Optional[BreakoutModel], device,
                       assignments: Optional[Dict[str, str]] = None
                       ) -> Tuple[ConnectorStatus, ...]:
    """Every connector on the block, crossed with the card behind it."""
    if model is None or device is None:
        return tuple()
    return tuple(connector_status(connector, device, assignments)
                 for connector in model.connectors)


def assignments_for_device(device_name: str, *, stream=None, laser=None,
                           ports=None) -> Dict[str, str]:
    """Which of this device's terminals the configuration already uses.

    Keyed by the bare terminal, so it can be crossed with a block whose data
    carries no device prefix.
    """
    found: Dict[str, str] = {}

    prefix = _normalize(device_name) + "/"

    def record(value, owner):
        text = _normalize(value)
        # A value naming another device, or naming none at all the way
        # PXI_CLK10 does, is not this block's business.
        if not text.startswith(prefix):
            return
        tail = text[len(prefix):]
        if tail:
            found.setdefault(tail, owner)

    for channel in getattr(stream, "channels", ()) or ():
        record(getattr(channel, "physical_channel", ""),
               getattr(channel, "name", "") or "the signal stream")

    for channel in getattr(laser, "channels", ()) or ():
        number = getattr(getattr(channel, "channel_id", None), "value", "?")
        for attribute, described in (
            ("analog_output", f"laser {number} output"),
            ("trigger_source", f"laser {number} trigger"),
            ("trigger_route_source", f"laser {number} trigger route"),
            ("shutter_output", f"laser {number} shutter"),
            ("diode_input", f"laser {number} diode"),
        ):
            record(getattr(channel, attribute, ""), described)

    for attribute, described in (
        ("tone1", "tone 1"), ("tone2", "tone 2"), ("tone3_r", "tone 3 right"),
        ("tone3_l", "tone 3 left"), ("cam_frames", "camera frames"),
        ("barcode", "barcode"),
    ):
        record(getattr(ports, attribute, ""), described)

    timing = getattr(ports, "timing", None)
    for attribute, described in (
        ("reference_clock_source", "the reference clock"),
        ("start_trigger_source", "the start trigger"),
        ("sample_clock_source", "the sample clock"),
        ("sample_clock_export_terminal", "the exported sample clock"),
        ("start_trigger_export_terminal", "the exported start trigger"),
    ):
        record(getattr(timing, attribute, ""), described)

    return found


def describe_terminal(model: Optional[BreakoutModel], terminal: str) -> str:
    """`ai3` as the label somebody can find on the block, when there is one."""
    if model is None:
        return ""
    connectors = model.connectors_for(terminal)
    if not connectors:
        return ""
    return " / ".join(
        f'{model.name} {connector.describe()}' for connector in connectors)


def unreachable_terminals(model: Optional[BreakoutModel], terminals: Iterable[str]
                          ) -> Tuple[str, ...]:
    """Configured terminals this block does not bring out anywhere.

    Not an error - a 68-pin cable to something else reaches them - but worth
    saying, because the alternative is looking for a label that is not there.
    """
    if model is None:
        return tuple()
    return tuple(terminal for terminal in terminals
                 if terminal and not model.offers(terminal))
