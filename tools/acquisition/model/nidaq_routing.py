"""Whether timing can cross from one NI board to another, and by what means.

A configuration that puts a laser's output on one board and its clock on
another is describing a route, and until now nothing checked that the route
exists. It failed inside a task instead, at -89125, "no registered trigger
lines could be found between the devices" - a message that names neither the
cause nor the remedy.

What decides it is the pair of devices, not the signal: their bus type, and
for PXI whether the chassis has been identified. This module answers that
from what discovery already reports, so a configuration can be refused at
load with the reason rather than at run with an error code.

One distinction here was measured rather than assumed. DAQmx will not
*reserve* a backplane trigger line without an identified chassis, but a line
driven by name works regardless: on christielab10, with the chassis reading
4294967295, `connect_terms("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")` and
arming the far board on its own `/PXI1Slot4/PXI_Trig0` carried the trigger and
fired the waveform. So an unidentified chassis does not make cross-device
timing impossible; it makes the automatic form impossible and leaves the
explicit form, which is what the laser controller now uses.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional, Tuple

#: What PXI Platform Services reports for a chassis or slot it has not
#: identified. It is 0xFFFFFFFF rather than a sentinel of NI's choosing, so
#: comparing against it is the only way to tell.
UNIDENTIFIED = 0xFFFFFFFF

SAME_DEVICE = "same_device"
PXI_BACKPLANE = "pxi_backplane"
PXI_BACKPLANE_EXPLICIT = "pxi_backplane_explicit"
RTSI = "rtsi"
IMPOSSIBLE = "impossible"


def chassis_is_identified(device) -> bool:
    """Whether PXI Platform Services has placed this board in a chassis.

    Both the chassis and the slot read 0xFFFFFFFF when it has not, and that
    is also when DAQmx declines to allocate a backplane trigger line of its
    own accord.
    """
    chassis = getattr(device, "pxi_chassis_number", None)
    return chassis is not None and chassis != UNIDENTIFIED


def _bus(device) -> str:
    return str(getattr(device, "bus_type", "") or "").upper()


def _is_pxi(device) -> bool:
    return "PXI" in _bus(device)


@dataclasses.dataclass(frozen=True)
class RoutingCapability:
    """How a timing signal can get from one board to another, if at all."""

    source: str
    destination: str
    means: str
    is_possible: bool
    reason: str
    #: What has to be arranged for this route to work, when it is possible
    #: but not automatic.
    requires: str = ""

    @property
    def is_automatic(self) -> bool:
        """Whether DAQmx will route it without being told how."""
        return self.is_possible and self.means in (SAME_DEVICE, PXI_BACKPLANE)


def capability_between(source, destination) -> RoutingCapability:
    """How a signal can cross from `source` to `destination`."""
    source_name = getattr(source, "name", str(source))
    destination_name = getattr(destination, "name", str(destination))

    if source_name == destination_name:
        return RoutingCapability(
            source_name, destination_name, SAME_DEVICE, True,
            "the same board, so nothing has to cross")

    if _is_pxi(source) and _is_pxi(destination):
        if chassis_is_identified(source) and chassis_is_identified(destination):
            if (getattr(source, "pxi_chassis_number", None)
                    != getattr(destination, "pxi_chassis_number", None)):
                return RoutingCapability(
                    source_name, destination_name, IMPOSSIBLE, False,
                    "the boards are in different PXI chassis, which share no "
                    "trigger bus")
            return RoutingCapability(
                source_name, destination_name, PXI_BACKPLANE, True,
                "both boards are in one identified PXI chassis")
        return RoutingCapability(
            source_name, destination_name, PXI_BACKPLANE_EXPLICIT, True,
            "the chassis is not identified, so DAQmx will not reserve a "
            "trigger line, but one driven by name still carries the signal",
            requires="drive a PXI_Trig line on the source board and read the "
                     "destination board's own view of it; identifying the "
                     "chassis would restore automatic routing")

    if _bus(source) == _bus(destination) and _bus(source):
        return RoutingCapability(
            source_name, destination_name, RTSI, False,
            f"both boards are on {_bus(source)}, which shares timing only "
            "over an RTSI cable",
            requires="an RTSI cable between the boards, registered with the "
                     "driver")

    return RoutingCapability(
        source_name, destination_name, IMPOSSIBLE, False,
        f"{source_name} is on {_bus(source) or 'an unknown bus'} and "
        f"{destination_name} is on {_bus(destination) or 'an unknown bus'}; "
        "they share no timing bus",
        requires="an RTSI cable between the boards, registered with the "
                 "driver")


def routing_matrix(devices: Iterable) -> Tuple[RoutingCapability, ...]:
    """Every ordered pair, so a plan can be checked without probing again."""
    devices = tuple(devices)
    return tuple(
        capability_between(source, destination)
        for source in devices
        for destination in devices
    )


def capability_for(devices: Iterable, source_name: str,
                   destination_name: str) -> Optional[RoutingCapability]:
    """The capability for one named pair, or None if a board is unknown."""
    by_name = {getattr(device, "name", str(device)): device
               for device in devices}
    source = by_name.get(source_name)
    destination = by_name.get(destination_name)
    if source is None or destination is None:
        return None
    return capability_between(source, destination)
