"""Whether a timing route between two boards exists, decided before a task does.

The failure this replaces is -89125 raised from inside a running task: "no
registered trigger lines could be found between the devices in the route",
which names neither which configuration line asked for it nor what to do.
"""

from types import SimpleNamespace

from tools.acquisition.model.nidaq_routing import (
    IMPOSSIBLE,
    PXI_BACKPLANE,
    PXI_BACKPLANE_EXPLICIT,
    RTSI,
    SAME_DEVICE,
    UNIDENTIFIED,
    capability_between,
    capability_for,
    chassis_is_identified,
    routing_matrix,
)


def _device(name, bus="BusType.PXI", chassis=1, slot=5):
    return SimpleNamespace(name=name, bus_type=bus,
                           pxi_chassis_number=chassis, pxi_slot_number=slot)


def test_a_board_needs_no_route_to_itself():
    device = _device("PXI1Slot5")

    capability = capability_between(device, device)

    assert capability.means == SAME_DEVICE
    assert capability.is_possible and capability.is_automatic


def test_one_identified_chassis_routes_automatically():
    capability = capability_between(
        _device("PXI1Slot5", chassis=1), _device("PXI1Slot4", chassis=1))

    assert capability.means == PXI_BACKPLANE
    assert capability.is_automatic


def test_an_unidentified_chassis_still_routes_but_not_by_itself():
    """Measured: a PXI_Trig line driven by name carries the signal anyway."""
    capability = capability_between(
        _device("PXI1Slot5", chassis=UNIDENTIFIED, slot=UNIDENTIFIED),
        _device("PXI1Slot4", chassis=UNIDENTIFIED, slot=UNIDENTIFIED))

    assert capability.means == PXI_BACKPLANE_EXPLICIT
    assert capability.is_possible
    # The distinction that matters: possible, but nothing will do it for you.
    assert not capability.is_automatic
    assert "PXI_Trig" in capability.requires


def test_separate_chassis_share_no_trigger_bus():
    capability = capability_between(
        _device("PXI1Slot5", chassis=1), _device("PXI2Slot3", chassis=2))

    assert capability.means == IMPOSSIBLE
    assert not capability.is_possible


def test_two_pcie_boards_need_a_cable_nobody_can_detect():
    capability = capability_between(
        _device("Dev1", bus="BusType.PCIE", chassis=None),
        _device("Dev2", bus="BusType.PCIE", chassis=None))

    assert capability.means == RTSI
    assert not capability.is_possible
    assert "RTSI" in capability.requires


def test_a_pcie_board_and_a_pxi_board_share_nothing_by_default():
    """The mixed topology that motivated this: it must not look supported."""
    capability = capability_between(
        _device("Dev1", bus="BusType.PCIE", chassis=None),
        _device("PXI1Slot4", bus="BusType.PXI", chassis=1))

    assert not capability.is_possible
    assert "share no timing bus" in capability.reason


def test_an_unidentified_chassis_is_reported_as_such():
    assert chassis_is_identified(_device("a", chassis=1))
    assert not chassis_is_identified(_device("a", chassis=UNIDENTIFIED))
    assert not chassis_is_identified(_device("a", chassis=None))


def test_the_matrix_covers_every_ordered_pair():
    devices = [_device("PXI1Slot5"), _device("PXI1Slot4")]

    matrix = routing_matrix(devices)

    assert len(matrix) == 4
    assert {(c.source, c.destination) for c in matrix} == {
        ("PXI1Slot5", "PXI1Slot5"), ("PXI1Slot5", "PXI1Slot4"),
        ("PXI1Slot4", "PXI1Slot5"), ("PXI1Slot4", "PXI1Slot4"),
    }


def test_asking_about_a_board_that_is_not_installed_returns_nothing():
    """A configuration naming an absent device must not get a confident answer."""
    devices = [_device("PXI1Slot5")]

    assert capability_for(devices, "PXI1Slot5", "PXI1Slot9") is None
