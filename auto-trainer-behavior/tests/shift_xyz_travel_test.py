"""The intersession shift is applied to every axis or to none: one shifted axis outside the
configured travel leaves the whole send position unshifted."""
import logging
import math
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.behavior import SystemMachine, PelletDeviceProtocol
from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig


def _machine(refused):
    device = mock.create_autospec(PelletDeviceProtocol)
    device.refuse_shift_outside_travel.return_value = refused
    algo = SimpleNamespace(intersession_pellet_shift_enabled=True, diamond_triangle_config=None)
    return SimpleNamespace(_algorithm=algo, _pellet_device=device), device


def test_a_shift_with_an_axis_outside_the_travel_shifts_no_axis(caplog):
    machine, device = _machine(refused=True)
    with caplog.at_level(logging.ERROR):
        refusal = SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0))
    assert "configured travel" in refusal
    flips = DiamondTriangleOffsetConfig.flips_motor_diamond
    (delta,), _ = device.refuse_shift_outside_travel.call_args
    assert tuple(delta) == (1.0 * flips[0], 6.0 * flips[1], 0.0)
    device.set_x.assert_not_called()
    device.set_y.assert_not_called()
    device.set_z.assert_not_called()
    assert "Not applying the pellet send_position shift" in caplog.text


def test_a_shift_inside_the_travel_is_applied_as_before():
    machine, device = _machine(refused=False)
    assert SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0)) is None
    flips = DiamondTriangleOffsetConfig.flips_motor_diamond
    device.set_x.assert_called_once_with(1.0 * flips[0], absolute=False, sender="processed_shift_xyz")
    device.set_y.assert_called_once_with(6.0 * flips[1], absolute=False, sender="processed_shift_xyz")
    device.set_z.assert_not_called()


# A refused shift is not recorded as applied: not in last_processed_shift_xyz (the pellet-cycle
# records' applied_shift), not as an applied intertrialPelletShift event, and on the
# tongue-eaten path its y-limit is not committed to the animal.

from autotrainer.behavior.pellet_shift import ShiftXYZHandler
from autotrainer.core.configuration.behavior_configuration import ShiftXYZHandlerConfig
from autotrainer.core.reach_event import ReachEvent
from autotrainer.inference.analysis import IntersessionResponse


def _handler(refusals):
    """A ShiftXYZHandler whose processed-shift handler answers `refusals` in turn."""
    algo = SimpleNamespace(active_config=SimpleNamespace(shift_xyz_handler=ShiftXYZHandlerConfig()),
                           pellet_shift_y_limit=None)
    handler = ShiftXYZHandler(algo=algo)
    applied = []

    def processed(project, shift):
        refusal = refusals.pop(0)
        if refusal is None:
            applied.append(shift)
        return refusal

    handler.set_processed_handler(processed)
    return handler, algo, applied


def _tongue_eaten():
    eaten = ReachEvent(init=0, end=60, max=65, method="tongue", outcome="eaten", delay_since_presented=0)
    return IntersessionResponse(reach_events=[eaten], rh_max_vp_list=[], other_events=[eaten])


def _project(send_y):
    return SimpleNamespace(dcs_send_position=Offset3DTuple(0.0, send_y, 0.0))


def test_a_refused_tongue_eaten_shift_is_not_recorded_and_keeps_the_y_limit():
    handler, algo, applied = _handler(["a shifted axis would leave the configured travel"])
    with mock.patch("autotrainer.behavior.pellet_shift.post_api_event_content") as post:
        handler.put_intersession_response(_project(34.8), _tongue_eaten())
    assert applied == []
    assert all(map(math.isnan, handler.last_processed_shift_xyz))  # the record shows no shift
    assert algo.pellet_shift_y_limit is None  # the animal's y-limit is not changed
    (_kind,), kwargs = post.call_args
    data = kwargs["data"]
    assert data["applied"] is False
    assert "configured travel" in data["refusal"]


def test_after_a_refused_shift_a_later_shift_inside_the_travel_is_applied():
    handler, algo, applied = _handler(["a shifted axis would leave the configured travel", None])
    with mock.patch("autotrainer.behavior.pellet_shift.post_api_event_content") as post:
        handler.put_intersession_response(_project(34.8), _tongue_eaten())
        handler.put_intersession_response(_project(20.0), _tongue_eaten())
    tongue_shift = ShiftXYZHandlerConfig().tongue_eaten_shift
    assert applied == [tongue_shift]
    assert handler.last_processed_shift_xyz == tongue_shift
    # The limit comes from the applied shift's send position, not from the refused one.
    assert algo.pellet_shift_y_limit == 20.0 + tongue_shift.y
    data = post.call_args.kwargs["data"]
    assert data["applied"] is True and data["refusal"] is None


def test_a_raising_shift_handler_posts_not_applied_and_leaves_no_pending_y_limit():
    """N-13: a relative SET on an unknown base raises (no travel needed). The shift is posted as
    not applied, with the error, and its tongue-eaten y-limit is dropped, not committed by
    the next shift."""
    handler, algo, applied = _handler([None])
    calls = {"n": 0}
    inner = handler._processed_shift_handler

    def processed(project, shift):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("Cannot SET axis with relative value")
        return inner(project, shift)

    handler.set_processed_handler(processed)
    with mock.patch("autotrainer.behavior.pellet_shift.post_api_event_content") as post:
        with pytest.raises(ValueError):
            handler.put_intersession_response(_project(34.8), _tongue_eaten())
        data = post.call_args.kwargs["data"]
        assert data["applied"] is False and "Cannot SET axis" in data["refusal"]
        assert algo.pellet_shift_y_limit is None
        assert all(map(math.isnan, handler.last_processed_shift_xyz))

        handler.put_intersession_response(_project(20.0), _tongue_eaten())
    tongue_shift = ShiftXYZHandlerConfig().tongue_eaten_shift
    assert algo.pellet_shift_y_limit == 20.0 + tongue_shift.y  # not the failed shift's 35.3


# N-14 (Ben, 2026-11-23): applied is true only when every shifted axis was sent to the board.

def test_a_disabled_shift_or_no_device_is_not_applied():
    machine, device = _machine(refused=False)
    machine._algorithm.intersession_pellet_shift_enabled = False
    assert "disabled" in SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0))
    device.set_x.assert_not_called()
    device.set_y.assert_not_called()
    machine._algorithm.intersession_pellet_shift_enabled = True
    machine._pellet_device = None
    assert "no pellet device" in SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0))


def test_a_shift_with_a_set_not_queued_is_not_applied():
    machine, device = _machine(refused=False)
    device.set_y.return_value = None
    refusal = SystemMachine._apply_processed_shift_xyz(machine, Offset3DTuple(1.0, 6.0, 0.0))
    assert "shift-Y was not queued" in refusal
    device.set_x.assert_called_once()  # the other axes are still sent, as before


def test_a_disabled_tongue_eaten_shift_records_nothing_and_keeps_the_y_limit():
    machine, device = _machine(refused=False)
    machine._algorithm.intersession_pellet_shift_enabled = False
    algo = SimpleNamespace(active_config=SimpleNamespace(shift_xyz_handler=ShiftXYZHandlerConfig()),
                           pellet_shift_y_limit=None)
    handler = ShiftXYZHandler(algo=algo)
    handler.set_processed_handler(
        lambda project, shift: SystemMachine._apply_processed_shift_xyz(machine, shift))
    with mock.patch("autotrainer.behavior.pellet_shift.post_api_event_content") as post:
        handler.put_intersession_response(_project(20.0), _tongue_eaten())
    assert algo.pellet_shift_y_limit is None
    assert all(map(math.isnan, handler.last_processed_shift_xyz))
    data = post.call_args.kwargs["data"]
    assert data["applied"] is False and "disabled" in data["refusal"]
