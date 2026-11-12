"""Pellet firmware 2.3.1 reports driver state in STEPPER_STATUS.status.

Bit 0 DRIVER_FAULT: this motor's driver raised DIAG, latched until a re-home.
Bit 1 ENABLE_OFF: the shared enable is off, for any reason. A v1.2 board
always sends 0 (release note v2.3.1, "Wire protocol").
"""

import logging
from types import SimpleNamespace
from uuid import uuid4

from autotrainer.core import MessageHandler, SystemCommandKind
from autotrainer.device import (
    CanInterface,
    CanTransportConfiguration,
    Motor,
    StepperStatus,
    Target,
)
from tools.acquisition.model.hardware_model import HardwareModel


def _interface():
    return CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(kind="socketcan", fd=True),
    )


def _wire(status=None, motor_id=1):
    fields = dict(motor_id=motor_id, homing_status=0, limit_switch=0, position=0.0, send_position=0.0)
    if status is not None:
        fields["status"] = status
    return SimpleNamespace(dst_id=0, stepper_status=SimpleNamespace(**fields))


def test_translate_stepper_status_carries_driver_fault_and_enable_off():
    faulted = _interface()._translate_stepper_status(_wire(status=0x03))
    assert faulted.motor is Motor.PELLET_Y_MOTOR
    assert faulted.status == 0x03
    assert faulted.driver_fault is True and faulted.enable_off is True

    other = _interface()._translate_stepper_status(_wire(status=0x02))
    assert other.driver_fault is False and other.enable_off is True

    driver_only = _interface()._translate_stepper_status(_wire(status=0x01))
    assert driver_only.driver_fault is True and driver_only.enable_off is False


def test_a_status_byte_of_zero_reports_neither_bit():
    status = _interface()._translate_stepper_status(_wire(status=0x00))
    assert status.status == 0
    assert status.driver_fault is False and status.enable_off is False


def test_a_stepper_status_without_a_status_field_reads_as_zero():
    # The compiled transport's stepper-status object may not carry the byte.
    status = _interface()._translate_stepper_status(_wire(status=None))
    assert status.status == 0
    assert status.driver_fault is False and status.enable_off is False


_PROPERTY = {"X": MessageHandler.STEPPER_X_PROPERTY, "Y": MessageHandler.STEPPER_Y_PROPERTY,
             "Z": MessageHandler.STEPPER_Z_PROPERTY}
_MOTOR = {"X": Motor.PELLET_X_MOTOR, "Y": Motor.PELLET_Y_MOTOR, "Z": Motor.PELLET_Z_MOTOR}


def _report(hardware_model, axis, status):
    hardware_model._message_handler_property_changed(_PROPERTY[axis], StepperStatus(
        Target.PELLET_DEVICE, _MOTOR[axis], 1.0, 2.0, False, status=status), None)


def _faults_published(hardware_model):
    published = []
    hardware_model.property_changed += (
        lambda name, value, old:
        published.append(value) if name == HardwareModel.PELLET_DRIVER_FAULT_PROPERTY else None)
    return published


def _driver_fault_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "driver fault" in r.getMessage()]


def test_a_driver_fault_bit_names_its_axis_until_the_board_clears_it(hardware_model, caplog):
    published = _faults_published(hardware_model)
    with caplog.at_level(logging.DEBUG, logger="tools.acquisition.model.hardware_model"):
        for axis in "XYZ":
            _report(hardware_model, axis, 0x02)
        _report(hardware_model, "Y", 0x03)  # 0x03 for the faulted driver, 0x02 for the others
        _report(hardware_model, "Y", 0x03)  # repeated while latched
    assert hardware_model.pellet_axes_with_driver_fault == ("Y",)
    assert published == [("Y",)]
    assert len(_driver_fault_warnings(caplog)) == 1, "a repeated, latched status logs once"

    _report(hardware_model, "Y", 0x00)  # a home re-proved the drivers
    assert hardware_model.pellet_axes_with_driver_fault == ()
    assert published == [("Y",), ()]


def test_two_faulted_axes_are_listed_in_order_and_logged_once(hardware_model, caplog):
    published = _faults_published(hardware_model)
    with caplog.at_level(logging.DEBUG, logger="tools.acquisition.model.hardware_model"):
        _report(hardware_model, "Z", 0x03)
        _report(hardware_model, "X", 0x03)  # arrives second, sorts first
        _report(hardware_model, "X", 0x03)
    assert hardware_model.pellet_axes_with_driver_fault == ("X", "Z")
    assert published == [("Z",), ("X", "Z")]
    assert len(_driver_fault_warnings(caplog)) == 1, "the second axis adds no second WARNING"


def test_enable_off_alone_raises_no_fault_and_no_needs_home(hardware_model, caplog):
    """A v1.3 board with no heartbeat reports ENABLE_OFF on every motor after 30 s idle."""
    published = _faults_published(hardware_model)
    with caplog.at_level(logging.DEBUG, logger="tools.acquisition.model.hardware_model"):
        for axis in "XYZ":
            _report(hardware_model, axis, 0x02)
    assert hardware_model.pellet_axes_with_driver_fault == ()
    assert hardware_model.pellet_axes_needing_home == ()
    assert published == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_fault_that_relatches_after_a_home_shows_again(hardware_model):
    """DIAG still high after a re-home latches again two polls after EN returns."""
    hardware_model._set_axes_needing_home(("X", "Y", "Z"))
    _report(hardware_model, "Z", 0x03)
    home = uuid4()
    hardware_model._pending_tokens[home] = (SystemCommandKind.SEND_HOME, 1.0)
    hardware_model._ack_received(home)
    assert hardware_model.pellet_axes_needing_home == (), "the home ACK clears needs-home"
    assert hardware_model.pellet_axes_with_driver_fault == ("Z",), "only the board's status clears a fault"
    _report(hardware_model, "Z", 0x00)
    assert hardware_model.pellet_axes_with_driver_fault == ()

    _report(hardware_model, "Z", 0x03)
    assert hardware_model.pellet_axes_with_driver_fault == ("Z",)
    assert hardware_model.pellet_axes_needing_home == (), "the fault is reported, needs-home is not inferred"
