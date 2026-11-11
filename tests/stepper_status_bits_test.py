"""Pellet firmware 2.3.1 reports driver state in STEPPER_STATUS.status.

Bit 0 DRIVER_FAULT: this motor's driver raised DIAG, latched until a re-home.
Bit 1 ENABLE_OFF: the shared enable is off, for any reason. A v1.2 board
always sends 0 (release note v2.3.1, "Wire protocol").
"""

from types import SimpleNamespace

from autotrainer.device import CanInterface, CanTransportConfiguration, Motor, Target


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


def test_a_status_byte_of_zero_reports_neither_bit():
    status = _interface()._translate_stepper_status(_wire(status=0x00))
    assert status.status == 0
    assert status.driver_fault is False and status.enable_off is False


def test_a_stepper_status_without_a_status_field_reads_as_zero():
    # The compiled transport's stepper-status object may not carry the byte.
    status = _interface()._translate_stepper_status(_wire(status=None))
    assert status.status == 0
    assert status.driver_fault is False and status.enable_off is False
