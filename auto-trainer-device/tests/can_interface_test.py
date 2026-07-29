"""Assumes/requires real hardware is available and actively sending messages of some sort."""
# This test should NOT control the motors. Testing of the motors should
# be done manually.

import time
import pytest

pytestmark = pytest.mark.canbus

from autotrainer.device import (CanInterface, Target, Motor, Heartbeat, ServoConfig, StepperConfig,
                                DigitalOutputs, PelletDigitalInputs, Tone,
                                AnalogOutputs, AnalogOutput,
                                ColorLed, AudioData, DoorData, ServoStatus, StepperStatus,
                                SensorStatus, target_of_motor, is_servo)


@pytest.fixture(scope="module")
def interface():
    print(f"DEBUG: Loading Interface")
    interface = CanInterface()

    if not interface.open():
        pytest.fail("Failed to open CAN interface")
    else:
        print(f"DEBUG: Interface Opened")

    time.sleep(1)

    tries = 0
    while not interface.are_addresses_valid() and tries < 100:
        print(f"DEBUG: Reading Incoming Messages #{tries + 1}")
        interface.read(20)
        tries += 1

    if not interface.are_addresses_valid():
        pytest.fail("CAN addresses never became valid after 100 tries")
    else:
        print(f"DEBUG: Interface Addresses are Valid")

    yield interface

    interface.close()


def test_simple_fixture(interface):
    print(f"DEBUG: Got interface: {interface}")
    assert interface is not None


@pytest.mark.parametrize("target", [
    Target.PELLET_DEVICE,
    Target.MAGNET_DEVICE
])
def test_heartbeat(interface: CanInterface, target: Target):
    print(f"DEBUG: test_heartbeat called with interface={interface}")
    """Verify ping"""
    assert interface.send_heartbeat(), f"Failed to send heartbeat to {target}"

    _get_response(interface, Heartbeat, target)


def _read_config(interface: CanInterface, motor: Motor):
    target = target_of_motor(motor)

    # Verify servo configuration can be read
    assert interface.request_motor_config(motor), (f"Failed to send motor cfg request for "
                                                   f"{motor}")

    config = _get_response(interface,
                           ServoConfig if is_servo(motor) else StepperConfig,
                           target)

    assert config.motor == motor, (f"Received motor config was not for {motor} but for "
                                   f"{config.motor}")

    return config


@pytest.mark.parametrize("motor", [
    Motor.TUNNEL_MAGNET_SERVO,
    Motor.TUNNEL_GATE_SERVO,
    Motor.PELLET_X_MOTOR,
    Motor.PELLET_Y_MOTOR,
    Motor.PELLET_Z_MOTOR,
    Motor.PELLET_LOAD_SERVO,
    Motor.PELLET_COVER_SERVO,
])
def test_read_config(interface: CanInterface, motor: Motor):
    _read_config(interface, motor)


@pytest.mark.parametrize("motor", [
    Motor.TUNNEL_MAGNET_SERVO,
    Motor.TUNNEL_GATE_SERVO,
    Motor.PELLET_LOAD_SERVO,
    Motor.PELLET_COVER_SERVO,
])
def test_write_servo_config(interface: CanInterface, motor: Motor):
    config = _read_config(interface, motor)

    orig_min = config.minimum_position
    orig_max = config.maximum_position

    config.maximum_position -= 10
    config.minimum_position += 10
    assert interface.set_motor_configuration(motor, config), (f"Failed to send motor set cfg for "
                                                              f"{motor}")

    new_config = _read_config(interface, config.motor)
    assert new_config.minimum_position == config.minimum_position, (f"Min position updated failed "
                                                                    f"for {motor}")
    assert new_config.maximum_position == config.maximum_position, (f"Max position updated failed "
                                                                    f"for {motor}")

    config.minimum_position = orig_min
    config.maximum_position = orig_max

    assert interface.set_motor_configuration(motor, config), (f"Failed to set configuration back "
                                                              f"to original for {motor}")


def _write_stepper_config(interface: CanInterface, motor: Motor, config: StepperConfig) -> bool:
    if not interface.set_motor_configuration(motor, config):
        return False

    new_config = _read_config(interface, motor)

    return new_config.microsteps == config.microsteps and \
        new_config.steps_per_revolution == config.steps_per_revolution


@pytest.mark.parametrize("motor", [
    Motor.PELLET_X_MOTOR,
    Motor.PELLET_Y_MOTOR,
    Motor.PELLET_Z_MOTOR,
])
def test_write_stepper_config(interface: CanInterface, motor: Motor):
    config = _read_config(interface, motor)

    orig_min = config.microsteps
    orig_steps = config.steps_per_revolution

    config.microsteps *= 2
    config.steps_per_revolution *= 2
    assert _write_stepper_config(interface, motor, config), (f"Failed to set config for "
                                                             f"{motor}")

    config.microsteps = orig_min
    config.steps_per_revolution = orig_steps
    assert _write_stepper_config(interface, motor, config), (f"Failed to restore config for "
                                                             f"{motor}")


@pytest.mark.parametrize("stim, state", [
    (DigitalOutputs.STIMULUS_1, True),
    (DigitalOutputs.STIMULUS_1, False),
    (DigitalOutputs.STIMULUS_2, True),
    (DigitalOutputs.STIMULUS_2, False),
    (DigitalOutputs.STIMULUS_3, True),
    (DigitalOutputs.STIMULUS_3, False),
    (DigitalOutputs.STIMULUS_4, True),
    (DigitalOutputs.STIMULUS_4, False),
])
def test_write_gpio(interface: CanInterface, stim, state: bool):
    assert interface.set_digital_output(stim, state), (f"Failed to send GPIO "
                                                       f"command")

    data = _get_response(interface, PelletDigitalInputs, Target.PELLET_DEVICE)

    if stim == DigitalOutputs.STIMULUS_1:
        assert data.stimulus_1 == state, f"Failed to set GPIO #1 output"
    elif stim == DigitalOutputs.STIMULUS_2:
        assert data.stimulus_2 == state, f"Failed to set GPIO #2 output"
    elif stim == DigitalOutputs.STIMULUS_3:
        assert data.stimulus_3 == state, f"Failed to set GPIO #3 output"
    elif stim == DigitalOutputs.STIMULUS_4:
        assert data.stimulus_4 == state, f"Failed to set GPIO #4 output"


@pytest.mark.parametrize("frequency_hz, duration_ms", [
    (2000, 400),
    (5000, 600),
])
def test_tone(interface: CanInterface, frequency_hz: int, duration_ms: int):
    assert interface.emit_tone(frequency_hz, duration_ms), f"Failed to send Emit Tone message"

    tone = _get_response(interface, Tone, Target.PELLET_DEVICE, sleep=0.2)
    assert tone.time_remaining_ms <= duration_ms, f"Tone generation not functional"


@pytest.mark.parametrize("value_mv", [
    1000,
    3000,
])
def test_analog_out(interface: CanInterface, value_mv: int):
    assert interface.set_analog_output(AnalogOutputs.STATUS_OUT, value_mv), (f"Failed to set "
                                                                             f"analog output")

    aout = _get_response(interface, AnalogOutput, Target.PELLET_DEVICE, sleep=1.0)
    assert aout.status_out_mv == value_mv, f"Failed to set analog output"


@pytest.mark.parametrize("red, green, blue", [
    (25, 25, 25),
    (25, 50, 75),
])
def test_color_led(interface: CanInterface, red: int, green: int, blue: int):
    assert interface.set_color_led(red, green, blue)

    led = _get_response(interface, ColorLed, Target.PELLET_DEVICE, sleep=1.0)
    assert led.red == red and led.green == green and led.blue == blue, (f"Failed to set colors on "
                                                                        f"color LED")


@pytest.mark.parametrize("type_of, target", [
    (AudioData, Target.MAGNET_DEVICE),
    (DoorData, Target.PELLET_DEVICE),
    (ServoStatus, Target.PELLET_DEVICE),
    (ServoStatus, Target.MAGNET_DEVICE),
    (StepperStatus, Target.PELLET_DEVICE),
    (SensorStatus, Target.MAGNET_DEVICE),
])
def test_streaming_data(interface: CanInterface, type_of, target):
    _get_response(interface, type_of, target)


def _get_response(interface: CanInterface, typeof, target: Target, timeout: float = 2.0,
                  sleep: float = 0.0):
    if sleep != 0:
        now = time.time()

        while time.time() - now < sleep:
            interface.read(1)

    now = time.time()

    while time.time() - now < timeout:
        messages = interface.read(1)
        if len(messages) > 0:
            for msg in messages:
                if isinstance(msg, typeof) and msg.target == target:
                    return msg

    assert False, f"Failed to get desired response of {typeof} from {target}"
