"""The finite pulse request, which the firmware will answer once it implements it."""

import struct

import pytest

from autotrainer.device.can_interface import CanInterface
from autotrainer.device.device_interface import DigitalOutputs, Target


class FakeJerryCan:
    def __init__(self):
        self.calls = []

    def GPIOPulse(self, dst_id, instance, gpio_idx, duration_us, uuid):
        self.calls.append((dst_id, instance, gpio_idx, duration_us))
        return 0


def make_interface(jerrycan):
    interface = CanInterface.__new__(CanInterface)
    interface._jc = jerrycan
    interface._tgt2addr = lambda target: 0
    return interface


@pytest.fixture
def interface():
    jerrycan = FakeJerryCan()
    return make_interface(jerrycan), jerrycan


def test_stim3_sends_gpio_index_seven(interface):
    api, jerrycan = interface

    assert api.pulse_digital_output(DigitalOutputs.STIMULUS_4, 1000) is True
    assert jerrycan.calls[0][2] == 7


def test_stim2_sends_gpio_index_six(interface):
    api, jerrycan = interface

    assert api.pulse_digital_output(DigitalOutputs.STIMULUS_3, 1000) is True
    assert jerrycan.calls[0][2] == 6


def test_a_tone_confirmation_line_is_refused(interface):
    api, jerrycan = interface

    with pytest.raises(ValueError, match="tone confirmation"):
        api.pulse_digital_output(DigitalOutputs.STIMULUS_1, 1000)

    assert jerrycan.calls == []


def test_the_second_tone_confirmation_line_is_refused(interface):
    api, jerrycan = interface

    with pytest.raises(ValueError, match="tone confirmation"):
        api.pulse_digital_output(DigitalOutputs.STIMULUS_2, 1000)

    assert jerrycan.calls == []


def test_a_duration_below_the_floor_is_refused(interface):
    api, _jerrycan = interface

    with pytest.raises(ValueError, match="100 us"):
        api.pulse_digital_output(DigitalOutputs.STIMULUS_4, 99)


def test_a_duration_above_the_ceiling_is_refused(interface):
    api, _jerrycan = interface

    with pytest.raises(ValueError, match="100 us"):
        api.pulse_digital_output(DigitalOutputs.STIMULUS_4, 5_000_001)


def test_a_transport_without_the_command_is_refused(interface):
    api, _jerrycan = interface
    api._jc = object()

    with pytest.raises(RuntimeError, match="finite GPIO pulse"):
        api.pulse_digital_output(DigitalOutputs.STIMULUS_4, 1000)


def pulse_status(gpio_idx, phase=2, error=0):
    """A GPIO_PULSE_STATUS frame as the board sends it, decoded and translated."""
    from autotrainer.device.socketcan_jerrycan import JerryCANCmdType, decode_frame

    can_id = (JerryCANCmdType.GPIO_PULSE_STATUS << 5) | 0x01
    payload = struct.pack("<BHIBi", 0, gpio_idx, 1000, phase, error)
    return CanInterface._translate_gpio_pulse_status(
        decode_frame(can_id, payload + bytes([23])))


@pytest.mark.parametrize("gpio_idx, channel", [
    (6, DigitalOutputs.STIMULUS_3),  # board STIM2
    (7, DigitalOutputs.STIMULUS_4),  # board STIM3
])
def test_a_pulse_status_names_the_line_the_board_reports(gpio_idx, channel):
    assert pulse_status(gpio_idx).channel == channel


def test_the_board_refusing_a_tone_line_is_recorded_against_that_line():
    # The qualification tool pulses STIM0 on purpose to see the board refuse it.
    status = pulse_status(4, phase=0, error=-1)

    assert status.channel == DigitalOutputs.STIMULUS_1
    assert status.error == -1


def test_a_pulse_status_for_an_unknown_line_names_none():
    assert pulse_status(2).channel is None
