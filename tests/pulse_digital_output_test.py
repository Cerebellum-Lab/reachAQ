"""The finite pulse request, which the firmware will answer once it implements it."""

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
