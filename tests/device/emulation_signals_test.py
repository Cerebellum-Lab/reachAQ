import time

import pytest

from reachaq.core.simulation import SimSignalBusReader
from reachaq.device import CanInterface, CanTransportConfiguration
from reachaq.device.device_interface import DigitalOutputs, PelletDigitalInputs
from reachaq.device.emulation_interface import EmulationInterface
from reachaq.device.motor_pellet_backend import MotorPelletBackend


@pytest.fixture
def bus(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(path))
    return SimSignalBusReader(path)


@pytest.fixture
def emulator():
    interface = EmulationInterface()
    interface.open()
    return interface


@pytest.mark.parametrize("frequency, line", [(5000, "tone1"), (6000, "tone2")])
def test_emit_tone_publishes_tone1_window(bus, emulator, frequency, line):
    before = time.perf_counter()
    assert emulator.emit_tone(frequency, 300)
    after = time.perf_counter()

    (event,) = bus.poll()
    assert event.line == line
    assert before + 0.001 <= event.start <= after + 0.001
    assert event.end - event.start == pytest.approx(0.300)
    assert event.value == 1.0


@pytest.mark.parametrize("frequency", [8000, 5001, None])
def test_unknown_frequency_publishes_nothing(bus, emulator, frequency):
    # Still acknowledged: the board plays the tone, it just raises no line.
    assert emulator.emit_tone(frequency, 300)
    assert bus.poll() == ()


def test_a_closed_emulator_publishes_nothing(bus):
    # The validator fails on an NI tone pulse with no PLAY_TONE row, so a
    # tone the emulator did not accept must leave the NI lines quiet.
    interface = EmulationInterface()
    assert not interface.emit_tone(5000, 300)
    assert not interface.pulse_digital_output(DigitalOutputs.STIMULUS_3, 1000)
    assert bus.poll() == ()


@pytest.mark.parametrize("gpio, line", [
    (DigitalOutputs.STIMULUS_3, "board_stim2"),
    (DigitalOutputs.STIMULUS_4, "board_stim3"),
])
def test_stim_pulse_publishes_readback(bus, emulator, gpio, line):
    before = time.perf_counter()
    assert emulator.pulse_digital_output(gpio, 2500)
    after = time.perf_counter()

    (event,) = bus.poll()
    assert event.line == line
    assert before + 0.001 <= event.start <= after + 0.001
    assert event.end - event.start == pytest.approx(0.0025)


def test_refused_stim_pulse_publishes_nothing(bus, emulator):
    with pytest.raises(ValueError):
        emulator.pulse_digital_output(DigitalOutputs.STIMULUS_1, 1000)
    assert bus.poll() == ()


def _status_messages(emulator):
    emulator._last_status_message = -float("inf")
    return emulator.read()


def test_a_simulation_reads_idle_inputs_stamped_on_receipt(emulator, monkeypatch):
    # The host decodes stimulus_1 as the tone1 GPIO: a constant high is a tone
    # the NI never saw. Unstamped, an ack would precede the SEND it answers.
    monkeypatch.setenv("REACHAQ_SIMULATION", "1")
    before = time.perf_counter_ns()
    messages = _status_messages(emulator)
    after = time.perf_counter_ns()

    (inputs,) = [m for m in messages if isinstance(m, PelletDigitalInputs)]
    assert not (inputs.stimulus_1 or inputs.stimulus_2 or inputs.stimulus_3 or inputs.stimulus_4)
    assert all(before <= m.index <= after and m.timestamp_ns > 0 for m in messages)


def test_emulation_outside_a_simulation_is_unchanged(emulator, monkeypatch):
    monkeypatch.delenv("REACHAQ_SIMULATION", raising=False)
    messages = _status_messages(emulator)

    (inputs,) = [m for m in messages if isinstance(m, PelletDigitalInputs)]
    assert (inputs.stimulus_1, inputs.stimulus_2, inputs.stimulus_3, inputs.stimulus_4) == (
        True, False, True, False)
    assert all(m.index == 0 for m in messages)


def test_emulation_interface_is_motor_pellet_backend():
    assert isinstance(EmulationInterface(), MotorPelletBackend)


def test_can_interface_is_motor_pellet_backend():
    # Never opened: the check reads attributes only.
    interface = CanInterface(can_transport=CanTransportConfiguration(kind="socketcan"))
    assert isinstance(interface, MotorPelletBackend)


def test_an_object_missing_a_command_is_not_a_motor_pellet_backend():
    class Partial:
        def open(self):
            return True

    assert not isinstance(Partial(), MotorPelletBackend)
