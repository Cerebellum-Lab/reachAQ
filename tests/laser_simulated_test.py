import time
from types import SimpleNamespace

import pytest

from autotrainer.core.simulation import ENV_SIM_BUS, ENV_SIMULATION, SimSignalBusReader
from autotrainer.device import LaserChannelConfiguration, LaserSystemConfiguration
from autotrainer.device.laser import (
    LaserPulseTrain,
    NullLaserController,
    SimulatedLaserController,
    make_laser_controller,
)
from tools.acquisition.model.laser_firing import LaserFiring
from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute

SOFTWARE = LaserFiring(1, LaserTriggerRoute.DIRECT_NI_SOFTWARE)


def _configuration(backend="nidaq"):
    return LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=1,
            analog_output="SimDAQ1/ao0",
            diode_input="SimDAQ1/ai0",
            command_copy_input="SimDAQ1/ai1",
            shutter_output="SimDAQ1/port0/line0",
        ),
    ), backend=backend, hardware_timed=True, sample_rate_hz=10_000)


def _recipe():
    return SimpleNamespace(
        session_id="session001", session_generation=3,
        protocol_id="training", protocol_revision=2,
        logical_trial_id=4, attempt_id=1, operation_id="trial-op",
    )


@pytest.fixture
def simulation(monkeypatch, tmp_path):
    bus = tmp_path / "bus.jsonl"
    monkeypatch.setenv(ENV_SIMULATION, "1")
    monkeypatch.setenv(ENV_SIM_BUS, str(bus))
    return SimSignalBusReader(bus)


def _by_line(events):
    lines = {}
    for event in events:
        lines.setdefault(event.line, []).append(event)
    return lines


def test_factory_hardware_class_when_not_simulating(monkeypatch):
    monkeypatch.delenv(ENV_SIMULATION, raising=False)
    built = []

    class _Nidaq:
        def __init__(self, configuration, **kwargs):
            built.append(kwargs)

    reader = object()
    controller = make_laser_controller(
        _configuration(), feedback_reader=reader, timing_plan=None,
        analog_terminal_config="rse", hardware_class=_Nidaq)

    assert isinstance(controller, _Nidaq)
    assert built == [{"feedback_reader": reader, "timing_plan": None,
                      "analog_terminal_config": "rse"}]
    assert type(make_laser_controller(_configuration("null"))) is NullLaserController
    with pytest.raises(RuntimeError, match="has no controller to open"):
        make_laser_controller(_configuration("disabled"))


def test_factory_simulated_class_when_simulating(simulation):
    built = []

    class _Nidaq:
        def __init__(self, configuration, **kwargs):
            built.append(kwargs)

    controller = make_laser_controller(
        _configuration(), feedback_reader=object(), timing_plan=None,
        analog_terminal_config="rse", hardware_class=_Nidaq)

    assert isinstance(controller, SimulatedLaserController)
    assert built == []
    # The null backend stays the null controller in simulation too.
    assert type(make_laser_controller(_configuration("null"))) is NullLaserController
    model = LaserModel()
    model.load_configuration(_configuration())
    assert isinstance(model._controller, SimulatedLaserController)
    assert isinstance(model.open_controller(_configuration()), SimulatedLaserController)
    model.close()


def test_simulated_train_publishes_on_windows(simulation):
    model = LaserModel()
    model.load_configuration(_configuration())
    profile = LaserPulseProfile(
        "pulse", 1, 2.5, 5, pulse_count=3, frequency_hz=20, baseline_ms=10)

    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    assert simulation.poll() == ()  # nothing before the start
    before = time.perf_counter()
    operation.trigger()
    assert operation.wait(2).value == "completed"
    model.close()

    lines = _by_line(simulation.poll())
    assert set(lines) == {"laser1_command_copy", "laser1_diode"}
    for events in lines.values():
        assert len(events) == 3
        for index, event in enumerate(events):
            assert event.end - event.start == pytest.approx(0.005)
            assert event.start - events[0].start == pytest.approx(index * 0.05)
            assert event.value == 2.5
        # Published with the start, so no window begins before it.
        assert events[0].start >= before + 0.010


def test_simulated_blocking_train_publishes_on_windows(simulation):
    controller = make_laser_controller(_configuration())
    before = time.perf_counter()
    controller.run_pulse_train(LaserPulseTrain(
        channel_id=1, amplitude_volts=1.5, duration_ms=2, pulse_count=2,
        frequency_hz=100))

    lines = _by_line(simulation.poll())
    assert set(lines) == {"laser1_command_copy", "laser1_diode"}
    for events in lines.values():
        assert [round(e.end - e.start, 6) for e in events] == [0.002, 0.002]
        assert events[1].start - events[0].start == pytest.approx(0.01)
        assert events[0].start >= before


def test_simulated_timing_status_is_simulated(simulation):
    model = LaserModel()
    model.load_configuration(_configuration())
    assert model.timing_status["status"] == "simulated"
    blocks = []
    model.trace_received = blocks.append
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)

    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    operation.trigger()
    assert operation.wait(2).value == "completed"
    model.close()

    status = operation.to_record()["timing_status"]
    assert status["status"] == "emulated_software_start"
    assert status["simulated"] is True
    prepared = [block for block in blocks if block.event == "prepared"]
    assert [block.timing_confidence for block in prepared] == ["simulated"]


def test_cancelled_train_clears_its_unfired_windows(simulation):
    controller = make_laser_controller(_configuration())
    profile = LaserPulseProfile(
        "pulse", 1, 2.5, 5, pulse_count=10, frequency_hz=10)
    model = LaserModel(controller)
    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    operation.trigger()
    time.sleep(0.15)
    operation.cancel()
    operation.wait(2)
    model.close()

    lines = _by_line(simulation.poll())
    for events in lines.values():
        pulses = [event for event in events if event.value == 2.5]
        cleared = [event for event in events if event.value == 0.0]
        assert len(pulses) == 10
        # One 0 V event, from the cancel to the train's end, published last.
        assert len(cleared) == 1 and events[-1] is cleared[0]
        assert pulses[0].start < cleared[0].start < pulses[-1].start
        assert cleared[0].end >= pulses[-1].end - 1e-9


def _record_publish_times(monkeypatch):
    """Wrap the laser's publish_signal so each event's publish time is kept."""
    from autotrainer.device import laser

    published = []
    original = laser.publish_signal

    def publish(line, start, end, value=1.0):
        published.append((time.perf_counter(), start))
        original(line, start, end, value)

    monkeypatch.setattr(laser, "publish_signal", publish)
    return published


def test_windows_start_after_they_are_published(simulation, monkeypatch):
    published = _record_publish_times(monkeypatch)
    controller = make_laser_controller(_configuration())
    controller.run_pulse_train(LaserPulseTrain(
        channel_id=1, amplitude_volts=1.5, duration_ms=2, pulse_count=2,
        frequency_hz=100))
    model = LaserModel(controller)
    operation = model.prepare_pulse_profile(
        LaserPulseProfile("pulse", 1, 2.5, 5, pulse_count=2, frequency_hz=20),
        SOFTWARE, _recipe())
    operation.trigger()
    assert operation.wait(2).value == "completed"
    model.close()

    assert len(published) == 8  # two pulses on two lines, blocking then async
    assert all(start > at for at, start in published)


def test_blocking_train_survives_a_bus_write_error(simulation, monkeypatch, caplog):
    from autotrainer.device import laser

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(laser, "publish_signal", broken)
    controller = make_laser_controller(_configuration())
    train = LaserPulseTrain(channel_id=1, amplitude_volts=1.5, duration_ms=2)
    with caplog.at_level("WARNING", logger="autotrainer.device.laser"):
        controller.run_pulse_train(train)
        controller.run_pulse_train(train)
    assert [r.getMessage() for r in caplog.records].count(
        "simulated laser windows not published on the bus: disk full") == 1


def test_async_train_survives_a_bus_write_error(simulation, monkeypatch, caplog):
    from autotrainer.device import laser

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(laser, "publish_signal", broken)
    controller = make_laser_controller(_configuration())
    model = LaserModel(controller)
    with caplog.at_level("WARNING", logger="autotrainer.device.laser"):
        operation = model.prepare_pulse_profile(
            LaserPulseProfile("pulse", 1, 2.5, 5, pulse_count=2, frequency_hz=20),
            SOFTWARE, _recipe())
        operation.trigger()
        assert operation.wait(2).value == "completed"
        # A cancelled train publishes its clearing window; that one fails too.
        cancelled = model.prepare_pulse_profile(
            LaserPulseProfile("pulse", 1, 2.5, 500, pulse_count=2, frequency_hz=20),
            SOFTWARE, _recipe())
        cancelled.trigger()
        time.sleep(0.05)
        cancelled.cancel()
        assert cancelled.wait(2).value == "cancelled"
    model.close()
    channel = controller._configuration.get_channel(1)
    assert controller._command_volts[channel.channel_id] == channel.minimum_command_volts
    assert [r.getMessage() for r in caplog.records].count(
        "simulated laser windows not published on the bus: disk full") == 1
