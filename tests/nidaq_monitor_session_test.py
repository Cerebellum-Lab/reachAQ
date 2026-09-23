"""The monitor's model half: what it streams, what it polls, what it drives.

Two of these pin behaviour that was measured on the rig rather than designed,
and both cost a working monitor until they were found.
"""

from types import SimpleNamespace

import pytest

from tools.acquisition.model import nidaq_monitor_session as module
from tools.acquisition.model.nidaq_monitor_session import NidaqMonitorSession


class _Stream:
    """Enough of NidaqSignalMonitorModel to watch what the session asks it."""

    IS_RUNNING = "is_running"
    ERROR_MESSAGE = "error_message"

    def __init__(self, starts=True):
        self.is_running = False
        self.error_message = ""
        self.sample_ring = object()
        self.loaded = None
        self.timing = None
        self.hardware_enabled = None
        self._starts = starts
        self.property_changed = _Event()

    def load_configuration(self, configuration):
        self.loaded = configuration

    def configure_timing(self, configuration, **kwargs):
        self.timing = (configuration, kwargs)

    def set_hardware_enabled(self, enabled, **_kwargs):
        self.hardware_enabled = enabled

    def start(self):
        self.is_running = self._starts
        return self._starts

    def stop(self):
        self.is_running = False

    def close(self):
        pass


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        if handler in self.handlers:
            self.handlers.remove(handler)
        return self

    def __call__(self, *args):
        for handler in list(self.handlers):
            handler(*args)


def _device(name="PXI1Slot5", analog=("ai0", "ai1"), digital_rate=1_000_000.0):
    digital = tuple(f"{name}/{port}/line{n}"
                    for port in ("port0", "port1") for n in range(2))
    return SimpleNamespace(
        name=name, analog_inputs=tuple(f"{name}/{c}" for c in analog),
        analog_outputs=(), digital_inputs=digital, digital_outputs=(),
        counter_inputs=(), counter_outputs=(), terminals=(),
        digital_input_max_rate=digital_rate)


def _configuration():
    return SimpleNamespace(
        nidaq_ports=SimpleNamespace(
            device_identities=(SimpleNamespace(
                logical_name="PXI1Slot5", runtime_name="PXI1Slot5",
                breakout="BNC-2090A"),),
            timing=SimpleNamespace(sync_mode="auto"),
            tone1=None, tone2=None, tone3_r=None, tone3_l=None,
            cam_frames=None, barcode=None),
        nidaq_stream=SimpleNamespace(channels=(), analog_terminal_config="rse"),
        laser=SimpleNamespace(channels=(SimpleNamespace(
            channel_id=SimpleNamespace(value=1), analog_output="PXI1Slot4/ao0",
            diode_input=None, shutter_output=None, trigger_route_source=None,
            minimum_command_volts=0.2),)),
    )


class _App:
    def __init__(self, configuration=None, hardware=None, laser=None):
        self.loaded_configuration = configuration or _configuration()
        self.hardware = hardware
        self.laser = laser


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(module, "NidaqSignalMonitorModel", _Stream)
    monkeypatch.setattr(module, "discover_nidaq_devices",
                        lambda: ((_device(),), None))
    return NidaqMonitorSession(_App())


def test_a_refresh_surveys_the_cards_and_locates_every_line(session):
    assert session.refresh()

    assert session.survey.devices == ("PXI1Slot5",)
    by_terminal = {line.terminal: line for line in session.survey.lines}
    assert by_terminal["ai0"].label == 'BNC-2090A "AI 0" (BNC)'
    assert "4 lines across 1 card(s)" not in session.status_message
    assert "streamed" in session.status_message


def test_discovery_failing_is_reported_rather_than_raised(monkeypatch):
    monkeypatch.setattr(module, "NidaqSignalMonitorModel", _Stream)
    monkeypatch.setattr(module, "discover_nidaq_devices",
                        lambda: ((), "the driver is not installed"))
    session = NidaqMonitorSession(_App())

    assert not session.refresh()
    assert "the driver is not installed" in session.error_message
    assert session.survey.lines == ()


def test_starting_streams_the_clocked_half_at_the_configured_referencing(session):
    session.refresh()

    assert session.start(sample_rate_hz=1234.0)

    loaded = session.stream.loaded
    assert {c.physical_channel for c in loaded.channels} == {
        "PXI1Slot5/ai0", "PXI1Slot5/ai1",
        "PXI1Slot5/port0/line0", "PXI1Slot5/port0/line1"}
    assert loaded.sample_rate_hz == 1234.0
    # Letting DAQmx choose would read half a 6221's range against pins it is
    # also reading directly, so the configured mode is carried over.
    assert loaded.analog_terminal_config == "rse"


def test_the_stream_is_input_only_and_declares_no_output_devices(session):
    """Naming the laser outputs would pull an output board into the graph."""
    session.refresh()
    session.start()

    _configuration_argument, kwargs = session.stream.timing
    assert "hardware_timed_output_devices" not in kwargs
    assert "hardware_timed_output_channels" not in kwargs
    assert kwargs["device_identities"]


def test_static_tasks_are_closed_before_the_stream_process_is_spawned(session):
    """Measured: with DI tasks open here, spawning the worker fails at EFAULT.

    The failure names the Python executable and says "Bad address", which
    points nowhere near the cause, so this is pinned rather than left to be
    rediscovered.
    """
    session.refresh()
    closed = []
    session._static_tasks = {("PXI1Slot5", "port1"): SimpleNamespace(
        close=lambda: closed.append("port1"))}

    session.start()

    assert closed == ["port1"]
    assert session._static_tasks == {}


def test_the_stream_reaching_running_is_forwarded(session):
    """It happens on the model's reader thread, after start() has returned."""
    session.refresh()
    seen = []
    session.property_changed += (
        lambda name, new, old: seen.append((name, new)))

    session.stream.property_changed(_Stream.IS_RUNNING, True, False)

    assert (NidaqMonitorSession.IS_RUNNING, True) in seen


def test_a_laser_hold_returns_the_channel_to_its_configured_rest(session):
    """Its own minimum, not zero: zero would leave a channel driven."""
    commands = []
    session._app_model.laser = SimpleNamespace(
        set_command_voltage=lambda channel, volts: commands.append(volts),
        set_shutter_open=lambda channel, state: None)

    assert session.hold_laser_command(1, 1.5, 0.05)

    assert commands == [1.5, 0.2]


def test_a_laser_hold_still_returns_to_rest_when_the_hold_fails(session):
    commands = []

    def set_voltage(channel, volts):
        commands.append(volts)
        if len(commands) == 1:
            raise RuntimeError("the output is not armed")

    session._app_model.laser = SimpleNamespace(
        set_command_voltage=set_voltage,
        set_shutter_open=lambda channel, state: None)

    assert not session.hold_laser_command(1, 1.5, 0.05)
    assert commands == [1.5, 0.2]
    assert "not armed" in session.error_message


def test_a_board_line_is_pulsed_through_the_existing_command_path(session):
    pulses = []

    def pulse_stim(duration_us, line):
        pulses.append((duration_us, line))
        return "token"  # what _send_with_token returns when it is queued

    session._app_model.hardware = SimpleNamespace(pulse_stim=pulse_stim)

    assert session.pulse_board_line(3, 0.25)

    assert pulses == [(250_000, 3)]


def test_a_board_line_the_device_does_not_queue_is_not_reported_as_driven():
    """pulse_stim returns None when nothing was sent, which is not success."""
    session = NidaqMonitorSession(_App(hardware=SimpleNamespace(
        pulse_stim=lambda duration_us, line: None)))

    assert not session.pulse_board_line(3)
    assert "not queued" in session.error_message


def test_driving_something_that_is_not_connected_says_so(session):
    session._app_model.hardware = None

    assert not session.pulse_board_line(3)
    assert "pellet device" in session.error_message
    assert not session.play_tone(8000, 0.3)


def test_the_wiring_test_command_names_the_record_it_will_write(session):
    command = session.wiring_test_command(open_shutters=True,
                                          command_volts=2.0)

    assert "--open-shutters" in command
    assert "--command-volts" in command and "2" in command
    assert any(part.endswith("verify_nidaq_wiring.py") for part in command)


def test_the_wiring_test_stops_the_stream_before_it_runs(session, monkeypatch):
    """Two clients reserving the same port is -89137 with no explanation."""
    session.refresh()
    session.start()
    assert session.is_running
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(
                            returncode=0, stdout="done", stderr=""))

    ok, output = session.run_wiring_test()

    assert ok and output == "done"
    assert not session.is_running


def test_a_wiring_test_that_never_finishes_is_reported(session, monkeypatch):
    def timeout(*_args, **_kwargs):
        raise module.subprocess.TimeoutExpired("verify", 1)

    monkeypatch.setattr(module.subprocess, "run", timeout)

    ok, message = session.run_wiring_test(timeout=1.0)

    assert not ok and "did not finish" in message
