import threading

import pytest

from autotrainer.device import LaserChannelConfiguration, LaserSystemConfiguration

from tools.acquisition.model.stim_bench_test import StimTestResult
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


class FakeOperation:
    def __init__(self):
        self._callbacks = []
        self.cancelled = False
        self.triggered = False

    def add_terminal_callback(self, callback):
        self._callbacks.append(callback)

    def trigger(self):
        # The software start, after which the armed waveform runs to completion.
        self.triggered = True
        threading.Timer(0.0, self.finish).start()

    def finish(self):
        for callback in tuple(self._callbacks):
            callback(self)

    def cancel(self):
        self.cancelled = True
        return True


class FakeLaser:
    def __init__(self, operation, backend="nidaq", channel_ids=(1,)):
        self.operation = operation
        self.prepared = []
        self.released = []
        self.configuration = LaserSystemConfiguration.from_channels(
            tuple(
                LaserChannelConfiguration(
                    channel_id=item,
                    analog_output=f"Dev1/ao{item - 1}",
                    diode_input=f"Dev1/ai{item - 1}",
                    shutter_output=f"Dev1/port0/line{item - 1}",
                    trigger_source="/Dev1/PFI0",
                    board_stim_line=3,
                )
                for item in channel_ids
            ),
            backend=backend,
        )

    def prepare_pulse_profile(self, profile, firing, recipe):
        self.prepared.append((profile, recipe))
        return self.operation

    def release_prepared_profile(self, operation):
        self.released.append(operation)


class FakeHardware:
    def __init__(self, operation):
        self.operation = operation
        self.pulses = []
        self.firmware_compatibility = {
            "reported_capabilities": ["finite_stim3_pulse"]
        }

    def pulse_stim(self, duration_us, stim_line=3):
        self.pulses.append((duration_us, stim_line))
        # The board acknowledges, and the armed waveform runs to completion.
        threading.Timer(0.0, self.operation.finish).start()
        return "token"

    def wait_pending_command_acked(self, token, timeout=None):
        return True


def make_profile(**overrides):
    values = dict(
        profile_id="stim-a",
        revision=1,
        channel_id=1,
        amplitude_volts=2.0,
        pulse_duration_ms=5.0,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
    )
    values.update(overrides)
    return LaserPulseProfile(**values)


@pytest.fixture
def bench(app_model):
    operation = FakeOperation()
    laser = FakeLaser(operation)
    hardware = FakeHardware(operation)
    app_model._laser = laser
    app_model._hardware = hardware
    app_model._laser_profiles = {"stim-a": make_profile()}
    return app_model, laser, hardware, operation


def test_a_bench_test_arms_the_output_before_pulsing_the_board(bench):
    model, laser, hardware, _operation = bench

    result = model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")

    assert laser.prepared, "the analog output must be armed first"
    assert hardware.pulses == [(1000, 3)]
    assert isinstance(result, StimTestResult)
    assert result.arm_to_terminal_ms is not None


def test_the_bench_recipe_claims_no_session(bench):
    model, laser, _hardware, _operation = bench

    model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")

    _profile, recipe = laser.prepared[0]
    assert recipe.session_generation == 0
    assert recipe.session_id == "bench"


def test_the_prepared_profile_is_always_released(bench):
    model, laser, _hardware, operation = bench

    model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")

    assert laser.released == [operation]


def test_no_profile_is_refused(bench):
    model, _laser, hardware, _operation = bench

    with pytest.raises(RuntimeError, match="profile"):
        model.run_stim_bench_test(None, 1, "hardware_stim3")

    assert hardware.pulses == []


def test_a_board_failure_cancels_the_armed_output(bench):
    model, laser, hardware, operation = bench

    def fail(_duration_us, stim_line=3):
        raise RuntimeError("bus down")

    hardware.pulse_stim = fail

    with pytest.raises(RuntimeError, match="bus down"):
        model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")

    assert operation.cancelled, "a failed test must not leave the output armed"
    assert laser.released == [operation]


def test_a_board_that_does_not_queue_the_pulse_cancels_the_output(bench):
    model, laser, hardware, operation = bench
    hardware.pulse_stim = lambda _duration_us, stim_line=3: None

    with pytest.raises(RuntimeError, match="not queued"):
        model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")

    assert operation.cancelled
    assert laser.released == [operation]


def test_the_capability_guard_reads_the_hardware_model_record(bench):
    model, _laser, hardware, _operation = bench
    hardware.firmware_compatibility = {"reported_capabilities": ["time_sync"]}

    with pytest.raises(RuntimeError, match="finite_stim3_pulse"):
        model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")


def test_a_board_that_reports_nothing_still_runs(bench):
    # Shipped firmware does not answer the capability request at all, so an
    # empty record must not block a path the trial route already drives.
    model, _laser, hardware, _operation = bench
    hardware.firmware_compatibility = {}

    result = model.run_stim_bench_test(model._laser_profiles["stim-a"], 1, "hardware_stim3")

    assert hardware.pulses == [(1000, 3)]
    assert result.profile_id == "stim-a"


def test_a_software_route_profile_is_started_from_the_host(bench):
    # The start a trial's stim-camera trigger makes, with no board involved.
    model, laser, hardware, operation = bench
    profile = make_profile(
        trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
        trigger_terminal="",
    )

    result = model.run_stim_bench_test(profile, 1, "direct_ni_software")

    assert laser.prepared, "the analog output must be armed first"
    assert operation.triggered
    assert hardware.pulses == []
    assert result.arm_to_terminal_ms is not None
    assert "software start" in str(result)
    assert laser.released == [operation]


def test_a_software_start_that_fails_cancels_the_armed_output(bench):
    model, laser, _hardware, operation = bench
    profile = make_profile(
        trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
        trigger_terminal="",
    )

    def fail():
        raise RuntimeError("not armed")

    operation.trigger = fail

    with pytest.raises(RuntimeError, match="not armed"):
        model.run_stim_bench_test(profile, 1, "direct_ni_software")

    assert operation.cancelled
    assert laser.released == [operation]


def test_a_stim2_profile_pulses_the_second_line(bench):
    model, _laser, hardware, _operation = bench
    profile = make_profile(profile_id="stim-b", stim_line=2)
    model._laser.configuration = LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=1, analog_output="Dev1/ao0", diode_input="Dev1/ai0",
            shutter_output="Dev1/port0/line0", trigger_source="/Dev1/PFI0",
            board_stim_line=2),
    ), backend="nidaq")

    result = model.run_stim_bench_test(profile, 1, "hardware_stim3")

    assert hardware.pulses == [(1000, 2)]
    assert "STIM2" in str(result)


def test_the_board_line_comes_from_the_laser_not_the_profile(bench):
    model, _laser, hardware, _operation = bench
    profile = make_profile(stim_line=3)
    model._laser.configuration = LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=2, analog_output="Dev1/ao1", diode_input="Dev1/ai1",
            shutter_output="Dev1/port0/line1", trigger_source="/Dev1/PXI_Trig2",
            board_stim_line=2),
    ), backend="nidaq")

    result = model.run_stim_bench_test(profile, 2, "hardware_stim3")

    assert hardware.pulses == [(1000, 2)]
    assert result.channel_id == 2
    assert result.trigger_terminal == "/Dev1/PXI_Trig2"


def test_a_board_test_on_a_laser_without_a_board_line_is_refused(bench):
    model, _laser, hardware, _operation = bench
    model._laser.configuration = LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=1, analog_output="Dev1/ao0", diode_input="Dev1/ai0",
            shutter_output="Dev1/port0/line0", trigger_source="/Dev1/PFI0"),
    ), backend="nidaq")

    with pytest.raises(RuntimeError, match="boardStimLine"):
        model.run_stim_bench_test(make_profile(), 1, "hardware_stim3")

    assert hardware.pulses == []
