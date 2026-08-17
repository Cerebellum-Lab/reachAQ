import pytest
import threading
from types import SimpleNamespace

from autotrainer.device import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserPulseTrain,
    LaserSystemConfiguration,
    NidaqLaserController,
    LaserOperationState,
    NullLaserController,
    LaserSynchronizedPulseTrain,
)
from autotrainer.core import NidaqTimingPlan


def make_channel(channel_id=LaserChannelId.LASER_1, command_copy_input="Dev1/ai1"):
    return LaserChannelConfiguration(
        channel_id=channel_id,
        analog_output="Dev1/ao0",
        diode_input="Dev1/ai0",
        shutter_output="Dev1/port0/line0",
        auxiliary_output="Dev1/port0/line1",
        command_copy_input=command_copy_input,
    )


def test_laser_channel_configuration_normalizes_channel_id():
    config = make_channel(1)

    assert config.channel_id == LaserChannelId.LASER_1


def test_laser_channel_configuration_clamps_command_voltage():
    config = make_channel()

    assert config.clamp_command_voltage(-1) == 0.0
    assert config.clamp_command_voltage(2.5) == 2.5
    assert config.clamp_command_voltage(10) == 5.0


def test_laser_system_configuration_limits_channels_to_four():
    channels = [
        make_channel(LaserChannelId.LASER_1),
        make_channel(LaserChannelId.LASER_2),
        make_channel(LaserChannelId.LASER_3),
        make_channel(LaserChannelId.LASER_4),
        make_channel(LaserChannelId.LASER_1),
    ]

    with pytest.raises(ValueError):
        LaserSystemConfiguration.from_channels(channels)


def test_laser_system_configuration_rejects_duplicate_channel_ids():
    channels = [make_channel(LaserChannelId.LASER_1), make_channel(LaserChannelId.LASER_1)]

    with pytest.raises(ValueError):
        LaserSystemConfiguration.from_channels(channels)


def test_null_laser_controller_tracks_outputs():
    system = LaserSystemConfiguration.from_channels([make_channel()])
    controller = NullLaserController(system)

    applied = controller.set_command_voltage(LaserChannelId.LASER_1, 2.25)
    controller.set_shutter_open(LaserChannelId.LASER_1, True)
    controller.set_auxiliary_output(LaserChannelId.LASER_1, True)

    assert applied == 2.25
    assert controller.read_diode_voltage(LaserChannelId.LASER_1) == 2.25
    assert controller.read_command_copy_voltage(LaserChannelId.LASER_1) == 2.25
    assert controller.read_feedback_sample(LaserChannelId.LASER_1).command_copy_volts == 2.25
    assert controller.is_shutter_open(LaserChannelId.LASER_1)
    assert controller.is_auxiliary_output_enabled(LaserChannelId.LASER_1)

    controller.close_all_shutters()

    assert not controller.is_shutter_open(LaserChannelId.LASER_1)


def test_null_laser_controller_reports_missing_command_copy_input():
    system = LaserSystemConfiguration.from_channels([make_channel(command_copy_input=None)])
    controller = NullLaserController(system)

    with pytest.raises(RuntimeError, match="AI command-copy input"):
        controller.read_command_copy_voltage(LaserChannelId.LASER_1)

    assert controller.read_feedback_sample(LaserChannelId.LASER_1).command_copy_volts is None


def test_laser_pulse_train_requires_frequency_for_multiple_pulses():
    with pytest.raises(ValueError, match="frequency_hz"):
        LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
            pulse_count=2,
        )


def test_null_laser_controller_runs_pulse_train():
    system = LaserSystemConfiguration.from_channels([make_channel()])
    controller = NullLaserController(system)

    controller.run_pulse_train(
        LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=2.0,
            duration_ms=10.0,
        )
    )

    assert controller.read_diode_voltage(LaserChannelId.LASER_1) == 0.0
    assert not controller.is_shutter_open(LaserChannelId.LASER_1)


def test_null_laser_controller_emulates_deferred_software_start():
    system = LaserSystemConfiguration.from_channels(
        [make_channel()], backend="null", hardware_timed=True,
    )
    controller = NullLaserController(system)
    operation = controller.run_synchronized_pulse_train(
        LaserSynchronizedPulseTrain(
            pulse_trains=(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1,
                amplitude_volts=2.0,
                duration_ms=1.0,
            ),),
            wait=False,
            defer_start=True,
        )
    )

    assert operation.to_record()["timing_status"]["status"] == (
        "emulated_software_start"
    )
    operation.trigger()
    assert operation.wait(1).value == "completed"
    assert controller.read_diode_voltage(LaserChannelId.LASER_1) == 0.0


def test_null_laser_controller_does_not_claim_hardware_synchronization():
    system = LaserSystemConfiguration.from_channels(
        [make_channel()], backend="null", hardware_timed=True,
    )
    controller = NullLaserController(system)

    with pytest.raises(RuntimeError, match="cannot emulate a hardware"):
        controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
            pulse_trains=(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1,
                amplitude_volts=2.0,
                duration_ms=1.0,
            ),),
            trigger_source="/Dev1/PFI0",
            wait=False,
        ))


def test_nidaq_laser_uses_shared_scaled_feedback_without_reserving_ai_tasks():
    channel = make_channel()
    configuration = LaserSystemConfiguration.from_channels(
        [channel],
        backend="nidaq",
        hardware_timed=True,
        sample_rate_hz=1000.0,
    )
    created_tasks = []

    class FakeTask:
        def __init__(self, name):
            self.name = name
            self.ai_channels = SimpleNamespace(
                add_ai_voltage_chan=lambda *_args, **_kwargs: None,
            )
            self.do_channels = SimpleNamespace(
                add_do_chan=lambda *_args, **_kwargs: None,
            )
            created_tasks.append(name)

    values = {"Dev1/ai0": 1.25, "Dev1/ai1": 2.5}
    controller = object.__new__(NidaqLaserController)
    controller._configuration = configuration
    controller._feedback_reader = values.__getitem__
    controller._nidaqmx = SimpleNamespace(Task=FakeTask)
    controller._command_volts = {LaserChannelId.LASER_1: 0.75}
    tasks = controller._create_channel_tasks(channel)
    controller._tasks = {LaserChannelId.LASER_1: tasks}

    sample = controller.read_feedback_sample(LaserChannelId.LASER_1)

    assert tasks.diode_input is None
    assert tasks.command_copy_input is None
    assert not any(name.endswith("_ai") for name in created_tasks)
    assert sample.command_volts == 0.75
    assert sample.diode_volts == 1.25
    assert sample.command_copy_volts == 2.5


def test_nidaq_laser_reports_on_demand_output_as_not_synchronized():
    channel = make_channel()
    controller = object.__new__(NidaqLaserController)
    controller._timing_plan = NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Input",
        sample_clock_source="/Input/ai/SampleClock",
        hardware_output_devices=("Dev1",),
        hardware_output_timing_status="declared_not_armed",
    )
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
    )

    kwargs, status = controller._resolve_pulse_timing((channel,), pulse)

    assert kwargs == {}
    assert status["status"] == "declared_not_armed"


def test_nidaq_laser_uses_shared_clock_only_with_future_hardware_trigger():
    channel = make_channel()
    controller = object.__new__(NidaqLaserController)
    controller._timing_plan = NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Input",
        reference_clock_source="PXI_CLK10",
        reference_clock_rate_hz=10_000_000.0,
        sample_clock_source="/Input/ai/SampleClock",
        hardware_output_devices=("Dev1",),
        hardware_output_timing_status="declared_not_armed",
    )
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        trigger_source="/Input/PXI_Trig0",
    )

    kwargs, status = controller._resolve_pulse_timing((channel,), pulse)

    assert kwargs == {"source": "/Input/ai/SampleClock"}
    assert status["status"] == "hardware_synchronized"
    assert status["referenceClockSource"] == "PXI_CLK10"


def test_nidaq_laser_labels_deferred_start_as_software_timed_without_plan():
    channel = make_channel()
    controller = object.__new__(NidaqLaserController)
    controller._timing_plan = None
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        wait=False,
        defer_start=True,
    )

    kwargs, status = controller._resolve_pulse_timing((channel,), pulse)

    assert kwargs == {}
    assert status["status"] == "software_start"


def test_nonblocking_laser_operation_is_owned_until_terminal(monkeypatch):
    channel = make_channel()
    controller = object.__new__(NidaqLaserController)
    controller._configuration = LaserSystemConfiguration.from_channels(
        [channel], backend="nidaq", hardware_timed=True, sample_rate_hz=1000.0,
    )
    controller._operation_lock = threading.RLock()
    controller._live_operations = {}
    release = threading.Event()

    def execute(_pulse, *, operation):
        operation._mark_armed()
        release.wait(2)
        operation._mark_triggered()

    monkeypatch.setattr(controller, "_execute_synchronized_pulse_train", execute)
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        wait=False,
    )

    operation = controller.run_synchronized_pulse_train(pulse)
    terminal = []
    operation.add_terminal_callback(lambda value: terminal.append(value.state))
    assert operation.state is LaserOperationState.ARMED
    assert operation.operation_id in controller._live_operations
    with pytest.raises(RuntimeError, match="already owned"):
        controller.run_synchronized_pulse_train(pulse)

    release.set()
    assert operation.wait(2) is LaserOperationState.COMPLETED
    assert terminal == [LaserOperationState.COMPLETED]
    assert operation.operation_id not in controller._live_operations


def test_nonblocking_laser_operation_cancel_is_terminal_after_worker_cleanup(monkeypatch):
    channel = make_channel()
    controller = object.__new__(NidaqLaserController)
    controller._configuration = LaserSystemConfiguration.from_channels(
        [channel], backend="nidaq", hardware_timed=True, sample_rate_hz=1000.0,
    )
    controller._operation_lock = threading.RLock()
    controller._live_operations = {}
    release = threading.Event()

    def execute(_pulse, *, operation):
        operation._mark_armed()
        release.wait(2)
        operation._require_not_cancelled()

    monkeypatch.setattr(controller, "_execute_synchronized_pulse_train", execute)
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        wait=False,
    ))

    assert operation.cancel()
    assert operation.state is LaserOperationState.CANCELLED
    assert operation.operation_id in controller._live_operations
    release.set()
    assert operation.wait(2) is LaserOperationState.CANCELLED
    assert operation.operation_id not in controller._live_operations


def test_synchronized_pulse_train_rejects_deferred_hardware_trigger():
    with pytest.raises(ValueError, match="hardware trigger"):
        LaserSynchronizedPulseTrain(
            pulse_trains=(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1,
                amplitude_volts=1.0,
                duration_ms=10.0,
            ),),
            trigger_source="/Dev1/PFI0",
            wait=False,
            defer_start=True,
        )
