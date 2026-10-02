import pytest
import threading

from autotrainer.device import (
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserPulseTrain,
    LaserSystemConfiguration,
    NidaqLaserController,
    LaserOperationState,
    NullLaserController,
    LaserSynchronizedPulseTrain,
)
from autotrainer.device import nidaq_laser
from autotrainer.core import NidaqTimingPlan

from nidaq_daqmx_fake import FakeDaqmx


def make_channel(channel_id=LaserChannelId.LASER_1, command_copy_input="Dev1/ai1"):
    return LaserChannelConfiguration(
        channel_id=channel_id,
        analog_output="Dev1/ao0",
        diode_input="Dev1/ai0",
        shutter_output="Dev1/port0/line0",
        auxiliary_output="Dev1/port0/line1",
        command_copy_input=command_copy_input,
    )


@pytest.fixture
def daq(monkeypatch):
    """The DAQmx fake the controllers here open on; nothing touches a board."""
    fake = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake)
    return fake


@pytest.fixture
def open_controller(daq):
    """Open a NidaqLaserController by its own __init__, on the fake.

    These tests built one with object.__new__ and set only the attributes
    the path under test read, and the controller kept a guard for each one
    they left out. Opened whole, it has every attribute __init__ sets. Each
    is closed after the test, and no task on the fake is left open.
    """
    opened = []

    def open_one(channel=None, *, timing_plan=None, feedback_reader=None,
                 **configuration):
        controller = NidaqLaserController(
            LaserSystemConfiguration.from_channels(
                [channel or make_channel()], backend="nidaq",
                hardware_timed=True, sample_rate_hz=1000.0, **configuration),
            timing_plan=timing_plan, feedback_reader=feedback_reader)
        opened.append(controller)
        return controller

    yield open_one
    for controller in opened:
        controller.close()
    assert [task.label for task in daq.tasks if not task.closed] == []


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
        sample_rate_hz=10_000,
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
        sample_rate_hz=10_000,
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


def test_nidaq_laser_uses_shared_scaled_feedback_without_reserving_ai_tasks(
    open_controller, daq,
):
    values = {"Dev1/ai0": 1.25, "Dev1/ai1": 2.5}
    controller = open_controller(feedback_reader=values.__getitem__)
    controller.set_command_voltage(LaserChannelId.LASER_1, 0.75)
    tasks = controller._tasks[LaserChannelId.LASER_1]

    sample = controller.read_feedback_sample(LaserChannelId.LASER_1)

    assert tasks.diode_input is None
    assert tasks.command_copy_input is None
    assert not any(task.label.endswith("_ai") for task in daq.tasks)
    assert sample.command_volts == 0.75
    assert sample.diode_volts == 1.25
    assert sample.command_copy_volts == 2.5


def test_nidaq_laser_reports_on_demand_output_as_not_synchronized(open_controller):
    channel = make_channel()
    controller = open_controller(channel, timing_plan=NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Input",
        sample_clock_source="/Input/ai/SampleClock",
        hardware_output_devices=("Dev1",),
        hardware_output_timing_status="declared_not_armed",
    ))
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


def test_nidaq_laser_uses_shared_clock_only_with_future_hardware_trigger(
    open_controller, daq,
):
    """The clock crosses to the output board, and is named there.

    This asserted the plan's clock verbatim until cross-board routing was
    added, and then failed for two years' worth of reasons at once: the
    controller it built by hand never grew the collaborators that path
    needs, and the answer it expected was the one from before there was a
    route. Both are the test's to fix; the behaviour is deliberate. It
    opens a whole controller on the DAQmx fake now.
    """
    channel = make_channel()
    plan = NidaqTimingPlan(
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
    controller = open_controller(
        channel, timing_plan=plan, backplane_clock_line="PXI_Trig1")
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        trigger_source="/Input/PXI_Trig0",
    )

    kwargs, status = controller._resolve_pulse_timing((channel,), pulse)

    # The clock is produced on Input and the output is on Dev1, so it is
    # driven onto a backplane line and read as Dev1's view of that line.
    # DAQmx will not route it across an unidentified chassis by name.
    assert daq.connected == [("/Input/ai/SampleClock", "/Input/PXI_Trig1")]
    assert kwargs == {"source": "/Dev1/PXI_Trig1"}
    assert status["status"] == "hardware_synchronized"
    assert status["referenceClockSource"] == "PXI_CLK10"
    # The plan's own clock is still reported, because that is what it is.
    assert status["sampleClockSource"] == "/Input/ai/SampleClock"


def test_a_clock_already_on_the_output_board_acquires_no_route(open_controller, daq):
    """A single-board rig must not reserve a backplane line it cannot use."""
    plan = NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Dev1",
        reference_clock_source="PXI_CLK10",
        reference_clock_rate_hz=10_000_000.0,
        sample_clock_source="/Dev1/ai/SampleClock",
        hardware_output_devices=("Dev1",),
        hardware_output_timing_status="declared_not_armed",
    )
    controller = open_controller(timing_plan=plan)
    pulse = LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        trigger_source="/Dev1/PXI_Trig0",
    )

    kwargs, _status = controller._resolve_pulse_timing((make_channel(),), pulse)

    assert kwargs == {"source": "/Dev1/ai/SampleClock"}
    assert daq.connected == []


def test_nidaq_laser_labels_deferred_start_as_software_timed_without_plan(
    open_controller,
):
    channel = make_channel()
    controller = open_controller(channel, timing_plan=None)
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


def test_nonblocking_laser_operation_is_owned_until_terminal(
    open_controller, monkeypatch,
):
    controller = open_controller()
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
    with pytest.raises(RuntimeError, match="refused while the pulse on laser 1 holds"):
        controller.run_synchronized_pulse_train(pulse)

    release.set()
    assert operation.wait(2) is LaserOperationState.COMPLETED
    assert terminal == [LaserOperationState.COMPLETED]
    assert operation.operation_id not in controller._live_operations


def test_nonblocking_laser_operation_cancel_is_terminal_after_worker_cleanup(
    open_controller, monkeypatch,
):
    controller = open_controller()
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


def _deferred_pulse(**fields):
    return LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1,
            amplitude_volts=1.0,
            duration_ms=10.0,
        ),),
        wait=False,
        **fields,
    )


@pytest.mark.parametrize("fields", [
    dict(defer_start=True, start_wait_seconds=0),
    dict(defer_start=True, start_wait_seconds=-1.0),
    dict(start_wait_seconds=1.0),
], ids=["zero", "negative", "not_deferred"])
def test_a_start_wait_is_positive_and_only_for_a_deferred_start(fields):
    with pytest.raises(ValueError, match="start_wait_seconds"):
        _deferred_pulse(**fields)


def test_a_start_wait_is_none_unless_given():
    assert _deferred_pulse(defer_start=True).start_wait_seconds is None
    assert _deferred_pulse(defer_start=True, start_wait_seconds=2.0).start_wait_seconds == 2.0


def test_the_null_controller_waits_for_a_start_no_longer_than_its_start_wait():
    system = LaserSystemConfiguration.from_channels(
        [make_channel()], backend="null", hardware_timed=True,
        sample_rate_hz=10_000,
    )
    controller = NullLaserController(system)
    operation = controller.run_synchronized_pulse_train(_deferred_pulse(
        defer_start=True, timeout_seconds=30.0, start_wait_seconds=0.05))

    assert operation.wait_until_finished(1.0)
    assert operation.state is LaserOperationState.FAILED
    assert isinstance(operation.error, TimeoutError)


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


def _ramp(**values):
    fields = dict(channel_id=LaserChannelId.LASER_1, start_volts=0.0,
                  stop_volts=5.0, steps=3, samples_per_step=100)
    fields.update(values)
    return LaserCalibrationRamp(**fields)


def test_a_calibration_ramp_settles_600_us_by_default():
    # A time, not a share of the step: the laser, the diode and the input
    # take as long to settle however long the step is.
    from autotrainer.device.laser import CALIBRATION_SETTLE_SECONDS

    assert _ramp().settle_seconds == CALIBRATION_SETTLE_SECONDS == 600e-6
    # round(settle x rate) samples of each step.
    assert _ramp().settle_sample_count(100_000.0) == 60
    assert _ramp().settle_sample_count(1_000.0) == 1
    assert _ramp(settle_seconds=26e-6).settle_sample_count(100_000.0) == 3
    assert _ramp(settle_seconds=0).settle_sample_count(100_000.0) == 0


def test_a_settle_that_leaves_no_sample_of_a_step_names_both():
    with pytest.raises(ValueError, match="600 µs is 60 samples at 100000 Hz.*60 samples of each step"):
        _ramp(samples_per_step=60).settle_sample_count(100_000.0)
    assert _ramp(samples_per_step=61).settle_sample_count(100_000.0) == 60


@pytest.mark.parametrize("settle", [-1e-6, True, "600e-6", float("nan"), float("inf")])
def test_a_calibration_ramp_takes_a_settle_of_zero_seconds_or_more(settle):
    with pytest.raises(ValueError, match="settle_seconds"):
        _ramp(settle_seconds=settle)
