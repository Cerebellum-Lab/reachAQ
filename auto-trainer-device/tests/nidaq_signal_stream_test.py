from types import SimpleNamespace

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqTimingPlan,
)
from autotrainer.device import nidaq_signal_stream
from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


class _FakeTiming:
    def __init__(self, task):
        self._task = task
        self.ref_clk_src = None
        self.ref_clk_rate = None

    def cfg_samp_clk_timing(self, **kwargs):
        self._task.timing_configuration = kwargs

    def cfg_implicit_timing(self, **kwargs):
        self._task.implicit_timing_configuration = kwargs


class _FakeDigitalChannels:
    def __init__(self, task):
        self._task = task
        self.all = self
        self.di_data_xfer_mech = None
        self.di_data_xfer_req_cond = None

    def add_di_chan(self, physical_channel, *, line_grouping):
        self._task.channels.append((physical_channel, line_grouping))


class _FakeAnalogChannels:
    def __init__(self, task):
        self._task = task

    def add_ai_voltage_chan(self, physical_channel, **kwargs):
        self._task.channels.append((physical_channel, kwargs))


class _FakeCounterChannels:
    def __init__(self, task):
        self._task = task

    def add_co_pulse_chan_freq(self, physical_channel, *, freq):
        self._task.counter_configuration = (physical_channel, freq)


class _FakeTask:
    def __init__(self, name: str, start_order):
        self.name = name
        self._start_order = start_order
        self.channels = []
        self.timing_configuration = None
        self.implicit_timing_configuration = None
        self.counter_configuration = None
        self.timing = _FakeTiming(self)
        self.ai_channels = _FakeAnalogChannels(self)
        self.di_channels = _FakeDigitalChannels(self)
        self.co_channels = _FakeCounterChannels(self)
        self.triggers = SimpleNamespace(
            start_trigger=SimpleNamespace(
                cfg_dig_edge_start_trig=self._set_start_trigger,
            ),
        )
        self.start_trigger_source = None
        self.started = False
        self.closed = False
        self.read_count = 0

    def start(self):
        self.started = True
        self._start_order.append(self.name)

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def read(self, *, number_of_samples_per_channel, timeout):
        assert timeout >= 1.0
        values = tuple(
            [
                (index + channel_index) % 2 == 0
                for index in range(number_of_samples_per_channel)
            ]
            for channel_index in range(len(self.channels))
        )
        self.read_count += number_of_samples_per_channel
        return values[0] if len(values) == 1 else list(values)

    def _set_start_trigger(self, source):
        self.start_trigger_source = source


class _FakeNidaqmx:
    constants = SimpleNamespace(
        AcquisitionType=SimpleNamespace(CONTINUOUS="continuous"),
        DataTransferActiveTransferMode=SimpleNamespace(INTERRUPT="interrupt"),
        InputDataTransferCondition=SimpleNamespace(
            ON_BOARD_MEMORY_NOT_EMPTY="not-empty",
        ),
        LineGrouping=SimpleNamespace(CHAN_PER_LINE="per-line"),
    )

    def __init__(self):
        self.tasks = []
        self.start_order = []

    def Task(self, name):
        task = _FakeTask(name, self.start_order)
        self.tasks.append(task)
        return task


def _digital_configuration() -> NidaqSignalStreamConfiguration:
    return NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                name="barcode",
                physical_channel="Dev1/port0/line7",
                kind="digital",
            ),
            NidaqSignalChannelConfiguration(
                name="cam_frames",
                physical_channel="Dev1/port0/line2",
                kind="digital",
            ),
        ),
        is_enabled=True,
        sample_rate_hz=1000.0,
        read_chunk_size=3,
    )


def test_digital_device_uses_hardware_counter_sample_clock(monkeypatch):
    fake_nidaqmx = _FakeNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: fake_nidaqmx)

    controller = NidaqSignalStreamController(_digital_configuration())
    try:
        controller.start()
        assert len(fake_nidaqmx.tasks) == 2
        digital_task, clock_task = fake_nidaqmx.tasks
        assert digital_task.timing_configuration["source"] == "/Dev1/Ctr0InternalOutput"
        assert digital_task.di_channels.di_data_xfer_mech == "interrupt"
        assert digital_task.di_channels.di_data_xfer_req_cond == "not-empty"
        assert clock_task.counter_configuration == ("Dev1/ctr0", 1000.0)
        assert clock_task.started

        block = controller.read_chunk()

        assert block.sample_count == 3
        assert block.values["barcode"] == (1.0, 0.0, 1.0)
        assert block.values["cam_frames"] == (0.0, 1.0, 0.0)
        assert digital_task.read_count == 3
    finally:
        controller.close()


def test_multi_device_tasks_arm_slave_before_master(monkeypatch):
    fake_nidaqmx = _FakeNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: fake_nidaqmx)
    configuration = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration("master_ai", "Acquire/ai0"),
            NidaqSignalChannelConfiguration("slave_ai", "Feedback/ai0"),
        ),
        is_enabled=True,
        sample_rate_hz=1000.0,
        read_chunk_size=3,
    )
    plan = NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Acquire",
        slave_devices=("Feedback",),
        reference_clock_source="PXI_CLK10",
        reference_clock_rate_hz=10_000_000.0,
        sample_clock_source="/Acquire/ai/SampleClock",
        start_trigger_source="/Acquire/ai/StartTrigger",
        task_start_order=("Feedback", "Acquire"),
        synchronization_quality="hardware_backplane",
    )

    controller = NidaqSignalStreamController(
        configuration,
        timing_plan=plan,
    )
    try:
        controller.start()
        master, slave = fake_nidaqmx.tasks

        assert slave.timing_configuration["source"] == "/Acquire/ai/SampleClock"
        assert "source" not in master.timing_configuration
        assert slave.start_trigger_source == "/Acquire/ai/StartTrigger"
        assert master.start_trigger_source is None
        assert slave.timing.ref_clk_src == "PXI_CLK10"
        assert slave.timing.ref_clk_rate == 10_000_000.0
        assert fake_nidaqmx.start_order == (
            ["reachaq_signal_stream_Feedback_ai", "reachaq_signal_stream_Acquire_ai"]
        )

        block = controller.read_chunk()

        assert block.sample_index == 0
        assert block.sample_count == 3
        assert block.epoch_perf_time is not None
        assert block.epoch_wall_time is not None
    finally:
        controller.close()


def test_digital_master_arms_all_inputs_before_counter_clock(monkeypatch):
    fake_nidaqmx = _FakeNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: fake_nidaqmx)
    configuration = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                "master", "Acquire/port0/line0", "digital",
            ),
            NidaqSignalChannelConfiguration(
                "slave", "Confirm/port0/line0", "digital",
            ),
        ),
        is_enabled=True,
        sample_rate_hz=1000.0,
        read_chunk_size=3,
    )
    plan = NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Acquire",
        slave_devices=("Confirm",),
        reference_clock_source="PXI_CLK10",
        reference_clock_rate_hz=10_000_000.0,
        sample_clock_source="/Acquire/Ctr0InternalOutput",
        start_trigger_source=None,
        task_start_order=("Confirm", "Acquire"),
        synchronization_quality="hardware_backplane",
        clock_producer="counter",
        clock_producer_device="Acquire",
        consumer_devices=("Acquire", "Confirm"),
    )

    controller = NidaqSignalStreamController(configuration, timing_plan=plan)
    try:
        controller.start()
        tasks = {task.name: task for task in fake_nidaqmx.tasks}
        assert tasks["reachaq_signal_stream_Confirm_di"].timing_configuration[
            "source"
        ] == "/Acquire/Ctr0InternalOutput"
        assert tasks["reachaq_signal_stream_Acquire_di"].timing_configuration[
            "source"
        ] == "/Acquire/Ctr0InternalOutput"
        assert fake_nidaqmx.start_order[-1] == (
            "reachaq_signal_stream_Acquire_clock"
        )
        assert fake_nidaqmx.start_order.index(
            "reachaq_signal_stream_Confirm_di"
        ) < fake_nidaqmx.start_order.index("reachaq_signal_stream_Acquire_clock")
        assert fake_nidaqmx.start_order.index(
            "reachaq_signal_stream_Acquire_di"
        ) < fake_nidaqmx.start_order.index("reachaq_signal_stream_Acquire_clock")
        assert tasks["reachaq_signal_stream_Acquire_clock"].timing.ref_clk_src == (
            "PXI_CLK10"
        )
    finally:
        controller.close()


def test_external_clock_and_start_trigger_apply_to_selected_master(monkeypatch):
    fake_nidaqmx = _FakeNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: fake_nidaqmx)
    configuration = NidaqSignalStreamConfiguration(
        channels=(NidaqSignalChannelConfiguration("input", "DevA/ai0"),),
        is_enabled=True,
        sample_rate_hz=1000.0,
        read_chunk_size=3,
    )
    plan = NidaqTimingPlan(
        requested_mode="external",
        resolved_mode="external",
        is_valid=True,
        master_device="DevA",
        sample_clock_source="/DevA/PFI1",
        start_trigger_source="/DevA/PFI2",
        task_start_order=("DevA",),
        synchronization_quality="hardware_external",
        clock_producer="external",
        consumer_devices=("DevA",),
    )

    controller = NidaqSignalStreamController(configuration, timing_plan=plan)
    try:
        task = fake_nidaqmx.tasks[0]
        assert task.timing_configuration["source"] == "/DevA/PFI1"
        assert task.start_trigger_source == "/DevA/PFI2"
    finally:
        controller.close()
