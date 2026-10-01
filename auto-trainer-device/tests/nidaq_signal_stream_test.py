from types import SimpleNamespace

import numpy
import pytest

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqTaskGraph,
    NidaqTaskSpecification,
    NidaqTimingPlan,
)
from autotrainer.device import nidaq_signal_stream
from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController

from nidaq_daqmx_fake import FakeDaqError

#: The signals a master exports for its slaves, as nidaqmx.constants.Signal.
_SIGNALS = SimpleNamespace(
    SAMPLE_CLOCK="sample_clock",
    COUNTER_OUTPUT_EVENT="counter_output_event",
    START_TRIGGER="start_trigger",
)
_CLOCK_SIGNALS = {_SIGNALS.SAMPLE_CLOCK, _SIGNALS.COUNTER_OUTPUT_EVENT}


def _board_of(terminal):
    """The board a terminal names, or None for a bare one such as PXI_Trig4."""
    parts = str(terminal).strip("/").split("/")
    return parts[0] if len(parts) > 1 else None


def _backplane_line(terminal):
    tail = str(terminal or "").strip("/").rsplit("/", 1)[-1].lower()
    return tail if tail.startswith("pxi_trig") else None


class _FakeTiming:
    def __init__(self, task):
        self._task = task
        self.ref_clk_src = None
        self.ref_clk_rate = None

    def cfg_samp_clk_timing(self, **kwargs):
        self._task.refuse_another_boards_terminal(kwargs.get("source"))
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
        self._task.is_digital = True
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
    def __init__(self, name: str, start_order, daq=None):
        self.name = name
        self._start_order = start_order
        self._daq = daq
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
        #: Each export_signal call, as (signal, output terminal).
        self.exports = []
        self.export_signals = SimpleNamespace(export_signal=self._export_signal)
        self.start_trigger_source = None
        self.started = False
        self.closed = False
        self.read_count = 0
        self.is_digital = False

    @property
    def devices(self):
        """What DAQmx offers a task, and how the board gets asked anything."""
        names = []
        for physical_channel, _ in self.channels:
            name = str(physical_channel).split("/", 1)[0]
            if name not in names:
                names.append(name)
        if self.counter_configuration is not None:
            name = _board_of(self.counter_configuration[0])
            if name not in names:
                names.append(name)
        return [SimpleNamespace(name=name) for name in names]

    def refuse_another_boards_terminal(self, terminal):
        """As christielab10's unidentified chassis answers a terminal named
        on another board: DAQmx would have to reserve a backplane line for
        the route, and without the chassis it reserves none (-89125). A
        terminal on the task's own board, or a bare one, is taken."""
        if not terminal:
            return
        board = _board_of(terminal)
        own = [device.name for device in self.devices]
        if board is not None and board not in own:
            raise FakeDaqError(
                -89125,
                "No registered trigger lines could be found between the "
                f"devices in the route. Source Device: {board} Destination "
                f"Device: {', '.join(own)} ({self.name})")

    def _export_signal(self, signal_id, output_terminal):
        self.refuse_another_boards_terminal(output_terminal)
        self.exports.append((signal_id, output_terminal))
        if self._daq is not None:
            self._daq.exports.append((signal_id, output_terminal))

    def _refuse_undriven_lines(self):
        """A task that reads a backplane line nothing drives never samples.

        DAQmx takes the local name and waits: the read times out (-200284).
        A clock line has to carry a clock and a trigger line a start trigger.
        """
        if self._daq is None:
            return
        driven = {}
        for signal_id, terminal in self._daq.exports:
            line = _backplane_line(terminal)
            if line is not None:
                driven.setdefault(line, set()).add(signal_id)
        source = (self.timing_configuration or {}).get("source")
        for terminal, wanted in ((source, _CLOCK_SIGNALS),
                                 (self.start_trigger_source, {_SIGNALS.START_TRIGGER})):
            line = _backplane_line(terminal)
            if line is not None and not driven.get(line, set()) & wanted:
                raise FakeDaqError(
                    -200284, "Some or all of the samples requested have not "
                    f"yet been acquired: {terminal} carries "
                    f"{sorted(driven.get(line, ())) or 'nothing'} ({self.name})")

    def start(self):
        self.started = True
        self._start_order.append(self.name)

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def read(self, *, number_of_samples_per_channel, timeout):
        assert timeout >= 1.0
        self._refuse_undriven_lines()
        if self.is_digital:
            # A port-grouped digital task reads whole port words, one row per
            # port - not one row per line. Alternating which single line is
            # high is what makes this a test: a reader that took a line's bit
            # from its position in the channel list instead of from its line
            # number would see nothing on either of these, and that defect
            # shipped once.
            values = tuple(
                [(1 << 7) if index % 2 == 0 else (1 << 2)
                 for index in range(number_of_samples_per_channel)]
                for _ in range(len(self.channels))
            )
        else:
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
        self.refuse_another_boards_terminal(source)
        self.start_trigger_source = source


class _FakeNidaqmx:
    constants = SimpleNamespace(
        AcquisitionType=SimpleNamespace(CONTINUOUS="continuous"),
        DataTransferActiveTransferMode=SimpleNamespace(INTERRUPT="interrupt"),
        InputDataTransferCondition=SimpleNamespace(
            ON_BOARD_MEMORY_NOT_EMPTY="not-empty",
        ),
        LineGrouping=SimpleNamespace(CHAN_PER_LINE="per-line",
                                     CHAN_FOR_ALL_LINES="all-lines"),
        TerminalConfiguration=SimpleNamespace(RSE="rse", NRSE="nrse",
                                              DIFF="diff"),
        Signal=_SIGNALS,
    )

    system = SimpleNamespace(
        Device=lambda name: SimpleNamespace(terminals=(
            f"/{name}/PXI_Clk10", f"/{name}/PFI0",
            f"/{name}/Ctr0InternalOutput")))

    def __init__(self):
        self.tasks = []
        self.start_order = []
        #: Every export made, by any task, as (signal, output terminal).
        self.exports = []

    def Task(self, name):
        task = _FakeTask(name, self.start_order, self)
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
        assert list(block.values["barcode"]) == [1.0, 0.0, 1.0]
        assert list(block.values["cam_frames"]) == [0.0, 1.0, 0.0]
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
    plan = _backplane_plan()

    controller = NidaqSignalStreamController(
        configuration,
        timing_plan=plan,
    )
    try:
        controller.start()
        master, slave = fake_nidaqmx.tasks

        # The master's clock and start trigger, each on the backplane line
        # the master exports it onto, named on the slave's own board: named
        # on the master, they are a cross-board route the fake refuses as
        # christielab10's unidentified chassis does (-89125).
        assert slave.timing_configuration["source"] == "/Feedback/PXI_Trig4"
        assert "source" not in master.timing_configuration
        assert slave.start_trigger_source == "/Feedback/PXI_Trig5"
        assert master.start_trigger_source is None
        assert master.exports == [
            ("sample_clock", "/Acquire/PXI_Trig4"),
            ("start_trigger", "/Acquire/PXI_Trig5"),
        ]
        assert slave.exports == []
        # What the laser reads off the plan is the master's own terminal.
        assert plan.sample_clock_source == "/Acquire/ai/SampleClock"
        assert slave.timing.ref_clk_src == "/Feedback/PXI_Clk10"
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


def _backplane_plan(**values):
    """Acquire's inputs master Feedback's, as build_nidaq_timing_plan has it."""
    fields = dict(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="Acquire",
        slave_devices=("Feedback",),
        reference_clock_source="PXI_CLK10",
        reference_clock_rate_hz=10_000_000.0,
        sample_clock_source="/Acquire/ai/SampleClock",
        start_trigger_source="/Acquire/ai/StartTrigger",
        sample_clock_export_terminal="/Acquire/PXI_Trig4",
        start_trigger_export_terminal="/Acquire/PXI_Trig5",
        task_start_order=("Feedback", "Acquire"),
        synchronization_quality="hardware_backplane",
        clock_producer="ai",
        clock_producer_device="Acquire",
        consumer_devices=("Acquire", "Feedback"),
    )
    fields.update(values)
    return NidaqTimingPlan(**fields)


def _two_board_inputs():
    return NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration("master_ai", "Acquire/ai0"),
            NidaqSignalChannelConfiguration("slave_ai", "Feedback/ai0"),
            NidaqSignalChannelConfiguration(
                "slave_line", "Feedback/port0/line0", "digital"),
        ),
        is_enabled=True,
        sample_rate_hz=1000.0,
        read_chunk_size=3,
    )


def test_a_slaves_digital_input_runs_on_its_own_clock_and_the_backplane_trigger(
        monkeypatch):
    # Beside an analog task on the same board, the digital task takes that
    # task's clock, on its own board as before; its start trigger is the
    # backplane line's.
    fake_nidaqmx = _FakeNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: fake_nidaqmx)

    controller = NidaqSignalStreamController(
        _two_board_inputs(), timing_plan=_backplane_plan())
    try:
        controller.start()
        tasks = {task.name: task for task in fake_nidaqmx.tasks}
        slave_ai = tasks["reachaq_signal_stream_Feedback_ai"]
        slave_di = tasks["reachaq_signal_stream_Feedback_di"]

        assert slave_ai.timing_configuration["source"] == "/Feedback/PXI_Trig4"
        assert slave_ai.start_trigger_source == "/Feedback/PXI_Trig5"
        assert slave_di.timing_configuration["source"] == "/Feedback/ai/SampleClock"
        assert slave_di.start_trigger_source == "/Feedback/PXI_Trig5"
        assert controller.read_chunk().sample_count == 3
    finally:
        controller.close()


@pytest.mark.parametrize(("values", "field"), [
    (dict(sample_clock_export_terminal=None), "sampleClockExportTerminal"),
    (dict(sample_clock_export_terminal="/Acquire/PFI3"), "sampleClockExportTerminal"),
    (dict(start_trigger_export_terminal=None), "startTriggerExportTerminal"),
])
def test_a_slave_with_no_backplane_line_is_refused_by_the_field(
        monkeypatch, values, field):
    # The plan refuses this first (build_nidaq_timing_plan); a plan that got
    # here anyway names the field rather than the master's terminal, which
    # DAQmx would refuse inside the task at -89125.
    fake_nidaqmx = _FakeNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: fake_nidaqmx)

    with pytest.raises(RuntimeError, match=f"NI-DAQ slave Feedback .*timing {field}"):
        NidaqSignalStreamController(
            _two_board_inputs(), timing_plan=_backplane_plan(**values))

    assert all(task.closed for task in fake_nidaqmx.tasks)


def test_a_slave_named_on_the_masters_terminal_is_refused_as_daqmx_does(monkeypatch):
    # The fake itself: the old naming meets -89125, as it would on the rig.
    fake_nidaqmx = _FakeNidaqmx()
    task = fake_nidaqmx.Task("reachaq_signal_stream_Feedback_ai")
    task.ai_channels.add_ai_voltage_chan("Feedback/ai0")

    with pytest.raises(FakeDaqError) as refused:
        task.timing.cfg_samp_clk_timing(source="/Acquire/ai/SampleClock")
    with pytest.raises(FakeDaqError):
        task.triggers.start_trigger.cfg_dig_edge_start_trig("/Acquire/ai/StartTrigger")

    assert refused.value.error_code == -89125
    task.timing.cfg_samp_clk_timing(source="/Feedback/PXI_Trig4")
    task.triggers.start_trigger.cfg_dig_edge_start_trig("/Feedback/PXI_Trig5")
    # Nothing drives either line yet, so nothing is ever sampled.
    with pytest.raises(FakeDaqError, match="-200284"):
        task.read(number_of_samples_per_channel=3, timeout=1.0)
    fake_nidaqmx.exports += [("sample_clock", "/Acquire/PXI_Trig4"),
                             ("start_trigger", "/Acquire/PXI_Trig5")]
    assert len(task.read(number_of_samples_per_channel=3, timeout=1.0)) == 3


def test_verified_multidevice_probe_uses_one_expanded_task(monkeypatch):
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
        task_start_order=("Feedback", "Acquire"),
        synchronization_quality="hardware_backplane",
        task_graph=NidaqTaskGraph(
            graph_id="g",
            strategy="auto_multidevice",
            tasks=(
                NidaqTaskSpecification(
                    "Acquire.ai", "Acquire", "ai",
                    ("Acquire/ai0", "Feedback/ai0"), "continuous_input",
                ),
            ),
        ),
        multidevice_probe_status="verified",
    )

    controller = NidaqSignalStreamController(configuration, timing_plan=plan)
    try:
        assert len(fake_nidaqmx.tasks) == 1
        assert [item[0] for item in fake_nidaqmx.tasks[0].channels] == [
            "Acquire/ai0", "Feedback/ai0",
        ]
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
        # A counter master has no start trigger to export: one line only.
        sample_clock_export_terminal="/Acquire/PXI_Trig4",
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
        ] == "/Confirm/PXI_Trig4"
        assert tasks["reachaq_signal_stream_Confirm_di"].start_trigger_source is None
        assert tasks["reachaq_signal_stream_Acquire_clock"].exports == [
            ("counter_output_event", "/Acquire/PXI_Trig4")]
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
            "/Acquire/PXI_Clk10"
        )
        # The slave samples: its line carries the counter's clock.
        assert controller.read_chunk().sample_count == 3
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


def test_preallocated_stream_reader_returns_exact_common_chunk():
    class Reader:
        def read_many_sample(self, buffer, **_kwargs):
            buffer[:] = ((1.0, 2.0, 3.0),)
            return 3

    controller = object.__new__(NidaqSignalStreamController)
    controller._analog_readers = {"Dev1": Reader()}
    controller._analog_buffers = {"Dev1": numpy.empty((1, 3))}
    controller._read_telemetry = {"short_reads": 0}

    values = controller._read_analog("Dev1", object(), 3, 1.0)

    assert values.tolist() == [[1.0, 2.0, 3.0]]
    assert controller._read_telemetry["short_reads"] == 0


def test_preallocated_stream_reader_rejects_short_chunk():
    class Reader:
        def read_many_sample_port_uint32(self, _buffer, **_kwargs):
            return 2

    controller = object.__new__(NidaqSignalStreamController)
    controller._digital_readers = {"Dev1": Reader()}
    controller._digital_buffers = {
        "Dev1": numpy.empty((1, 3), dtype=numpy.uint32)}
    controller._read_telemetry = {"short_reads": 0}

    try:
        controller._read_digital("Dev1", object(), 3, 1.0)
    except RuntimeError as error:
        assert "returned 2 samples" in str(error)
    else:
        raise AssertionError("short read was accepted")
    assert controller._read_telemetry["short_reads"] == 1
