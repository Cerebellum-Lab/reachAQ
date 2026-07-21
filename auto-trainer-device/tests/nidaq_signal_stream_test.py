from types import SimpleNamespace

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from autotrainer.device import nidaq_signal_stream
from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController


class _FakeTiming:
    def __init__(self, task):
        self._task = task

    def cfg_samp_clk_timing(self, **kwargs):
        self._task.timing_configuration = kwargs

    def cfg_implicit_timing(self, **kwargs):
        self._task.implicit_timing_configuration = kwargs


class _FakeDigitalChannels:
    def __init__(self, task):
        self._task = task

    def add_di_chan(self, physical_channel, *, line_grouping):
        self._task.channels.append((physical_channel, line_grouping))


class _FakeCounterChannels:
    def __init__(self, task):
        self._task = task

    def add_co_pulse_chan_freq(self, physical_channel, *, freq):
        self._task.counter_configuration = (physical_channel, freq)


class _FakeTask:
    def __init__(self, name: str):
        self.name = name
        self.channels = []
        self.timing_configuration = None
        self.implicit_timing_configuration = None
        self.counter_configuration = None
        self.timing = _FakeTiming(self)
        self.di_channels = _FakeDigitalChannels(self)
        self.co_channels = _FakeCounterChannels(self)
        self.started = False
        self.closed = False
        self.read_count = 0

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def read(self, *, number_of_samples_per_channel, timeout):
        assert timeout >= 1.0
        values = [
            [index % 2 == 0 for index in range(number_of_samples_per_channel)],
            [index % 2 == 1 for index in range(number_of_samples_per_channel)],
        ]
        self.read_count += number_of_samples_per_channel
        return values


class _FakeNidaqmx:
    constants = SimpleNamespace(
        AcquisitionType=SimpleNamespace(CONTINUOUS="continuous"),
        LineGrouping=SimpleNamespace(CHAN_PER_LINE="per-line"),
    )

    def __init__(self):
        self.tasks = []

    def Task(self, name):
        task = _FakeTask(name)
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
        assert clock_task.counter_configuration == ("Dev1/ctr0", 1000.0)
        assert clock_task.started

        block = controller.read_chunk()

        assert block.sample_count == 3
        assert block.values["barcode"] == (1.0, 0.0, 1.0)
        assert block.values["cam_frames"] == (0.0, 1.0, 0.0)
        assert digital_task.read_count == 3
    finally:
        controller.close()
