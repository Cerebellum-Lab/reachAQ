"""Where a pulse train leaves the laser's output when its generation ends.

After a finite generation the PXI-6713 holds the last sample it wrote. The
pulse-train waveform added low samples only between pulses, so with no
post-stim it ended on the amplitude, and a single pulse was high samples
alone. The output then stayed at the amplitude until the cleanup's reset,
which comes after the task's stop and close. On christielab10 (2026-10-01,
efe173a2) a 23 ms last pulse measured 28.45-35.0 ms and a 5 ms one
12.0-12.4 ms: each too long by the trace's gap from done to the reset.

The 6713 also refuses a buffer of an odd number of samples per channel times
channels: the one sample that ended B2's 105,740-sample train on the minimum
made it 105,741, and the write failed with -200692 (christielab10,
2026-10-01). So the buffer ends on the minimum and is even.

Nothing here touches a driver or a board (nidaq_daqmx_fake).
"""

import dataclasses

import pytest

from autotrainer.core import NidaqTimingPlan
from autotrainer.device import (
    LaserChannelId,
    LaserOperationState,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import FakeDaqError, FakeDaqmx, rig_lasers


#: The pellet board's STIM line, as the 6713 sees it (PXI_Trig0).
BOARD_STIM = "/PXI1Slot4/PXI_Trig0"
#: Built at the stream's rate, so that each waveform can be written out.
RATE = 10_000.0


@pytest.fixture
def daq(monkeypatch):
    fake = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake)
    return fake


def _synchronized_plan():
    """christielab10's: the 6221 clocks the stream, the 6713 is its output."""
    return NidaqTimingPlan(
        requested_mode="auto",
        resolved_mode="backplane",
        is_valid=True,
        master_device="PXI1Slot5",
        slave_devices=("PXI1Slot4",),
        sample_clock_source="/PXI1Slot5/ai/SampleClock",
        sample_clock_rate_hz=10_000.0,
        start_trigger_source="/PXI1Slot5/ai/StartTrigger",
        hardware_output_devices=("PXI1Slot4",),
        hardware_output_timing_status="declared_not_armed",
    )


def _built(pulse, **channel):
    """The waveform the controller builds for `pulse` at RATE."""
    controller = NidaqLaserController(rig_lasers(**channel))
    try:
        return controller._build_pulse_train_waveform(
            controller.configuration.get_channel(pulse.channel_id), pulse, RATE)
    finally:
        controller.close()


def _train(**fields):
    return LaserPulseTrain(channel_id=LaserChannelId.LASER_1, **fields)


def _last_samples(daq):
    """Each channel's last sample in the pulse's AO buffer, as written."""
    analog = daq.task("laser_sync_pulse_ao")
    buffer, = analog.writes
    per_channel = buffer if len(analog.channels) > 1 else [buffer]
    # The buffer is the generation: as many samples as the task runs for,
    # and an even number of them, which the 6713 requires (-200692).
    samples = analog.timing_kwargs["samps_per_chan"]
    assert {len(channel) for channel in per_channel} == {samples}
    assert samples % 2 == 0
    return [channel[-1] for channel in per_channel]


# ------------------------------------------------------- the waveform builder


def test_an_odd_train_with_no_post_stim_gains_one_sample_at_the_minimum(daq):
    # Three 1 ms pulses at 100 Hz after 0.5 ms of baseline: 215 samples,
    # ending on the last pulse's tenth high sample. One at the minimum ends
    # it there and makes it even.
    waveform = _built(_train(
        amplitude_volts=2.0, duration_ms=1.0, baseline_ms=0.5,
        pulse_count=3, frequency_hz=100.0))

    period = [2.0] * 10 + [0.0] * 90
    assert waveform == [0.0] * 5 + period * 2 + [2.0] * 10 + [0.0]


def test_an_even_train_with_no_post_stim_gains_two_samples_at_the_minimum(daq):
    # A single pulse was high samples alone, so the output held the
    # amplitude from the generation's first sample until the reset. One
    # sample ends it on the minimum, and a second keeps it even.
    assert _built(_train(amplitude_volts=2.0, duration_ms=1.0)) == [2.0] * 10 + [0.0] * 2


def test_an_odd_train_whose_post_stim_ends_it_on_the_minimum_gains_one(daq):
    # Two pulses at 500 Hz and 0.3 ms of post-stim: 33 samples, which the
    # 6713 refuses although they end on the minimum.
    waveform = _built(_train(
        amplitude_volts=2.0, duration_ms=1.0, post_stim_ms=0.3,
        pulse_count=2, frequency_hz=500.0))

    assert waveform == [2.0] * 10 + [0.0] * 10 + [2.0] * 10 + [0.0] * 4


def test_an_even_train_whose_post_stim_ends_it_on_the_minimum_is_unchanged(daq):
    # 0.4 ms of post-stim: 34 samples, ending on the minimum already.
    waveform = _built(_train(
        amplitude_volts=2.0, duration_ms=1.0, post_stim_ms=0.4,
        pulse_count=2, frequency_hz=500.0))

    assert waveform == [2.0] * 10 + [0.0] * 10 + [2.0] * 10 + [0.0] * 4


@pytest.mark.parametrize("duration_ms, samples", [(1.0, 10), (1.1, 12)])
def test_a_train_at_the_minimum_is_only_made_even(daq, duration_ms, samples):
    # The Pulse Builder's default draft is 0 V (bench check B1): every
    # sample is the minimum, so it ends there already. 1.1 ms is 11 samples.
    assert _built(_train(amplitude_volts=0.0, duration_ms=duration_ms)) == [0.0] * samples


def test_a_train_ends_on_a_minimum_that_is_not_zero(daq):
    # On the channel's own minimum, as the samples between its pulses are.
    waveform = _built(
        _train(amplitude_volts=2.0, duration_ms=1.0, pulse_count=2, frequency_hz=500.0),
        minimum_command_volts=0.25)

    assert waveform == [2.0] * 10 + [0.25] * 10 + [2.0] * 10 + [0.25] * 2


# ------------------------------------- what each path writes to the AO task


def test_run_pulse_leaves_the_output_on_the_minimum(daq):
    # Bench check B2's saved profile, 31 x 23 ms at 29 Hz and 2.0 V, run as
    # Run Pulse runs it: waited for, on the output's own 100 kHz clock. Its
    # 105,740 samples and the one that ends it on 0 V were refused by the
    # 6713 (-200692); a second makes them even.
    controller = NidaqLaserController(rig_lasers())

    controller.run_pulse_train(_train(
        amplitude_volts=2.0, duration_ms=23.0, pulse_count=31, frequency_hz=29.0))

    assert _last_samples(daq) == [0.0]
    assert daq.task("laser_sync_pulse_ao").timing_kwargs["samps_per_chan"] == 105_742
    controller.close()


def test_an_armed_pulse_leaves_the_output_on_the_minimum(daq):
    # Armed, then started by software, as Test stim's software route (B3)
    # and a protocol's direct NI route arm it (prepare_pulse_profile). B3's
    # pulse, 5 ms at 10 Hz and 1.0 V, five of its fifty.
    controller = NidaqLaserController(rig_lasers())
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(_train(
            amplitude_volts=1.0, duration_ms=5.0, pulse_count=5, frequency_hz=10.0),),
        wait=False, defer_start=True))
    assert operation.state is LaserOperationState.ARMED

    operation.trigger()

    assert operation.wait(5.0) is LaserOperationState.COMPLETED
    assert _last_samples(daq) == [0.0]
    controller.close()


def test_a_synchronized_pulse_leaves_the_output_on_the_minimum(daq):
    # A trial's pulse, and Test stim on the board STIM route (B4): armed on
    # the STIM, on the input stream's 10 kHz clock.
    controller = NidaqLaserController(
        rig_lasers(trigger_source=BOARD_STIM, trigger_route_source="/PXI1Slot5/PFI0"),
        timing_plan=_synchronized_plan())
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(_train(
            amplitude_volts=2.0, duration_ms=23.0, pulse_count=31, frequency_hz=29.0),),
        trigger_source=BOARD_STIM, wait=False))

    assert operation.wait(5.0) is LaserOperationState.COMPLETED
    assert operation.to_record()["timing_status"]["status"] == "hardware_synchronized"
    assert daq.task("laser_sync_pulse_ao").timing_kwargs["rate"] == 10_000.0
    assert _last_samples(daq) == [0.0]
    controller.close()


def _two_lasers():
    """Both lasers on the 6713, laser 2's minimum not zero."""
    lasers = rig_lasers(trigger_source=BOARD_STIM, trigger_route_source="/PXI1Slot5/PFI0")
    laser_2 = dataclasses.replace(
        lasers.channels[0], channel_id=LaserChannelId.LASER_2,
        analog_output="PXI1Slot4/ao1", diode_input="PXI1Slot5/ai9",
        shutter_output="PXI1Slot5/port0/line5", command_copy_input=None,
        trigger_source="/PXI1Slot4/PXI_Trig2", trigger_route_source="/PXI1Slot5/PFI1",
        minimum_command_volts=0.25)
    return dataclasses.replace(lasers, channels=(lasers.channels[0], laser_2))


@pytest.mark.parametrize(
    "lead_ms, lag_ms", [(0.0, 0.0), (0.5, 0.5), (0.5, 0.0)],
    ids=["no PMT margins", "PMT margins", "an odd PMT lead alone"])
def test_both_lasers_of_a_synchronized_train_line_up_and_end_on_their_minimum(
    daq, lead_ms, lag_ms,
):
    # The shorter channel is padded with its minimum to the longer one's
    # length, and PMT margins pad both, so only the longer channel, with no
    # margins, ended on its amplitude. Padded, both still start together.
    # A 5-sample lead alone made the even waveforms odd: one more sample at
    # each minimum after them. The PMT line runs for as many samples.
    controller = NidaqLaserController(_two_lasers(), timing_plan=_synchronized_plan())
    margins = dict(
        enable_pmt_shutter=lead_ms > 0 or lag_ms > 0,
        pmt_shutter_open_delay_ms=lead_ms, pmt_shutter_close_delay_ms=lag_ms)

    controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(
            _train(amplitude_volts=2.0, duration_ms=1.0, **margins),
            LaserPulseTrain(
                channel_id=LaserChannelId.LASER_2, amplitude_volts=1.0,
                duration_ms=1.0, pulse_count=3, frequency_hz=500.0, **margins),
        ),
        trigger_source=BOARD_STIM, timeout_seconds=5.0))

    laser_1, laser_2 = daq.task("laser_sync_pulse_ao").writes[0]
    lead, lag = (int(ms * RATE / 1000.0) for ms in (lead_ms, lag_ms))
    even = (lead + lag) % 2
    train_2 = ([1.0] * 10 + [0.25] * 10) * 2 + [1.0] * 10 + [0.25] * 2
    assert laser_2 == [0.25] * lead + train_2 + [0.25] * (lag + even)
    assert laser_1 == (
        [0.0] * lead + [2.0] * 10 + [0.0] * (len(train_2) - 10 + lag + even))
    assert _last_samples(daq) == [0.0, 0.25]
    if margins["enable_pmt_shutter"]:
        pmt = daq.task("laser_pmt_shutter_do")
        pmt_samples, = pmt.writes
        assert len(pmt_samples) == pmt.timing_kwargs["samps_per_chan"] == len(laser_1)
    controller.close()


# ------------------------------------------- the pulse's clocked digital lines
#
# They run on the output's clock for as many samples, and hold their last one
# after it as the output does (NI's finite generation; christielab10 configures
# none of these lines, so this is not measured there).


@pytest.mark.parametrize("lag_ms, parity", [(0.03, 0), (0.0, 1)],
                         ids=["lead and lag", "an odd lead alone"])
def test_the_pmt_shutter_line_ends_low_after_its_close_lag(daq, lag_ms, parity):
    # It was high for the whole output, so it stayed high after it until the
    # cleanup wrote it low, last of everything, and the close lag was the
    # host's. A 0.05 ms lead and a 0.1 ms pulse at 100 kHz, two samples at
    # 0 V that end it there and keep it even, then the lag: low on the
    # output's last sample. The 5-sample lead alone made the output odd,
    # which the 6713 refuses: one more sample at 0 V.
    controller = NidaqLaserController(rig_lasers())

    controller.run_pulse_train(_train(
        amplitude_volts=2.0, duration_ms=0.1, enable_pmt_shutter=True,
        pmt_shutter_open_delay_ms=0.05, pmt_shutter_close_delay_ms=lag_ms))

    analog, = daq.task("laser_sync_pulse_ao").writes
    pmt, = daq.task("laser_pmt_shutter_do").writes
    after = 2 + int(lag_ms * 100) + parity
    assert analog == [0.0] * 5 + [2.0] * 10 + [0.0] * after
    assert pmt == [True] * (5 + 10 + after - 1) + [False]
    controller.close()


#: Each trigger line: its configuration field, the pulse's field, its task.
TRIGGER_LINES = {
    "trigger_output": ("emit_trigger_output", "laser_1_trigger_do"),
    "timing_trigger_output": ("emit_timing_trigger_output", "laser_1_timing_trigger_do"),
}


def _trigger_line_written(daq, output, **pulse):
    emit, task = TRIGGER_LINES[output]
    controller = NidaqLaserController(rig_lasers(**{output: "PXI1Slot5/port0/line7"}))
    controller.run_pulse_train(_train(amplitude_volts=2.0, **pulse, **{emit: True}))
    controller.close()
    line, = daq.task(task).writes
    return line


@pytest.mark.parametrize("output", sorted(TRIGGER_LINES))
def test_a_trigger_line_longer_than_the_output_ends_low(daq, output):
    # Its default 1 ms beside a 0.5 ms pulse filled the output and ended
    # high, and nothing writes these lines low: it stayed high after the
    # pulse, and the next pulse's trigger had no rising edge. It is cut to
    # all but the output's last sample, of 52.
    assert _trigger_line_written(daq, output, duration_ms=0.5) == [True] * 51 + [False]


@pytest.mark.parametrize("output", sorted(TRIGGER_LINES))
@pytest.mark.parametrize("duration_ms", [1.0, 2.0])
def test_a_trigger_line_the_output_outlasts_keeps_its_width(daq, output, duration_ms):
    # 1 ms at 100 kHz, then low to the output's end: the pulse and its two
    # samples at 0 V. Beside a 1 ms pulse it filled the output until the
    # output gained those samples.
    pulse_samples = int(duration_ms * 100)
    assert _trigger_line_written(daq, output, duration_ms=duration_ms) == (
        [True] * 100 + [False] * (pulse_samples + 2 - 100))


# ------------------------------------------------------ the fake's own rule


def _timed_output(daq, channels):
    task = daq.Task("probe_ao")
    for channel in channels:
        task.ao_channels.add_ao_voltage_chan(channel)
    task.timing.cfg_samp_clk_timing(rate=RATE, sample_mode="finite", samps_per_chan=3)
    return task


def test_the_fake_refuses_an_odd_ao_buffer_as_the_6713_does(daq):
    # A guard on the stand-in, not on the controller: christielab10's
    # PXI-6713 refused a 105,741-sample buffer with -200692 (2026-10-01), so
    # an odd one fails here as it does there, and writes nothing.
    task = _timed_output(daq, ["PXI1Slot4/ao0"])

    with pytest.raises(FakeDaqError) as refused:
        task.write([0.0] * 3)

    assert refused.value.error_code == -200692
    assert str(refused.value) == (
        "DAQmx -200692: Number of samples per channel to write multiplied by "
        "the number of channels in the task cannot be an odd number for this device.")
    assert task.writes == []
    # Samples per channel times channels: two channels of three are six.
    _timed_output(daq, ["PXI1Slot4/ao0", "PXI1Slot4/ao1"]).write([[0.0] * 3] * 2)
    task.write([0.0] * 4)


def test_the_fake_takes_an_on_demand_ao_sample(daq):
    # Not timed: the command reset's one sample, which the 6713 takes.
    task = daq.Task("laser_1_manual_ao")
    task.ao_channels.add_ao_voltage_chan("PXI1Slot4/ao0")

    task.write(0.0, auto_start=True)

    assert task.writes == [0.0]
