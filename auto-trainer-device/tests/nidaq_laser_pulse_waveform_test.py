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
import itertools

import numpy
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
from autotrainer.device.laser import LaserPulseRefused

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
    # 105,740 samples plus the one that ends it on 0 V, 105,741 in all, were
    # refused by the 6713 (-200692); a second makes them even.
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


# ----------------------------------------- the pulse's buffer, built with numpy
#
# One float64 array, filled once, where the pulse path built a list per
# channel, padded each by list concatenation, and left nidaqmx to run
# numpy.asarray over it inside the write: 4.1 ms at 105,742 samples and
# 19.5 ms at 490,502 on a core of christielab10 (latency-report.md 3.3, 7),
# on the Run Pulse click path. Its values are the list's.


def _listed(channel, pulse_train, sample_rate_hz):
    """The list builder as it stood before the array one, copied verbatim.

    The specification the array builder is held to; the controller's own
    _build_pulse_train_waveform now returns the array's samples as a list,
    so it cannot be its own oracle.
    """
    baseline_samples = nidaq_laser._samples_from_ms(pulse_train.baseline_ms, sample_rate_hz)
    high_samples = max(1, nidaq_laser._samples_from_ms(pulse_train.duration_ms, sample_rate_hz))
    post_stim_samples = nidaq_laser._samples_from_ms(pulse_train.post_stim_ms, sample_rate_hz)
    minimum = channel.minimum_command_volts
    amplitude = pulse_train.amplitude_volts
    waveform = [minimum] * baseline_samples
    if pulse_train.pulse_count == 1:
        waveform.extend([amplitude] * high_samples)
    else:
        period_samples = max(1, int(round(sample_rate_hz / pulse_train.frequency_hz)))
        if high_samples > period_samples:
            raise ValueError(
                f"laser pulse duration {pulse_train.duration_ms} ms exceeds pulse period "
                f"at {pulse_train.frequency_hz} Hz"
            )
        low_samples = period_samples - high_samples
        for pulse_index in range(pulse_train.pulse_count):
            waveform.extend([amplitude] * high_samples)
            if pulse_index < pulse_train.pulse_count - 1:
                waveform.extend([minimum] * low_samples)
    waveform.extend([minimum] * post_stim_samples)
    if not waveform:
        raise ValueError("laser pulse train waveform is empty")
    if waveform[-1] != minimum:
        waveform.append(minimum)
    if len(waveform) % 2:
        waveform.append(minimum)
    return waveform


def _listed_buffers(channels, pulses, sample_rate_hz, lead_ms, lag_ms):
    """Each channel's whole AO buffer as the list path wrote it: margins, pads."""
    waveforms = [
        _listed(channel, pulse, sample_rate_hz) for channel, pulse in zip(channels, pulses)]
    pre = nidaq_laser._samples_from_ms(lead_ms, sample_rate_hz)
    post = nidaq_laser._samples_from_ms(lag_ms, sample_rate_hz)
    longest = max(len(waveform) for waveform in waveforms)
    parity = (pre + longest + post) % 2
    return [
        [channel.minimum_command_volts] * pre
        + waveform
        + [channel.minimum_command_volts] * (longest - len(waveform) + post + parity)
        for channel, waveform in zip(channels, waveforms)
    ]


def _outcome(build):
    """What a builder gives, or the refusal it raises."""
    try:
        return build()
    except ValueError as error:
        return ("refused", str(error))


def test_the_array_builder_matches_the_list_rules():
    # Every sample the list builder gave, and its refusals, over a grid that
    # takes in the pads (odd and even, ending on the amplitude and not), a
    # post-stim, a baseline, a minimum that is not zero, an amplitude that is
    # the minimum, a pulse the period cannot hold, and the 10 kHz stream's
    # rate as well as the output's own. 29 Hz is a period that is not a whole
    # number of samples (344.8 at 10 kHz), which rounds.
    channels = {
        minimum: rig_lasers(minimum_command_volts=minimum).channels[0] for minimum in (0.0, 0.5)}
    trains = [(1, None)] + [
        (count, hz) for count in (2, 3, 7) for hz in (10.0, 29.0, 100.0, 500.0)]
    grid = itertools.product(
        channels, (False, True), (10_000.0, 100_000.0), trains,
        (0.01, 1.0, 5.0, 23.0), (0.0, 0.5), (0.0, 0.3, 0.4))
    ran = refused = 0
    for minimum, at_minimum, rate, (count, frequency), duration, baseline, post_stim in grid:
        channel = channels[minimum]
        pulse = _train(
            amplitude_volts=minimum if at_minimum else 2.0, duration_ms=duration,
            baseline_ms=baseline, post_stim_ms=post_stim,
            pulse_count=count, frequency_hz=frequency)
        where = (minimum, at_minimum, rate, count, frequency, duration, baseline, post_stim)
        expected = _outcome(lambda: _listed(channel, pulse, rate))
        samples = _outcome(lambda: nidaq_laser._pulse_train_samples(channel, pulse, rate))
        if isinstance(samples, numpy.ndarray):
            assert samples.dtype == numpy.float64, where
            assert samples.ndim == 1, where
            samples = samples.tolist()
        assert samples == expected, where
        # The list builder, which the pulse path no longer calls, is still the
        # list it was, and callable unbound as the bench harness calls it.
        legacy = _outcome(lambda: NidaqLaserController._build_pulse_train_waveform(
            None, channel, pulse, rate))
        assert legacy == expected, where
        if isinstance(expected, list):
            assert isinstance(legacy, list), where
            ran += 1
        else:
            refused += 1
    # Neither side of the grid is empty: 2,064 build and 432 are refused.
    assert (ran, refused) == (2064, 432)


@pytest.mark.parametrize("rate", [10_000.0, 100_000.0])
@pytest.mark.parametrize("post_stim_ms", [0.0, 0.3])
@pytest.mark.parametrize("duration_ms", [0.1, 1.0])
def test_a_long_train_is_laid_out_pulse_for_pulse(rate, post_stim_ms, duration_ms):
    # A thousand pulses, which the grid above does not reach: the first 999
    # are written in one assignment, and the last on its own.
    channel = rig_lasers(minimum_command_volts=0.5).channels[0]
    pulse = _train(
        amplitude_volts=2.0, duration_ms=duration_ms, baseline_ms=0.5,
        post_stim_ms=post_stim_ms, pulse_count=1000, frequency_hz=500.0)

    samples = nidaq_laser._pulse_train_samples(channel, pulse, rate)

    assert samples.tolist() == _listed(channel, pulse, rate)


#: Each representative profile: its lasers, the timing plan and trigger it runs
#: with, its pulses' fields, and its PMT margins in ms. The first three are
#: christielab10's: Run Pulse's saved profile (bench check B2), B3's burst
#: and a trial's pulse on the input stream's clock (B4).
_PROFILES = {
    "laser1 Run Pulse, 31 x 23 ms at 29 Hz, 100 kHz": dict(
        pulses=(dict(amplitude_volts=2.0, duration_ms=23.0, pulse_count=31, frequency_hz=29.0),),
        samples=105_742, rate=100_000.0),
    "a 100 kHz burst, 50 x 5 ms at 10 Hz": dict(
        pulses=(dict(amplitude_volts=1.0, duration_ms=5.0, pulse_count=50, frequency_hz=10.0),),
        samples=490_502, rate=100_000.0),
    "a 10 kHz stream-clocked train": dict(
        pulses=(dict(amplitude_volts=2.0, duration_ms=23.0, pulse_count=31, frequency_hz=29.0),),
        samples=10_582, rate=10_000.0, stream=True),
    "PMT margins, both even": dict(
        pulses=(dict(amplitude_volts=2.0, duration_ms=1.0, pulse_count=3, frequency_hz=500.0),),
        samples=602, rate=100_000.0, lead_ms=0.5, lag_ms=0.5),
    "an odd PMT lead alone": dict(
        pulses=(dict(amplitude_volts=2.0, duration_ms=1.0, pulse_count=3, frequency_hz=500.0),),
        samples=508, rate=100_000.0, lead_ms=0.05),
    "two lasers with PMT margins, minima 0 and 0.25 V": dict(
        pulses=(dict(amplitude_volts=2.0, duration_ms=1.0),
                dict(amplitude_volts=1.0, duration_ms=1.0, pulse_count=3, frequency_hz=500.0,
                     channel_id=LaserChannelId.LASER_2)),
        samples=62, rate=10_000.0, lead_ms=0.5, lag_ms=0.5, stream=True, two=True),
}


def _fire(daq, pulses, *, lasers=None, stream=False, lead_ms=0.0, lag_ms=0.0):
    """Run `pulses`, each a dict of LaserPulseTrain fields, as the pulse path does.

    Returns the AO task it made, the pulses and their channels. Laser 1 is the
    channel a pulse names none for.
    """
    margins = dict(
        enable_pmt_shutter=lead_ms > 0 or lag_ms > 0,
        pmt_shutter_open_delay_ms=lead_ms, pmt_shutter_close_delay_ms=lag_ms)
    pulses = tuple(
        LaserPulseTrain(**{"channel_id": LaserChannelId.LASER_1, **fields, **margins})
        for fields in pulses)
    controller = NidaqLaserController(
        lasers or rig_lasers(), timing_plan=_synchronized_plan() if stream else None)
    try:
        channels = [controller.configuration.get_channel(pulse.channel_id) for pulse in pulses]
        controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
            pulse_trains=pulses, trigger_source=BOARD_STIM if stream else None,
            timeout_seconds=5.0))
    finally:
        controller.close()
    return daq.task("laser_sync_pulse_ao"), pulses, channels


@pytest.mark.parametrize("name", sorted(_PROFILES))
def test_the_buffer_is_the_list_paths_buffer_element_for_element(daq, name):
    # What the output is given is what the list path gave it, margins, pads
    # and all: the same samples, in the same order, for the profiles the rig
    # runs and for the PMT margins that shift them. Compared as the lists
    # the fake records, so that it held on the list path as well.
    profile = _PROFILES[name]
    lead_ms, lag_ms = profile.get("lead_ms", 0.0), profile.get("lag_ms", 0.0)
    lasers = None
    if profile.get("two"):
        lasers = _two_lasers()
    elif profile.get("stream"):
        lasers = rig_lasers(trigger_source=BOARD_STIM, trigger_route_source="/PXI1Slot5/PFI0")

    ao, pulses, channels = _fire(
        daq, profile["pulses"], lasers=lasers, stream=profile.get("stream", False),
        lead_ms=lead_ms, lag_ms=lag_ms)

    expected = _listed_buffers(channels, pulses, profile["rate"], lead_ms, lag_ms)
    assert ao.timing_kwargs["rate"] == profile["rate"]
    assert ao.timing_kwargs["samps_per_chan"] == profile["samples"] == len(expected[0])
    written, = ao.writes
    assert written == (expected if len(expected) > 1 else expected[0])


@pytest.mark.parametrize("two", [False, True], ids=["one laser", "two lasers"])
def test_the_ao_write_is_one_float64_array(daq, two):
    # The array nidaqmx is given as it is: float64, in one piece, one write.
    # A (channels, samples) array for two lasers; a transposed one would
    # interleave them. One laser is its row.
    profile = _PROFILES["two lasers with PMT margins, minima 0 and 0.25 V"]
    one_laser = _PROFILES["laser1 Run Pulse, 31 x 23 ms at 29 Hz, 100 kHz"]
    ao, _pulses, _channels = (
        _fire(daq, profile["pulses"], lasers=_two_lasers(), stream=True,
              lead_ms=0.5, lag_ms=0.5)
        if two else _fire(daq, one_laser["pulses"]))

    written, = ao.write_types
    samples = ao.timing_kwargs["samps_per_chan"]
    assert isinstance(written, numpy.ndarray)
    assert written.dtype == numpy.float64
    assert written.flags["C_CONTIGUOUS"]
    assert written.shape == ((2, samples) if two else (samples,))
    assert samples == (62 if two else 105_742)


#: A sample outside the range, as a function of the channel: past its maximum,
#: below its minimum, and not a number, which fails every comparison.
_OUT_OF_RANGE = {
    "above the maximum": lambda channel: channel.maximum_command_volts + 0.1,
    "below the minimum": lambda channel: channel.minimum_command_volts - 0.1,
    "not a number": lambda channel: float("nan"),
}


@pytest.mark.parametrize("wait", [True, False], ids=["waited for", "not waited for"])
@pytest.mark.parametrize("sample", sorted(_OUT_OF_RANGE))
def test_a_buffer_outside_the_lasers_range_is_refused_before_any_task(
    daq, monkeypatch, wait, sample,
):
    # Defence in depth: the amplitude is refused before the operation is made
    # (run_synchronized_pulse_train), so the buffer never leaves the range by
    # that. A builder that lets a sample out of it is stopped before a task
    # exists, so the output keeps the command reset's value, and nothing is
    # written to the board.
    monkeypatch.setattr(
        nidaq_laser, "_pulse_train_samples",
        lambda channel, pulse_train, sample_rate_hz: numpy.array(
            [_OUT_OF_RANGE[sample](channel)]))
    controller = NidaqLaserController(rig_lasers())
    tasks_before, writes_before = len(daq.tasks), len(daq.writes)
    try:
        with pytest.raises(LaserPulseRefused, match=r"outside the configured range 0.0..5.0 V"):
            controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
                pulse_trains=(_train(amplitude_volts=2.0, duration_ms=1.0),),
                wait=wait, timeout_seconds=5.0))

        assert daq.tasks[tasks_before:] == []
        assert daq.writes[writes_before:] == []
        assert controller._live_operations == {}
    finally:
        controller.close()


def _narrow_second_laser():
    """Both lasers on the 6713, laser 2's range 0.25..1.0 V and laser 1's 0..5 V."""
    lasers = _two_lasers()
    laser_2 = dataclasses.replace(lasers.channels[1], maximum_command_volts=1.0)
    return dataclasses.replace(lasers, channels=(lasers.channels[0], laser_2))


def test_each_row_of_the_buffer_is_held_to_its_own_lasers_range(daq, monkeypatch):
    # Laser 1 at 2 V is outside laser 2's range, and laser 2 at its maximum
    # is inside its own: the buffer is taken, the second at exactly its
    # maximum. A sample 0.1 V over laser 2's maximum is refused although it
    # is well inside laser 1's.
    pulses = (
        dict(channel_id=LaserChannelId.LASER_1, amplitude_volts=2.0, duration_ms=1.0),
        dict(channel_id=LaserChannelId.LASER_2, amplitude_volts=1.0, duration_ms=1.0))
    ao, _pulses, _channels = _fire(daq, pulses, lasers=_narrow_second_laser(), stream=True)
    laser_1, laser_2 = ao.write_types[-1]
    assert laser_1.max() == 2.0 and laser_2.max() == 1.0

    real = nidaq_laser._pulse_train_samples

    def laser_2_over(channel, pulse_train, sample_rate_hz):
        samples = real(channel, pulse_train, sample_rate_hz)
        if channel.channel_id is LaserChannelId.LASER_2:
            samples[0] = channel.maximum_command_volts + 0.1
        return samples

    monkeypatch.setattr(nidaq_laser, "_pulse_train_samples", laser_2_over)
    tasks_before = len(daq.tasks)
    with pytest.raises(LaserPulseRefused, match=r"laser channel 2 .*0.25..1.0 V"):
        _fire(daq, pulses, lasers=_narrow_second_laser(), stream=True)
    assert [task for task in daq.tasks[tasks_before:]
            if task.label == "laser_sync_pulse_ao"] == []


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


def test_the_fake_counts_the_samples_of_a_numpy_array(daq):
    # The rule is on samples per channel times channels. An array was not a
    # list, so it counted as the one sample of a scalar: every array was
    # odd, and a numpy pulse buffer would have been refused whatever its
    # length.
    task = _timed_output(daq, ["PXI1Slot4/ao0"])

    with pytest.raises(FakeDaqError) as refused:
        task.write(numpy.zeros(3))

    assert refused.value.error_code == -200692
    assert task.writes == []
    task.write(numpy.zeros(4))
    assert len(task.writes) == 1
    # Two channels of three samples are six, which it takes; three channels
    # of three are nine, which it refuses, lists and arrays alike.
    _timed_output(daq, ["PXI1Slot4/ao0", "PXI1Slot4/ao1"]).write(numpy.zeros((2, 3)))
    three = _timed_output(daq, ["PXI1Slot4/ao0", "PXI1Slot4/ao1", "PXI1Slot4/ao2"])
    with pytest.raises(FakeDaqError) as refused:
        three.write(numpy.zeros((3, 3)))
    assert refused.value.error_code == -200692
    with pytest.raises(FakeDaqError):
        three.write([[0.0] * 3] * 3)
    assert three.writes == []


def test_the_fake_records_an_array_as_lists_and_keeps_what_was_passed(daq):
    # What the tests already compare against are lists, so a written array is
    # recorded as one, wherever the fake keeps writes; the object passed is
    # kept apart, for a test of what the driver is given.
    one = _timed_output(daq, ["PXI1Slot4/ao0"])
    array = numpy.array([0.0, 2.0, 2.0, 0.0])
    one.write(array)
    two = _timed_output(daq, ["PXI1Slot4/ao1", "PXI1Slot4/ao2"])
    block = numpy.array([[0.0, 1.0], [0.25, 0.5]])
    two.write(block)

    assert one.writes == [[0.0, 2.0, 2.0, 0.0]]
    assert isinstance(one.writes[0], list)
    assert two.writes == [[[0.0, 1.0], [0.25, 0.5]]]
    assert isinstance(two.writes[0][0], list)
    assert [write.data for write in daq.writes] == one.writes + two.writes
    assert daq.writes_to("PXI1Slot4/ao0") == one.writes
    assert one.write_types[-1] is array
    assert two.write_types[-1] is block
    # A list is passed on as it is.
    listed = [0.0, 1.0]
    one.write(listed)
    assert one.writes[-1] == listed
    assert one.write_types[-1] is listed


# nidaqmx 1.6.0's Task.write (read on christielab10, task/_task.py) refuses an
# analog output write by its layout before the board is reached. The fake
# mirrors those rules, so that a layout the driver would refuse is not taken.


def test_the_fake_refuses_data_of_the_wrong_shape_for_the_tasks_channels(daq):
    # One channel takes a 1-D array, and several take (channels, samples), with
    # as many rows as the task has channels: anything else is -200524. A (1, 6)
    # array on one channel, a transposed (6, 2) one or a (3, 6) one on two
    # channels, and lists laid out the same wrong ways, each raise it, and
    # write nothing, before the odd-count rule is reached.
    one = _timed_output(daq, ["PXI1Slot4/ao0"])
    two = _timed_output(daq, ["PXI1Slot4/ao1", "PXI1Slot4/ao2"])
    refused = [
        (one, numpy.zeros((1, 6))),
        (one, [[0.0] * 6]),
        (two, numpy.zeros((6, 2))),
        (two, numpy.zeros((3, 6))),
        (two, numpy.zeros(6)),
        (two, [[0.0] * 4] * 3),
    ]
    for task, data in refused:
        with pytest.raises(FakeDaqError) as error:
            task.write(data)
        assert error.value.error_code == -200524, data
    assert one.writes == two.writes == []
    # The layouts it takes: a row for one channel, (channels, samples) for two.
    one.write(numpy.zeros(6))
    two.write(numpy.zeros((2, 6)))
    two.write([[0.0] * 4] * 2)
    assert len(one.writes) == 1 and len(two.writes) == 2


def test_the_fake_refuses_an_array_that_is_not_c_contiguous(daq):
    # The driver's ctypes argument takes a C-contiguous array only: a strided
    # view or a transposed one raises TypeError. Each of these has an even
    # number of samples and the right shape, so only the layout is wrong.
    one = _timed_output(daq, ["PXI1Slot4/ao0"])
    two = _timed_output(daq, ["PXI1Slot4/ao1", "PXI1Slot4/ao2"])
    strided = [
        (one, numpy.zeros(8)[::2]),
        (two, numpy.zeros((2, 8))[:, ::2]),
        (two, numpy.zeros((4, 2)).T),
    ]
    for task, data in strided:
        assert data.shape[-1] == 4 and not data.flags["C_CONTIGUOUS"]
        with pytest.raises(TypeError, match="C_CONTIGUOUS"):
            task.write(data)
    assert one.writes == two.writes == []
    # A slice that is one piece is taken.
    one.write(numpy.zeros(8)[:4])
    assert len(one.writes) == 1
