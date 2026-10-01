"""Where a pulse train leaves the laser's output when its generation ends.

After a finite generation the PXI-6713 holds the last sample it wrote. The
pulse-train waveform added low samples only between pulses, so with no
post-stim it ended on the amplitude, and a single pulse was high samples
alone. The output then stayed at the amplitude until the cleanup's reset,
which comes after the task's stop and close. On christielab10 (2026-10-01,
efe173a2) a 23 ms last pulse measured 28.45-35.0 ms and a 5 ms one
12.0-12.4 ms: each too long by the trace's gap from done to the reset.

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

from nidaq_daqmx_fake import FakeDaqmx, rig_lasers


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
    # The buffer is the generation: as many samples as the task runs for.
    assert {len(samples) for samples in per_channel} == {
        analog.timing_kwargs["samps_per_chan"]}
    return [samples[-1] for samples in per_channel]


# ------------------------------------------------------- the waveform builder


def test_a_train_with_no_post_stim_ends_on_the_minimum(daq):
    # Three 1 ms pulses at 100 Hz after 0.5 ms of baseline: 10 samples high
    # and 90 low each period. It ended on the last pulse's tenth high sample.
    waveform = _built(_train(
        amplitude_volts=2.0, duration_ms=1.0, baseline_ms=0.5,
        pulse_count=3, frequency_hz=100.0))

    period = [2.0] * 10 + [0.0] * 90
    assert waveform == [0.0] * 5 + period * 2 + [2.0] * 10 + [0.0]


def test_a_single_pulse_ends_on_the_minimum(daq):
    # It was high samples alone, so the output held the amplitude from the
    # generation's first sample until the reset.
    assert _built(_train(amplitude_volts=2.0, duration_ms=1.0)) == [2.0] * 10 + [0.0]


def test_a_train_whose_post_stim_ends_it_on_the_minimum_is_unchanged(daq):
    # It ends on the minimum already, so no sample is added to it.
    waveform = _built(_train(
        amplitude_volts=2.0, duration_ms=1.0, post_stim_ms=0.3,
        pulse_count=2, frequency_hz=500.0))

    assert waveform == [2.0] * 10 + [0.0] * 10 + [2.0] * 10 + [0.0] * 3


def test_a_train_at_the_minimum_is_unchanged(daq):
    # The Pulse Builder's default draft is 0 V (bench check B1): every
    # sample is the minimum, so it ends there already.
    assert _built(_train(amplitude_volts=0.0, duration_ms=1.0)) == [0.0] * 10


def test_a_train_ends_on_a_minimum_that_is_not_zero(daq):
    # On the channel's own minimum, as the samples between its pulses are.
    waveform = _built(
        _train(amplitude_volts=2.0, duration_ms=1.0, pulse_count=2, frequency_hz=500.0),
        minimum_command_volts=0.25)

    assert waveform == [2.0] * 10 + [0.25] * 10 + [2.0] * 10 + [0.25]


# ------------------------------------- what each path writes to the AO task


def test_run_pulse_leaves_the_output_on_the_minimum(daq):
    # Bench check B2's saved profile, 31 x 23 ms at 29 Hz and 2.0 V, run as
    # Run Pulse runs it: waited for, on the output's own 100 kHz clock.
    controller = NidaqLaserController(rig_lasers())

    controller.run_pulse_train(_train(
        amplitude_volts=2.0, duration_ms=23.0, pulse_count=31, frequency_hz=29.0))

    assert _last_samples(daq) == [0.0]
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


@pytest.mark.parametrize("margin_ms", [0.0, 0.5], ids=["no PMT margins", "PMT margins"])
def test_both_lasers_of_a_synchronized_train_line_up_and_end_on_their_minimum(
    daq, margin_ms,
):
    # The shorter channel is padded with its minimum to the longer one's
    # length, and PMT margins pad both, so only the longer channel, with no
    # margins, ended on its amplitude. Padded, both still start together.
    controller = NidaqLaserController(_two_lasers(), timing_plan=_synchronized_plan())
    margins = dict(
        enable_pmt_shutter=margin_ms > 0,
        pmt_shutter_open_delay_ms=margin_ms, pmt_shutter_close_delay_ms=margin_ms)

    controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(
            _train(amplitude_volts=2.0, duration_ms=1.0, **margins),
            LaserPulseTrain(
                channel_id=LaserChannelId.LASER_2, amplitude_volts=1.0,
                duration_ms=1.0, pulse_count=3, frequency_hz=500.0, **margins),
        ),
        trigger_source=BOARD_STIM, timeout_seconds=5.0))

    laser_1, laser_2 = daq.task("laser_sync_pulse_ao").writes[0]
    margin = int(margin_ms * RATE / 1000.0)
    train_2 = ([1.0] * 10 + [0.25] * 10) * 2 + [1.0] * 10 + [0.25]
    assert laser_2 == [0.25] * margin + train_2 + [0.25] * margin
    assert laser_1 == (
        [0.0] * margin + [2.0] * 10 + [0.0] * (len(train_2) - 10 + margin))
    assert _last_samples(daq) == [0.0, 0.25]
    controller.close()
