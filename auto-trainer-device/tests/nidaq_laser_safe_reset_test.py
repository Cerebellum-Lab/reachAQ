"""The laser controller drives each laser safe even when one of its writes fails.

close() and a failed open are what the laser process's safe trip and its reset
child run. A shutter or auxiliary write that is refused must not stop the
command reset of the same laser, and a laser that fails to open must not stop
the lasers after it from being driven safe.

Nothing here touches a driver or a board (nidaq_daqmx_fake).
"""

import pytest

from autotrainer.device import (
    LaserChannelConfiguration,
    LaserChannelId,
    LaserSystemConfiguration,
    NidaqLaserController,
)
from autotrainer.device import nidaq_laser

from nidaq_daqmx_fake import FakeDaqmx, rig_lasers


@pytest.fixture
def daq(monkeypatch):
    fake = FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake)
    return fake


def _two_lasers():
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="PXI1Slot4/ao0",
                diode_input="PXI1Slot5/ai8",
                shutter_output="PXI1Slot5/port0/line4",
            ),
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_2,
                analog_output="PXI1Slot4/ao1",
                diode_input="PXI1Slot5/ai9",
                shutter_output="PXI1Slot5/port0/line5",
            ),
        ),
        backend="nidaq",
        hardware_timed=True,
        sample_rate_hz=100_000.0,
    )


def test_close_resets_the_command_when_the_shutter_write_fails(daq):
    controller = NidaqLaserController(rig_lasers())
    controller.set_command_voltage(LaserChannelId.LASER_1, 3.0)
    since = len(daq.writes)
    daq.failing_write = "laser_1_shutter"
    try:
        with pytest.raises(RuntimeError, match="shutter"):
            controller.close()
    finally:
        daq.failing_write = None

    # The command is the only barrier left once the shutter cannot close.
    assert daq.writes_to("PXI1Slot4/ao0", since=since)[-1:] == [0.0]


def test_a_laser_that_fails_to_open_does_not_stop_the_next_one_being_driven_safe(daq):
    daq.failing_write = "laser_1_shutter"
    try:
        with pytest.raises(Exception):
            NidaqLaserController(_two_lasers())
    finally:
        daq.failing_write = None

    # Laser 2 was never opened, but the reset child relies on this open to
    # put every configured laser back to its minimum and close its shutter.
    assert daq.writes_to("PXI1Slot4/ao1")[-1:] == [0.0]
    assert daq.writes_to("PXI1Slot5/port0/line5")[-1:] == [False]
    assert not [task for task in daq.tasks if not task.closed]
