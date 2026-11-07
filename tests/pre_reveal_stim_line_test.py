"""A pre-reveal trial pulses its own laser's board STIM line.

On christielab10 laser 2 is wired to board STIM2 (boardStimLine 2, PFI1 to
PXI_Trig2). The pre-reveal pulse was hard-coded to STIM3, so a laser 2
pre-reveal trial armed laser 2 while the board pulsed laser 1's line, and the
trial failed and was retried (bench check, 2026-10-01).
"""

from unittest import mock

import pytest

from autotrainer.core import SystemCommandKind
from autotrainer.device import (
    CanDevice,
    DeviceApi,
    DigitalOutputs,
    LaserChannelConfiguration,
    LaserSystemConfiguration,
    Target,
)

from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.trial_action import (
    LaserPulseProfile,
    TrialActionCompiler,
    TrialCompileContext,
)
from tools.acquisition.model.trial_protocol_schedule import TrialProtocolRow


# christielab10's wiring: laser 1 on board STIM3, laser 2 on board STIM2.
LASERS = LaserSystemConfiguration.from_channels((
    LaserChannelConfiguration(
        channel_id=1,
        analog_output="Dev4/ao0",
        diode_input="Dev4/ai0",
        shutter_output="Dev4/port0/line0",
        trigger_source="/Dev4/PXI_Trig0",
        board_stim_line=3,
    ),
    LaserChannelConfiguration(
        channel_id=2,
        analog_output="Dev4/ao1",
        diode_input="Dev4/ai1",
        shutter_output="Dev4/port0/line1",
        trigger_source="/Dev4/PXI_Trig2",
        board_stim_line=2,
    ),
))


def _pre_reveal_recipe(channel_id):
    row = TrialProtocolRow(trial_id=1).with_updates({
        "enabled": True,
        "cover_policy": "reveal",
        "laser_profile_id": "pulse",
        "laser_phase": "embedded_in_sequence",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": channel_id,
        "stimulus_assignment": "always",
        "stimulus_trigger": "pre_reveal",
        "pre_reveal_ms": 200,
    })
    compiler = TrialActionCompiler(
        laser_profiles={"pulse": LaserPulseProfile("pulse", 3, 2.5, 5.0)},
        laser_configuration=LASERS,
        dcs_to_motor=lambda values: values,
    )
    return compiler.compile(row, TrialCompileContext(
        session_id="session001",
        session_generation=4,
        protocol_id="p",
        protocol_revision=2,
        logical_trial_id=1,
        attempt_id=1,
        session_seed=42,
        animal_base_dcs=(10.0, 20.0, 30.0),
        lane_offsets_dcs={
            "center": (0.0, 0.0, 0.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
        },
    ))


def _send_data_queued_by_the_app(app_model, recipe):
    """Prepare the trial's cover through the app and SEND; the data queued.

    The pellet machine is the app's real one. Its device is a HardwareModel
    whose queueing is captured, so this is the data the CAN device receives
    and the payload the session's outbound device event records.
    """
    sent = []
    hardware = object.__new__(HardwareModel)
    hardware._device_conn = mock.sentinel.device_conn
    hardware._send_with_token = (
        lambda device, kind, data=None: sent.append((kind, data)) or "token"
    )
    pellet = app_model._behavior.system_machine.pellet
    pellet._pellet_device = hardware

    app_model._configure_protocol_cover("reveal", recipe)
    pellet.send_pellet(force=True)

    (kind, data), = sent
    assert kind is SystemCommandKind.SEND_PELLET
    return data


def _run_on_the_emulated_board(data):
    """Start that SEND on an emulated CAN device; its pulse and its report."""
    operations = []
    device = CanDevice(
        api=DeviceApi(message_callback=lambda kind, data: None),
        force_emulation=True,
        required_targets=(Target.PELLET_DEVICE,),
        operation_callback=lambda *args: operations.append(args),
    )
    interface = device.device_interface
    interface.open()
    interface.pulse_digital_output = mock.Mock(wraps=interface.pulse_digital_output)

    assert device._start_send_pellet_sequence(data)

    return interface.pulse_digital_output, operations


@pytest.mark.parametrize("channel_id, stim_line, output", [
    (2, 2, DigitalOutputs.STIMULUS_3),
    (1, 3, DigitalOutputs.STIMULUS_4),
])
def test_a_pre_reveal_trial_pulses_its_own_lasers_board_line(
    app_model, channel_id, stim_line, output,
):
    recipe = _pre_reveal_recipe(channel_id)
    assert recipe.laser_firing.stim_line == stim_line

    data = _send_data_queued_by_the_app(app_model, recipe)
    pulse, operations = _run_on_the_emulated_board(data)

    pulse.assert_called_once_with(output, 1000)
    kind, reported, *_ = operations[0]
    assert kind is SystemCommandKind.PULSE_DIGITAL_OUTPUT
    assert reported == (int(output.value), 1000)
    # The SEND itself names the line, so the recorded command does too.
    assert data == {"pre_reveal_stimulus": (200, 1000, stim_line)}
