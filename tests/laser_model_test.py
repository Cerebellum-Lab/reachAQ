from types import SimpleNamespace

from autotrainer.device import LaserChannelConfiguration, LaserSystemConfiguration
from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


class _Controller:
    def __init__(self):
        self.configuration = LaserSystemConfiguration.from_channels((
            LaserChannelConfiguration(
                channel_id=1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line0",
            ),
        ), backend="null", hardware_timed=True, sample_rate_hz=10_000)
        self.pulse = None
        self.operation = object()

    def run_synchronized_pulse_train(self, pulse):
        self.pulse = pulse
        return self.operation

    def close_all_shutters(self):
        pass

    def close(self):
        pass


def _recipe():
    return SimpleNamespace(
        session_id="session001", session_generation=3,
        protocol_id="training", protocol_revision=2,
        logical_trial_id=4, attempt_id=1, operation_id="trial-op",
    )


def test_prepare_hardware_laser_profile_freezes_context_and_trigger():
    controller = _Controller()
    model = LaserModel(controller)
    profile = LaserPulseProfile(
        "pulse", 1, 1, 2.5, 5, pulse_count=2, frequency_hz=20,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PFI0",
    )

    assert model.prepare_pulse_profile(profile, _recipe()) is controller.operation
    assert controller.pulse.trigger_source == "/Dev1/PFI0"
    assert not controller.pulse.defer_start
    assert controller.pulse.operation_context["trial_operation_id"] == "trial-op"


def test_prepare_direct_laser_profile_defers_start():
    controller = _Controller()
    model = LaserModel(controller)
    profile = LaserPulseProfile(
        "pulse", 1, 1, 2.5, 5,
        trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
    )

    model.prepare_pulse_profile(profile, _recipe())
    assert controller.pulse.trigger_source is None
    assert controller.pulse.defer_start
