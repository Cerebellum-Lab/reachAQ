import queue
import threading
import time
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
        self.operation = SimpleNamespace(trigger_count=0)
        self.operation.trigger = lambda: setattr(
            self.operation, "trigger_count", self.operation.trigger_count + 1
        )

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


def test_direct_trigger_receiver_validates_nonce_and_starts_prepared_operation():
    controller = _Controller()
    model = LaserModel(controller)
    profile = LaserPulseProfile(
        "pulse", 1, 1, 2.5, 5,
        trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
    )
    model.prepare_pulse_profile(profile, _recipe())
    model.bind_direct_trigger_nonce("trial-op", "once")
    trigger_queue = queue.Queue(maxsize=1)
    result_ready = threading.Event()
    results = []
    model.start_direct_trigger_receiver(
        trigger_queue,
        lambda result: (results.append(result), result_ready.set()),
    )
    try:
        trigger_queue.put_nowait({
            "operation_id": "trial-op",
            "session_generation": 3,
            "logical_trial_id": 4,
            "attempt_id": 1,
            "nonce": "once",
            "stim_frame_id": 9,
            "ipc_send_perf_time": time.perf_counter(),
        })
        assert result_ready.wait(1)
        assert results[0]["accepted"]
        assert results[0]["timing_confidence"] == "software_start"
        assert controller.operation.trigger_count == 1
    finally:
        model.stop_direct_trigger_receiver()
