import queue
import threading
import time
from types import SimpleNamespace

import pytest

from autotrainer.device import LaserChannelConfiguration, LaserSystemConfiguration
from tools.acquisition.model.laser_firing import LaserFiring
from tools.acquisition.model.laser_model import LaserModel
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


BOARD = LaserFiring(1, LaserTriggerRoute.HARDWARE_STIM3, "/Dev1/PFI0", 3, 1000)
SOFTWARE = LaserFiring(1, LaserTriggerRoute.DIRECT_NI_SOFTWARE)


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
        self.operation.cancel = lambda: True
        self.operation.to_record = lambda: {
            "timing_status": {
                "status": (
                    "software_start"
                    if self.pulse is not None and self.pulse.defer_start
                    else "hardware_synchronized"
                )
            }
        }

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
        "pulse", 1, 2.5, 5, pulse_count=2, frequency_hz=20,
    )

    assert model.prepare_pulse_profile(profile, BOARD, _recipe()) is controller.operation
    assert controller.pulse.trigger_source == "/Dev1/PFI0"
    assert not controller.pulse.defer_start
    assert controller.pulse.operation_context["trial_operation_id"] == "trial-op"


def test_prepare_direct_laser_profile_defers_start():
    controller = _Controller()
    model = LaserModel(controller)
    profile = LaserPulseProfile("pulse", 1, 2.5, 5)

    model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    assert controller.pulse.trigger_source is None
    assert controller.pulse.defer_start


def test_prepare_direct_profile_accepts_explicit_simulated_timing():
    configuration = LaserSystemConfiguration.from_channels((
        LaserChannelConfiguration(
            channel_id=1,
            analog_output="Dev1/ao0",
            diode_input="Dev1/ai0",
            shutter_output="Dev1/port0/line0",
        ),
    ), backend="null", hardware_timed=True, sample_rate_hz=10_000)
    model = LaserModel()
    model.configure_null(configuration)
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)

    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    operation.trigger()

    assert operation.wait(1).value == "completed"
    assert operation.to_record()["timing_status"]["emulated"] is True


def test_prepare_hardware_laser_rejects_unverified_timing():
    controller = _Controller()
    controller.operation.to_record = lambda: {
        "timing_status": {
            "status": "unsupported",
            "reason": "trigger route was not verified",
        }
    }
    model = LaserModel(controller)
    profile = LaserPulseProfile("pulse", 1, 2.5, 5)

    with pytest.raises(RuntimeError, match="trigger route was not verified"):
        model.prepare_pulse_profile(profile, BOARD, _recipe())


def test_direct_trigger_receiver_validates_nonce_and_starts_prepared_operation():
    controller = _Controller()
    model = LaserModel(controller)
    profile = LaserPulseProfile("pulse", 1, 2.5, 5)
    model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
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


def test_direct_trigger_receiver_restarts_after_acquisition_stop():
    model = LaserModel(_Controller())
    trigger_queue = queue.Queue(maxsize=1)
    observer = lambda _result: None

    model.start_direct_trigger_receiver(trigger_queue, observer)
    first = model._direct_trigger_thread
    model.start_direct_trigger_receiver(trigger_queue, observer)
    assert model._direct_trigger_thread is first

    model.stop_direct_trigger_receiver()
    model.start_direct_trigger_receiver(trigger_queue, observer)
    try:
        assert model._direct_trigger_thread is not first
        assert model._direct_trigger_thread.is_alive()
    finally:
        model.stop_direct_trigger_receiver()


def test_prepare_takes_the_laser_and_terminal_from_the_firing_not_the_profile():
    controller = _Controller()
    model = LaserModel(controller)
    profile = LaserPulseProfile("pulse", 1, 2.5, 5)

    model.prepare_pulse_profile(
        profile, LaserFiring(1, LaserTriggerRoute.HARDWARE_STIM3, "/Dev1/PXI_Trig0", 3), _recipe())

    assert controller.pulse.trigger_source == "/Dev1/PXI_Trig0"
    assert controller.pulse.operation_context["laser_channel_id"] == 1


def _null_hardware_timed_model():
    model = LaserModel()
    model.configure_null(_Controller().configuration)
    return model


def test_closing_the_model_leaves_the_shutters_to_the_controllers_close():
    # The model drove the shutters itself before the controller's close had
    # marked the controller closed: a driver hung in that write held the
    # close before anything was marked, with every cancel and abort behind
    # it. NidaqLaserController.close() closes them once it is marked.
    controller = _Controller()
    calls = []
    controller.close_all_shutters = lambda: calls.append("close_all_shutters")
    controller.close = lambda: calls.append("close")
    model = LaserModel(controller)

    model.close()

    assert calls == ["close"]
    assert not model.is_connected


def test_a_waited_for_pulse_leaves_the_command_at_its_minimum():
    # The last command stayed at the pulse's amplitude after the pulse had
    # returned to baseline, and a close that hung then named that amplitude
    # as what the output may hold.
    from autotrainer.device import LaserChannelId, LaserPulseTrain

    model = _null_hardware_timed_model()

    model.run_pulse_train(LaserPulseTrain(
        channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0))

    assert model.last_command_volts == {1: 0.0}


def test_a_pulse_that_fails_keeps_its_amplitude_as_the_last_command():
    # Its cleanup may not have reset the output, so the amplitude is what
    # the output may hold.
    from autotrainer.device import LaserChannelId, LaserPulseTrain

    model = _null_hardware_timed_model()
    controller = model._controller

    def refused(_pulse_train):
        raise RuntimeError("DAQmx refused to start laser_sync_pulse_ao")

    controller.run_pulse_train = refused
    with pytest.raises(RuntimeError, match="refused"):
        model.run_pulse_train(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0))

    assert model.last_command_volts == {1: 2.5}


def test_the_model_names_what_a_closed_controller_left_running():
    # A controller whose close gave up waiting on a pulse or a ramp says so
    # (NidaqLaserController.work_left_running); the model keeps asking it
    # after the close, until nothing is left.
    controller = _Controller()
    left = ["laser operation 1234"]
    ended = threading.Event()
    controller.work_left_running = lambda: tuple(left)
    controller.wait_for_work_left_running = lambda timeout=None: ended.wait(timeout)
    model = LaserModel(controller)

    model.close()

    assert model.work_left_running_after_close() == ("laser operation 1234",)
    assert model.wait_for_work_left_running_after_close(0.05) is False
    left.clear()
    ended.set()
    assert model.wait_for_work_left_running_after_close(1.0) is True
    assert model.work_left_running_after_close() == ()
