import json
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


def test_a_trial_pulse_leaves_the_command_at_its_minimum_once_it_completes():
    # Only a waited-for pulse recorded its end: a trial's, armed and run on
    # its own thread, left its amplitude as the last command, and a close
    # that hung after the trial named it as what the output may hold.
    model = _null_hardware_timed_model()
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)

    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    assert model.last_command_volts == {1: 2.5}
    operation.trigger()
    assert operation.wait(1).value == "completed"

    deadline = time.monotonic() + 2.0
    while model.last_command_volts != {1: 0.0} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert model.last_command_volts == {1: 0.0}


def test_a_trial_pulse_that_is_cancelled_keeps_its_amplitude():
    model = _null_hardware_timed_model()
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)

    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    operation.cancel()
    operation.wait(1)
    time.sleep(0.05)

    assert model.last_command_volts == {1: 2.5}


def test_a_pulse_that_ends_after_a_newer_command_leaves_that_command():
    # The end of a pulse is its own: once a newer command has been recorded,
    # the pulse's completion does not overwrite it with the minimum.
    from autotrainer.device import LaserChannelId

    model = _null_hardware_timed_model()
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)
    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    model._record_command(LaserChannelId.LASER_1, 1.5)

    operation.trigger()
    operation.wait(1)
    time.sleep(0.05)

    assert model.last_command_volts == {1: 1.5}


def test_a_command_recorded_as_a_pulse_ends_is_not_overwritten():
    # The completion looked for a newer command, then recorded the minimum,
    # as two steps: a command recorded between them was overwritten.
    from autotrainer.device import LaserChannelId

    model = _null_hardware_timed_model()
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)
    operation = model.prepare_pulse_profile(profile, SOFTWARE, _recipe())
    in_reset, go_on = threading.Event(), threading.Event()
    record_baseline = model._record_baseline

    def held(channel_ids):
        channel_ids = tuple(channel_ids)  # the look, before the record
        in_reset.set()
        go_on.wait(5.0)
        record_baseline(channel_ids)

    model._record_baseline = held
    operation.trigger()
    assert in_reset.wait(5.0)
    newer = threading.Thread(
        target=model._record_command, args=(LaserChannelId.LASER_1, 1.5), daemon=True)
    newer.start()
    newer.join(0.2)
    go_on.set()
    newer.join(5.0)
    operation.wait(1)
    time.sleep(0.05)

    assert model.last_command_volts == {1: 1.5}


def test_test_stims_pulse_says_it_is_test_stim():
    # Its recipe claims no trial, and trial 0 is what its number reads.
    from tools.acquisition.model.stim_bench_test import BenchRecipe

    model = _null_hardware_timed_model()
    profile = LaserPulseProfile("pulse", 1, 2.5, 1)

    operation = model.prepare_pulse_profile(profile, SOFTWARE, BenchRecipe())

    assert operation.context["operation_label"] == "Test stim"
    operation.cancel()


def test_the_models_nidaq_controllers_take_the_streams_terminal_config(monkeypatch):
    # The laser's own inputs are referenced as the stream references its
    # inputs: the value is passed to each NI-DAQ controller the model opens.
    from tools.acquisition.model import laser_model as laser_model_module

    made = []

    class _Controller:
        def __init__(self, configuration, **kwargs):
            made.append(kwargs.get("analog_terminal_config"))
            self.configuration = configuration

        def close(self):
            pass

        def work_left_running(self):
            return ()

    monkeypatch.setattr(laser_model_module, "NidaqLaserController", _Controller)
    configuration = LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(
            channel_id=1, analog_output="Dev1/ao0", diode_input="Dev1/ai0",
            shutter_output="Dev1/port0/line2"),),
        backend="nidaq")
    model = LaserModel()

    model.open_controller(configuration, analog_terminal_config="rse")
    model.load_configuration(configuration, analog_terminal_config="nrse")

    assert made == ["rse", "nrse"]



def _daqmx_fake():
    """The laser device tests' stand-in for NI-DAQmx, loaded from its file."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1].joinpath(
        "auto-trainer-device", "tests", "nidaq_daqmx_fake.py")
    spec = importlib.util.spec_from_file_location("nidaq_daqmx_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_pulse_the_board_rule_refuses_leaves_the_last_command(monkeypatch):
    # The amplitude was recorded before the controller was asked: a pulse
    # refused, which drove nothing, left it as what the output "may still
    # hold", and a close that failed named it.
    from autotrainer.device import (
        LaserChannelId, LaserPulseTrain, LaserSynchronizedPulseTrain, NidaqLaserController)
    from autotrainer.device import nidaq_laser

    fake = _daqmx_fake()
    daq = fake.FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    model = LaserModel()
    model.set_controller(NidaqLaserController(fake.rig_lasers()))
    armed = model.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0),),
        wait=False, defer_start=True, timeout_seconds=30.0))
    try:
        assert model.last_command_volts == {1: 2.5}

        with pytest.raises(RuntimeError, match="refused while"):
            model.run_pulse_train(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0, duration_ms=1.0))

        assert model.last_command_volts == {1: 2.5}
    finally:
        armed.cancel()
        armed.wait_until_finished(5.0)
        model.close()


# ------------------------------------------------ workstream D: the follow-ups


def _nidaq_model(monkeypatch, configuration=None):
    """A model on a NI-DAQ controller over the device tests' DAQmx fake."""
    from autotrainer.device import NidaqLaserController
    from autotrainer.device import nidaq_laser

    fake = _daqmx_fake()
    daq = fake.FakeDaqmx()
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    model = LaserModel()
    model.set_controller(NidaqLaserController(configuration or fake.rig_lasers()))
    return model, daq


@pytest.mark.parametrize("path", ["run_pulse", "trial"])
def test_a_pulse_outside_the_lasers_range_leaves_the_last_command(monkeypatch, path):
    # Refused in the pulse's own thread, as its operation's ValueError, it
    # left its amplitude as what the output "may still hold", and a close
    # that failed named it. It is refused before anything exists now.
    from autotrainer.device import LaserChannelId, LaserPulseTrain
    from autotrainer.device.laser import LaserPulseRefused

    model, _daq = _nidaq_model(monkeypatch)
    try:
        with pytest.raises(LaserPulseRefused, match="outside the configured range"):
            if path == "run_pulse":
                model.run_pulse_train(LaserPulseTrain(
                    channel_id=LaserChannelId.LASER_1, amplitude_volts=6.0, duration_ms=1.0))
            else:
                model.prepare_pulse_profile(
                    LaserPulseProfile("hot", 1, 6.0, 1.0), SOFTWARE, _recipe())

        assert model.last_command_volts == {1: 0.0}
    finally:
        model.close()


def test_an_armed_pulse_records_its_baseline_after_a_refused_pulse_on_its_laser(monkeypatch):
    # The refusal takes back its record and its count. Taken back without
    # the count, the armed pulse's completion found a newer command than its
    # own, and left its 2.5 V as what the output may hold.
    from autotrainer.device import (
        LaserChannelId, LaserOperationState, LaserPulseTrain, LaserSynchronizedPulseTrain)
    from autotrainer.device.laser import LaserPulseRefused

    model, _daq = _nidaq_model(monkeypatch)
    armed = model.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0),),
        wait=False, defer_start=True, timeout_seconds=30.0))
    try:
        with pytest.raises(LaserPulseRefused, match="refused while"):
            model.run_pulse_train(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0, duration_ms=1.0))
        assert model.last_command_volts == {1: 2.5}

        armed.trigger()
        assert armed.wait(5.0) is LaserOperationState.COMPLETED
        deadline = time.monotonic() + 2.0
        while model.last_command_volts != {1: 0.0} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert model.last_command_volts == {1: 0.0}
    finally:
        armed.cancel()
        armed.wait_until_finished(5.0)
        model.close()


# ------------------------------------------------ the PMT rule (Ben, 2026-09-30)


def _with_margins(lead=5.0, lag=5.0):
    return LaserPulseProfile(
        "pmt", 1, 2.5, 1.0, pmt_open_lead_ms=lead, pmt_close_lag_ms=lag)


def _pmt_line(controller):
    import dataclasses

    controller.configuration = dataclasses.replace(
        controller.configuration, pmt_shutter_output="Dev1/port0/line6")
    return controller


def _pmt_warnings(caplog):
    return [record.getMessage() for record in caplog.records
            if record.levelname == "WARNING" and "PMT" in record.getMessage()]


def test_a_trial_drives_the_pmt_shutter_by_its_margins_with_a_pmt_line():
    controller = _pmt_line(_Controller())
    model = LaserModel(controller)

    model.prepare_pulse_profile(_with_margins(), BOARD, _recipe())
    pulse, = controller.pulse.pulse_trains
    assert pulse.enable_pmt_shutter
    assert (pulse.pmt_shutter_open_delay_ms, pulse.pmt_shutter_close_delay_ms) == (5.0, 5.0)

    for lead, lag in ((5.0, 0.0), (0.0, 5.0)):
        model.prepare_pulse_profile(_with_margins(lead, lag), BOARD, _recipe())
        assert controller.pulse.pulse_trains[0].enable_pmt_shutter
    model.prepare_pulse_profile(_with_margins(0.0, 0.0), BOARD, _recipe())
    assert not controller.pulse.pulse_trains[0].enable_pmt_shutter


def test_without_a_pmt_line_a_trials_margins_are_ignored_and_said_once(caplog):
    # christielab10 configures no PMT shutter line, and a trial with margins
    # failed there: "PMT shutter output requested, but laser
    # pmt_shutter_output is not configured". It fires without them.
    controller = _Controller()
    model = LaserModel(controller)

    with caplog.at_level("WARNING"):
        for _ in range(2):
            model.prepare_pulse_profile(_with_margins(), BOARD, _recipe())
            assert not controller.pulse.pulse_trains[0].enable_pmt_shutter
        warning, = _pmt_warnings(caplog)
        assert "Laser 1" in warning and "pmtShutterOutput" in warning

        # Once per controller: the next one says it again.
        controller = _Controller()
        model.set_controller(controller)
        model.prepare_pulse_profile(_with_margins(), BOARD, _recipe())
        assert not controller.pulse.pulse_trains[0].enable_pmt_shutter
        assert len(_pmt_warnings(caplog)) == 2


def test_a_trial_with_pmt_margins_fires_on_a_rig_with_no_pmt_line(monkeypatch):
    # christielab10's own case, through the NI-DAQ controller on the fake:
    # the margins are ignored, so the output is the train alone, and no PMT
    # line is driven.
    import dataclasses

    fake = _daqmx_fake()
    model, daq = _nidaq_model(
        monkeypatch, dataclasses.replace(fake.rig_lasers(), pmt_shutter_output=None))
    try:
        operation = model.prepare_pulse_profile(_with_margins(), SOFTWARE, _recipe())
        operation.trigger()
        assert operation.wait(5.0).value == "completed"

        # 1 ms at 100 kHz, with no 5 ms margin before or after it.
        assert daq.task("laser_sync_pulse_ao").timing_kwargs["samps_per_chan"] == 100
        assert not [task.label for task in daq.tasks if "pmt" in task.label]
    finally:
        model.close()


# ------------------------------- a manual Run Pulse, told as a session event (D-D)

# What Run Pulse's tab hands the model (LaserChannelTab._manual_pulse_context).
MANUAL = {"profile_id": "burst", "profile_revision": 3, "trigger_mode": "internal"}


def _told(model):
    """Every trace and event the model tells, as a recording would take it."""
    told = []
    model.trace_received += told.append
    return told


def _manual_events(told):
    return [trace for trace in told if trace.source == "manual pulse"]


def test_a_manual_run_pulse_is_told_as_requested_then_completed():
    # Ben, 2026-09-30: Run Pulse stays allowed while a session records, and
    # is kept in it as a marked manual event. It was kept only as its
    # waveform, unmarked, and stamped as that was told: after the train.
    from autotrainer.device import LaserChannelId, LaserPulseTrain

    model = _null_hardware_timed_model()
    told = _told(model)
    run_pulse_train = model._controller.run_pulse_train
    called = []

    def controller_run_pulse_train(pulse_train):
        called.append(time.perf_counter())
        run_pulse_train(pulse_train)

    model._controller.run_pulse_train = controller_run_pulse_train
    before = time.perf_counter()
    model.run_pulse_train(LaserPulseTrain(
        channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0,
        baseline_ms=2.0), manual_context=MANUAL)
    returned = time.perf_counter()

    assert [trace.event for trace in told] == ["requested", "trace", "completed"]
    requested, completed = _manual_events(told)
    waveform = told[1]
    assert requested.operation_id.startswith("manual-")
    assert completed.operation_id == requested.operation_id
    # Taken as the controller is called: the output starts at or after it.
    assert before <= requested.origin_perf_time <= called[0]
    assert called[0] <= completed.origin_perf_time <= returned
    assert (requested.timestamp_method, requested.timing_confidence) == (
        "manual_pulse_call_perf_counter", "before_output_start")
    assert (completed.timestamp_method, completed.timing_confidence) == (
        "manual_pulse_call_perf_counter", "after_output_end")
    # The waveform starts where the pulse was asked for, not after it ended,
    # and says so: each of its points was output at or after its time.
    assert waveform.source == "internal pulse"
    assert (waveform.origin_perf_time, waveform.origin_wall_time) == (
        requested.origin_perf_time, requested.origin_wall_time)
    assert (waveform.timestamp_method, waveform.timing_confidence) == (
        "manual_pulse_call_perf_counter", "before_output")
    assert waveform.operation_id == ""
    context = json.loads(requested.context_json)
    assert context == {
        "manual": True,
        "laser_channel_id": 1,
        "profile_id": "burst",
        "profile_revision": 3,
        "amplitude_volts": 2.5,
        "trigger_mode": "internal",
        "route": {"trigger_terminal": "", "trigger_edge": None, "stim_line": None},
    }
    # Returned: its output is back at the minimum.
    assert json.loads(completed.context_json) == {**context, "last_command_volts": 0.0}


def test_a_manual_pulse_that_fails_is_told_as_failed_with_its_error_class():
    # It may still hold its amplitude: the failed row says what the model
    # keeps as what the output may hold.
    from autotrainer.device import LaserChannelId, LaserPulseTrain

    model = _null_hardware_timed_model()
    told = _told(model)

    def fails(_pulse_train):
        raise RuntimeError("DAQmx refused to start laser_sync_pulse_ao")

    model._controller.run_pulse_train = fails
    with pytest.raises(RuntimeError, match="refused to start"):
        model.run_pulse_train(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0,
            trigger_source="/Dev1/PFI0", trigger_edge="falling"),
            manual_context={**MANUAL, "trigger_mode": "external"})

    # No waveform: it did not run.
    assert [trace.event for trace in told] == ["requested", "failed"]
    requested, failed = told
    assert failed.operation_id == requested.operation_id
    assert failed.timing_confidence == "operation_failure"
    context = json.loads(failed.context_json)
    assert context["route"] == {
        "trigger_terminal": "/Dev1/PFI0", "trigger_edge": "falling", "stim_line": None}
    assert context["trigger_mode"] == "external"
    assert context["error_class"] == "RuntimeError"
    assert context["error"] == "DAQmx refused to start laser_sync_pulse_ao"
    assert context["last_command_volts"] == 2.5
    assert model.last_command_volts == {1: 2.5}


def test_a_refused_manual_pulse_is_told_as_refused_and_leaves_the_baseline_logic(
    monkeypatch,
):
    # A refused pulse drove nothing, and must never read as one that fired.
    # Its refusal still takes back its record and count
    # (_undo_refused_pulse), so the armed pulse's completion finds its mark.
    from autotrainer.device import (
        LaserChannelId, LaserOperationState, LaserPulseTrain, LaserSynchronizedPulseTrain)
    from autotrainer.device.laser import LaserPulseRefused

    model, _daq = _nidaq_model(monkeypatch)
    armed = model.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0),),
        wait=False, defer_start=True, timeout_seconds=30.0))
    told = _told(model)
    try:
        with pytest.raises(LaserPulseRefused, match="refused while"):
            model.run_pulse_train(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0, duration_ms=1.0),
                manual_context=MANUAL)

        assert [trace.event for trace in told] == ["requested", "refused"]
        requested, refused = told
        assert refused.operation_id == requested.operation_id
        assert refused.timing_confidence == "operation_refused"
        context = json.loads(refused.context_json)
        assert context["error_class"] == "LaserPulseRefused"
        assert context["amplitude_volts"] == 1.0
        # The armed pulse's 2.5 V is still what the output may hold.
        assert context["last_command_volts"] == 2.5
        assert model.last_command_volts == {1: 2.5}

        armed.trigger()
        assert armed.wait(5.0) is LaserOperationState.COMPLETED
        deadline = time.monotonic() + 2.0
        while model.last_command_volts != {1: 0.0} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert model.last_command_volts == {1: 0.0}
    finally:
        armed.cancel()
        armed.wait_until_finished(5.0)
        model.close()


def test_every_other_run_pulse_train_caller_is_told_only_the_train():
    # The protocol path, the hardware tool and the tests pass no manual
    # context: one unmarked trace, stamped as it is told, as before.
    from autotrainer.device import LaserChannelId, LaserPulseTrain
    from autotrainer.device.laser import LaserPulseRefused

    model = _null_hardware_timed_model()
    told = _told(model)
    model.run_pulse_train(LaserPulseTrain(
        channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0))

    trace, = told
    assert (trace.event, trace.source, trace.operation_id, trace.context_json) == (
        "trace", "internal pulse", "", "{}")
    assert (trace.timestamp_method, trace.timing_confidence) == (
        "laser_event_perf_counter", "host_timestamp")
    assert (trace.origin_perf_time, trace.origin_wall_time) == (None, None)

    def refused(_pulse_train):
        raise LaserPulseRefused("refused while another pulse holds the board")

    model._controller.run_pulse_train = refused
    with pytest.raises(LaserPulseRefused):
        model.run_pulse_train(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0))
    assert len(told) == 1


def test_a_manual_pulse_event_needs_a_train_that_is_waited_for():
    # A train that is not waited for returns before it ends, so "completed"
    # would be a guess. Refused before anything is recorded or driven.
    from autotrainer.device import LaserChannelId, LaserPulseTrain

    model = _null_hardware_timed_model()
    told = _told(model)

    with pytest.raises(ValueError, match="waited for"):
        model.run_pulse_train(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=2.5, duration_ms=1.0,
            wait=False), manual_context=MANUAL)

    assert told == []
    assert model.last_command_volts == {1: 0.0}
