import importlib.util, sys, dataclasses
from pathlib import Path
import pytest
from autotrainer.device import LaserChannelId, nidaq_laser

_REPO = Path(__file__).resolve().parents[1]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module

fake = _load("nidaq_daqmx_fake", _REPO / "auto-trainer-device" / "tests" / "nidaq_daqmx_fake.py")
victim = _load("laser_kill_victim", _REPO / "tools" / "hardware" / "laser_kill_victim.py")

@pytest.fixture
def daq(monkeypatch):
    daq = fake.FakeDaqmx(block_wait=True, hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    yield daq
    daq.waits_released.set()

def _system(laser):
    from types import SimpleNamespace
    return SimpleNamespace(laser=laser, nidaq_stream=SimpleNamespace(analog_terminal_config="rse"))

def _args(**overrides):
    # pulse_seconds=2.0: the fake keeps every write as a list, and a 30 s
    # train at 100 kHz is three million samples. 2 s is ten pulses.
    values = dict(timing="hardware", waveform="dc", mode="clean", laser=1, level=2.0,
                  pulse_seconds=2.0, ready_settle=0.0, cue_timeout=1.0, config="unused")
    values.update(overrides)
    return victim.parse_args([f"--{k.replace('_', '-')}={v}" for k, v in values.items()])

def _run(daq, args, cued=True):
    lines = []
    code = victim.run(args, load=lambda path, save_backup: _system(fake.rig_lasers()),
                      announce=lines.append, wait_for_cue=lambda timeout: cued)
    return code, lines

def test_the_software_timed_configuration_keeps_every_channel():
    original = fake.rig_lasers()
    software = victim.laser_configuration_for(original, "software")
    assert (software.hardware_timed, software.sample_rate_hz) == (False, None)
    assert software.channels == original.channels
    assert dataclasses.replace(software, hardware_timed=True, sample_rate_hz=100_000.0) == original

def test_a_hardware_row_refuses_a_configuration_without_a_rate():
    untimed = dataclasses.replace(fake.rig_lasers(), hardware_timed=False, sample_rate_hz=None)
    with pytest.raises(ValueError, match="hardwareTimed"):
        victim.laser_configuration_for(untimed, "hardware")

def test_the_pulse_train_is_thirty_seconds_of_five_hertz_at_ninety_percent():
    train = victim.pulse_train_for(LaserChannelId.LASER_1, 2.0, 30.0)
    assert (train.pulse_count, train.frequency_hz, train.duration_ms) == (150, 5.0, 180.0)
    assert (train.open_shutter, train.wait) == (True, False)

def test_a_software_dc_victim_holds_its_on_demand_task_open_at_ready(daq):
    code, lines = _run(daq, _args(timing="software", mode="clean"))
    assert lines[0].startswith("READY pid=") and "level=2.0000" in lines[0]
    task = daq.task("laser_1_ao")
    assert daq.writes_to("PXI1Slot4/ao0", task_suffix="laser_1_ao")[-2] == 2.0
    assert [t.label for t in daq.tasks if t.ai_added] == []   # no 6221 input task
    assert code == 0 and task.closed                            # clean closed it

def test_a_hardware_dc_victim_leaves_no_task_on_the_output(daq):
    code, lines = _run(daq, _args(timing="hardware", waveform="dc", mode="clean"))
    assert daq.task("laser_1_manual_ao").closed
    assert daq.writes_to("PXI1Slot4/ao0")[-2:] == [2.0, 0.0]

def test_a_hardware_pulse_victim_is_ready_with_its_train_running(daq):
    code, lines = _run(daq, _args(timing="hardware", waveform="pulse", mode="clean"))
    assert "waveform=pulse" in lines[0]
    assert daq.task("laser_sync_pulse_ao").started or daq.task("laser_sync_pulse_ao").aborted.is_set()
    assert daq.task("laser_1_shutter").writes[-1] is False and code == 0

# Not in the plan's list: NidaqLaserController.run_pulse_train returns None, so
# hold() gets the operation from run_synchronized_pulse_train, and the clean
# mode must end the train through it.
def test_a_pulse_hold_returns_the_running_operation_with_the_shutter_open_and_clean_ends_it(daq):
    system = _system(fake.rig_lasers())
    model, controller = victim.open_controller(system, system.laser)
    operation = victim.hold(model, controller, LaserChannelId.LASER_1, "pulse", 2.0, 2.0)
    assert operation is not None and operation.state not in operation.TERMINAL
    assert daq.task("laser_1_shutter").writes[-1] is True
    assert victim.act("clean", model=model, operation=operation) == 0
    assert operation.state is nidaq_laser.LaserOperationState.CANCELLED
    assert daq.task("laser_1_shutter").writes[-1] is False

def test_a_software_pulse_victim_is_refused_and_closes(daq):
    code, lines = _run(daq, _args(timing="software", waveform="pulse"))
    assert code == 2 and lines[0].startswith("REFUSED ")
    assert all(t.closed for t in daq.tasks)

def test_the_exception_mode_raises_out_and_leaves_the_output_task_open(daq):
    with pytest.raises(RuntimeError, match="exception mode"):
        _run(daq, _args(timing="software", mode="exception"))
    assert not daq.task("laser_1_ao").closed

def test_an_uncued_victim_closes_and_says_so(daq):
    code, lines = _run(daq, _args(timing="software", mode="sigkill"), cued=False)
    assert code == 3 and lines[-1] == "NOT CUED" and daq.task("laser_1_ao").closed

def test_act_dispatches_each_mode_without_touching_the_process():
    calls = []
    assert victim.act("sigkill", model=None, operation=None,
                      pause=lambda s: calls.append(("pause", s))) == 3
    victim.act("abort", model=None, operation=None, abort=lambda: calls.append("abort"))
    assert calls == [("pause", 3600), "abort"]

# Not in the plan's list: the hang mode's SIGTERM ignore and the cue the main
# thread polls are what the orchestrator's rows lean on, and nothing above runs them.
def test_the_hang_mode_ignores_sigterm_before_it_sleeps_forever():
    calls = []

    def pause(seconds):
        calls.append(("pause", seconds))
        if len(calls) > 3:
            raise KeyboardInterrupt   # stands in for the SIGKILL that ends the row

    with pytest.raises(KeyboardInterrupt):
        victim.act("hang", model=None, operation=None,
                   ignore_sigterm=lambda: calls.append("ignore"), pause=pause)
    assert calls == ["ignore", ("pause", 1), ("pause", 1), ("pause", 1)]

def test_wait_for_cue_reports_a_cue_and_a_timeout():
    victim._cued.clear()
    assert victim.wait_for_cue(0.05) is False
    victim._on_cue(0, None)
    try:
        assert victim.wait_for_cue(0.05) is True
    finally:
        victim._cued.clear()
