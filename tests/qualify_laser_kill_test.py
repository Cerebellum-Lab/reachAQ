import hashlib, importlib.util, io, json, os, queue, signal, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import pytest
from autotrainer.core import SystemConfiguration
from autotrainer.device import nidaq_laser
from tools.acquisition.model.laser_kill_matrix import (
    HELD_UNTIL_RESET, SAFE_ON_DEATH, UNKNOWN, KillCase, LevelSample, from_record, judge)

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hardware" / "qualify_laser_kill.py"
_spec = importlib.util.spec_from_file_location("qualify_laser_kill", _TOOL)
tool = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("qualify_laser_kill", tool)
_spec.loader.exec_module(tool)

_REAL_MAKE_SPAWN = tool.make_spawn   # the autouse guard below replaces the module's

_REPO = Path(__file__).resolve().parents[1]
_fake_spec = importlib.util.spec_from_file_location(
    "nidaq_daqmx_fake", _REPO / "auto-trainer-device" / "tests" / "nidaq_daqmx_fake.py")
fake = importlib.util.module_from_spec(_fake_spec)
sys.modules.setdefault("nidaq_daqmx_fake", fake)
_fake_spec.loader.exec_module(fake)


@pytest.fixture(autouse=True)
def _no_real_hardware(monkeypatch):
    """Nothing here may open a real NI task or start a real victim.

    The tests inject fakes. If one reaches for the real driver or a real
    process it fails instead of driving the output.
    """
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test reached for real hardware or a real process")
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", refuse)
    monkeypatch.setattr(tool, "_load_nidaqmx", refuse)
    monkeypatch.setattr(tool, "make_spawn", refuse)
    # What an earlier test gave up on must not be waited for by this one, and a
    # test whose fake call never returns must not make the recovery wait 20 s.
    monkeypatch.setattr(tool, "_outstanding", [])
    monkeypatch.setattr(tool, "OUTSTANDING_WAIT_S", 0.1)
    # And no test may signal a real process: a call that leaves out kill= would
    # otherwise reach os.kill with a made-up pid.
    for function in (tool.apply_kill, tool.end_victim, tool.kill_victim, tool.run_case,
                     tool.tracking_check, tool.run_matrix, tool.execute):
        monkeypatch.setitem(function.__kwdefaults__, "kill", refuse)


@pytest.fixture
def daq(monkeypatch):
    daq = fake.FakeDaqmx()
    # The fake has no continuous acquisition, and nothing like Device.
    daq.constants.AcquisitionType.CONTINUOUS = "continuous"
    daq.resets = []

    class _Device:
        def __init__(self, name):
            self.name = name
        def reset_device(self):
            daq.resets.append(self.name)
    daq.system = SimpleNamespace(Device=_Device)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    yield daq
    daq.waits_released.set()


class _Process:
    def __init__(self, pid=4242, ready="READY pid=4242 timing=software waveform=dc mode=sigkill level=2.0000",
                 ignores_sigterm=False, alive=True, returncode=None):
        self.pid, self.lines, self.alive = pid, queue.Queue(), alive
        self.ignores_sigterm, self.returncode, self.signals = ignores_sigterm, returncode, []
        self.lines.put(ready)
    def wait(self, timeout=None):
        if not self.alive:
            return self.returncode
        raise tool.subprocess.TimeoutExpired("victim", timeout)
    def kill_by(self, pid, sig):
        assert pid == self.pid
        self.signals.append(sig)
        if sig == signal.SIGKILL or (sig == signal.SIGTERM and not self.ignores_sigterm):
            self.alive, self.returncode = False, -int(sig)

class _CuedProcess(_Process):
    """A clean victim: the SIGUSR1 cue makes it close and exit 0."""
    def kill_by(self, pid, sig):
        super().kill_by(pid, sig)
        if sig == signal.SIGUSR1:
            self.alive, self.returncode = False, 0

class _Clock:
    def __init__(self):
        self.t = 100.0
    def now(self):
        return self.t
    def sleep(self, seconds):
        self.t += seconds

def _log(levels):
    """A reader log: (time, level) pairs, each a flat 10 ms chunk."""
    log = tool.LevelLog()
    for t, level in levels:
        log.append(LevelSample(t, level, level, level))
    return log

def _run(case, process, log, clock, reopen=lambda: None, reset=lambda: None, **extra):
    return tool.run_case(case, spawn=lambda c: process, log=log, now=clock.now,
                         sleep=clock.sleep, kill=process.kill_by, reopen=reopen,
                         reset=reset, hold_seconds=5.0, instrument="ai3", **extra)

def test_without_the_disconnected_flag_the_tool_refuses():
    with pytest.raises(SystemExit):
        tool.parse_args(["--laser", "1"])
    assert tool.parse_args(["--dry-run"]).dry_run

def test_the_reader_is_the_command_copy_never_the_floating_diode():
    laser = SimpleNamespace(channel_id=SimpleNamespace(value=1), command_copy_input="PXI1Slot5/ai3",
                            diode_input="PXI1Slot5/ai8")
    configuration = SimpleNamespace(
        laser=SimpleNamespace(get_channel=lambda _id: laser),
        nidaq_stream=SimpleNamespace(analog_terminal_config="rse"))
    assert tool.reader_channel_for(configuration, 1, None) == ("PXI1Slot5/ai3", "rse")
    with pytest.raises(ValueError, match="floats"):
        tool.reader_channel_for(configuration, 1, "PXI1Slot5/ai8")

def test_apply_kill_signals_only_the_recorded_pid():
    sent = []
    for mode, expected in (("sigterm", signal.SIGTERM), ("sigkill", signal.SIGKILL),
                           ("exception", signal.SIGUSR1), ("hang", signal.SIGUSR1)):
        tool.apply_kill(KillCase(mode, "software", "dc"), 4242, kill=lambda pid, s: sent.append((pid, s)))
        assert sent[-1] == (4242, expected)

def test_a_victim_that_ignores_sigterm_is_ended_by_its_own_pid():
    process = _Process(ignores_sigterm=True)
    assert tool.end_victim(process, kill=process.kill_by, timeout=0.01) == "killed after hang (sigkill by pid)"
    assert process.signals == [signal.SIGKILL]

def test_run_case_reads_each_instant_off_the_log_around_the_kill():
    clock = _Clock()
    # READY at 100.0; kill at 101.0; hold read over (101.15, 106.0]; reopen at
    # 106.0 and read at 106.5. Held at 2 V until the reopen, then 0 V.
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    process = _Process()
    observation = _run(KillCase("sigkill", "software", "dc"), process, log, clock)
    assert observation.before.mean == 2.0 and observation.after_kill.mean == 2.0
    assert observation.after_reopen.mean == 0.0 and observation.victim_exit == "signal 9"
    assert process.signals == [signal.SIGKILL]

def test_run_case_completes_when_the_log_has_no_samples_after_the_kill():
    clock = _Clock()
    log = _log([(100.5, 2.0)])
    process = _Process()
    observation = _run(KillCase("sigkill", "software", "dc"), process, log, clock)
    assert observation.after_kill is None
    assert judge(observation, minimum_volts=0.0, level_volts=2.0) == UNKNOWN
    assert observation.victim_exit == "signal 9"

def test_a_refused_reopen_is_recorded_and_the_row_goes_on():
    clock = _Clock()
    # Still 2 V when the refused reopen is read at 106.5; 0 V by the reset
    # reading at 107.0.
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.7 else 0.0) for i in range(0, 900)])
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock,
                       reopen=lambda: "DaqError: -50103 The specified resource is reserved")
    assert observation.reopen_error.startswith("DaqError: -50103")
    assert judge(observation, minimum_volts=0.0, level_volts=2.0) == HELD_UNTIL_RESET

def test_a_refused_victim_is_a_row_with_its_reason_and_no_kill():
    # A refused victim closes and exits 2 by itself; nothing is signalled.
    process = _Process(ready="REFUSED run_synchronized_pulse_train refuses ...",
                       alive=False, returncode=2)
    observation = _run(KillCase("clean", "software", "pulse"), process, _log([]), _Clock())
    assert observation.notes[0].startswith("run_synchronized_pulse_train")
    assert observation.victim_exit == "exit 2" and process.signals == []

def test_the_victim_command_names_the_row():
    args = tool.parse_args(["--dry-run", "--config", "/tmp/c.yaml", "--level", "1.5"])
    command = tool.victim_command("python", Path("/w/tools/hardware/laser_kill_victim.py"), args,
                                  KillCase("abort", "hardware", "pulse", 2))
    assert command[:2] == ["python", "/w/tools/hardware/laser_kill_victim.py"]
    for flag in ("--timing=hardware", "--waveform=pulse", "--mode=abort", "--level=1.5", "--config=/tmp/c.yaml"):
        assert flag in command


# Not in the plan's list from here on.

# Ruling T1-I1b: a hold window the reader did not cover cannot show the output
# stayed down, so the row carries no hold reading, and the judge decides what
# that means.
def test_a_log_that_stops_a_second_into_the_hold_window_gives_no_hold_reading():
    clock = _Clock()
    # The hold window is (101.15, 106.0]. 2 V all through, then the reader goes.
    log = _log([(100 + i / 100, 2.0) for i in range(0, 216)])  # last chunk 102.15
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock)
    assert observation.hold is None
    assert observation.after_kill is not None and observation.after_kill.mean == 2.0
    assert observation.before.mean == 2.0

@pytest.mark.parametrize("stops_early_s, covered", [(0.30, True), (0.60, False)])
def test_the_hold_reading_needs_ninety_percent_of_the_window(stops_early_s, covered):
    clock = _Clock()
    # 485 chunks of 10 ms fill the window; 90 % of that is 436.5 chunks.
    last_chunk = round(600 - stops_early_s * 100)
    log = _log([(100 + i / 100, 2.0) for i in range(0, last_chunk + 1)])
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock)
    assert (observation.hold is not None) is covered
    if covered:
        assert observation.hold.sample_count >= 437

def test_the_whole_hold_window_is_one_reading_with_its_chunk_count():
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock)
    assert observation.hold.sample_count == pytest.approx(485, abs=1)
    assert (observation.hold.minimum, observation.hold.maximum) == (2.0, 2.0)

def test_the_row_runs_on_the_documented_timeline():
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    process = _Process()
    events = []
    def kill(pid, sig):
        events.append(("kill", clock.t))
        process.kill_by(pid, sig)
    tool.run_case(KillCase("sigkill", "software", "dc"), spawn=lambda c: process, log=log,
                  now=clock.now, sleep=clock.sleep, kill=kill,
                  reopen=lambda: events.append(("reopen", clock.t)),
                  reset=lambda: events.append(("reset", clock.t)),
                  hold_seconds=5.0, instrument="ai3")
    assert [name for name, _ in events] == ["kill", "reopen", "reset"]
    assert [t for _, t in events] == pytest.approx([101.0, 106.0, 106.5])
    assert clock.t == pytest.approx(107.0)

def test_dmm_values_are_kept_at_the_five_instants_and_a_blank_is_not_stored():
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    answers = {"before": 2.0, "after_kill": None, "after_hold": 1.9,
               "after_reopen": 0.0, "after_reset": 0.0}
    asked = []
    def dmm(instant):
        asked.append(instant)
        return answers[instant]
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock, dmm=dmm)
    assert asked == ["before", "after_kill", "after_hold", "after_reopen", "after_reset"]
    assert observation.dmm == {"before": 2.0, "after_hold": 1.9, "after_reopen": 0.0, "after_reset": 0.0}

def test_a_slow_dmm_does_not_move_the_instants_read_off_the_log():
    clock = _Clock()
    # 2 V until 104.0, 0.5 V until 110.0, then 3 V. The after-kill prompt takes
    # 20 s of the operator's time; the hold window and the 5 s instant are still
    # read at their own times, not at the end of the prompt.
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 104.0 else 0.5 if 100 + i / 100 < 110.0 else 3.0)
                for i in range(0, 4000)])
    def dmm(instant):
        if instant == "after_kill":
            clock.sleep(20.0)
        return None
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock, dmm=dmm)
    assert observation.hold.minimum == 0.5 and observation.hold.maximum == 2.0
    assert observation.after_hold.mean == 0.5   # read at 106.0, not at 126.0 or later

def test_a_skipped_reopen_and_reset_leave_no_reading_and_say_so():
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock,
                       reopen=None, reset=None)
    assert observation.after_reopen is None and observation.after_reset is None
    assert observation.reopen_error is None and observation.reset_error is None
    assert "reopen skipped" in observation.notes and "reset skipped" in observation.notes
    assert clock.t == pytest.approx(106.0)   # no half-second waits for readings never taken

def test_a_ready_line_naming_another_pid_is_not_a_row_and_signals_only_the_spawned_pid():
    process = _Process(pid=4242, ready="READY pid=9999 timing=software waveform=dc mode=sigkill level=2.0000")
    observation = _run(KillCase("sigkill", "software", "dc"), process, _log([]), _Clock())
    assert process.signals == [signal.SIGKILL]   # the end of the spawned process, nothing else
    assert observation.before is None and observation.after_kill is None
    assert "9999" in observation.notes[0] and "4242" in observation.notes[0]

def test_a_victim_that_never_says_ready_is_ended_by_pid_and_recorded(monkeypatch):
    monkeypatch.setattr(tool, "READY_TIMEOUT_S", 0.05)
    process = _Process()
    process.lines = queue.Queue()
    observation = _run(KillCase("sigkill", "software", "dc"), process, _log([]), _Clock())
    assert process.signals == [signal.SIGKILL]
    assert observation.after_kill is None and "READY" in observation.notes[0]
    # Not "after hang": the row never cued it, and it never said READY.
    assert "sigkill by pid" in observation.victim_exit and "hang" not in observation.victim_exit

def test_a_victim_whose_output_ends_before_ready_is_a_row_without_waiting_for_the_timeout():
    process = _Process(ready=tool.STREAM_CLOSED, alive=False, returncode=1)
    observation = _run(KillCase("sigkill", "software", "dc"), process, _log([]), _Clock())
    assert observation.victim_exit == "exit 1" and process.signals == []
    assert "before READY" in observation.notes[0]

def test_a_failure_mid_row_still_ends_the_victim_by_its_pid():
    process = _Process()
    def broken_reopen():
        raise RuntimeError("the orchestrator's own bug")
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    with pytest.raises(RuntimeError, match="own bug"):
        _run(KillCase("hang", "software", "dc"), process, log, clock, reopen=broken_reopen)
    # The cue, then the end of the row's process by its pid.
    assert process.signals == [signal.SIGUSR1, signal.SIGKILL]

def test_make_spawn_starts_the_victim_with_its_row_and_puts_its_lines_on_the_queue(tmp_path):
    started = []
    class _Popen:
        def __init__(self, command, **kwargs):
            started.append((command, kwargs))
            self.pid = 31337
            self.stdout = io.StringIO(
                "READY pid=31337 timing=hardware waveform=dc mode=clean level=4.0000\n")
    args = tool.parse_args(["--dry-run", f"--record-dir={tmp_path}", "--config=/tmp/c.yaml"])
    spawn = _REAL_MAKE_SPAWN(args, stamp="S", python="py", victim_path=Path("/w/v.py"), popen=_Popen)
    process = spawn(KillCase("clean", "hardware", "dc"), 4.0)
    (command, kwargs), = started
    assert command[:2] == ["py", "/w/v.py"]
    assert "--level=4.0" in command and "--timing=hardware" in command   # the override, not --level's 2.0
    assert kwargs["stdin"] is tool.subprocess.DEVNULL and kwargs["stdout"] is tool.subprocess.PIPE
    assert tool.wait_for_ready(process.lines, 2.0)["pid"] == 31337
    assert process.lines.get(timeout=2.0) == tool.STREAM_CLOSED
    assert (tmp_path / "laser-kill-S-clean-hardware-dc-4V.stderr.log").exists()

def test_wait_for_ready_parses_ready_refused_and_skips_other_lines():
    lines = queue.Queue()
    for text in ("some noise", "READY pid=77 timing=hardware waveform=pulse mode=abort level=1.2500"):
        lines.put(text)
    ready = tool.wait_for_ready(lines, 1.0)
    assert ready["pid"] == 77 and ready["level"] == 1.25 and ready["mode"] == "abort"
    lines.put("REFUSED because of a thing")
    assert tool.wait_for_ready(lines, 1.0) == {"refused": "because of a thing"}
    with pytest.raises(TimeoutError):
        tool.wait_for_ready(queue.Queue(), 0.02)

def test_the_matrix_of_the_dry_run_is_thirty_one_rows_and_a_case_filter_keeps_the_matrix_order():
    cases = tool.select_cases(tool.kill_cases(pulse_repeats=3), None)
    assert len(cases) == 31
    chosen = tool.select_cases(cases, "hang-hardware-pulse-r1,sigkill-software-dc")
    assert [c.case_id for c in chosen] == ["sigkill-software-dc", "hang-hardware-pulse-r1"]
    with pytest.raises(ValueError, match="no-such-case"):
        tool.select_cases(cases, "sigkill-software-dc,no-such-case")
    with pytest.raises(ValueError, match="no case id"):
        tool.select_cases(cases, ",")

def test_the_reader_needs_a_channel_from_the_config_or_the_command_line():
    laser = SimpleNamespace(channel_id=SimpleNamespace(value=2), command_copy_input=None,
                            diode_input="PXI1Slot5/ai4")
    configuration = SimpleNamespace(
        laser=SimpleNamespace(get_channel=lambda _id: laser),
        nidaq_stream=SimpleNamespace(analog_terminal_config=""))
    with pytest.raises(ValueError, match="laser 2"):
        tool.reader_channel_for(configuration, 2, None)
    assert tool.reader_channel_for(configuration, 2, "PXI1Slot5/ai6") == ("PXI1Slot5/ai6", None)

def test_the_dmm_prompt_returns_a_float_none_for_blank_and_asks_again_on_a_typo():
    answers = iter(["", "x", "1.98"])
    prompts = []
    def ask(prompt):
        prompts.append(prompt)
        return next(answers)
    assert tool.prompt_dmm("sigkill-software-dc", "before", ask=ask) is None
    assert tool.prompt_dmm("sigkill-software-dc", "after_kill", ask=ask) == 1.98
    assert prompts[0] == "DMM reading for sigkill-software-dc before (V): "
    assert len(prompts) == 3

def test_the_reader_logs_each_chunk_and_keeps_going_after_a_read_error(daq):
    calls = []
    def read_samples(task, count):
        calls.append(count)
        if len(calls) == 2:
            raise RuntimeError("DAQmx -200279: the buffer overflowed")
        time.sleep(0.002)
        return [1.0, 2.0, 3.0]
    daq.read_samples = read_samples
    log, stop = tool.LevelLog(), threading.Event()
    thread = tool.start_reader(daq, "PXI1Slot5/ai3", "rse", log, stop)
    deadline = time.monotonic() + 5.0
    while len(log.samples()) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(5.0)
    task = daq.task("laser_kill_reader")
    assert task.ai_added == [("PXI1Slot5/ai3", {"terminal_config": "rse"})]
    assert task.timing_kwargs == {"rate": 10_000.0, "sample_mode": "continuous", "samps_per_chan": 100_000}
    assert set(calls) == {tool.READER_CHUNK}
    samples = log.samples()
    assert len(samples) >= 3
    assert (samples[0].mean, samples[0].minimum, samples[0].maximum) == (2.0, 1.0, 3.0)
    assert log.error.startswith("RuntimeError: DAQmx -200279")
    # The error stopped and started the task again, and the log says so.
    assert daq.starts.count("laser_kill_reader") == 2
    assert len(log.restarts) == 1 and "-200279" in log.restarts[0] and "restarted" in log.restarts[0]
    assert not thread.is_alive() and task.closed

def test_a_reader_with_no_terminal_configuration_leaves_it_to_daqmx(daq):
    stop = threading.Event()
    stop.set()
    thread = tool.start_reader(daq, "PXI1Slot5/ai3", None, tool.LevelLog(), stop)
    thread.join(5.0)
    assert daq.task("laser_kill_reader").ai_added == [("PXI1Slot5/ai3", {})]

def test_a_reader_that_cannot_start_closes_its_task_and_says_why(daq):
    daq.failing_task = "laser_kill_reader"
    with pytest.raises(RuntimeError, match="refused to start"):
        tool.start_reader(daq, "PXI1Slot5/ai3", "rse", tool.LevelLog(), threading.Event())
    assert daq.task("laser_kill_reader").closed

def test_an_unknown_terminal_configuration_is_refused_not_left_to_daqmx(daq):
    with pytest.raises(ValueError, match="sideways"):
        tool.start_reader(daq, "PXI1Slot5/ai3", "sideways", tool.LevelLog(), threading.Event())

def _system(laser):
    return SimpleNamespace(laser=laser, nidaq_stream=SimpleNamespace(analog_terminal_config="rse"))

def test_reopen_opens_the_live_configuration_and_closes_it_leaving_the_output_at_its_minimum(daq):
    assert tool.reopen(_system(fake.rig_lasers()), 1) is None
    # __init__ writes the minimum, close() writes it again.
    assert daq.writes_to("PXI1Slot4/ao0") == [0.0, 0.0]
    assert daq.task("laser_1_shutter").writes[-1] is False
    assert all(task.closed for task in daq.tasks)
    assert [t.label for t in daq.tasks if t.ai_added] == []   # no input task of its own

def test_a_reopen_the_driver_refuses_is_returned_as_text_not_raised(daq):
    daq.failing_write = "manual_ao"
    text = tool.reopen(_system(fake.rig_lasers()), 1)
    assert text.startswith("FakeDaqError: ") and "-50103" in text

def test_reopen_of_a_laser_the_configuration_does_not_have_is_reported(daq):
    text = tool.reopen(_system(fake.rig_lasers()), 3)
    assert text.startswith("KeyError") and daq.tasks == []

def test_reset_device_resets_the_named_device_and_returns_text_on_failure():
    resets = []
    class _Device:
        def __init__(self, name):
            self.name = name
        def reset_device(self):
            resets.append(self.name)
            if self.name == "PXI1Slot9":
                raise RuntimeError("DAQmx -50103: reserved")
    daq = SimpleNamespace(system=SimpleNamespace(Device=_Device))
    assert tool.reset_device(daq, "PXI1Slot4") is None
    assert tool.reset_device(daq, "PXI1Slot9") == "RuntimeError: DAQmx -50103: reserved"
    assert resets == ["PXI1Slot4", "PXI1Slot9"]

def test_the_tracking_check_reads_the_reader_and_the_dmm_at_each_level_and_cues_each_victim(capsys):
    clock = _Clock()
    by_pid = {}
    levels = tool.TRACKING_LEVELS
    # One second per level: the reader holds levels[k] over (100 + k, 101 + k].
    log = _log([(100 + i / 100, levels[(i - 1) // 100]) for i in range(1, 401)])
    processes, spawned_at = [], []
    def spawn(case, level=None):
        spawned_at.append((case, level))
        process = _CuedProcess(pid=5000 + len(processes),
                               ready=f"READY pid={5000 + len(processes)} timing=software waveform=dc mode=clean level={level:.4f}")
        processes.append(process)
        by_pid[process.pid] = process
        return process
    configuration = SimpleNamespace(
        laser=SimpleNamespace(get_channel=lambda _id: SimpleNamespace(
            channel_id=SimpleNamespace(value=1), command_copy_input="PXI1Slot5/ai3",
            diode_input="PXI1Slot5/ai8")),
        nidaq_stream=SimpleNamespace(analog_terminal_config="rse"))
    dmm_labels = []
    def dmm(label):
        dmm_labels.append(label)
        return float(label.split()[0]) - 0.01
    rows = tool.tracking_check(tool.parse_args(["--dry-run"]), configuration, log, spawn, dmm,
                               now=clock.now, sleep=clock.sleep,
                               kill=lambda pid, sig: by_pid[pid].kill_by(pid, sig))
    assert [level for level, _, _ in rows] == list(levels)
    assert [round(reading.mean, 6) for _, reading, _ in rows] == list(levels)
    assert [value for _, _, value in rows] == pytest.approx([level - 0.01 for level in levels])
    assert all(case == KillCase("clean", "software", "dc") for case, _ in spawned_at)
    assert [level for _, level in spawned_at] == list(levels)
    assert [p.signals for p in processes] == [[signal.SIGUSR1]] * 4
    assert dmm_labels == ["0 V", "1 V", "2 V", "4 V"]
    text = capsys.readouterr().out
    assert "reader" in text and "DMM" in text

def test_write_record_writes_the_json_and_the_table_beside_it(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "repository_revision", lambda: "abc1234-dirty")
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    observation = _run(KillCase("sigkill", "software", "dc"), _Process(), log, clock,
                       dmm=lambda instant: 1.98 if instant == "after_kill" else None)
    args = tool.parse_args(["--dry-run", f"--record-dir={tmp_path / 'rec'}", "--hold-seconds=5"])
    started = tool.datetime(2026, 11, 5, 14, 30, 0, tzinfo=tool.timezone.utc)
    extra = dict(reader_channel="PXI1Slot5/ai3", level_volts=2.0, minimum_volts=0.0,
                 config_sha256="deadbeef")
    path = tool.write_record(args, [observation], started, extra)
    assert path == tmp_path / "rec" / "laser-kill-20261105T143000Z.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value in (("host", tool.socket.gethostname()), ("reachaq_revision", "abc1234-dirty"),
                       ("reader_channel", "PXI1Slot5/ai3"), ("level_volts", 2.0),
                       ("minimum_volts", 0.0), ("hold_seconds", 5.0), ("config_sha256", "deadbeef")):
        assert record[key] == value
    (round_trip,) = from_record(record)
    assert round_trip == observation
    table = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "sigkill-software-dc" in table and "dmm=1.98" in table

def test_a_dry_run_loads_the_config_lists_the_matrix_and_opens_nothing(tmp_path, capsys):
    configuration = SystemConfiguration()
    configuration.laser = fake.rig_lasers()
    configuration.save_file(tmp_path / "system_configuration", as_yaml=True)
    path = tmp_path / "system_configuration.yaml"
    before = path.read_bytes()
    code = tool.main(["--dry-run", f"--config={path}", f"--record-dir={tmp_path / 'rec'}"])
    text = capsys.readouterr().out
    assert code == 0
    assert "31 cases" in text and "PXI1Slot5/ai3" in text and "rse" in text
    assert "sigkill-hardware-pulse-r3" in text
    assert path.read_bytes() == before and not (tmp_path / "rec").exists()
    # The autouse guard fails the test if main reached the driver or a process.

def test_a_dry_run_with_an_unknown_case_says_so_and_stops(tmp_path, capsys):
    configuration = SystemConfiguration()
    configuration.laser = fake.rig_lasers()
    configuration.save_file(tmp_path / "system_configuration", as_yaml=True)
    code = tool.main(["--dry-run", f"--config={tmp_path / 'system_configuration.yaml'}", "--cases=nope"])
    captured = capsys.readouterr()
    assert code == 2 and "nope" in captured.out + captured.err

def _matrix_args(tmp_path, *flags):
    return tool.parse_args(["--dry-run", f"--record-dir={tmp_path}", *flags])

def test_run_matrix_runs_each_row_with_its_own_process_and_the_recoveries_bound_to_the_config(daq, tmp_path, capsys):
    clock = _Clock()
    # 2 V until the reopen at 106.0, then 0 V: every row reads held-until-reopen.
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    processes = []
    def spawn(case, level=None):
        process = _Process(ready=f"READY pid=4242 timing={case.timing} waveform={case.waveform} mode={case.mode} level=2.0000")
        processes.append((case, process))
        return process
    class _Device:
        names = []
        def __init__(self, name):
            _Device.names.append(name)
        def reset_device(self):
            pass
    daq.system = SimpleNamespace(Device=_Device)
    observations = []
    tool.run_matrix(
        [KillCase("sigkill", "software", "dc")], observations,
        args=_matrix_args(tmp_path), configuration=_system(fake.rig_lasers()), nidaqmx=daq,
        spawn=spawn, log=log, instrument="PXI1Slot5/ai3", level_volts=2.0, minimum_volts=0.0,
        now=clock.now, sleep=clock.sleep, kill=lambda pid, sig: processes[0][1].kill_by(pid, sig))
    (observation,) = observations
    assert observation.reopen_error is None and observation.reset_error is None
    assert _Device.names == ["PXI1Slot4"]                       # the 6713, from the AO channel
    assert daq.writes_to("PXI1Slot4/ao0") == [0.0, 0.0]          # the reopen ran on the live config
    assert "sigkill-software-dc" in capsys.readouterr().out
    assert observation.after_reopen.mean == 0.0

def test_run_matrix_skips_the_recoveries_it_is_told_to(daq, tmp_path):
    clock = _Clock()
    # The output is down by the 5 s instant: with both recoveries skipped, that
    # is the last thing the reader sees, and it is at the minimum.
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 105.0 else 0.0) for i in range(0, 900)])
    process = _Process()
    observations = []
    tool.run_matrix(
        [KillCase("sigkill", "software", "dc")], observations,
        args=_matrix_args(tmp_path, "--skip-reopen", "--skip-reset"),
        configuration=_system(fake.rig_lasers()), nidaqmx=SimpleNamespace(),
        spawn=lambda case, level=None: process, log=log, instrument="PXI1Slot5/ai3",
        level_volts=2.0, minimum_volts=0.0, now=clock.now, sleep=clock.sleep, kill=process.kill_by)
    assert observations[0].after_reopen is None and observations[0].after_reset is None
    assert daq.tasks == []                                       # nothing was reopened

def test_run_matrix_keeps_the_rows_it_finished_when_one_fails(daq, tmp_path):
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    first = _Process()
    def spawn(case, level=None):
        if case.mode == "sigkill":
            return first
        raise OSError("fork failed")
    observations = []
    with pytest.raises(OSError, match="fork failed"):
        tool.run_matrix(
            [KillCase("sigkill", "software", "dc"), KillCase("abort", "software", "dc")], observations,
            args=_matrix_args(tmp_path, "--skip-reopen", "--skip-reset"),
            configuration=_system(fake.rig_lasers()), nidaqmx=SimpleNamespace(),
            spawn=spawn, log=log, instrument="PXI1Slot5/ai3", level_volts=2.0, minimum_volts=0.0,
            now=clock.now, sleep=clock.sleep, kill=first.kill_by)
    assert [o.case.mode for o in observations] == ["sigkill"]


# ---------------------------------------------------------------------------
# Fix round 1 (Task 3 review).

class _Immortal(_Process):
    """A victim stuck in the driver: SIGKILL is recorded and does nothing."""
    def kill_by(self, pid, sig):
        assert pid == self.pid
        self.signals.append(sig)

def _config_file(tmp_path):
    configuration = SystemConfiguration()
    configuration.laser = fake.rig_lasers()
    configuration.save_file(tmp_path / "system_configuration", as_yaml=True)
    return tmp_path / "system_configuration.yaml"

def _plan(tmp_path, *flags):
    path = _config_file(tmp_path)
    args = tool.parse_args(["--dry-run", f"--config={path}", f"--record-dir={tmp_path / 'rec'}", *flags])
    # execute() refuses a namespace that has not attested the laser is unable to
    # emit. These tests hand it fakes only, so the attestation is set on the
    # namespace here; the parser's --laser-disconnected is never passed.
    args.laser_disconnected = True
    configuration = _system(fake.rig_lasers())
    return args, configuration, tool.make_plan(args, configuration)

_STARTED = tool.datetime(2026, 11, 5, 14, 30, 0, tzinfo=tool.timezone.utc)
_STAMP = "20261105T143000Z"

def _samples(level_at, count=3000):
    return [LevelSample(100 + i / 100, level_at(100 + i / 100), level_at(100 + i / 100),
                        level_at(100 + i / 100)) for i in range(count)]

def _held_until_the_reopen(t):
    return 2.0 if t < 106.2 else 0.0

class _FakeReader:
    """start_reader's stand-in: fills the log with chosen samples, lives until stopped."""
    def __init__(self, samples=()):
        self.samples, self.calls, self.thread = list(samples), [], None
    def __call__(self, nidaqmx, channel, terminal, log, stop):
        self.calls.append((channel, terminal))
        for sample in self.samples:
            log.append(sample)
        self.thread = threading.Thread(target=stop.wait, daemon=True)
        self.thread.start()
        return self.thread

def _factory(make):
    def spawn_factory(args, *, stamp):
        return lambda case, level=None: make(case, level)
    return spawn_factory

def _execute(daq, tmp_path, make, *flags, samples=None, reader=None, clock=None, **seams):
    args, configuration, plan = _plan(tmp_path, *flags)
    clock = clock or _Clock()
    by_pid = {}
    def tracked(case, level):
        process = make(case, level)
        by_pid[process.pid] = process
        return process
    reader = reader or _FakeReader(_samples(_held_until_the_reopen) if samples is None else samples)
    seams.setdefault("kill", lambda pid, sig: by_pid[pid].kill_by(pid, sig))
    reader.by_pid = by_pid
    try:
        status = tool.execute(
            args, configuration, plan, _STARTED, load_nidaqmx=lambda: daq,
            spawn_factory=_factory(tracked), reader_factory=reader,
            now=clock.now, sleep=clock.sleep, **seams)
    except KeyboardInterrupt as error:
        # Left to propagate it would abort the whole pytest session.
        pytest.fail(f"execute let {error!r} escape")
    return status, args, reader

def _killable(case, level):
    if case.refusal:
        return _Process(pid=4300, ready="REFUSED run_synchronized_pulse_train refuses ...",
                        alive=False, returncode=2)
    return _Process(pid=4242, ready=f"READY pid=4242 timing={case.timing} waveform={case.waveform} mode={case.mode} level=2.0000")

# C1: a slow before prompt cannot make a dead train read safe.

def _run_with_a_before_prompt(prompt_seconds, level_at, hold_start=101.0):
    clock = _Clock()
    log = _log([(100 + i / 100, level_at(100 + i / 100)) for i in range(0, 6000)])
    process = _Process()
    events = []
    def kill(pid, sig):
        events.append(("kill", clock.t))
        process.kill_by(pid, sig)
    def dmm(instant):
        events.append((instant, clock.t))
        if instant == "before":
            clock.sleep(prompt_seconds)
        return None
    observation = tool.run_case(
        KillCase("sigkill", "software", "dc"), spawn=lambda c: process, log=log, now=clock.now,
        sleep=clock.sleep, kill=kill, reopen=lambda: None, reset=lambda: None,
        hold_seconds=5.0, instrument="ai3", dmm=dmm)
    return observation, events

def test_the_before_prompt_is_asked_first_and_before_is_read_at_the_kill():
    # The output drops while the operator reads the meter (15 s, under the limit):
    # before, read after the prompt, shows it, and the row is not safe.
    observation, events = _run_with_a_before_prompt(
        15.0, lambda t: 2.0 if t < 109.5 else 0.0)
    assert [name for name, _ in events][:2] == ["before", "kill"]
    assert events[1][1] >= 116.0                       # the kill came after the prompt
    assert observation.before.maximum == 0.0           # read after it, at the kill
    verdict = judge(observation, minimum_volts=0.0, level_volts=2.0)
    assert verdict != SAFE_ON_DEATH and verdict == UNKNOWN

def test_a_forty_second_before_prompt_leaves_the_row_unknown_with_a_note():
    # The requested case: 2 V until 129.5 and a 40 s prompt.
    observation, _ = _run_with_a_before_prompt(40.0, lambda t: 2.0 if t < 129.5 else 0.0)
    assert observation.before is None
    assert "before prompt took 40 s" in observation.notes[0]
    assert judge(observation, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_before_prompt_over_twenty_seconds_cannot_read_safe_when_the_output_drops_just_after_it():
    # The output holds until the prompt ends and drops a moment later: before
    # reads 2 V, the kill lands on a dead output, and everything after reads 0 V.
    observation, _ = _run_with_a_before_prompt(25.0, lambda t: 2.0 if t < 126.02 else 0.0)
    assert observation.after_kill.mean == 0.0 and observation.hold.maximum == 0.0
    assert observation.before is None and "over 20 s" in observation.notes[0]
    assert "it read 2.000 V" in observation.notes[0]
    assert judge(observation, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

def test_a_before_prompt_within_the_limit_keeps_its_reading_and_has_no_note():
    observation, _ = _run_with_a_before_prompt(19.0, lambda t: 2.0 if t < 125.0 else 0.0)
    assert observation.before.maximum == 2.0 and observation.notes == ()

def test_a_victim_that_closed_itself_during_the_prompt_is_not_a_safe_row():
    # The 60 s cue timeout: the victim closed, the signal lands on a zombie.
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 160.0 else 0.0) for i in range(0, 9000)])
    process = _Process(alive=False, returncode=3)
    def dmm(instant):
        if instant == "before":
            clock.sleep(65.0)
    observation = tool.run_case(
        KillCase("sigkill", "software", "dc"), spawn=lambda c: process, log=log, now=clock.now,
        sleep=clock.sleep, kill=lambda pid, sig: None, reopen=lambda: None, reset=lambda: None,
        hold_seconds=5.0, instrument="ai3", dmm=dmm)
    assert observation.victim_exit == "exit 3"
    assert judge(observation, minimum_volts=0.0, level_volts=2.0) == UNKNOWN

# C2: the orchestrator's own death.

def test_the_first_interrupt_raises_and_later_ones_are_only_recorded():
    gate = tool.InterruptGate(raising=True)
    with pytest.raises(KeyboardInterrupt, match="15"):
        gate(signal.SIGTERM, None)
    gate(signal.SIGINT, None)        # a second one cannot cut the recovery short
    gate(signal.SIGTERM, None)
    assert gate.raised == [signal.SIGTERM] and gate.pending == [signal.SIGINT, signal.SIGTERM]

def test_a_gate_that_is_not_raising_only_records():
    gate = tool.InterruptGate()
    gate(signal.SIGTERM, None)
    assert gate.pending == [signal.SIGTERM] and gate.raised == []

def test_the_run_region_raises_once_and_leaves_the_gate_not_raising():
    gate = tool.InterruptGate()
    with pytest.raises(KeyboardInterrupt):
        with gate.interruptible():
            gate(signal.SIGINT, None)
    assert gate.raising is False
    gate(signal.SIGINT, None)                    # after the run: held
    assert gate.pending == [signal.SIGINT]

def test_a_signal_during_a_deferred_stretch_is_raised_when_the_stretch_ends_inside_the_run():
    gate = tool.InterruptGate()
    with gate.interruptible():
        with gate.deferred():
            gate(signal.SIGTERM, None)           # held while a victim is started
            assert gate.pending == [signal.SIGTERM]
        assert gate.raising is True              # back to what it interrupted
        with pytest.raises(KeyboardInterrupt, match="15"):
            gate.raise_if_pending()
    assert gate.pending == [] and gate.raised == [signal.SIGTERM]

def test_a_signal_held_before_the_run_raises_on_entering_it():
    gate = tool.InterruptGate()
    gate(signal.SIGHUP, None)
    with pytest.raises(KeyboardInterrupt, match="1"):
        with gate.interruptible():
            raise AssertionError("the run must not begin")

def test_sigint_sigterm_and_sighup_all_become_the_one_interrupt():
    installed = {}
    handler = tool.install_interrupt_handlers(install=lambda number, h: installed.__setitem__(number, h))
    assert set(installed) == {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    assert all(h is handler for h in installed.values())

def _matrix(daq, tmp_path, process, log, clock, *flags, cases=(KillCase("sigkill", "software", "dc"),),
            stops=None):
    """run_matrix over `cases`; with `stops`, expects it to stop with RunStopped matching that."""
    observations = []
    run = lambda: tool.run_matrix(
        list(cases), observations,
        args=tool.parse_args(["--dry-run", f"--record-dir={tmp_path}", *flags]),
        configuration=_system(fake.rig_lasers()), nidaqmx=daq,
        spawn=lambda case, level=None: process, log=log, instrument="PXI1Slot5/ai3",
        level_volts=2.0, minimum_volts=0.0, now=clock.now, sleep=clock.sleep, kill=process.kill_by)
    if stops is None:
        run()
    else:
        with pytest.raises(tool.RunStopped, match=stops):
            run()
    return observations

def test_each_row_line_names_the_victims_pid_and_its_notes(daq, tmp_path, capsys):
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    _matrix(daq, tmp_path, _Process(), log, _Clock(), "--skip-reset")
    text = capsys.readouterr().out
    assert "sigkill-software-dc (victim pid 4242)" in text
    assert "note: reset skipped" in text

def test_a_refused_reopen_is_printed_on_its_row(daq, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: "DaqError: -50103 reserved")
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.7 else 0.0) for i in range(0, 900)])
    # The only row is the last, and a refusal in the last row stops the run (N2),
    # after its line has been printed.
    _matrix(daq, tmp_path, _Process(), log, _Clock(), stops="last row")
    assert "reopen: DaqError: -50103 reserved" in capsys.readouterr().out

# C3: the failure path.

def test_a_row_that_fails_sigkills_its_victim_at_once_and_says_if_it_survives(capsys):
    process = _Immortal()
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    def broken_reopen():
        raise RuntimeError("the orchestrator's own bug")
    with pytest.raises(RuntimeError, match="own bug"):
        _run(KillCase("hang", "software", "dc"), process, log, clock, reopen=broken_reopen)
    assert process.signals == [signal.SIGUSR1, signal.SIGKILL]
    text = capsys.readouterr().out
    assert "STILL ALIVE" in text and "4242" in text

def test_kill_victim_does_not_signal_a_pid_that_has_already_been_reaped():
    process = _Process(alive=False, returncode=0)
    assert tool.kill_victim(process, kill=process.kill_by, timeout=0.01) is True
    assert process.signals == []

def test_end_victim_says_what_happened_not_always_hang():
    quiet = _Process(ignores_sigterm=True)
    text = tool.end_victim(quiet, kill=quiet.kill_by, timeout=0.01, mode="clean")
    assert "sigkill by pid" in text and "hang" not in text
    immortal = _Immortal()
    assert tool.end_victim(immortal, kill=immortal.kill_by, timeout=0.01) == (
        "still alive after sigkill by pid 4242")

def test_a_failed_run_recovers_and_names_what_it_cannot_confirm(daq, tmp_path, capsys):
    def make(case, level):
        if case.mode == "abort":
            raise OSError("fork failed")
        return _killable(case, level)
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc,abort-software-dc")
    text = capsys.readouterr().out
    assert status == 1
    # Row 1's own reopen and reset, then the recovery's: the output was written
    # to its minimum and the 6713 reset again.
    assert daq.writes_to("PXI1Slot4/ao0") == [0.0] * 4
    assert daq.resets == ["PXI1Slot4", "PXI1Slot4"]
    assert "DID NOT COMPLETE" in text and "NOT CONFIRMED SAFE" in text
    assert "PXI1Slot5/port0/line4" in text and "NOT READABLE by this tool" in text
    assert "0.000 V" in text and "DO NOT RECONNECT" in text
    # The row that finished is kept, in the file as it ended and in the record.
    rows = [json.loads(line) for line in (args.record_dir / f"laser-kill-{_STAMP}.rows.jsonl").read_text().splitlines()]
    assert [list(r)[0] for r in rows] == ["header", "observation", "recovery"]
    record = json.loads((args.record_dir / f"laser-kill-{_STAMP}.json").read_text())
    assert record["recovery"]["reopen"] == "ok" and record["recovery"]["volts"] == pytest.approx(0.0)

def test_an_interrupt_during_a_row_gets_the_same_recovery(daq, tmp_path, capsys):
    def make(case, level):
        raise KeyboardInterrupt("signal 15")
    status, _, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc")
    text = capsys.readouterr().out
    assert status == 1 and "KeyboardInterrupt" in text and "NOT CONFIRMED SAFE" in text
    assert daq.writes_to("PXI1Slot4/ao0") == [0.0, 0.0] and daq.resets == ["PXI1Slot4"]

def test_the_recovery_banner_reports_a_level_it_read_above_the_minimum(daq, tmp_path, capsys):
    def make(case, level):
        raise OSError("fork failed")
    stuck_high = [LevelSample(100 + i / 100, 2.0, 2.0, 2.0) for i in range(0, 3000)]
    _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", samples=stuck_high)
    text = capsys.readouterr().out
    assert "2.000 V" in text and "ABOVE THE MINIMUM" in text

def test_the_recovery_banner_says_when_the_reader_has_nothing(daq, tmp_path, capsys):
    def make(case, level):
        raise OSError("fork failed")
    _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", samples=[])
    assert "NO READING" in capsys.readouterr().out

def test_a_recovery_the_driver_refuses_is_named_in_the_banner(daq, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: "DaqError: -50103 reserved")
    def make(case, level):
        raise OSError("fork failed")
    _execute(daq, tmp_path, make, "--cases=sigkill-software-dc")
    text = capsys.readouterr().out
    assert "reopen: DaqError: -50103 reserved" in text and "closes it" in text

def test_a_run_that_skipped_the_reopen_ends_with_the_shutter_warning(daq, tmp_path, capsys):
    status, _, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc", "--skip-reopen")
    text = capsys.readouterr().out
    assert status == 0
    assert "--skip-reopen" in text and "shutter line" in text and "DO NOT RECONNECT" in text

def test_a_run_that_reopened_does_not_print_that_warning(daq, tmp_path, capsys):
    status, _, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc")
    assert status == 0 and "--skip-reopen" not in capsys.readouterr().out

# I1: a wedged driver cannot hold the row.

def test_a_call_that_blocks_is_given_up_on_and_reported_as_timed_out():
    release = threading.Event()
    started = time.monotonic()
    text = tool.bounded(lambda: release.wait(5.0) and None, 0.05, "reopen")
    release.set()
    assert text.startswith("timed out") and time.monotonic() - started < 2.0

def test_a_call_that_finishes_or_fails_is_reported_as_it_is():
    assert tool.bounded(lambda: None, 1.0, "reopen") is None
    assert tool.bounded(lambda: "DaqError: x", 1.0, "reopen") == "DaqError: x"
    def boom():
        raise RuntimeError("driver said no")
    assert tool.bounded(boom, 1.0, "reset") == "RuntimeError: driver said no"

def test_a_reset_that_never_returns_does_not_hold_the_row(daq, tmp_path, monkeypatch):
    release = threading.Event()
    class _Wedged:
        def __init__(self, name):
            pass
        def reset_device(self):
            release.wait(5.0)
    daq.system = SimpleNamespace(Device=_Wedged)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    observations = _matrix(daq, tmp_path, _Process(), log, _Clock(), stops="reset timed out")
    release.set()
    # The row is kept, and is the last: the run stops with the reset outstanding.
    assert observations[0].reset_error.startswith("timed out")
    assert observations[0].victim_exit == "signal 9"

def test_a_reopen_that_never_returns_does_not_hold_the_row(daq, tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: release.wait(5.0) and None)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    observations = _matrix(daq, tmp_path, _Process(), log, _Clock(), stops="reopen timed out")
    release.set()
    assert observations[0].reopen_error.startswith("timed out")
    assert observations[0].reset_error is None and observations[0].victim_exit == "signal 9"

def test_a_reopen_that_never_returns_is_reported_by_the_same_bound(daq):
    daq.hang = {"task_close"}
    daq.held_wait_limit = 5.0
    text = tool.bounded(lambda: tool.reopen(_system(fake.rig_lasers()), 1), 0.05, "reopen")
    daq.hang_released.set()
    assert text.startswith("timed out")

# I2

def test_the_preconditions_name_the_laser_being_run_and_every_channel_the_reopen_rewrites():
    text = tool.preconditions(2, "PXI1Slot4")
    assert "laser 2 key OFF" in text and "laser 1" not in text
    assert "EVERY configured laser channel" in text and "PXI1Slot4" in text
    assert "minimum" in text and "shutter closed" in text

# I3

def test_a_level_too_close_to_the_minimum_is_refused_when_the_run_is_planned(tmp_path, capsys):
    path = _config_file(tmp_path)
    args = tool.parse_args(["--dry-run", f"--config={path}", "--level=0.1"])
    with pytest.raises(ValueError, match="tell apart"):
        tool.make_plan(args, _system(fake.rig_lasers()))
    assert tool.main(["--dry-run", f"--config={path}", "--level=0.1"]) == 2
    assert "tell apart" in capsys.readouterr().out

# I5: rows survive, the table comes first, and a broken stdout loses nothing.

def test_the_table_is_printed_before_the_record_is_written(daq, tmp_path, capsys, monkeypatch):
    def write_record_fails(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(tool, "write_record", write_record_fails)
    status, args, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc")
    text = capsys.readouterr().out
    assert "| case |" in text and "sigkill-software-dc" in text and "verdicts:" in text
    # The failed write is reported, with where the rows are.
    assert status == 1 and "could not write the record: disk full" in text and ".rows.jsonl" in text

def test_a_row_is_on_disk_as_it_ends_not_only_in_the_final_record(daq, tmp_path, monkeypatch):
    seen = []
    real = tool._append_row
    def spy(path, observation):
        real(path, observation)
        seen.append(len(path.read_text().splitlines()))
    monkeypatch.setattr(tool, "_append_row", spy)
    _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc,clean-software-pulse")
    assert seen == [2, 3]    # header + row 1, then + row 2

def test_a_broken_stdout_does_not_lose_rows_or_the_record(daq, tmp_path, monkeypatch):
    class _Closed(io.StringIO):
        def write(self, text):
            raise BrokenPipeError("tee is gone")
        def flush(self):
            raise BrokenPipeError("tee is gone")
    monkeypatch.setattr(sys, "stdout", _Closed())
    status, args, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc,clean-software-pulse")
    assert status == 0
    lines = (args.record_dir / f"laser-kill-{_STAMP}.rows.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert (args.record_dir / f"laser-kill-{_STAMP}.json").exists()

# I6: main past the dry run, with fakes.

@pytest.mark.parametrize("argv", [[], ["--tracking-check"], ["--cases", "sigkill-software-dc"],
                                  ["--pause-for-dmm", "--skip-reopen"]])
def test_every_way_of_running_without_dry_run_still_needs_the_disconnected_flag(argv):
    with pytest.raises(SystemExit):
        tool.parse_args(argv)

def test_execute_runs_the_rows_prints_the_table_then_writes_the_record(daq, tmp_path, capsys):
    status, args, reader = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc,clean-software-pulse")
    text = capsys.readouterr().out
    assert status == 0 and reader.calls == [("PXI1Slot5/ai3", "rse")]
    assert text.index("| case |") < text.index("record: ")
    assert "refused_by_design" in text and "verdicts:" in text
    record = json.loads((args.record_dir / f"laser-kill-{_STAMP}.json").read_text())
    assert [c["case"]["mode"] for c in record["cases"]] == ["sigkill", "clean"]
    assert (args.record_dir / f"laser-kill-{_STAMP}.md").exists()
    reader.thread.join(2.0)
    assert not reader.thread.is_alive()                  # stopped with the run

def test_execute_stops_with_a_message_when_the_reader_cannot_start(daq, tmp_path, capsys):
    def reader(*args):
        raise RuntimeError("DAQmx -50103: the specified resource is reserved")
    def make(case, level):
        raise AssertionError("no victim may start without the reader")
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", reader=reader)
    text = capsys.readouterr().out
    assert status == 1 and "reachAQ may be running" in text and "-50103" in text
    assert not args.record_dir.exists()

def test_execute_runs_the_tracking_check_and_stops(daq, tmp_path, capsys):
    def make(case, level):
        pid = 5000 + int(level)
        return _CuedProcess(pid=pid, ready=f"READY pid={pid} timing=software waveform=dc mode=clean level={level:.4f}")
    status, args, _ = _execute(daq, tmp_path, make, "--tracking-check",
                               samples=_samples(lambda t: float(int(t - 100.0))))
    text = capsys.readouterr().out
    assert status == 0 and "reader (V)" in text and "victim pid 5000" in text
    assert not list(args.record_dir.glob("*.json"))

def test_the_config_is_hashed_before_the_run_not_after(daq, tmp_path):
    original = {}
    def make(case, level):
        # The run itself changes the file, as a rig run that saved on exit would.
        path = tmp_path / "system_configuration.yaml"
        original.setdefault("sha", hashlib.sha256(path.read_bytes()).hexdigest())
        path.write_text(path.read_text() + "# changed during the run\n")
        return _killable(case, level)
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc")
    record = json.loads((args.record_dir / f"laser-kill-{_STAMP}.json").read_text())
    assert record["config_sha256"] == original["sha"]

# Minors.

def test_an_empty_cases_option_is_an_error_not_every_row(tmp_path, capsys):
    path = _config_file(tmp_path)
    assert tool.main(["--dry-run", f"--config={path}", "--cases="]) == 2
    assert "no case id" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no case id"):
        tool.select_cases(tool.kill_cases(), "")

def test_an_unreadable_ready_line_is_a_row_without_a_kill_like_the_other_no_ready_cases():
    process = _Process(ready="READY pid=not-a-number level=2.0")
    observation = _run(KillCase("sigkill", "software", "dc"), process, _log([]), _Clock())
    assert "unreadable READY" in observation.notes[0]
    assert process.signals == [signal.SIGKILL]     # the end of the spawned process only
    assert observation.before is None

def test_the_reader_restarts_its_task_after_a_read_error_but_only_so_often(daq):
    def always_fails(task, count):
        raise RuntimeError("DAQmx -200279")
    daq.read_samples = always_fails
    log, stop = tool.LevelLog(), threading.Event()
    thread = tool.start_reader(daq, "PXI1Slot5/ai3", "rse", log, stop)
    deadline = time.monotonic() + 5.0
    while len(log.restarts) < tool.READER_MAX_RESTARTS and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)
    stop.set()
    thread.join(5.0)
    assert len(log.restarts) == tool.READER_MAX_RESTARTS
    assert log.error.startswith("RuntimeError: DAQmx -200279")

def test_a_restart_that_fails_is_recorded_and_the_reader_goes_on(daq):
    calls = []
    def read_samples(task, count):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("DAQmx -200279")
        time.sleep(0.002)
        return [1.0]
    daq.read_samples = read_samples
    starts = []
    real_fail_start = daq.fail_start
    def fail_every_start_after_the_first(task):
        starts.append(task.label)
        if len(starts) > 1:
            raise RuntimeError(f"DAQmx refused to start {task.label}")
    daq.fail_start = fail_every_start_after_the_first
    log, stop = tool.LevelLog(), threading.Event()
    thread = tool.start_reader(daq, "PXI1Slot5/ai3", "rse", log, stop)
    deadline = time.monotonic() + 5.0
    while len(log.samples()) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(5.0)
    assert any("failed" in text and "refused to start" in text for text in log.restarts)
    assert len(log.samples()) >= 2

def test_the_victim_command_passes_the_disconnected_attestation_and_this_pid():
    args = tool.parse_args(["--dry-run"])
    command = tool.victim_command("python", Path("/w/v.py"), args, KillCase("clean", "software", "dc"))
    assert "--laser-disconnected" in command
    assert f"--orchestrator-pid={os.getpid()}" in command


# ---------------------------------------------------------------------------
# Fix round 2 (re-review of Task 3). The scenarios of the reviewers' probe, as
# tests: real signals, sent by the test to its own process, with the handlers
# that main installs; fakes for the driver and the victims.

@pytest.fixture
def gate():
    """The real handlers on SIGINT, SIGTERM and SIGHUP, as main installs them."""
    numbers = [getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP")]
    saved = {number: signal.getsignal(number) for number in numbers}
    handler = tool.install_interrupt_handlers()
    yield handler
    for number, previous in saved.items():
        signal.signal(number, previous)

def _signal_me(gate, number=signal.SIGTERM):
    # Never signal this process unless the handler under test is the one in place.
    assert signal.getsignal(number) is gate
    os.kill(os.getpid(), number)

def _read_jsonl(args):
    path = args.record_dir / f"laser-kill-{_STAMP}.rows.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]

_FULL_BANNER = ("DID NOT COMPLETE", "NOT CONFIRMED SAFE", "reopen: ", "command output on",
                "NOT READABLE", "DO NOT RECONNECT")

def _clock_that_signals(gate, at, number=signal.SIGTERM):
    """A clock whose sleep sends a real signal the first time it reaches `at`."""
    class _SignallingClock(_Clock):
        sent = False
        def sleep(self, seconds):
            super().sleep(seconds)
            if not self.sent and self.t >= at:
                self.sent = True
                _signal_me(gate, number)
    return _SignallingClock()

class _SignalOnWait(_Process):
    """wait() sends a real signal the first time it is called: main is then inside end_victim."""
    def __init__(self, gate, **kwargs):
        super().__init__(**kwargs)
        self.gate, self.signalled = gate, False
    def wait(self, timeout=None):
        if not self.signalled:
            self.signalled = True
            _signal_me(self.gate, signal.SIGTERM)
        return super().wait(timeout)

# C3a: the first signal can arrive during the recovery.

def test_a_signal_during_the_recovery_is_held_and_the_banner_and_record_finish(daq, tmp_path, capsys, monkeypatch, gate):
    def reopen_that_is_interrupted(configuration, laser):
        _signal_me(gate, signal.SIGINT)      # lands on the main thread, which waits in bounded()
        time.sleep(0.2)
    monkeypatch.setattr(tool, "reopen", reopen_that_is_interrupted)
    def make(case, level):
        raise OSError("fork failed")
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", interrupts=gate)
    text = capsys.readouterr().out
    assert status == 1
    for needle in _FULL_BANNER:
        assert needle in text, needle
    assert "arrived after the run had stopped or finished" in text
    assert gate.raised == [] and gate.pending == [signal.SIGINT]
    recovery = _read_jsonl(args)[-1]["recovery"]
    assert recovery["reopen"] == "ok"
    assert recovery["signals_held"] == {"count": 1, "by_signal": {"SIGINT": 1}}

def test_a_signal_during_the_report_after_a_finished_run_cannot_cut_the_record(daq, tmp_path, capsys, monkeypatch, gate):
    real = tool.write_record
    def write_record_while_signalled(*args, **kwargs):
        _signal_me(gate, signal.SIGTERM)
        return real(*args, **kwargs)
    monkeypatch.setattr(tool, "write_record", write_record_while_signalled)
    status, args, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc", interrupts=gate)
    text = capsys.readouterr().out
    assert status == 0 and "record: " in text and "held until everything above was complete" in text
    assert (args.record_dir / f"laser-kill-{_STAMP}.json").exists()

def test_a_signal_in_the_hold_window_stops_the_row_and_the_recovery_still_finishes(daq, tmp_path, capsys, gate):
    # The control: the first signal lands in the run itself, while main sleeps.
    clock = _clock_that_signals(gate, at=101.1)
    def make(case, level):
        return _Process(pid=4242, ready="READY pid=4242 timing=software waveform=dc mode=hang level=2.0000",
                        ignores_sigterm=True)
    status, args, reader = _execute(daq, tmp_path, make, "--cases=hang-software-dc", interrupts=gate, clock=clock)
    text = capsys.readouterr().out
    assert status == 1 and "stopped: KeyboardInterrupt: signal 15" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle
    assert reader.by_pid[4242].signals == [signal.SIGUSR1, signal.SIGKILL]   # cued, then killed at once
    assert "victim pid 4242" in text and gate.raised == [signal.SIGTERM]

# C3b: no instant at which a victim exists and nothing owns it.

def test_a_signal_just_after_the_victim_exists_still_ends_it_and_prints_its_pid(daq, tmp_path, capsys, gate):
    def make(case, level):
        process = _killable(case, level)
        _signal_me(gate, signal.SIGTERM)     # lands before run_case's own work has begun
        return process
    status, args, reader = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", interrupts=gate)
    text = capsys.readouterr().out
    victim = reader.by_pid[4242]
    assert status == 1 and victim.signals == [signal.SIGKILL] and not victim.alive
    assert "(victim pid 4242)" in text and "ended by sigkill after the failure" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle

def test_a_signal_while_the_tracking_check_waits_for_a_cued_victim_still_kills_it(daq, tmp_path, capsys, gate):
    def make(case, level):
        pid = 5000 + int(level)
        return _SignalOnWait(gate, pid=pid, ready=f"READY pid={pid} timing=software waveform=dc mode=clean level={level:.4f}")
    status, args, reader = _execute(daq, tmp_path, make, "--tracking-check", interrupts=gate,
                                    samples=_samples(lambda t: 0.0))
    text = capsys.readouterr().out
    victim = reader.by_pid[5000]
    assert status == 1
    assert victim.signals == [signal.SIGUSR1, signal.SIGKILL] and not victim.alive
    assert "tracking 0 V (victim pid 5000)" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle

def test_a_signal_just_after_a_tracking_victim_exists_still_ends_it(daq, tmp_path, capsys, gate):
    def make(case, level):
        pid = 5000 + int(level)
        process = _CuedProcess(pid=pid, ready=f"READY pid={pid} timing=software waveform=dc mode=clean level={level:.4f}")
        _signal_me(gate, signal.SIGTERM)
        return process
    status, _, reader = _execute(daq, tmp_path, make, "--tracking-check", interrupts=gate)
    assert status == 1 and reader.by_pid[5000].signals == [signal.SIGKILL]

def test_a_signal_before_the_run_starts_starts_nothing(daq, tmp_path, capsys, gate):
    gate(signal.SIGHUP, None)          # held: it arrived where it may not raise
    def make(case, level):
        raise AssertionError("no victim may start")
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", interrupts=gate)
    assert status == 1 and "before any victim was started" in capsys.readouterr().out
    assert daq.writes_to("PXI1Slot4/ao0") == [] and daq.resets == []

# N1: nothing runs while a stale driver call is outstanding.

def test_a_timed_out_reopen_stops_the_run_before_the_next_row(daq, tmp_path, capsys, monkeypatch):
    release, calls, spawned = threading.Event(), [], []
    def reopen(configuration, laser):
        calls.append(1)
        if len(calls) == 1:
            release.wait(5.0)                   # the row's own reopen: stale, still running
    monkeypatch.setattr(tool, "reopen", reopen)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    def make(case, level):
        spawned.append(case.case_id)
        return _killable(case, level)
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    release.set()
    text = capsys.readouterr().out
    assert status == 1 and spawned == ["sigkill-software-dc"]        # the second row never started
    assert "RunStopped" in text and "the reopen timed out" in text and "outstanding" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle
    assert [list(r)[0] for r in _read_jsonl(args)] == ["header", "observation", "recovery"]

def test_a_timed_out_reset_stops_the_run_too(daq, tmp_path, capsys, monkeypatch):
    release, spawned = threading.Event(), []
    monkeypatch.setattr(tool, "reset_device", lambda nidaqmx, device: release.wait(5.0) and None)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    def make(case, level):
        spawned.append(case.case_id)
        return _killable(case, level)
    status, _, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    release.set()
    assert status == 1 and spawned == ["sigkill-software-dc"]
    assert "the reset timed out" in capsys.readouterr().out

def test_a_victim_still_alive_after_sigkill_stops_the_run_before_the_next_row(daq, tmp_path, capsys):
    spawned = []
    def make(case, level):
        spawned.append(case.case_id)
        return _Immortal()
    status, _, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    text = capsys.readouterr().out
    assert status == 1 and spawned == ["sigkill-software-dc"]
    assert "still alive after sigkill by pid 4242" in text

def test_a_refusal_in_an_earlier_row_is_still_a_measurement_not_a_reason_to_stop(daq, tmp_path, monkeypatch):
    # -50103 from the reopen is the table's evidence, not a wedge: the run goes
    # on. (The same refusal in the LAST row is not confirmed safe: see below.)
    calls, spawned = [], []
    def reopen(configuration, laser):
        calls.append(1)
        return "DaqError: -50103 reserved" if len(calls) == 1 else None
    monkeypatch.setattr(tool, "reopen", reopen)
    def make(case, level):
        spawned.append(case.case_id)
        return _killable(case, level)
    status, _, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    assert status == 0 and len(spawned) == 2

# N2: a run that ends with its last recovery failed is not a success.

def test_a_run_whose_last_row_leaves_everything_wedged_exits_1_with_the_banner(daq, tmp_path, capsys, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: release.wait(5.0) and None)
    monkeypatch.setattr(tool, "reset_device", lambda nidaqmx, device: release.wait(5.0) and None)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    status, args, _ = _execute(daq, tmp_path, lambda case, level: _Immortal(), "--cases=sigkill-hardware-dc")
    release.set()
    text = capsys.readouterr().out
    assert status == 1
    for needle in _FULL_BANNER:
        assert needle in text, needle
    assert "reopen: timed out" in text and "still alive after sigkill by pid 4242" in text
    record = json.loads((args.record_dir / f"laser-kill-{_STAMP}.json").read_text())
    assert record["recovery"]["reopen"].startswith("timed out")

def test_a_tracking_victim_that_survives_sigkill_stops_the_check(daq, tmp_path, capsys):
    def make(case, level):
        pid = 5000 + int(level)
        return _Immortal(pid=pid, ready=f"READY pid={pid} timing=software waveform=dc mode=clean level={level:.4f}")
    status, _, _ = _execute(daq, tmp_path, make, "--tracking-check")
    text = capsys.readouterr().out
    assert status == 1 and "still alive after sigkill" in text and "NOT CONFIRMED SAFE" in text

# M3, M5, and the reader's message.

def test_skip_reopen_says_the_recovery_closed_the_shutter_line_when_it_did(daq, tmp_path, capsys):
    def make(case, level):
        raise OSError("fork failed")
    _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", "--skip-reopen")
    text = capsys.readouterr().out
    assert "the recovery's reopen above did, and said ok" in text
    assert "nothing in it closed the shutter line" not in text

def test_skip_reopen_says_so_when_the_recovery_could_not_close_it_either(daq, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: "DaqError: -50103 reserved")
    def make(case, level):
        raise OSError("fork failed")
    _execute(daq, tmp_path, make, "--cases=sigkill-software-dc", "--skip-reopen")
    text = capsys.readouterr().out
    assert "nothing in it closed the shutter line" in text and "did not succeed either" in text

def test_execute_refuses_a_namespace_that_has_not_attested_the_laser_is_disconnected(tmp_path):
    args, configuration, plan = _plan(tmp_path)
    args.laser_disconnected = False
    # No seams given: past the check, the autouse guard would raise instead.
    with pytest.raises(ValueError, match="laser_disconnected"):
        tool.execute(args, configuration, plan, _STARTED)
    with pytest.raises(ValueError, match="laser_disconnected"):
        tool.execute(SimpleNamespace(), configuration, plan, _STARTED)

def test_a_reader_that_fails_for_another_reason_is_not_blamed_on_reachaq(daq, tmp_path, capsys):
    def reader(*args):
        raise ValueError("unknown analog terminal configuration 'sideways'")
    status, _, _ = _execute(daq, tmp_path, lambda case, level: _killable(case, level),
                            "--cases=sigkill-software-dc", reader=reader)
    text = capsys.readouterr().out
    assert status == 1 and "unknown analog terminal configuration 'sideways'" in text
    assert "reachAQ" not in text and "nothing was started" in text


# ---------------------------------------------------------------------------
# Fix round 3 (second re-review of Task 3).

def _high(count=3000):
    return [LevelSample(100 + i / 100, 2.0, 2.0, 2.0) for i in range(count)]

class _ExplodingClock(_Clock):
    """A clock whose sleep fails (a bug in the orchestrator, mid-row) once it reaches `at`."""
    def __init__(self, at):
        super().__init__()
        self.at, self.exploded = at, False
    def sleep(self, seconds):
        super().sleep(seconds)
        if self.t >= self.at and not self.exploded:
            self.exploded = True            # once: the recovery's own sleeps must work
            raise RuntimeError("the clock failed")

@pytest.fixture
def stale_close(daq, monkeypatch):
    """The real controller on the fake driver, with a refused open whose own close hangs in the driver."""
    monkeypatch.setattr(nidaq_laser, "_FAILED_OPEN_CLOSE_TIMEOUT_S", 0.05)
    daq.failing_write = "manual_ao"                      # the open is refused (-50103)
    release = threading.Event()
    real_sick = daq.sick
    def sick(call):
        if call == "task_close" and threading.current_thread().name == "NidaqLaserFailedOpenClose":
            release.wait(5.0)
        else:
            real_sick(call)
    daq.sick = sick
    yield release
    release.set()
    for thread in threading.enumerate():
        if thread.name == "NidaqLaserFailedOpenClose":
            thread.join(2.0)

# N1: a controller whose failed open is still closing.

def test_a_refused_open_whose_close_is_still_running_is_wedged_and_outstanding(daq, stale_close):
    text = tool.reopen(_system(fake.rig_lasers()), 1)
    assert isinstance(text, tool.WedgedText)
    assert text.startswith("LaserControllerStillClosing")
    assert tool.outstanding_calls() == ["the controller's close after a refused open"]
    stale_close.set()
    deadline = time.monotonic() + 5.0
    while tool.outstanding_calls() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tool.outstanding_calls() == []                 # it ended, and is no longer listed

def test_a_still_closing_error_wrapped_by_another_is_found_on_its_chain(daq, monkeypatch):
    closing = nidaq_laser.LaserControllerStillClosing(RuntimeError("x"), threading.Event())
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = closing
    assert tool._still_closing(wrapper) is closing
    assert tool._still_closing(RuntimeError("plain")) is None

def test_a_stale_close_of_a_refused_open_stops_the_run_before_the_next_row(daq, stale_close, tmp_path, capsys, monkeypatch):
    # The real reopen, the real controller, a fake driver. Row 2 must not start
    # while the close can still write the minimum, as late as just after its kill.
    monkeypatch.setattr(tool, "OUTSTANDING_WAIT_S", 0.2)
    spawned = []
    def make(case, level):
        spawned.append(case.case_id)
        return _killable(case, level)
    status, args, _ = _execute(daq, tmp_path, make, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    text = capsys.readouterr().out
    assert status == 1 and spawned == ["sigkill-software-dc"]
    assert "RunStopped" in text and "LaserControllerStillClosing" in text and "outstanding" in text
    assert "the controller's close after a refused open is STILL RUNNING" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle
    assert [list(r)[0] for r in _read_jsonl(args)] == ["header", "observation", "recovery"]

# N2: the last row's recovery.

def _run_a_last_row(daq, tmp_path, capsys, monkeypatch, *flags, reopen=None, reset=None,
                    samples=None, cases="--cases=sigkill-hardware-pulse-r1"):
    if reopen is not None:
        monkeypatch.setattr(tool, "reopen", reopen)
    if reset is not None:
        monkeypatch.setattr(tool, "reset_device", reset)
    status, args, _ = _execute(daq, tmp_path, _killable, cases, *flags, samples=samples)
    return status, args, capsys.readouterr().out

def _refused(configuration, laser):
    return "DaqError: -50103 reserved"

def test_a_last_row_whose_reopen_was_refused_is_not_confirmed_safe_even_at_0_v(daq, tmp_path, capsys, monkeypatch):
    # reopen -50103, reset ok, 0 V: held_until_reset. Nothing closed the shutter line.
    samples = _samples(lambda t: 2.0 if t < 106.7 else 0.0)
    status, args, text = _run_a_last_row(daq, tmp_path, capsys, monkeypatch, reopen=_refused, samples=samples)
    assert status == 1
    assert "RunStopped" in text and "its reopen said: DaqError: -50103" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle
    assert _read_jsonl(args)[-1]["recovery"]["shutter_line"] == "PXI1Slot5/port0/line4"

def test_a_last_row_whose_reopen_and_reset_were_both_refused_with_the_output_up_exits_1(daq, tmp_path, capsys, monkeypatch):
    status, _, text = _run_a_last_row(
        daq, tmp_path, capsys, monkeypatch, reopen=_refused,
        reset=lambda nidaqmx, device: "DaqError: reset refused", samples=_high())
    assert status == 1
    assert "its reopen said" in text and "ABOVE THE MINIMUM" in text and "DO NOT RECONNECT" in text

def test_a_last_row_with_both_recoveries_ok_but_the_output_still_up_exits_1(daq, tmp_path, capsys, monkeypatch):
    status, _, text = _run_a_last_row(daq, tmp_path, capsys, monkeypatch, samples=_high())
    assert status == 1
    assert "the output read 2.000 V after it" in text
    for needle in _FULL_BANNER:
        assert needle in text, needle

def test_a_last_row_run_with_both_recoveries_skipped_and_the_output_up_exits_1(daq, tmp_path, capsys, monkeypatch):
    status, _, text = _run_a_last_row(
        daq, tmp_path, capsys, monkeypatch, "--skip-reopen", "--skip-reset", samples=_high())
    assert status == 1 and "the output read 2.000 V after it" in text

def test_a_last_row_that_the_reader_never_saw_is_not_confirmed_either(daq, tmp_path, capsys, monkeypatch):
    status, _, text = _run_a_last_row(daq, tmp_path, capsys, monkeypatch, samples=[])
    assert status == 1 and "the reader saw nothing after it" in text and "NO READING" in text

def test_a_last_row_with_a_clean_recovery_at_the_minimum_still_exits_0(daq, tmp_path, capsys, monkeypatch):
    status, _, text = _run_a_last_row(daq, tmp_path, capsys, monkeypatch)
    assert status == 0 and "DO NOT RECONNECT" not in text

def test_a_last_row_the_controller_refused_by_design_is_not_held_to_it(daq, tmp_path, capsys, monkeypatch):
    status, _, text = _run_a_last_row(
        daq, tmp_path, capsys, monkeypatch, cases="--cases=clean-software-pulse", samples=_high())
    assert status == 0 and "DO NOT RECONNECT" not in text

# a. no abbreviations

@pytest.mark.parametrize("argv", [["--laser-d"], ["--dry-run", "--laser-d"], ["--dry"],
                                  ["--laser-disconnect"]])
def test_an_abbreviation_cannot_attest_or_stand_in_for_a_flag(argv, capsys):
    with pytest.raises(SystemExit):
        tool.parse_args(argv)
    assert "unrecognized arguments" in capsys.readouterr().err

# b. a signal in the first instants of the failure handler

class _SignalledDuringAbandon(tool.InterruptGate):
    """A real signal landing as the failure handler starts: the second deferral is the abandon's."""
    def __init__(self):
        super().__init__(raising=True)
        self.calls = 0
    def deferred(self):
        self.calls += 1
        if self.calls == 2:
            self(signal.SIGTERM, None)       # raising here: becomes a KeyboardInterrupt
        return super().deferred()

def test_a_signal_in_the_first_instants_of_the_failure_handler_does_not_skip_the_kill():
    gate = _SignalledDuringAbandon()
    process = _Process()
    def broken_reopen():
        raise RuntimeError("the orchestrator's own bug")
    clock = _Clock()
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
    with pytest.raises(RuntimeError, match="own bug"):          # the original failure still propagates
        _run(KillCase("hang", "software", "dc"), process, log, clock,
             reopen=broken_reopen, interrupts=gate)
    assert process.signals == [signal.SIGUSR1, signal.SIGKILL] and not process.alive
    assert gate.raised == [signal.SIGTERM]
    assert process.pid == 4242

def test_the_signal_that_landed_on_a_non_signal_failure_is_named_in_the_stop_line(daq, tmp_path, capsys):
    status, _, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc",
                            interrupts=_SignalledDuringAbandon(), clock=_ExplodingClock(at=101.1))
    text = capsys.readouterr().out
    assert status == 1
    assert "stopped: RuntimeError: the clock failed (and 1 signal(s) (SIGTERM x1) arrived)" in text

# c. a storm of signals is a count

def test_signals_are_summarised_by_name_and_count():
    assert tool._signal_summary([signal.SIGINT] * 1000 + [signal.SIGTERM] * 5) == (
        "1005 signal(s) (SIGINT x1000, SIGTERM x5)")
    assert tool._signal_counts([signal.SIGHUP]) == {"count": 1, "by_signal": {"SIGHUP": 1}}

def test_a_storm_of_signals_is_reported_as_a_count_not_a_list(daq, tmp_path, capsys, monkeypatch, gate):
    real = tool.write_record
    def write_record_in_a_storm(*args, **kwargs):
        for _ in range(1000):
            gate(signal.SIGINT, None)
        for _ in range(5):
            gate(signal.SIGTERM, None)
        return real(*args, **kwargs)
    monkeypatch.setattr(tool, "write_record", write_record_in_a_storm)
    status, _, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc", interrupts=gate)
    text = capsys.readouterr().out
    assert status == 0
    assert "1005 signal(s) (SIGINT x1000, SIGTERM x5)" in text and "[2, 2, 2" not in text

# d. the recovery waits for what an earlier row left running, and says so

def test_the_recovery_waits_for_a_call_an_earlier_row_left_running_and_names_it(daq, tmp_path, capsys, monkeypatch):
    release, calls = threading.Event(), []
    def reopen(configuration, laser):
        calls.append(1)
        if len(calls) == 1:
            release.wait(5.0)                       # the row's own reopen, given up on
    monkeypatch.setattr(tool, "reopen", reopen)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(tool, "OUTSTANDING_WAIT_S", 3.0)
    threading.Timer(0.3, release.set).start()
    status, args, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    text = capsys.readouterr().out
    assert status == 1
    assert "the reopen call finished after" in text and "before the recovery made calls of its own" in text
    outstanding = _read_jsonl(args)[-1]["recovery"]["outstanding"]
    assert [(o["call"], o["finished"]) for o in outstanding] == [("the reopen call", True)]
    assert len(calls) == 2                           # the recovery's own reopen came after it

def test_the_recovery_says_when_the_earlier_call_is_still_running_and_may_collide(daq, tmp_path, capsys, monkeypatch):
    release = threading.Event()
    calls = []
    def reopen(configuration, laser):
        calls.append(1)
        if len(calls) == 1:
            release.wait(5.0)
    monkeypatch.setattr(tool, "reopen", reopen)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(tool, "OUTSTANDING_WAIT_S", 0.1)
    status, args, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc,sigkill-hardware-dc")
    release.set()
    text = capsys.readouterr().out
    assert status == 1
    assert "the reopen call is STILL RUNNING after" in text and "collide" in text
    assert _read_jsonl(args)[-1]["recovery"]["outstanding"][0]["finished"] is False

def test_nothing_outstanding_means_no_waiting_and_no_line_about_it(daq, tmp_path, capsys):
    def make(case, level):
        raise OSError("fork failed")
    _execute(daq, tmp_path, make, "--cases=sigkill-software-dc")
    text = capsys.readouterr().out
    assert "still running in the driver from earlier" not in text

# e. a surviving victim's pid and fate on disk

def test_a_surviving_victims_pid_and_fate_are_in_the_jsonl(daq, tmp_path, capsys):
    status, args, _ = _execute(daq, tmp_path, lambda case, level: _Immortal(),
                               "--cases=sigkill-software-dc", clock=_ExplodingClock(at=101.1))
    text = capsys.readouterr().out
    assert status == 1 and "VICTIM pid 4242 IS STILL ALIVE" in text
    victim = _read_jsonl(args)[-1]["recovery"]["victim"]
    assert victim["pid"] == 4242 and victim["ended"] is False and "STILL ALIVE" in victim["detail"]

def test_a_victim_ended_after_a_failure_is_in_the_jsonl_too(daq, tmp_path, capsys):
    status, args, _ = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc",
                               clock=_ExplodingClock(at=101.1))
    victim = _read_jsonl(args)[-1]["recovery"]["victim"]
    assert status == 1 and victim["pid"] == 4242 and victim["ended"] is True

# f. a failed tracking check on disk

def test_a_failed_tracking_check_writes_its_banner_content_to_disk(daq, tmp_path, capsys):
    def make(case, level):
        pid = 5000 + int(level)
        return _Immortal(pid=pid, ready=f"READY pid={pid} timing=software waveform=dc mode=clean level={level:.4f}")
    status, args, _ = _execute(daq, tmp_path, make, "--tracking-check")
    capsys.readouterr()
    path = args.record_dir / f"laser-kill-{_STAMP}.tracking.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert status == 1 and [list(line)[0] for line in lines] == ["header", "recovery"]
    recovery = lines[1]["recovery"]
    assert "still alive after sigkill" in recovery["reason"]
    assert recovery["shutter_line"] == "PXI1Slot5/port0/line4" and recovery["reopen"] == "ok"

def test_a_tracking_check_that_finished_writes_no_file(daq, tmp_path, capsys):
    def make(case, level):
        pid = 5000 + int(level)
        return _CuedProcess(pid=pid, ready=f"READY pid={pid} timing=software waveform=dc mode=clean level={level:.4f}")
    status, args, _ = _execute(daq, tmp_path, make, "--tracking-check",
                               samples=_samples(lambda t: float(int(t - 100.0))))
    assert status == 0 and not list(args.record_dir.glob("*.jsonl"))

def test_a_signal_while_a_row_waits_on_its_reopen_leaves_that_call_named_in_the_banner(daq, tmp_path, capsys, monkeypatch, gate):
    # The first signal lands while main waits in bounded() on the row's own
    # reopen, which goes on in the driver. The recovery's calls would collide
    # with it, so the recovery waits for it first and says so.
    release, calls = threading.Event(), []
    def reopen(configuration, laser):
        calls.append(1)
        if len(calls) == 1:
            _signal_me(gate, signal.SIGTERM)       # lands on main, inside bounded()'s join
            release.wait(5.0)
    monkeypatch.setattr(tool, "reopen", reopen)
    status, args, reader = _execute(daq, tmp_path, _killable, "--cases=sigkill-software-dc", interrupts=gate)
    release.set()
    text = capsys.readouterr().out
    assert status == 1 and "stopped: KeyboardInterrupt: signal 15" in text
    assert "the reopen call is STILL RUNNING after" in text and "collide" in text
    assert reader.by_pid[4242].signals == [signal.SIGKILL] and not reader.by_pid[4242].alive
    outstanding = _read_jsonl(args)[-1]["recovery"]["outstanding"]
    assert [(o["call"], o["finished"]) for o in outstanding] == [("the reopen call", False)]
