import hashlib, importlib.util, io, json, queue, signal, sys, threading, time
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
    log = _log([(100 + i / 100, 2.0) for i in range(0, 900)])
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
    status = tool.execute(
        args, configuration, plan, _STARTED, load_nidaqmx=lambda: daq,
        spawn_factory=_factory(tracked), reader_factory=reader,
        now=clock.now, sleep=clock.sleep, **seams)
    return status, args, reader

def _killable(case, level):
    if case.refusal:
        return _Process(pid=4300, ready="REFUSED run_synchronized_pulse_train refuses ...",
                        alive=False, returncode=2)
    return _Process(pid=4242, ready=f"READY pid=4242 timing={case.timing} waveform={case.waveform} mode={case.mode} level=2.0000")

# C1: a slow before prompt cannot make a dead train read safe.

def test_the_before_prompt_is_asked_first_and_before_is_read_at_the_kill():
    clock = _Clock()
    # 2 V until 129.5: the operator's before prompt takes 40 s, and the train
    # (30 s) is over by the time the kill lands.
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 129.5 else 0.0) for i in range(0, 5500)])
    process = _Process()
    events = []
    def kill(pid, sig):
        events.append(("kill", clock.t))
        process.kill_by(pid, sig)
    def dmm(instant):
        events.append((instant, clock.t))
        if instant == "before":
            clock.sleep(40.0)
        return None
    observation = tool.run_case(
        KillCase("sigkill", "software", "dc"), spawn=lambda c: process, log=log, now=clock.now,
        sleep=clock.sleep, kill=kill, reopen=lambda: None, reset=lambda: None,
        hold_seconds=5.0, instrument="ai3", dmm=dmm)
    assert [name for name, _ in events][:2] == ["before", "kill"]
    assert events[1][1] >= 141.0                       # the kill came after the prompt
    assert observation.before.maximum == 0.0           # read after it, at the kill
    verdict = judge(observation, minimum_volts=0.0, level_volts=2.0)
    assert verdict != SAFE_ON_DEATH and verdict == UNKNOWN

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

def test_the_first_interrupt_raises_and_later_ones_are_ignored():
    handler = tool.InterruptOnce()
    with pytest.raises(KeyboardInterrupt, match="15"):
        handler(signal.SIGTERM, None)
    handler(signal.SIGINT, None)        # a second one cannot cut the recovery short
    handler(signal.SIGTERM, None)

def test_sigint_sigterm_and_sighup_all_become_the_one_interrupt():
    installed = {}
    handler = tool.install_interrupt_handlers(install=lambda number, h: installed.__setitem__(number, h))
    assert set(installed) == {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    assert all(h is handler for h in installed.values())

def _matrix(daq, tmp_path, process, log, clock, *flags, cases=(KillCase("sigkill", "software", "dc"),)):
    observations = []
    tool.run_matrix(
        list(cases), observations,
        args=tool.parse_args(["--dry-run", f"--record-dir={tmp_path}", *flags]),
        configuration=_system(fake.rig_lasers()), nidaqmx=daq,
        spawn=lambda case, level=None: process, log=log, instrument="PXI1Slot5/ai3",
        level_volts=2.0, minimum_volts=0.0, now=clock.now, sleep=clock.sleep, kill=process.kill_by)
    return observations

def test_each_row_line_names_the_victims_pid_and_its_notes(daq, tmp_path, capsys):
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.7 else 0.0) for i in range(0, 900)])
    _matrix(daq, tmp_path, _Process(), log, _Clock(), "--skip-reset")
    text = capsys.readouterr().out
    assert "sigkill-software-dc (victim pid 4242)" in text
    assert "note: reset skipped" in text

def test_a_refused_reopen_is_printed_on_its_row(daq, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: "DaqError: -50103 reserved")
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.7 else 0.0) for i in range(0, 900)])
    _matrix(daq, tmp_path, _Process(), log, _Clock())
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
    observations = _matrix(daq, tmp_path, _Process(), log, _Clock())
    release.set()
    assert observations[0].reset_error.startswith("timed out")
    assert observations[0].victim_exit == "signal 9"

def test_a_reopen_that_never_returns_does_not_hold_the_row(daq, tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(tool, "reopen", lambda configuration, laser: release.wait(5.0) and None)
    monkeypatch.setattr(tool, "RECOVERY_TIMEOUT_S", 0.05)
    log = _log([(100 + i / 100, 2.0 if 100 + i / 100 < 106.2 else 0.0) for i in range(0, 900)])
    observations = _matrix(daq, tmp_path, _Process(), log, _Clock())
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

def test_the_victim_command_passes_the_disconnected_attestation():
    args = tool.parse_args(["--dry-run"])
    command = tool.victim_command("python", Path("/w/v.py"), args, KillCase("clean", "software", "dc"))
    assert "--laser-disconnected" in command
