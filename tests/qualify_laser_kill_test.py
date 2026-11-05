import importlib.util, io, json, queue, signal, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import pytest
from autotrainer.core import SystemConfiguration
from autotrainer.device import nidaq_laser
from tools.acquisition.model.laser_kill_matrix import (
    HELD_UNTIL_RESET, UNKNOWN, KillCase, LevelSample, from_record, judge)

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
    for function in (tool.apply_kill, tool.end_victim, tool.run_case,
                     tool.tracking_check, tool.run_matrix):
        monkeypatch.setitem(function.__kwdefaults__, "kill", refuse)


@pytest.fixture
def daq(monkeypatch):
    daq = fake.FakeDaqmx()
    # The fake has no continuous acquisition, and nothing like Device.
    daq.constants.AcquisitionType.CONTINUOUS = "continuous"
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
    assert observation.victim_exit == "killed after hang (sigkill by pid)"

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
