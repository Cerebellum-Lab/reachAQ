"""What the laser command output does when the process holding it dies.

For each row of the matrix in tools/acquisition/model/laser_kill_matrix.py this
starts a victim (tools/hardware/laser_kill_victim.py), which opens the output
the way reachAQ does and holds it in the row's state, kills it the way the row
says, and then tries the two recoveries an operator has: restart reachAQ (a new
NidaqLaserController opened and closed on the live configuration) and reset the
device. What the output reads at each instant comes from a reader owned by this
process: a continuous 10 kHz analog input on the other board (the 6221), on the
copy of the command the laser's commandCopyInput carries. It shares no handle
with the victim, so a victim's death does not touch it. A DMM on the command
output is the cross-check for a driver wedged hard enough to stall the reader;
--pause-for-dmm prompts for its readings.

    python qualify_laser_kill.py --dry-run
    python qualify_laser_kill.py --laser-disconnected --tracking-check --pause-for-dmm
    python qualify_laser_kill.py --laser-disconnected
    python qualify_laser_kill.py --laser-disconnected --cases sigkill-software-dc

Everything but --dry-run drives the real command output and opens the shutter
line, so the laser must be unable to emit first (PRECONDITIONS, printed on
every run), and the operator's reachAQ must not be running: it holds the 6221
inputs and the laser. --dry-run loads the configuration, lists the rows and the
reader channel, and stops: it starts no process and opens no NI task. Nothing
here saves the configuration (it is loaded with save_backup=False and no
AppModel is built). The only process this signals is the victim it spawned, by
the pid it was given.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    ".",
    "auto-trainer-core/src",
    "auto-trainer-device/src",
    "auto-trainer-model/src",
    "auto-trainer-pyside/src",
    "auto-trainer-behavior/src",
    "auto-trainer-inference/src",
    "auto-trainer-video/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _path))

from autotrainer.core.configuration import SystemConfiguration
from autotrainer.device import NidaqLaserController
from autotrainer.device.nidaq_signal_stream import resolve_analog_terminal_config
from tools.acquisition.model.laser_kill_matrix import (
    KillCase, KillObservation, LevelReading, LevelSample, format_table, judge,
    kill_cases, level_at, level_between, to_record,
)

READER_RATE_HZ = 10_000.0
#: Samples per read: 10 ms at the reader's rate.
READER_CHUNK = 100
CHUNK_S = READER_CHUNK / READER_RATE_HZ
#: Each instant is the mean, minimum and maximum over this window.
WINDOW_S = 0.05
AFTER_KILL_S = 0.10
READY_TIMEOUT_S = 30.0
END_TIMEOUT_S = 5.0
#: From READY to the kill, and after each recovery to its reading.
SETTLE_S = 1.0
RECOVERY_SETTLE_S = 0.5
#: Ruling T1-I1b: a hold window the reader's chunks cover less than this much
#: of cannot show the output stayed down.
HOLD_COVERAGE = 0.9
TRACKING_LEVELS = (0.0, 1.0, 2.0, 4.0)
DEFAULT_RECORD_DIR = Path.home() / "Autotrainer" / "laser-kill"
#: Pushed on the victim's line queue when its output ends.
STREAM_CLOSED = "STREAM CLOSED"
VICTIM_PATH = Path(__file__).with_name("laser_kill_victim.py")

PRECONDITIONS = """\
Before any row: the laser must be unable to emit.
  - laser 1 key OFF
  - the command BNC unplugged at the laser driver
  - the shutter cable unplugged (pulse rows open the shutter line)
  - the command output on a BNC tee: a dummy load of 1 kOhm or more (10 kOhm
    termination; the 6713 drives +-5 mA) on one arm, the DMM on the other
  - the operator's reachAQ not running (it holds the 6221 inputs and the laser)"""

_REFUSAL = (
    "refusing to drive the laser output: pass --laser-disconnected once the key "
    "is off, the command BNC and shutter cable are unplugged and the output is "
    "on the dummy load"
)


def _say(text: str = "") -> None:
    print(text, flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kill a process that holds the laser command output and read the output.")
    parser.add_argument(
        "--config", default=os.path.expanduser("~/Autotrainer/system_configuration.yaml"))
    parser.add_argument("--laser", type=int, default=1)
    parser.add_argument("--level", type=float, default=2.0, help="command volts")
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--pulse-repeats", type=int, default=3)
    parser.add_argument("--cases", help="comma-separated case ids; default all")
    parser.add_argument(
        "--reader-channel", help="the input that reads the command; default the laser's commandCopyInput")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list the rows and the reader channel and stop; opens nothing")
    parser.add_argument(
        "--laser-disconnected", action="store_true",
        help="the laser is unable to emit (see the preconditions); required to run")
    parser.add_argument("--tracking-check", action="store_true")
    parser.add_argument("--pause-for-dmm", action="store_true")
    parser.add_argument("--skip-reopen", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.laser_disconnected:
        parser.error(_REFUSAL)
    return args


def _same_channel(first: Optional[str], second: Optional[str]) -> bool:
    if not first or not second:
        return False
    return first.lstrip("/").lower() == second.lstrip("/").lower()


def reader_channel_for(
    configuration, laser: int, override: Optional[str],
) -> Tuple[str, Optional[str]]:
    """The channel that reads the command, and how the stream references its inputs."""
    channel = configuration.laser.get_channel(laser)
    chosen = override or channel.command_copy_input
    if not chosen:
        raise ValueError(
            f"laser {laser} has no commandCopyInput; pass --reader-channel with an "
            "input that reads its command")
    if _same_channel(chosen, channel.diode_input):
        raise ValueError(
            f"{chosen} is laser {laser}'s diode input, which floats on this rig; "
            "pass --reader-channel with an input that reads its command")
    return chosen, configuration.nidaq_stream.analog_terminal_config or None


def select_cases(cases: Sequence[KillCase], spec: Optional[str]) -> Tuple[KillCase, ...]:
    """The cases `spec` names (comma-separated ids), in the matrix's order."""
    if not spec:
        return tuple(cases)
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    if not wanted:
        raise ValueError("--cases names no case id")
    known = {case.case_id for case in cases}
    unknown = [item for item in wanted if item not in known]
    if unknown:
        raise ValueError(
            f"unknown case id(s): {', '.join(unknown)}; the matrix has: "
            f"{', '.join(case.case_id for case in cases)}")
    return tuple(case for case in cases if case.case_id in set(wanted))


def device_of(physical_channel: str) -> str:
    """The device in a channel name: PXI1Slot4 in PXI1Slot4/ao0."""
    return physical_channel.lstrip("/").split("/")[0]


class LevelLog:
    """What the reader has seen, shared with the thread that appends to it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: List[LevelSample] = []
        #: The reader's most recent read error, as text; None if it has had none.
        self.error: Optional[str] = None

    def append(self, sample: LevelSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def samples(self) -> Tuple[LevelSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def at(self, time: float, instrument: str) -> Optional[LevelReading]:
        return level_at(self.samples(), time, window_s=WINDOW_S, instrument=instrument)

    def between(self, start: float, end: float, instrument: str) -> Optional[LevelReading]:
        return level_between(self.samples(), start, end, instrument=instrument)


def _terminal_enum(nidaqmx, terminal_config):
    """The stream's analogTerminalConfig as nidaqmx's constant; None for none given."""
    if terminal_config is None or not isinstance(terminal_config, str):
        return terminal_config
    resolved = resolve_analog_terminal_config(nidaqmx, terminal_config)
    if resolved is None and terminal_config:
        # Letting DAQmx choose is not neutral on a 6221 (resolve_analog_terminal_config).
        raise ValueError(f"unknown analog terminal configuration {terminal_config!r}")
    return resolved


def _close_quietly(task) -> None:
    try:
        task.close()
    except Exception:
        pass


def _read_until(task, log: LevelLog, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            try:
                values = task.read(number_of_samples_per_channel=READER_CHUNK, timeout=1.0)
            except Exception as error:
                log.error = f"{type(error).__name__}: {error}"
                stop.wait(0.05)
                continue
            if len(values) == 0:
                continue
            log.append(LevelSample(
                time.perf_counter(), sum(values) / len(values), min(values), max(values)))
    finally:
        _close_quietly(task)


def start_reader(
    nidaqmx, physical_channel: str, terminal_config, log: LevelLog, stop: threading.Event,
) -> threading.Thread:
    """Sample `physical_channel` continuously into `log` until `stop` is set.

    The task is made and started here, so a refusal (-50103: something else
    holds the input) is raised to the caller, not lost in the thread.
    """
    terminal = _terminal_enum(nidaqmx, terminal_config)
    task = nidaqmx.Task("laser_kill_reader")
    try:
        if terminal is None:
            task.ai_channels.add_ai_voltage_chan(physical_channel)
        else:
            task.ai_channels.add_ai_voltage_chan(physical_channel, terminal_config=terminal)
        task.timing.cfg_samp_clk_timing(
            rate=READER_RATE_HZ,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=100_000,
        )
        task.start()
    except BaseException:
        _close_quietly(task)
        raise
    thread = threading.Thread(
        target=_read_until, args=(task, log, stop), name="LaserKillReader", daemon=True)
    thread.start()
    return thread


def victim_command(python: str, victim_path: Path, args, case: KillCase) -> List[str]:
    return [
        python, str(victim_path),
        f"--config={args.config}",
        f"--laser={args.laser}",
        f"--timing={case.timing}",
        f"--waveform={case.waveform}",
        f"--mode={case.mode}",
        f"--level={args.level}",
    ]


def wait_for_ready(lines: "queue.Queue[str]", timeout: float) -> dict:
    """What the victim's first protocol line says, within `timeout` seconds.

    READY gives {"pid": int, "level": float, ...}, REFUSED {"refused": text};
    the end of its output before either gives {"ended": text}. Other lines are
    skipped. TimeoutError when nothing decisive arrives in time.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError(f"no READY or REFUSED line within {timeout:g} s") from None
        line = line.strip()
        if line.startswith("READY"):
            fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
            try:
                return {**fields, "pid": int(fields["pid"]), "level": float(fields["level"])}
            except (KeyError, ValueError):
                raise ValueError(f"unreadable READY line: {line!r}") from None
        if line.startswith("REFUSED"):
            return {"refused": line[len("REFUSED"):].strip()}
        if line == STREAM_CLOSED:
            return {"ended": "victim output ended before READY"}


def apply_kill(case: KillCase, pid: int, *, kill=os.kill) -> None:
    """Signal `pid`, and only it: SIGTERM or SIGKILL as the row says, else the SIGUSR1 cue."""
    if case.mode == "sigterm":
        kill(pid, signal.SIGTERM)
    elif case.mode == "sigkill":
        kill(pid, signal.SIGKILL)
    else:
        kill(pid, signal.SIGUSR1)


def _exit_text(code: int) -> str:
    return f"signal {-code}" if code < 0 else f"exit {code}"


def end_victim(process, *, kill=os.kill, timeout: float = END_TIMEOUT_S) -> str:
    """Wait for the victim to end; SIGKILL it by its own pid if it does not."""
    try:
        return _exit_text(process.wait(timeout))
    except subprocess.TimeoutExpired:
        pass
    try:
        kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # gone between the wait and the signal
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        return f"still alive after sigkill by pid {process.pid}"
    return "killed after hang (sigkill by pid)"


def reopen(system_configuration, laser: int) -> Optional[str]:
    """What restarting reachAQ does to the output: open the controller, close it.

    Opening writes every channel's minimum command and closes its shutter.
    `laser` names the row's channel, which the configuration must have. None
    when it worked; otherwise the error as text, for the record.
    """
    try:
        system_configuration.laser.get_channel(laser)
        controller = NidaqLaserController(
            system_configuration.laser,
            feedback_reader=lambda _channel: 0.0,
            timing_plan=None,
            analog_terminal_config=system_configuration.nidaq_stream.analog_terminal_config or None,
        )
        controller.close()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def reset_device(nidaqmx, device_name: str) -> Optional[str]:
    try:
        nidaqmx.system.Device(device_name).reset_device()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _hold_reading(
    log: LevelLog, start: float, end: float, instrument: str,
) -> Optional[LevelReading]:
    """The hold window's reading, or None when the reader did not cover it.

    A partial window cannot prove the output stayed down (ruling T1-I1b).
    """
    reading = log.between(start, end, instrument)
    if reading is None or reading.sample_count < HOLD_COVERAGE * (end - start) / CHUNK_S:
        return None
    return reading


def _row_without_readings(
    case: KillCase, notes: Tuple[str, ...], victim_exit: str,
) -> KillObservation:
    return KillObservation(
        case=case, before=None, after_kill=None, hold=None, after_hold=None,
        after_reopen=None, after_reset=None, reopen_error=None, reset_error=None,
        victim_exit=victim_exit, notes=notes)


def run_case(
    case: KillCase,
    *,
    spawn: Callable,
    log: LevelLog,
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    kill=os.kill,
    reopen: Optional[Callable[[], Optional[str]]],
    reset: Optional[Callable[[], Optional[str]]],
    hold_seconds: float,
    instrument: str,
    dmm: Optional[Callable[[str], Optional[float]]] = None,
) -> KillObservation:
    """One row: start the victim, kill it, read the output at each instant, recover.

    `reopen` and `reset` are the two recoveries; None skips one and leaves its
    reading empty. Every wait is `sleep`, every instant is read off `log` at
    the time the row defines (not when the reading is taken, so an operator
    slow over the DMM prompt moves nothing), and the only process signalled is
    the one `spawn` returned, by its pid. A victim is never left behind: any
    failure ends it by its pid before it propagates.
    """
    process = spawn(case)
    try:
        problem = None
        try:
            ready = wait_for_ready(process.lines, READY_TIMEOUT_S)
        except TimeoutError:
            problem = f"no READY line within {READY_TIMEOUT_S:g} s"
        else:
            if "refused" in ready:
                problem = ready["refused"]
            elif "ended" in ready:
                problem = ready["ended"]
            elif ready["pid"] != process.pid:
                problem = (f"READY names pid {ready['pid']}, not the spawned pid "
                           f"{process.pid}; the row was not run")
        if problem is not None:
            return _row_without_readings(
                case, (problem,), end_victim(process, kill=kill))

        dmm_values = {}

        def ask(instant: str) -> None:
            if dmm is not None:
                value = dmm(instant)
                if value is not None:
                    dmm_values[instant] = value

        sleep(SETTLE_S)
        before = log.at(now(), instrument)
        ask("before")

        t_kill = now()
        apply_kill(case, process.pid, kill=kill)
        after_kill_at = t_kill + AFTER_KILL_S + WINDOW_S
        sleep(AFTER_KILL_S + WINDOW_S)
        after_kill = log.at(after_kill_at, instrument)
        ask("after_kill")

        hold_end = t_kill + hold_seconds
        sleep(max(0.0, hold_end - now()))
        hold = _hold_reading(log, after_kill_at, hold_end, instrument)
        after_hold = log.at(hold_end, instrument)
        ask("after_hold")

        notes = []
        if reopen is None:
            reopen_error, after_reopen = None, None
            notes.append("reopen skipped")
        else:
            reopen_error = reopen()
            sleep(RECOVERY_SETTLE_S)
            after_reopen = log.at(now(), instrument)
            ask("after_reopen")
        if reset is None:
            reset_error, after_reset = None, None
            notes.append("reset skipped")
        else:
            reset_error = reset()
            sleep(RECOVERY_SETTLE_S)
            after_reset = log.at(now(), instrument)
            ask("after_reset")

        return KillObservation(
            case=case, before=before, after_kill=after_kill, hold=hold,
            after_hold=after_hold, after_reopen=after_reopen, after_reset=after_reset,
            reopen_error=reopen_error, reset_error=reset_error,
            victim_exit=end_victim(process, kill=kill),
            dmm=dmm_values, notes=tuple(notes))
    except BaseException:
        try:
            end_victim(process, kill=kill)
        except Exception:
            pass
        raise


def prompt_dmm(case_id: str, instant: str, *, ask=input) -> Optional[float]:
    """Ask the operator for the DMM's reading; None for a blank answer."""
    while True:
        try:
            answer = ask(f"DMM reading for {case_id} {instant} (V): ").strip()
        except EOFError:
            return None
        if not answer:
            return None
        try:
            return float(answer)
        except ValueError:
            _say("not a number: type the voltage, or press Enter for no reading")


def tracking_check(
    args,
    configuration,
    log: LevelLog,
    spawn: Callable,
    dmm: Optional[Callable[[str], Optional[float]]],
    *,
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    kill=os.kill,
) -> List[Tuple[float, Optional[LevelReading], Optional[float]]]:
    """Does the reader follow the command? One clean software-DC victim per level.

    Read at READY plus a second, beside the DMM's reading when there is one.
    The victim is then cued to close and exit, and is ended by its pid if it
    does not.
    """
    instrument, _ = reader_channel_for(configuration, args.laser, args.reader_channel)
    case = KillCase("clean", "software", "dc")
    rows = []
    for level in TRACKING_LEVELS:
        process = spawn(case, level)
        try:
            ready = wait_for_ready(process.lines, READY_TIMEOUT_S)
            if ready.get("pid") != process.pid:
                raise RuntimeError(f"the victim at {level:g} V did not report ready: {ready}")
            sleep(SETTLE_S)
            reading = log.at(now(), instrument)
            value = dmm(f"{level:g} V") if dmm is not None else None
            apply_kill(case, process.pid, kill=kill)
        finally:
            end_victim(process, kill=kill)
        rows.append((level, reading, value))
    _say(f"{'level (V)':>10}  {'reader (V)':>11}  {'DMM (V)':>9}")
    for level, reading, value in rows:
        _say(f"{level:>10g}  "
             f"{'n/a' if reading is None else format(reading.mean, '.4f'):>11}  "
             f"{'n/a' if value is None else format(value, '.4f'):>9}")
    return rows


def run_matrix(
    cases: Sequence[KillCase],
    observations: List[KillObservation],
    *,
    args,
    configuration,
    nidaqmx,
    spawn: Callable,
    log: LevelLog,
    instrument: str,
    level_volts: float,
    minimum_volts: float,
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    kill=os.kill,
    prompt: Callable[[str, str], Optional[float]] = prompt_dmm,
) -> None:
    """Run each case in turn, appending its observation to `observations`.

    Appended as each row ends, so the rows finished before a failure survive
    it in the caller's list.
    """
    channel = configuration.laser.get_channel(args.laser)
    reopen_hook = None if args.skip_reopen else functools.partial(
        reopen, configuration, args.laser)
    reset_hook = None if args.skip_reset else functools.partial(
        reset_device, nidaqmx, device_of(channel.analog_output))
    for index, case in enumerate(cases, 1):
        _say(f"[{index}/{len(cases)}] {case.case_id}")
        dmm = functools.partial(prompt, case.case_id) if args.pause_for_dmm else None
        observation = run_case(
            case, spawn=spawn, log=log, now=now, sleep=sleep, kill=kill,
            reopen=reopen_hook, reset=reset_hook, hold_seconds=args.hold_seconds,
            instrument=instrument, dmm=dmm)
        observations.append(observation)
        verdict = judge(observation, minimum_volts=minimum_volts, level_volts=level_volts)
        _say(f"    {verdict}  (victim: {observation.victim_exit})")


def repository_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def write_record(args, observations, started: datetime, extra: dict) -> Path:
    """The JSON record and, beside it, the Markdown table; the JSON's path."""
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "generated_on": started.isoformat(),
        "host": socket.gethostname(),
        "reachaq_revision": repository_revision(),
        "laser": args.laser,
        "hold_seconds": args.hold_seconds,
        **extra,
    }
    record = to_record(observations, **header)
    path = args.record_dir / f"laser-kill-{stamp}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    table = format_table(
        observations, minimum_volts=extra["minimum_volts"], level_volts=extra["level_volts"])
    lines = [f"# Laser kill test {stamp}", ""]
    lines += [f"- {key}: {value}" for key, value in header.items()]
    path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n\n" + table + "\n", encoding="utf-8")
    return path


def _pump(stream, lines: "queue.Queue[str]") -> None:
    try:
        for line in stream:
            lines.put(line.rstrip("\r\n"))
    finally:
        lines.put(STREAM_CLOSED)
        try:
            stream.close()
        except Exception:
            pass


def make_spawn(
    args, *, stamp: str, python: str = sys.executable, victim_path: Path = VICTIM_PATH,
    popen=subprocess.Popen,
) -> Callable:
    """spawn(case, level=None): a victim process with its stdout lines on `.lines`.

    Its stderr goes to a file per row under --record-dir, and it has no stdin
    (the DMM prompt is this process's). `level` overrides --level for the
    tracking check.
    """
    def spawn(case: KillCase, level: Optional[float] = None):
        spawn_args = args if level is None else argparse.Namespace(**{**vars(args), "level": level})
        args.record_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if level is None else f"-{level:g}V"
        stderr_path = args.record_dir / f"laser-kill-{stamp}-{case.case_id}{suffix}.stderr.log"
        with stderr_path.open("a", encoding="utf-8") as stderr:
            process = popen(
                victim_command(python, victim_path, spawn_args, case),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr, text=True)
        process.lines = queue.Queue()
        threading.Thread(
            target=_pump, args=(process.stdout, process.lines),
            name=f"LaserKillVictimOutput-{process.pid}", daemon=True).start()
        return process

    return spawn


def _load_nidaqmx():
    import nidaqmx
    import nidaqmx.constants
    import nidaqmx.system
    return nidaqmx


def _describe_plan(args, cases, reader_channel, terminal_config, level_volts, minimum_volts) -> None:
    dc = sum(1 for case in cases if case.waveform == "dc")
    refused = sum(1 for case in cases if case.refusal)
    pulse = len(cases) - dc - refused
    _say(f"laser {args.laser}: reader {reader_channel} (analog terminal "
         f"{terminal_config or 'chosen by DAQmx'}), {READER_RATE_HZ:g} Hz in "
         f"{READER_CHUNK}-sample chunks")
    _say(f"command {level_volts:g} V, minimum {minimum_volts:g} V, hold {args.hold_seconds:g} s")
    _say(f"{len(cases)} cases ({dc} dc, {pulse} pulse, {refused} refused by design):")
    for case in cases:
        _say(f"  {case.case_id}")


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    _say(PRECONDITIONS)
    configuration = SystemConfiguration.load_yaml_file(args.config, save_backup=False)
    try:
        channel = configuration.laser.get_channel(args.laser)
        reader_channel, terminal_config = reader_channel_for(
            configuration, args.laser, args.reader_channel)
        cases = select_cases(kill_cases(pulse_repeats=args.pulse_repeats), args.cases)
    except (KeyError, ValueError) as error:
        _say(f"cannot plan the run: {error}")
        return 2
    level_volts = channel.clamp_command_voltage(args.level)
    minimum_volts = channel.minimum_command_volts
    _describe_plan(args, cases, reader_channel, terminal_config, level_volts, minimum_volts)
    if args.dry_run:
        _say("dry run: no process started, no NI task opened, nothing written")
        return 0

    nidaqmx = _load_nidaqmx()
    log, stop = LevelLog(), threading.Event()
    try:
        reader = start_reader(nidaqmx, reader_channel, terminal_config, log, stop)
    except Exception as error:
        _say(f"the reader on {reader_channel} did not start: {type(error).__name__}: {error}")
        _say("the operator's reachAQ may be running: it holds the 6221 inputs (-50103). "
             "Have its owner stop it; nothing was started or signalled.")
        return 1
    spawn = make_spawn(args, stamp=started.strftime("%Y%m%dT%H%M%SZ"))
    observations: List[KillObservation] = []
    status = 0
    try:
        if args.tracking_check:
            dmm = functools.partial(prompt_dmm, "tracking") if args.pause_for_dmm else None
            tracking_check(args, configuration, log, spawn, dmm)
        else:
            run_matrix(
                cases, observations, args=args, configuration=configuration, nidaqmx=nidaqmx,
                spawn=spawn, log=log, instrument=reader_channel, level_volts=level_volts,
                minimum_volts=minimum_volts)
    except (Exception, KeyboardInterrupt) as error:
        # Keep what was measured; the row that failed has ended its victim.
        _say(f"stopped: {type(error).__name__}: {error}")
        status = 1
    finally:
        stop.set()
        reader.join(END_TIMEOUT_S)
    if log.error:
        _say(f"the reader had read errors; the last: {log.error}")
    if observations:
        extra = {
            "reader_channel": reader_channel,
            "terminal_config": terminal_config,
            "level_volts": level_volts,
            "minimum_volts": minimum_volts,
            "command_copy_scale": channel.command_copy_scale,
            "config": str(args.config),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        }
        path = write_record(args, observations, started, extra)
        _say()
        _say(format_table(observations, minimum_volts=minimum_volts, level_volts=level_volts))
        _say()
        counts = Counter(
            judge(o, minimum_volts=minimum_volts, level_volts=level_volts) for o in observations)
        _say("verdicts: " + ", ".join(f"{verdict} {n}" for verdict, n in sorted(counts.items())))
        _say(f"record: {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
