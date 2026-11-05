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
line, so the laser must be unable to emit first (preconditions(), printed on
every run), and the operator's reachAQ must not be running: it holds the 6221
inputs and the laser. --dry-run loads the configuration, lists the rows and the
reader channel, and stops: it starts no process and opens no NI task. Nothing
here saves the configuration (it is loaded with save_backup=False and no
AppModel is built). The only process this signals is the victim it spawned, by
the pid it was given.

When a run stops early (an error, a row that leaves a reopen or reset
outstanding in the driver or a victim alive, or SIGINT, SIGTERM or SIGHUP, which
all become one KeyboardInterrupt) the victim in hand is sent SIGKILL by its pid,
from the moment it exists, and the run then tries, with a time bound on each, to
reopen the controller and reset the command device, reads the output once more,
and prints a banner naming what it could not confirm: the shutter line cannot be
read from here, and only the reopen closes it. A signal can stop the run only
inside the run itself (InterruptGate); during the recovery and the report it is
held, so they always finish. The rows are appended to a .rows.jsonl file as each
ends, so a lost terminal or a broken `| tee` loses none of them. The exit status
is 0 only for a run that finished with no reopen or reset still running in the
driver, no victim alive, and the last row's reopen and reset clean and its output
read at the minimum. A refusal (-50103) in an earlier row is data and the run goes
on; one in the last row, or an output still up, is not confirmed safe, so it goes
down the same path as a failure: recovery, banner, status 1.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
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
from autotrainer.device.nidaq_laser import LaserControllerStillClosing
from autotrainer.device.nidaq_signal_stream import resolve_analog_terminal_config
from tools.acquisition.model.laser_kill_matrix import (
    EXPECTED_VICTIM_EXIT, KillCase, KillObservation, LevelReading, LevelSample,
    format_table, judge, kill_cases, level_at, level_between, to_record,
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
#: A before-prompt (--pause-for-dmm) that took longer than this leaves the
#: output at the kill unknown: it could have dropped just after the reading.
BEFORE_PROMPT_LIMIT_S = 20.0
#: The longest one reopen or one device reset is waited for. A kill can leave
#: nipalk wedged, and a call into it then never returns.
RECOVERY_TIMEOUT_S = 20.0
#: How long the recovery waits for a call an earlier row gave up on (a timed-out
#: reopen or reset, a controller's close after a refused open) before it makes
#: calls of its own, which could collide with it.
OUTSTANDING_WAIT_S = 20.0
#: The same bound on the reader's restart of its task, and how many restarts in
#: a row (with no good read between) it will make.
RESTART_TIMEOUT_S = 5.0
READER_MAX_RESTARTS = 5
#: Less than this between the command level and the minimum and the judge cannot
#: tell a held output from a safe one (its tolerance is 0.05 V).
MIN_LEVEL_SPAN_V = 0.5
TRACKING_LEVELS = (0.0, 1.0, 2.0, 4.0)
DEFAULT_RECORD_DIR = Path.home() / "Autotrainer" / "laser-kill"
#: Pushed on the victim's line queue when its output ends.
STREAM_CLOSED = "STREAM CLOSED"
VICTIM_PATH = Path(__file__).with_name("laser_kill_victim.py")



def preconditions(laser: int, command_device: str = "the command device") -> str:
    return f"""\
Before any row: laser {laser} must be unable to emit.
  - laser {laser} key OFF
  - laser {laser}'s command BNC unplugged at the laser driver
  - laser {laser}'s shutter cable unplugged (pulse rows open the shutter line)
  - the command output on a BNC tee: a dummy load of 1 kOhm or more (10 kOhm
    termination; the 6713 drives +-5 mA) on one arm, the DMM on the other
  - the operator's reachAQ not running (it holds the 6221 inputs and the laser)
Every row reopens the laser controller as reachAQ does, which rewrites
EVERY configured laser channel, not only laser {laser}: each command to its
minimum, each shutter closed. The device reset after it resets every output
on {command_device}."""

_REFUSAL = (
    "refusing to drive the laser output: pass --laser-disconnected once the key "
    "is off, the command BNC and shutter cable are unplugged and the output is "
    "on the dummy load"
)


def _say(text: str = "") -> None:
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        # stdout is gone (a `| tee` that closed, a dropped ssh). The rows are on
        # disk as they end, so carry on; point stdout at nowhere so that neither
        # the next print nor the flush at exit raises again.
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass


def parse_args(argv=None) -> argparse.Namespace:
    # No abbreviations: --laser-d must not be taken for --laser-disconnected.
    parser = argparse.ArgumentParser(
        description="Kill a process that holds the laser command output and read the output.",
        allow_abbrev=False)
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
    """The cases `spec` names (comma-separated ids), in the matrix's order.

    None is every case. An empty or all-comma `spec` names none, which is an
    error, not every case.
    """
    if spec is None:
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
        self._restarts: List[str] = []

    def append(self, sample: LevelSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def samples(self) -> Tuple[LevelSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def record_restart(self, text: str) -> None:
        with self._lock:
            self._restarts.append(text)

    @property
    def restarts(self) -> Tuple[str, ...]:
        """What the reader did about its read errors, one line each."""
        with self._lock:
            return tuple(self._restarts)

    def at(self, time: float, instrument: str) -> Optional[LevelReading]:
        return level_at(self.samples(), time, window_s=WINDOW_S, instrument=instrument)

    def between(self, start: float, end: float, instrument: str) -> Optional[LevelReading]:
        return level_between(self.samples(), start, end, instrument=instrument)


class WedgedText(str):
    """An error text that says something is still outstanding.

    A call into the driver that was given up on and may yet act, or a victim
    that survived SIGKILL. It is made at the source, from what happened (a
    timeout, a LaserControllerStillClosing, a survivor), and the run stops on
    its type, never on its wording.
    """


#: Calls given up on that may still be running, as (what, waiter), where
#: waiter(timeout) is True once the call has ended. Under the lock.
_outstanding_lock = threading.Lock()
_outstanding: List[Tuple[str, Callable[[float], bool]]] = []


def _note_outstanding(what: str, waiter: Callable[[float], bool]) -> None:
    with _outstanding_lock:
        _outstanding.append((what, waiter))


def outstanding_calls() -> List[str]:
    """What was given up on and is still running; the ones that have ended are dropped."""
    with _outstanding_lock:
        _outstanding[:] = [(what, waiter) for what, waiter in _outstanding if not waiter(0)]
        return [what for what, _ in _outstanding]


def wait_for_outstanding(timeout: float) -> List[dict]:
    """Wait, up to `timeout` s in all, for each call still running; what became of each.

    {"call", "finished", "waited_s"} for every call that was still running when
    this began. A call that had already ended is not listed.
    """
    with _outstanding_lock:
        running = [(what, waiter) for what, waiter in _outstanding if not waiter(0)]
    deadline = time.monotonic() + timeout
    results = []
    for what, waiter in running:
        began = time.monotonic()
        finished = bool(waiter(max(0.0, deadline - began)))
        results.append({"call": what, "finished": finished,
                        "waited_s": round(time.monotonic() - began, 2)})
    outstanding_calls()
    return results


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return str(number)


def _signal_counts(signals: Sequence[int]) -> dict:
    """{"count": n, "by_signal": {"SIGINT": k, ...}}, which a storm of signals does not blow up."""
    by_signal: dict = {}
    for number in signals:
        name = _signal_name(number)
        by_signal[name] = by_signal.get(name, 0) + 1
    return {"count": len(signals), "by_signal": by_signal}


def _signal_summary(signals: Sequence[int]) -> str:
    counts = _signal_counts(signals)
    detail = ", ".join(f"{name} x{n}" for name, n in sorted(counts["by_signal"].items()))
    return f"{counts['count']} signal(s) ({detail})"


def bounded(call: Callable[[], Optional[str]], timeout: float, what: str) -> Optional[str]:
    """`call()` on a daemon thread, waited for at most `timeout` seconds.

    Its result (None, or an error as text); an exception becomes its text; and
    a call still running at the timeout gives a WedgedText, "timed out ...", and
    is left running in the driver, which is all that can be done about it. It is
    noted as outstanding, for the recovery to wait for first and name.
    """
    outcome: List[Optional[str]] = []

    def work() -> None:
        try:
            outcome.append(call())
        except Exception as error:
            outcome.append(f"{type(error).__name__}: {error}")

    thread = threading.Thread(target=work, name=f"LaserKill-{what}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        _note_outstanding(
            f"the {what} call", lambda wait: (thread.join(wait), not thread.is_alive())[1])
        return WedgedText(f"timed out after {timeout:g} s (the {what} call is still running)")
    return outcome[0]


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


def _restart_task(task, error: Exception) -> str:
    """Stop and start the reader's task after a read error, bounded; what happened."""
    def restart() -> None:
        task.stop()
        task.start()

    why = f"{type(error).__name__}: {error}"
    outcome = bounded(restart, RESTART_TIMEOUT_S, "reader restart")
    if outcome is None:
        return f"reader task restarted after {why}"
    return f"reader task restart after {why} failed: {outcome}"


def _read_until(task, log: LevelLog, stop: threading.Event) -> None:
    restarts_in_a_row = 0
    try:
        while not stop.is_set():
            try:
                values = task.read(number_of_samples_per_channel=READER_CHUNK, timeout=1.0)
            except Exception as error:
                log.error = f"{type(error).__name__}: {error}"
                # An error such as -200279 (the buffer overflowed) leaves the task
                # useless: stop and start it, a few times, and say so on the log.
                if restarts_in_a_row < READER_MAX_RESTARTS:
                    restarts_in_a_row += 1
                    log.record_restart(_restart_task(task, error))
                stop.wait(0.05)
                continue
            restarts_in_a_row = 0
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
        # The victim refuses to run without it. This process refused first,
        # unless --dry-run, which never starts a victim.
        "--laser-disconnected",
        # So that it closes the output and exits if this process goes away,
        # even one that went while the victim was still starting up.
        f"--orchestrator-pid={os.getpid()}",
    ]


def wait_for_ready(lines: "queue.Queue[str]", timeout: float) -> dict:
    """What the victim's first protocol line says, within `timeout` seconds.

    READY gives {"pid": int, "level": float, ...}, REFUSED {"refused": text};
    a READY it cannot read gives {"unreadable": line}; the end of the victim's
    output before any of them gives {"ended": text}. Other lines are skipped.
    TimeoutError when nothing decisive arrives in time.
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
                return {"unreadable": line}
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


def kill_victim(process, *, kill=os.kill, timeout: float = END_TIMEOUT_S) -> bool:
    """SIGKILL the victim by its own pid; True once it has ended within `timeout`.

    A process already reaped (its returncode is set) is not signalled: its pid
    may belong to someone else by now.
    """
    if getattr(process, "returncode", None) is None:
        try:
            kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # gone between the check and the signal
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def end_victim(
    process, *, kill=os.kill, timeout: float = END_TIMEOUT_S, mode: str = "hang",
) -> str:
    """Wait for the victim to end; SIGKILL it by its own pid if it does not.

    The text says what happened. "killed after hang" is the `hang` mode's own
    outcome (it ignores SIGTERM and is ended by this, by design); a victim of
    any other mode that had to be killed did not exit when it should have.
    """
    try:
        return _exit_text(process.wait(timeout))
    except subprocess.TimeoutExpired:
        pass
    if not kill_victim(process, kill=kill, timeout=timeout):
        return WedgedText(f"still alive after sigkill by pid {process.pid}")
    if mode == "hang":
        return "killed after hang (sigkill by pid)"
    return f"did not exit within {timeout:g} s; ended by sigkill by pid"


def reopen(system_configuration, laser: int) -> Optional[str]:
    """What restarting reachAQ does to the output: open the controller, close it.

    Opening writes every channel's minimum command and closes its shutter.
    `laser` names the row's channel, which the configuration must have. None
    when it worked; otherwise the error as text, for the record. A refused open
    whose own close is still running in the driver (LaserControllerStillClosing:
    laser work over the same lines must wait for it) gives a WedgedText and is
    noted as outstanding: it can write the minimum at any moment.
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
        text = f"{type(error).__name__}: {error}"
        closing = _still_closing(error)
        if closing is not None:
            _note_outstanding(
                "the controller's close after a refused open", closing.still_closing.wait)
            return WedgedText(f"{text} (its close is still running in the driver)")
        return text
    return None


def _still_closing(error: BaseException) -> Optional[LaserControllerStillClosing]:
    """The LaserControllerStillClosing in `error`'s chain, however it was wrapped."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, LaserControllerStillClosing):
            return error
        seen.add(id(error))
        error = error.__cause__ or error.__context__
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


class InterruptGate:
    """Where SIGINT, SIGTERM and SIGHUP may stop the run, and where they only wait.

    The handler raises one KeyboardInterrupt, and only while the gate is
    `raising`: inside the run itself (`interruptible()`), which is what ends
    the row in hand and starts the recovery. Everywhere else a signal is
    recorded in `pending` and does not raise: while a victim is being started
    and its pid taken (`deferred()`, so it can never exist without an owner),
    while a failed row's victim is killed, through the whole recovery, and from
    the end of the run to the last line it prints. A second Ctrl-C, or the
    hang-up that follows a closed terminal, therefore cannot cut the recovery
    or the record short; the recovery is time-bounded, and kill -9 is still
    there. The gate is spent by the first: `raising` is cleared before the
    exception is raised, so no later signal can raise another.
    """

    def __init__(self, raising: bool = False) -> None:
        self.raising = raising
        #: Signals that became a KeyboardInterrupt.
        self.raised: List[int] = []
        #: Signals that arrived where they may not raise.
        self.pending: List[int] = []

    def __call__(self, signum, frame) -> None:
        if self.raising:
            self.raising = False
            self.raised.append(signum)
            raise KeyboardInterrupt(f"signal {signum}")
        self.pending.append(signum)

    def raise_if_pending(self) -> None:
        """Turn a signal that arrived while deferred into the interrupt, if one may be raised."""
        if self.raising and self.pending:
            signum = self.pending[0]
            self.pending.clear()
            self.raising = False
            self.raised.append(signum)
            raise KeyboardInterrupt(f"signal {signum}")

    def interruptible(self) -> "_GateRegion":
        return _GateRegion(self, True)

    def deferred(self) -> "_GateRegion":
        return _GateRegion(self, False)


class _GateRegion:
    def __init__(self, gate: InterruptGate, raising: bool) -> None:
        self.gate, self.raising, self.previous = gate, raising, False

    def __enter__(self) -> InterruptGate:
        self.previous = self.gate.raising
        self.gate.raising = self.raising
        if self.raising:
            self.gate.raise_if_pending()
        return self.gate

    def __exit__(self, *exception) -> None:
        # Leaving the run leaves the gate not raising for good; leaving a
        # deferred stretch goes back to what it interrupted.
        self.gate.raising = False if self.raising else self.previous


def install_interrupt_handlers(install=signal.signal) -> InterruptGate:
    """SIGINT, SIGTERM and SIGHUP all go to one gate, which starts out not raising."""
    gate = InterruptGate()
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None:
            install(number, gate)
    return gate


class RunStopped(RuntimeError):
    """A row left something outstanding that no further row could be trusted past."""


def _wedged(observation: KillObservation) -> Optional[str]:
    """What a row left outstanding in the driver, or None.

    A reopen or reset still running (its call timed out) can write the minimum
    at any moment, as late as just after the next row's kill, and make that row
    read safe; a victim that survived SIGKILL may still hold the output. Neither
    is a measurement: the run stops there, with the recovery and its banner.
    """
    for label, text in (("reopen", observation.reopen_error),
                        ("reset", observation.reset_error)):
        if isinstance(text, WedgedText):
            return f"the {label} {text}"
    if isinstance(observation.victim_exit, WedgedText):
        return f"the victim {observation.victim_exit}"
    return None


def _not_recovered(observation: KillObservation, minimum_volts: float) -> Optional[str]:
    """Why the output after the LAST row is not confirmed at its minimum, or None.

    An earlier row's refusal is data and the next row's victim starts the
    output afresh. After the last, nothing does: its reopen or reset refused,
    or the last thing the reader saw was above the minimum, or it saw nothing,
    and nothing has closed the shutter line if the reopen refused. A row the
    controller refused by design ran nothing and is not held to this.
    """
    if observation.case.refusal:
        return None
    for label, text in (("reopen", observation.reopen_error),
                        ("reset", observation.reset_error)):
        if text:
            return f"its {label} said: {text}"
    last = observation.after_reset or observation.after_reopen or observation.after_hold
    if last is None:
        return "the reader saw nothing after it"
    if last.mean > minimum_volts + LEVEL_TOLERANCE_V:
        return (f"the output read {last.mean:.3f} V after it "
                f"(its minimum is {minimum_volts:g} V)")
    return None


@contextlib.contextmanager
def _victim_in_hand(spawn_call: Callable, announce: Optional[Callable], kill, gate: InterruptGate):
    """Start a victim and own it from the moment it exists to the end of the body.

    Signals are held while it is started and its pid announced, so there is no
    instant at which it exists and nothing is set to end it; one that arrived
    meanwhile ends the row here, with the victim in hand. A failure anywhere in
    the body, a signal included, sends it SIGKILL by its pid at once
    (_abandon_victim, itself undisturbed), and the failure then propagates. The
    body ends the victim itself when it succeeds.
    """
    process = None
    try:
        with gate.deferred():
            process = spawn_call()
            if announce is not None:
                announce(process)
        gate.raise_if_pending()
        yield process
    except BaseException as error:
        if process is not None:
            fate = _abandon_protected(process, kill, gate)
            if fate is not None:
                # Where the recovery finds it, for the record (a survivor's pid
                # must not live only on a terminal that may have gone).
                try:
                    error.victim_fate = fate
                except Exception:
                    pass
        raise


def _abandon_protected(process, kill, gate: InterruptGate) -> Optional[dict]:
    """_abandon_victim with signals held, tried once more if one got through.

    After a failure that was not a signal the gate may still be raising for the
    first instants of this, so a signal can raise here before the deferral takes
    hold. That spends the gate, and the second try cannot be interrupted.
    """
    for _ in range(2):
        try:
            with gate.deferred():
                return _abandon_victim(process, kill)
        except KeyboardInterrupt:
            continue
    return None


def _abandon_victim(process, kill) -> dict:
    """A row has failed: SIGKILL its victim by its pid now, and say if it survives.

    Returns {"pid", "ended", "detail"} for the record. What this prints is the
    other report of the victim's fate on a failure path, so a survivor (stuck in
    the driver, holding the output) is named loudly.
    """
    try:
        ended = kill_victim(process, kill=kill)
    except Exception as error:
        detail = f"could not end the victim: {type(error).__name__}: {error}"
        _say(f"    could not end the victim, pid {process.pid}: {type(error).__name__}: {error}")
        return {"pid": process.pid, "ended": False, "detail": detail}
    if ended:
        detail = "ended by sigkill after the failure"
        _say(f"    victim pid {process.pid} ended by sigkill after the failure")
    else:
        detail = "STILL ALIVE after sigkill: it may still hold the output"
        _say(f"    VICTIM pid {process.pid} IS STILL ALIVE after sigkill: "
             "it may still hold the output")
    return {"pid": process.pid, "ended": ended, "detail": detail}


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
    announce: Optional[Callable] = None,
    interrupts: Optional[InterruptGate] = None,
) -> KillObservation:
    """One row: start the victim, kill it, read the output at each instant, recover.

    `reopen` and `reset` are the two recoveries; None skips one and leaves its
    reading empty. Every wait is `sleep`, every instant is read off `log` at
    the time the row defines (not when the reading is taken, so an operator
    slow over the DMM prompt moves nothing), and the only process signalled is
    the one `spawn` returned, by its pid. The before-prompt is asked first and
    `before` read just ahead of the kill, so what it records is the output at
    the kill; if that prompt took over BEFORE_PROMPT_LIMIT_S the output at the
    kill is not known (it may have dropped just after the reading), so no
    `before` is recorded and the row reads unknown. `announce(process)` is
    called as soon as the victim exists. From then on (_victim_in_hand) any
    failure, a signal included, sends the victim SIGKILL by its pid at once, and
    a victim that survives that is reported, not assumed gone (_abandon_victim);
    the failure then propagates.
    """
    gate = interrupts or InterruptGate()
    with _victim_in_hand(lambda: spawn(case), announce, kill, gate) as process:
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
            elif "unreadable" in ready:
                problem = f"unreadable READY line: {ready['unreadable']!r}"
            elif ready["pid"] != process.pid:
                problem = (f"READY names pid {ready['pid']}, not the spawned pid "
                           f"{process.pid}; the row was not run")
        if problem is not None:
            return _row_without_readings(
                case, (problem,), end_victim(process, kill=kill, mode="none"))

        dmm_values = {}

        def ask(instant: str) -> None:
            if dmm is not None:
                value = dmm(instant)
                if value is not None:
                    dmm_values[instant] = value

        sleep(SETTLE_S)
        # The prompt first: it can take as long as the operator takes, and a
        # `before` read ahead of it would describe an output that may be gone
        # by the kill (a 30 s train over, a victim closed at its 60 s cue timeout).
        prompt_started = now()
        ask("before")
        prompt_seconds = now() - prompt_started
        before = log.at(now(), instrument)
        notes = []
        if prompt_seconds > BEFORE_PROMPT_LIMIT_S:
            # Read at the end of a long prompt, it can still show the level a
            # moment before the output drops; nothing then says it was up at the
            # kill. The row cannot be judged (no before reads unknown).
            shown = "no reading" if before is None else f"{before.mean:.3f} V"
            notes.append(
                f"before prompt took {prompt_seconds:.0f} s (over "
                f"{BEFORE_PROMPT_LIMIT_S:g} s): the output at the kill is not known, "
                f"so no before reading is recorded (it read {shown})")
            before = None

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
            victim_exit=end_victim(process, kill=kill, mode=case.mode),
            dmm=dmm_values, notes=tuple(notes))


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
    interrupts: Optional[InterruptGate] = None,
) -> List[Tuple[float, Optional[LevelReading], Optional[float]]]:
    """Does the reader follow the command? One clean software-DC victim per level.

    Read at READY plus a second, beside the DMM's reading when there is one.
    The victim is then cued to close and exit, and is ended by its pid if it
    does not. From the moment a victim exists, a failure (a signal included)
    sends it SIGKILL by its pid at once, as in run_case; a victim that survives
    that, or SIGKILL at the end, stops the run (RunStopped).
    """
    gate = interrupts or InterruptGate()
    instrument, _ = reader_channel_for(configuration, args.laser, args.reader_channel)
    case = KillCase("clean", "software", "dc")
    rows = []
    for level in TRACKING_LEVELS:
        def announce(process, level=level):
            _say(f"tracking {level:g} V (victim pid {process.pid})")

        with _victim_in_hand(lambda: spawn(case, level), announce, kill, gate) as process:
            ready = wait_for_ready(process.lines, READY_TIMEOUT_S)
            if ready.get("pid") != process.pid:
                raise RuntimeError(f"the victim at {level:g} V did not report ready: {ready}")
            sleep(SETTLE_S)
            reading = log.at(now(), instrument)
            value = dmm(f"{level:g} V") if dmm is not None else None
            apply_kill(case, process.pid, kill=kill)
            ended = end_victim(process, kill=kill, mode="clean")
        if isinstance(ended, WedgedText):
            raise RunStopped(f"the victim at {level:g} V {ended}")
        if ended != EXPECTED_VICTIM_EXIT["clean"]:
            _say(f"    the victim at {level:g} V did not end cleanly: {ended}")
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
    on_row: Optional[Callable[[KillObservation], None]] = None,
    interrupts: Optional[InterruptGate] = None,
) -> None:
    """Run each case in turn, appending its observation to `observations`.

    Appended as each row ends (and handed to `on_row`, which writes it to
    disk), so the rows finished before a failure survive it. Each row's line
    names its victim's pid, so that process can be found by hand if this one
    dies, then its verdict, notes and refusals. A row that leaves a reopen or
    reset outstanding in the driver, or a victim alive after SIGKILL, is kept and
    then stops the run (RunStopped): no further row starts while that is so.
    """
    channel = configuration.laser.get_channel(args.laser)
    # Both recoveries are bounded: a kill can wedge the driver, and a call into
    # it then never returns (the row would never reach end_victim).
    reopen_hook = None if args.skip_reopen else functools.partial(
        bounded, functools.partial(reopen, configuration, args.laser),
        RECOVERY_TIMEOUT_S, "reopen")
    reset_hook = None if args.skip_reset else functools.partial(
        bounded,
        functools.partial(reset_device, nidaqmx, device_of(channel.analog_output)),
        RECOVERY_TIMEOUT_S, "reset")
    for index, case in enumerate(cases, 1):
        def announce(process, index=index, case=case):
            _say(f"[{index}/{len(cases)}] {case.case_id} (victim pid {process.pid})")

        dmm = functools.partial(prompt, case.case_id) if args.pause_for_dmm else None
        observation = run_case(
            case, spawn=spawn, log=log, now=now, sleep=sleep, kill=kill,
            reopen=reopen_hook, reset=reset_hook, hold_seconds=args.hold_seconds,
            instrument=instrument, dmm=dmm, announce=announce, interrupts=interrupts)
        observations.append(observation)
        if on_row is not None:
            on_row(observation)
        verdict = judge(observation, minimum_volts=minimum_volts, level_volts=level_volts)
        _say(f"    {verdict}  (victim: {observation.victim_exit})")
        for note in observation.notes:
            _say(f"    note: {note}")
        if observation.reopen_error:
            _say(f"    reopen: {observation.reopen_error}")
        if observation.reset_error:
            _say(f"    reset: {observation.reset_error}")
        wedged = _wedged(observation)
        if wedged:
            raise RunStopped(
                f"{case.case_id}: {wedged}; the run stops here so that no row starts "
                "while it is outstanding")
    if cases:
        reason = _not_recovered(observation, minimum_volts)
        if reason:
            raise RunStopped(
                f"{case.case_id}, the last row: {reason}; the output is not confirmed at "
                "its minimum and the shutter line is not confirmed closed")


def repository_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _record_header(args, started: datetime, extra: dict) -> dict:
    return {
        "generated_on": started.isoformat(),
        "host": socket.gethostname(),
        "reachaq_revision": repository_revision(),
        "laser": args.laser,
        "hold_seconds": args.hold_seconds,
        **extra,
    }


def write_record(args, observations, started: datetime, extra: dict) -> Path:
    """The JSON record and, beside it, the Markdown table; the JSON's path."""
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    header = _record_header(args, started, extra)
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


def _append_json(path: Path, payload: dict) -> None:
    """One JSON line, on disk before this returns."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_row(path: Path, observation: KillObservation) -> None:
    _append_json(path, {"observation": dataclasses.asdict(observation)})


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


@dataclasses.dataclass(frozen=True)
class Plan:
    """What a run will do, worked out from the arguments and the configuration."""

    cases: Tuple[KillCase, ...]
    channel: object
    reader_channel: str
    terminal_config: Optional[str]
    level_volts: float
    minimum_volts: float


def make_plan(args, configuration) -> Plan:
    """KeyError for a laser the configuration lacks, ValueError for the rest."""
    channel = configuration.laser.get_channel(args.laser)
    reader_channel, terminal_config = reader_channel_for(
        configuration, args.laser, args.reader_channel)
    cases = select_cases(kill_cases(pulse_repeats=args.pulse_repeats), args.cases)
    level_volts = channel.clamp_command_voltage(args.level)
    minimum_volts = channel.minimum_command_volts
    if level_volts - minimum_volts < MIN_LEVEL_SPAN_V:
        raise ValueError(
            f"command level {level_volts:g} V is too close to the minimum "
            f"{minimum_volts:g} V to tell apart from it (the judge needs at least "
            f"{MIN_LEVEL_SPAN_V:g} V between them): pass a higher --level")
    return Plan(cases, channel, reader_channel, terminal_config, level_volts, minimum_volts)


def _describe_plan(args, plan: Plan) -> None:
    cases = plan.cases
    dc = sum(1 for case in cases if case.waveform == "dc")
    refused = sum(1 for case in cases if case.refusal)
    pulse = len(cases) - dc - refused
    _say(f"laser {args.laser}: reader {plan.reader_channel} (analog terminal "
         f"{plan.terminal_config or 'chosen by DAQmx'}), {READER_RATE_HZ:g} Hz in "
         f"{READER_CHUNK}-sample chunks")
    _say(f"command {plan.level_volts:g} V, minimum {plan.minimum_volts:g} V, "
         f"hold {args.hold_seconds:g} s")
    _say(f"{len(cases)} cases ({dc} dc, {pulse} pulse, {refused} refused by design):")
    for case in cases:
        _say(f"  {case.case_id}")


#: The judge's own tolerance: a reading within this of the minimum is at it.
LEVEL_TOLERANCE_V = 0.05

_DO_NOT_RECONNECT = (
    "    DO NOT RECONNECT the laser's command or shutter cable, or turn its key on, "
    "until the output is confirmed at its minimum on the DMM and the shutter line "
    "is confirmed closed."
)


def recover_after_failure(
    why: str,
    *,
    args,
    configuration,
    plan: Plan,
    nidaqmx,
    log: LevelLog,
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """After a run that stopped early: put the output back if possible, and say what is unconfirmed.

    Tries the reopen, then the device reset (each time-bounded, and whatever
    --skip-reopen and --skip-reset say: those are for measuring, and this is
    not), reads the output once more, and prints a banner. The output is read
    off the reader; the shutter line is not readable from here (a digital input
    on a port0 line would reconfigure it), only the reopen closes it, and the
    reset covers the command device only. Returns what the banner says.
    """
    command_device = device_of(plan.channel.analog_output)
    shutter_line = plan.channel.shutter_output
    _say()
    _say(f"!!! THE RUN DID NOT COMPLETE ({why}). THE LASER IS NOT CONFIRMED SAFE.")
    running = outstanding_calls()
    waited: List[dict] = []
    if running:
        _say(f"    still running in the driver from earlier: {'; '.join(running)}. "
             f"Waiting up to {OUTSTANDING_WAIT_S:g} s for it before the recovery's own calls ...")
        waited = wait_for_outstanding(OUTSTANDING_WAIT_S)
        for entry in waited:
            if entry["finished"]:
                _say(f"    {entry['call']} finished after {entry['waited_s']:g} s, "
                     "before the recovery made calls of its own.")
            else:
                _say(f"    {entry['call']} is STILL RUNNING after {entry['waited_s']:g} s: the "
                     "recovery's own calls may collide with it, and a refusal below may be that "
                     "collision, not a fault of the hardware.")
    _say("    reopening the controller, then resetting the command device ...")
    reopen_error = bounded(
        functools.partial(reopen, configuration, args.laser), RECOVERY_TIMEOUT_S, "reopen")
    reset_error = bounded(
        functools.partial(reset_device, nidaqmx, command_device), RECOVERY_TIMEOUT_S, "reset")
    sleep(RECOVERY_SETTLE_S)
    reading = log.at(now(), plan.reader_channel)
    volts = None if reading is None else reading.mean
    _say(f"    reopen: {'ok' if reopen_error is None else reopen_error}")
    _say(f"    reset of {command_device}: {'ok' if reset_error is None else reset_error}")
    if volts is None:
        _say(f"    command output on {plan.reader_channel}: NO READING (the reader has "
             "no samples just now)")
    else:
        above = volts > plan.minimum_volts + LEVEL_TOLERANCE_V
        _say(f"    command output on {plan.reader_channel}: {volts:.3f} V (its minimum is "
             f"{plan.minimum_volts:g} V){' - ABOVE THE MINIMUM' if above else ''}")
    _say(f"    shutter line {shutter_line}: NOT READABLE by this tool. Only the reopen "
         f"closes it ({command_device}'s reset does not touch it), and it is closed only "
         f"if the reopen above says ok.")
    _say(_DO_NOT_RECONNECT)
    return {
        "reason": why,
        "reopen": "ok" if reopen_error is None else reopen_error,
        "reset": "ok" if reset_error is None else reset_error,
        "volts": volts,
        "reader_channel": plan.reader_channel,
        "shutter_line": shutter_line,
        "outstanding": waited,
    }


def warn_reopen_skipped(plan: Plan, recovery: Optional[dict] = None) -> None:
    """The --skip-reopen warning, saying what is true of this run's shutter line.

    No row reopened the controller, so no row's end closed the line. If the run
    then failed and the recovery's own reopen succeeded, that closed it.
    """
    line = plan.channel.shutter_output
    _say()
    if recovery is not None and recovery.get("reopen") == "ok":
        _say(f"--skip-reopen: no row reopened the controller, so none closed the shutter "
             f"line {line}; the recovery's reopen above did, and said ok. The output still "
             "needs confirming at its minimum on the DMM.")
        return
    after = ""
    if recovery is not None:
        after = " The recovery's reopen did not succeed either (see above)."
    _say("!!! --skip-reopen: this run never reopened the controller after a row, so "
         f"nothing in it closed the shutter line {line} (a pulse victim opens it and a "
         "killed one never closes it), and the output was left as the last row left it "
         f"unless the device reset moved it.{after}")
    _say(_DO_NOT_RECONNECT)


def execute(
    args,
    configuration,
    plan: Plan,
    started: datetime,
    *,
    load_nidaqmx: Optional[Callable] = None,
    spawn_factory: Optional[Callable] = None,
    reader_factory: Optional[Callable] = None,
    now: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    kill=os.kill,
    prompt: Callable[[str, str], Optional[float]] = prompt_dmm,
    interrupts: Optional[InterruptGate] = None,
) -> int:
    """The run itself, after the plan is made and --dry-run is out of the way.

    This opens NI tasks and starts victims. The seams (`load_nidaqmx`,
    `spawn_factory`, `reader_factory`, the clock, `kill`, `prompt`) are for the
    tests, which hand it fakes; None takes the real one at the time of the call.
    `interrupts` is the gate the signal handlers feed (None: one nothing feeds).
    Returns the exit status: 0 for a run that finished, 1 for any that did not,
    including one that ended with a reopen or reset still outstanding in the
    driver, a victim still alive, or the last row's reopen or reset refused or its
    output read above the minimum (a refusal in an earlier row is data, and the
    run goes on).

    It refuses a namespace that has not attested --laser-disconnected, so that
    no caller can reach it around the parser.
    """
    if not getattr(args, "laser_disconnected", False):
        raise ValueError(
            "execute refuses to run: args.laser_disconnected is not set (the key off, "
            "the command BNC and shutter cable unplugged, the output on the dummy load)")
    gate = interrupts or InterruptGate()
    load_nidaqmx = load_nidaqmx or _load_nidaqmx
    spawn_factory = spawn_factory or make_spawn
    reader_factory = reader_factory or start_reader
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    # Hashed now, before anything has run: the record must name the file the
    # run started from, whatever the run (or something else on the rig) did to it.
    extra = {
        "reader_channel": plan.reader_channel,
        "terminal_config": plan.terminal_config,
        "level_volts": plan.level_volts,
        "minimum_volts": plan.minimum_volts,
        "command_copy_scale": plan.channel.command_copy_scale,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
    }
    try:
        nidaqmx = load_nidaqmx()
    except Exception as error:
        _say(f"cannot load nidaqmx: {type(error).__name__}: {error}")
        return 1
    log, stop = LevelLog(), threading.Event()
    try:
        reader = reader_factory(nidaqmx, plan.reader_channel, plan.terminal_config, log, stop)
    except Exception as error:
        _say(f"the reader on {plan.reader_channel} did not start: "
             f"{type(error).__name__}: {error}")
        if "-50103" in str(error):
            _say("-50103: something else holds that input. The operator's reachAQ may be "
                 "running: it holds the 6221 inputs. Have its owner stop it.")
        _say("nothing was started or signalled.")
        return 1

    rows_path = args.record_dir / f"laser-kill-{stamp}.rows.jsonl"
    tracking_path = args.record_dir / f"laser-kill-{stamp}.tracking.jsonl"
    observations: List[KillObservation] = []
    status = 0
    recovery = None
    if not args.tracking_check:
        try:
            args.record_dir.mkdir(parents=True, exist_ok=True)
            _append_json(rows_path, {"header": _record_header(args, started, extra)})
        except OSError as error:
            _say(f"cannot write {rows_path}: {error}; nothing was started")
            stop.set()
            reader.join(END_TIMEOUT_S)
            return 1

    def on_row(observation: KillObservation) -> None:
        try:
            _append_row(rows_path, observation)
        except OSError as error:
            _say(f"    could not append this row to {rows_path}: {error}")

    spawn = spawn_factory(args, stamp=stamp)
    if gate.pending:
        # A signal during the setup. Nothing has been started, so nothing needs
        # putting back.
        _say(f"interrupted ({_signal_summary(gate.pending)}) before any victim was "
             "started; nothing was started or signalled.")
        stop.set()
        reader.join(END_TIMEOUT_S)
        return 1
    try:
        # Only inside this `with` may a signal stop the run. Out of it, to the
        # last line this function prints, signals are held, so the recovery, the
        # banner, the table and the record always finish.
        with gate.interruptible():
            if args.tracking_check:
                dmm = functools.partial(prompt, "tracking") if args.pause_for_dmm else None
                tracking_check(args, configuration, log, spawn, dmm, now=now, sleep=sleep,
                               kill=kill, interrupts=gate)
            else:
                run_matrix(
                    plan.cases, observations, args=args, configuration=configuration,
                    nidaqmx=nidaqmx, spawn=spawn, log=log, instrument=plan.reader_channel,
                    level_volts=plan.level_volts, minimum_volts=plan.minimum_volts,
                    now=now, sleep=sleep, kill=kill, prompt=prompt, on_row=on_row,
                    interrupts=gate)
    except (Exception, KeyboardInterrupt) as error:
        # The victim in hand has been sent SIGKILL by its pid (run_case, or
        # tracking_check), and a survivor reported there. What is left is the
        # output and the shutter line, which the recovery tries to put back and
        # reports on.
        status = 1
        why = f"{type(error).__name__}: {error}"
        if gate.raised and not isinstance(error, KeyboardInterrupt):
            why += f" (and {_signal_summary(gate.raised)} arrived)"
        _say(f"stopped: {why}")
        try:
            recovery = recover_after_failure(
                why, args=args, configuration=configuration, plan=plan, nidaqmx=nidaqmx,
                log=log, now=now, sleep=sleep)
        except Exception as recovery_error:
            _say("the recovery itself failed: "
                 f"{type(recovery_error).__name__}: {recovery_error}")
            recovery = {"reason": why, "recovery_failed": str(recovery_error)}
        fate = getattr(error, "victim_fate", None)
        if fate is not None:
            recovery["victim"] = fate
        if gate.pending:
            recovery["signals_held"] = _signal_counts(gate.pending)
        # On disk, so that a dropped terminal loses none of it. A failed tracking
        # check has no rows file, so it gets one of its own.
        failure_path = tracking_path if args.tracking_check else rows_path
        try:
            args.record_dir.mkdir(parents=True, exist_ok=True)
            if args.tracking_check:
                _append_json(failure_path, {"header": _record_header(args, started, extra)})
            _append_json(failure_path, {"recovery": recovery})
        except OSError as io_error:
            _say(f"could not write the recovery to {failure_path}: {io_error}")
    finally:
        stop.set()
        reader.join(END_TIMEOUT_S)

    if log.error:
        _say(f"the reader had read errors; the last: {log.error}")
    for text in log.restarts:
        _say(f"reader: {text}")
    if observations:
        # The table first: it is what the operator reads, and nothing after
        # it (the record write) may stand between them and it.
        _say()
        _say(format_table(
            observations, minimum_volts=plan.minimum_volts, level_volts=plan.level_volts))
        _say()
        counts = Counter(
            judge(o, minimum_volts=plan.minimum_volts, level_volts=plan.level_volts)
            for o in observations)
        _say("verdicts: " + ", ".join(f"{verdict} {n}" for verdict, n in sorted(counts.items())))
        record_extra = dict(extra)
        if recovery is not None:
            record_extra["recovery"] = recovery
        try:
            path = write_record(args, observations, started, record_extra)
        except OSError as error:
            _say(f"could not write the record: {error}; the rows are in {rows_path}")
            status = 1
        else:
            _say(f"record: {path}")
    if args.skip_reopen and not args.tracking_check:
        warn_reopen_skipped(plan, recovery)
    if gate.pending:
        _say(f"{_signal_summary(gate.pending)} arrived after the run had stopped or finished "
             "and were held until everything above was complete.")
    return status


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    configuration = SystemConfiguration.load_yaml_file(args.config, save_backup=False)
    try:
        plan = make_plan(args, configuration)
    except (KeyError, ValueError) as error:
        _say(f"cannot plan the run: {error}")
        return 2
    _say(preconditions(args.laser, device_of(plan.channel.analog_output)))
    _describe_plan(args, plan)
    if args.dry_run:
        _say("dry run: no process started, no NI task opened, nothing written")
        return 0
    return execute(args, configuration, plan, started, interrupts=install_interrupt_handlers())


if __name__ == "__main__":
    sys.exit(main())
