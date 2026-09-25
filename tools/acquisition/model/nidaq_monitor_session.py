"""The monitor's own view of the rig: every line, live, plus something to poke.

Watching is only half of it. Deciding which line a stimulus arrives on means
driving something and seeing what moves, which is why this owns the stimulus
side as well - a tone, a laser command held at a voltage, a board output held
high. Without that the window shows forty-eight flat traces and answers
nothing.

Three deliberate separations:

  * the clocked lines go through a NidaqSignalMonitorModel, which runs the
    stream in its own process exactly as the application's does. This is a
    second instance rather than a borrowed one, because it streams a
    different set of lines at its own rate. The application's own stream,
    which now runs by itself while idle, is paused for as long as the session
    is open: both would reserve the same port and counter;
  * the static lines - the PFI pins, and every line of a board that cannot
    clock digital input - are read as a level by a short-lived task. They
    have no sample clock, so there is no waveform to stream;
  * the wiring test is the existing tools/hardware/verify_nidaq_wiring.py,
    run as a subprocess. Reimplementing it inside the window would give two
    things that could disagree about what the rig is doing, and the one with
    a terminal is the one that has been exercised.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from autotrainer.core import ObservableObject

from tools.acquisition.model.nidaq_breakout import breakout_for_device
from tools.acquisition.model.nidaq_discovery import discover_nidaq_devices
from tools.acquisition.model.nidaq_monitor_survey import (
    DEFAULT_SURVEY_CHUNK,
    DEFAULT_SURVEY_RATE_HZ,
    MonitorSurvey,
    build_survey,
    survey_stream_configuration,
)
from tools.acquisition.model.nidaq_signal_monitor_model import NidaqSignalMonitorModel
from tools.acquisition.model.nidaq_validation import (
    chassis_identification_note,
    validate_nidaq_configuration,
)
from tools.acquisition.model.nidaq_wiring_report import build_report
from tools.acquisition.model.nidaq_wiring_verification import WiringVerification

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VERIFY_TOOL = _REPO_ROOT / "tools" / "hardware" / "verify_nidaq_wiring.py"


class NidaqMonitorSession(ObservableObject):
    """Everything the monitor window needs, with no Qt in it."""

    SURVEY = "survey"
    IS_RUNNING = "is_running"
    STATUS_MESSAGE = "status_message"
    ERROR_MESSAGE = "error_message"
    STATIC_LEVELS = "static_levels"
    REPORT = "report"

    def __init__(self, app_model):
        super().__init__()
        self._app_model = app_model
        self._holds_app_stream = False
        # Set while the verify tool runs; the application's stream is not let
        # go until it finishes, even when the window closes first.
        self._wiring_lock = threading.Lock()
        self._wiring_active = False
        self._release_after_wiring = False
        self._survey = MonitorSurvey()
        self._devices: Tuple = tuple()
        self._issues: Tuple[str, ...] = tuple()
        self._notes: Tuple[str, ...] = tuple()
        self._status_message = "not started"
        self._error_message = ""
        self._static_levels: Dict[str, int] = {}
        self._report = None
        self._stream = NidaqSignalMonitorModel()
        self._static_tasks: Dict[Tuple[str, str], object] = {}
        self._nidaqmx = None
        # The stream reaches running a moment after start() returns, because
        # the worker has to come up first. Forwarding its own event is the
        # difference between a button that is right and one that is right
        # eventually.
        self._stream.property_changed += self._on_stream_changed
        # Last, so a session that fails to build holds nothing, and before
        # anything here opens a task: the application's stream would still
        # hold the lines this polls and streams, and the two fail each other
        # at -89137 with nothing saying why. The application refuses while
        # System Mode runs, because then the stream is the acquisition's.
        pause = getattr(app_model, "pause_nidaq_stream", None)
        if callable(pause):
            try:
                pause(self, "the DAQ Monitor is open")
            except Exception:
                self._stream.property_changed -= self._on_stream_changed
                self._stream.close()
                raise
            self._holds_app_stream = True

    # ------------------------------------------------------------- what is here

    @property
    def survey(self) -> MonitorSurvey:
        return self._survey

    @property
    def devices(self) -> Tuple:
        return self._devices

    @property
    def issues(self) -> Tuple[str, ...]:
        return self._issues

    @property
    def notes(self) -> Tuple[str, ...]:
        return self._notes

    @property
    def stream(self) -> NidaqSignalMonitorModel:
        return self._stream

    @property
    def sample_ring(self):
        return self._stream.sample_ring

    @property
    def is_running(self) -> bool:
        return self._stream.is_running

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def static_levels(self) -> Dict[str, int]:
        return dict(self._static_levels)

    @property
    def report(self):
        return self._report

    def breakout_for(self, device_name: str):
        return breakout_for_device(
            getattr(self._configuration(), "nidaq_ports", None), device_name)

    def _configuration(self):
        return getattr(self._app_model, "loaded_configuration", None)

    # ------------------------------------------------------------------ discover

    def refresh(self) -> bool:
        """Ask the driver what is installed and build the survey from it."""
        configuration = self._configuration()
        devices, error = discover_nidaq_devices()
        if error:
            self._set_error(f"NI-DAQ discovery failed: {error}")
            self._devices = tuple()
            self._set_survey(MonitorSurvey())
            return False
        self._devices = tuple(devices)
        self._set_survey(build_survey(devices, configuration))

        issues, notes = tuple(), []
        if configuration is not None:
            issues = tuple(issue.describe() for issue in validate_nidaq_configuration(
                devices,
                stream=getattr(configuration, "nidaq_stream", None),
                ports=getattr(configuration, "nidaq_ports", None),
                laser=getattr(configuration, "laser", None),
            ))
            note = chassis_identification_note(devices)
            if note:
                notes.append(note)
        for device in self._survey.devices:
            notes.extend(self._survey.notes_for(device))
        self._issues = issues
        self._notes = tuple(notes)
        self._set_error("")
        self._set_status(
            f"{len(self._survey.lines)} lines across {len(self._survey.devices)} "
            f"card(s); {len(self._survey.streamed())} streamed, "
            f"{len(self._survey.static())} polled")
        self.refresh_report()
        return True

    # -------------------------------------------------------------- the stream

    def start(self, *, sample_rate_hz: float = DEFAULT_SURVEY_RATE_HZ,
              read_chunk_size: int = DEFAULT_SURVEY_CHUNK) -> bool:
        """Start the clocked half of the survey.

        The static tasks are closed first, and that is not tidiness. Measured
        on christielab10: with three DI tasks open in this process, spawning
        the stream worker fails at its first subprocess with "[Errno 14] Bad
        address" naming the Python executable, and with the same tasks closed
        the identical call starts and streams. The cause is somewhere in the
        NI-DAQmx runtime's effect on this process rather than anything the
        stream does, so this does not try to explain it - it keeps the
        process clean across the one call that forks, and the poll reopens
        what it needs on its next tick.
        """
        self._close_static_tasks()
        if not self._survey.lines and not self.refresh():
            return False
        configuration = self._configuration()
        stream = survey_stream_configuration(
            self._survey,
            sample_rate_hz=sample_rate_hz,
            read_chunk_size=read_chunk_size,
            # Passed through rather than defaulted: letting DAQmx choose gives
            # a 6221 ai0-ai7 differential against ai8-ai15, so a survey of the
            # whole range would show one signal in three places.
            analog_terminal_config=getattr(
                getattr(configuration, "nidaq_stream", None),
                "analog_terminal_config", "") or None,
        )
        if not stream.is_enabled:
            self._set_error("no clocked lines to stream on this hardware")
            return False
        try:
            self._stream.load_configuration(stream)
            if configuration is not None:
                # Input only. The laser outputs are deliberately not declared
                # as hardware-timed here: the survey watches lines, and
                # naming them would pull an output board into the task graph
                # for a stream that never writes to it.
                ports = getattr(configuration, "nidaq_ports")
                self._stream.configure_timing(
                    ports.timing,
                    device_identities=getattr(ports, "device_identities", ()),
                )
            self._stream.set_hardware_enabled(True)
            started = self._stream.start()
        except Exception as error:
            logger.exception("monitor stream failed to start")
            self._set_error(f"monitor stream failed to start: {error}")
            return False
        if not started:
            self._set_error(self._stream.error_message or "monitor stream did not start")
            return False
        self._set_error("")
        self._set_status(f"streaming {len(stream.channels)} lines at "
                         f"{stream.sample_rate_hz:g} Hz")
        return True

    def stop(self) -> None:
        try:
            self._stream.stop()
        except Exception:
            logger.exception("monitor stream did not stop cleanly")
        self._close_static_tasks()
        self._set_status("stopped")

    def close(self) -> None:
        try:
            try:
                self._stream.property_changed -= self._on_stream_changed
            except Exception:
                pass
            self.stop()
            try:
                self._stream.close()
            except Exception:
                logger.exception("monitor stream did not close cleanly")
        finally:
            # Only once every task here is closed: the application's stream
            # is started as soon as it is let go.
            self._release_app_stream_unless_wiring()

    def _release_app_stream_unless_wiring(self) -> None:
        # Not while the verify tool still runs: the window gives it five
        # seconds at close, the tool can take three minutes, and it drives
        # the same lines. run_wiring_test lets go when it finishes.
        with self._wiring_lock:
            if self._wiring_active:
                self._release_after_wiring = True
                return
        self._release_app_stream()

    def _release_app_stream(self) -> None:
        if self._holds_app_stream:
            self._holds_app_stream = False
            self._app_model.resume_nidaq_stream(self)

    # --------------------------------------------------------- the static lines

    def _load_nidaqmx(self):
        if self._nidaqmx is None:
            import nidaqmx  # noqa: WPS433 - deliberately late
            self._nidaqmx = nidaqmx
        return self._nidaqmx

    def _close_static_tasks(self) -> None:
        for task in self._static_tasks.values():
            try:
                task.close()
            except Exception:
                pass
        self._static_tasks.clear()

    def poll_static_levels(self) -> Dict[str, int]:
        """The level on every line that has no sample clock.

        One task per device and port, held open between polls: a PFI pin is a
        level, and reopening a task to read one is most of the cost.
        """
        wanted = {}
        counters = []
        for line in self._survey.static():
            if line.kind == "counter":
                counters.append(line)
                continue
            port = line.terminal.split("/", 1)[0]
            wanted.setdefault((line.device, port), []).append(line)
        if not wanted and not counters:
            return {}

        try:
            nidaqmx = self._load_nidaqmx()
        except Exception as error:
            self._set_error(f"cannot read static lines: {error}")
            return {}

        levels: Dict[str, int] = {}
        for (device, port), lines in wanted.items():
            task = self._static_tasks.get((device, port))
            if task is None:
                try:
                    task = nidaqmx.Task(f"reachaq_monitor_{device}_{port}")
                    task.di_channels.add_di_chan(f"{device}/{port}")
                    self._static_tasks[(device, port)] = task
                except Exception:
                    # A board without this port simply has nothing to show.
                    try:
                        task.close()
                    except Exception:
                        pass
                    self._static_tasks.pop((device, port), None)
                    continue
            try:
                word = int(task.read())
            except Exception as error:
                logger.debug("static read of %s/%s failed: %s", device, port, error)
                continue
            for line in lines:
                bit = _line_number(line.terminal)
                if bit is not None:
                    levels[line.name] = (word >> bit) & 1

        for line in counters:
            task = self._static_tasks.get((line.device, line.terminal))
            if task is None:
                try:
                    task = nidaqmx.Task(
                        f"reachaq_monitor_{line.device}_{line.terminal}")
                    task.ci_channels.add_ci_count_edges_chan(
                        line.physical_channel)
                    task.start()
                    self._static_tasks[(line.device, line.terminal)] = task
                except Exception:
                    # The stream takes a counter for its digital sample
                    # clock, so one being reserved is ordinary rather than
                    # broken; it simply has nothing to show while that runs.
                    try:
                        task.close()
                    except Exception:
                        pass
                    self._static_tasks.pop((line.device, line.terminal), None)
                    continue
            try:
                levels[line.name] = int(task.read())
            except Exception as error:
                logger.debug("counter read of %s failed: %s",
                             line.physical_channel, error)

        previous, self._static_levels = self._static_levels, levels
        if levels != previous:
            self.property_changed(self.STATIC_LEVELS, levels, previous)
        return dict(levels)

    # ------------------------------------------------------------- the stimulus

    def board_stimulus_lines(self) -> Tuple[Tuple[str, int], ...]:
        """The stimulus lines the board can pulse, labelled as it labels them.

        Resolved through BOARD_STIM_LINE_OUTPUTS rather than restated, because
        that is the one place the board's numbering and the host's enum are
        reconciled. STIM0 and STIM1 are absent on purpose: the firmware tone
        confirmations drive those.
        """
        try:
            from autotrainer.device.device_interface import BOARD_STIM_LINE_OUTPUTS
        except Exception:
            return tuple()
        return tuple((f"board STIM{line}", int(line))
                     for line in sorted(BOARD_STIM_LINE_OUTPUTS))

    def pulse_board_line(self, stim_line: int, seconds: float = 0.25) -> bool:
        """Pulse one board stimulus line, long enough to see on a trace.

        Firmware-timed through the existing command path rather than held high
        from here, so this exercises the same route a trial takes. A quarter
        of a second is five hundred samples at the survey rate, which is
        unmistakable beside a line that did not move.
        """
        hardware = getattr(self._app_model, "hardware", None)
        pulse = getattr(hardware, "pulse_stim", None)
        if pulse is None:
            self._set_error("board stimulus lines are unavailable; "
                            "is the pellet device connected?")
            return False
        duration_us = int(max(0.01, min(float(seconds), 5.0)) * 1_000_000)
        try:
            token = pulse(duration_us, int(stim_line))
        except Exception as error:
            self._set_error(f"board STIM{stim_line}: {error}")
            return False
        if token is None:
            self._set_error(f"board STIM{stim_line} was not queued")
            return False
        return True

    def play_tone(self, frequency_hz: int, duration_s: float) -> bool:
        hardware = getattr(self._app_model, "hardware", None)
        play = getattr(hardware, "play_tone", None)
        if play is None:
            self._set_error("the tone generator is unavailable")
            return False
        token = play(int(frequency_hz), float(duration_s))
        if token is None:
            self._set_error("the tone was not queued")
            return False
        return True

    def hold_laser_command(self, channel_number: int, volts: float,
                           seconds: float, *, open_shutter: bool = False) -> bool:
        """Hold one laser's command at a voltage, so its copy can be found.

        DC rather than a pulse train on purpose: this is for finding which
        input a cable lands on, and a steady level is what a survey trace at a
        few kHz can show unambiguously.
        """
        laser = getattr(self._app_model, "laser", None)
        if laser is None:
            self._set_error("the laser model is unavailable")
            return False
        try:
            from autotrainer.device import LaserChannelId
            channel = LaserChannelId(int(channel_number))
        except Exception as error:
            self._set_error(f"laser channel {channel_number}: {error}")
            return False
        resting = self._resting_volts(channel_number)
        try:
            if open_shutter:
                laser.set_shutter_open(channel, True)
            laser.set_command_voltage(channel, float(volts))
            time.sleep(max(0.05, min(float(seconds), 5.0)))
        except Exception as error:
            self._set_error(f"laser {channel_number}: {error}")
            return False
        finally:
            try:
                laser.set_command_voltage(channel, resting)
                if open_shutter:
                    laser.set_shutter_open(channel, False)
            except Exception:
                logger.exception("laser %s did not return to rest", channel_number)
        return True

    def _resting_volts(self, channel_number: int) -> float:
        """Where this channel sits when it is not being driven.

        Its own configured minimum, not zero: a channel whose minimum is not
        zero would be left driven by an assumption.
        """
        configuration = self._configuration()
        for channel in getattr(getattr(configuration, "laser", None),
                               "channels", ()) or ():
            if getattr(getattr(channel, "channel_id", None), "value", None) == channel_number:
                return float(getattr(channel, "minimum_command_volts", 0.0) or 0.0)
        return 0.0

    def laser_channels(self) -> Tuple[int, ...]:
        configuration = self._configuration()
        return tuple(
            int(channel.channel_id.value)
            for channel in getattr(getattr(configuration, "laser", None),
                                   "channels", ()) or ())

    # ----------------------------------------------------------- the wiring test

    def refresh_report(self):
        """The report as it stands, from the record on disk."""
        configuration = self._configuration()
        if configuration is None:
            return None
        report = build_report(
            configuration,
            WiringVerification.load(self._record_path()),
            issues=self._issues,
            notes=self._notes,
        )
        previous, self._report = self._report, report
        self.property_changed(self.REPORT, report, previous)
        return report

    def _record_path(self) -> Path:
        path = getattr(self._app_model, "_wiring_record_path", None)
        if callable(path):
            return Path(path())
        if path:
            return Path(path)
        return Path.home() / "Autotrainer" / "nidaq_wiring_verification.json"

    def wiring_test_command(self, *, open_shutters: bool = False,
                            command_volts: float = 1.0,
                            dry_run: bool = False) -> Tuple[str, ...]:
        """Exactly what will be run, so the window can show it before it runs."""
        command = [sys.executable, str(_VERIFY_TOOL),
                   "--record", str(self._record_path()),
                   "--command-volts", f"{command_volts:g}"]
        if open_shutters:
            command.append("--open-shutters")
        if dry_run:
            command.append("--dry-run")
        return tuple(command)

    def run_wiring_test(self, *, open_shutters: bool = False,
                        command_volts: float = 1.0, dry_run: bool = False,
                        timeout: float = 180.0) -> Tuple[bool, str]:
        """Run the verification tool and read back what it wrote.

        The stream is stopped first. The tool opens its own tasks on the same
        lines, and two clients reserving the same digital port is how this
        fails at -89137 rather than telling anyone why.

        While it runs, closing the session does not hand the lines back to
        the application; this does, once the tool has exited. subprocess.run
        kills the tool itself on a timeout, so it has exited either way.
        """
        with self._wiring_lock:
            self._wiring_active = True
        try:
            return self._run_wiring_test(
                open_shutters=open_shutters, command_volts=command_volts,
                dry_run=dry_run, timeout=timeout)
        finally:
            with self._wiring_lock:
                self._wiring_active = False
                release = self._release_after_wiring
                self._release_after_wiring = False
            if release:
                self._release_app_stream()

    def _run_wiring_test(self, *, open_shutters: bool, command_volts: float,
                         dry_run: bool, timeout: float) -> Tuple[bool, str]:
        if self.is_running:
            self.stop()
        command = self.wiring_test_command(
            open_shutters=open_shutters, command_volts=command_volts,
            dry_run=dry_run)
        try:
            finished = subprocess.run(
                command, cwd=str(_REPO_ROOT), capture_output=True, text=True,
                timeout=timeout,
                # See nidaq_discovery: with NI-DAQmx resident in a Qt
                # process, a child started with the inherited environment
                # fails at exec with "Bad address".
                env=os.environ.copy())
        except subprocess.TimeoutExpired:
            return False, f"the wiring test did not finish within {timeout:g}s"
        except Exception as error:
            return False, f"the wiring test could not be run: {error}"
        output = (finished.stdout or "") + (finished.stderr or "")
        self.refresh_report()
        if finished.returncode != 0:
            return False, output or f"the wiring test exited {finished.returncode}"
        return True, output

    def _on_stream_changed(self, name, new_value, old_value) -> None:
        if name == NidaqSignalMonitorModel.IS_RUNNING:
            self.property_changed(self.IS_RUNNING, new_value, old_value)
        elif name == NidaqSignalMonitorModel.ERROR_MESSAGE and new_value:
            self._set_error(str(new_value))

    # ------------------------------------------------------------------ plumbing

    def _set_survey(self, value: MonitorSurvey) -> None:
        previous, self._survey = self._survey, value
        self.property_changed(self.SURVEY, value, previous)

    def _set_status(self, value: str) -> None:
        previous, self._status_message = self._status_message, value
        if previous != value:
            self.property_changed(self.STATUS_MESSAGE, value, previous)

    def _set_error(self, value: str) -> None:
        previous, self._error_message = self._error_message, value
        if previous != value:
            self.property_changed(self.ERROR_MESSAGE, value, previous)


def _line_number(terminal: str) -> Optional[int]:
    """The bit a line occupies in its port word, which is its line number."""
    tail = str(terminal or "").rsplit("/", 1)[-1].strip().lower()
    if tail.startswith("line") and tail[4:].isdigit():
        return int(tail[4:])
    return None
