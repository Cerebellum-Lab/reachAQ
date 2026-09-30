from __future__ import annotations

import dataclasses
import enum
import logging
import numbers
import threading
import time
import uuid
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from autotrainer.core import NidaqTimingPlan
from autotrainer.core.logging import log_hardware_initialization

from .laser import (
    LaserCalibrationPoint,
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserFeedbackSample,
    LaserPulseRefused,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


from autotrainer.device.nidaq_reference_clock import (
    apply_reference_clock,
)
from autotrainer.device.nidaq_signal_stream import resolve_analog_terminal_config

logger = logging.getLogger(__name__)

#: How long close() waits for an aborted calibration ramp to close its own
#: tasks before resetting the laser; the pulse path waits as long for its
#: operation's owner.
_CALIBRATION_RELEASE_TIMEOUT_S = 5.0
#: How long close() waits for each pulse train it cancels to end.
_OPERATION_CANCEL_TIMEOUT_S = 5.0
#: How long a path that cancels a pulse and goes on waits for it to end,
#: so that what it does next finds the board free; the one bound for every
#: such path, the application's too. The abort ends it in milliseconds
#: (christielab10: the wait woke 21-35 ms after the cancel, H8a and H8c on
#: c12189cd); this bounds a sick driver.
CANCELLED_OPERATION_WAIT_S = 2.0
#: How long a stop or close keeps trying while an abort is still finishing.
#: DAQmx refused a stop so, -88710, made while a task never started, its
#: buffer written, was being aborted (H8b); a close then is taken to be
#: refused as well, unmeasured. A running task's owner, woken by its abort,
#: stops and clears it cleanly before the abort returns (H8a, H8c).
_ABORT_FINISHING_S = 0.2
#: DAQmx's status for "a task is in the process of being aborted".
_ABORT_IN_PROGRESS = -88710
#: How nidaqmx's text for the warning an abort of a running task gives,
#: DaqWarning 200010 (H5b, H8a), begins; the one filter that keeps it off
#: stderr matches it (_ignore_the_aborts_warning).
_ABORT_WARNING_PATTERN = r"\s*Warning 200010 occurred"
#: How long a controller that failed to open waits for its own close, of
#: what it had opened, before it raises; as long as the application waits
#: for any laser close.
_FAILED_OPEN_CLOSE_TIMEOUT_S = 15.0
#: How long a caller waiting for another's route to settle sleeps between
#: looks; a settle, or close(), wakes it at once.
_ROUTE_SETTLE_WAIT_S = 1.0
#: How long it waits in all before it gives up: a driver hung in the other
#: caller's connect or release held this one with it, without end.
_ROUTE_PENDING_GIVE_UP_S = 5.0


class LaserControllerStillClosing(RuntimeError):
    """A controller failed to open, and its close of what it had opened goes on.

    Inside the driver, past the constructor's bound. Raised from the open's
    own error, whose text it keeps, as its cause. `still_closing` is set when
    that close ends; until then laser work over the same lines must wait. A
    caller that wraps this keeps it on the error's chain, where it is found
    whatever the wrapping (the application walks __cause__ and __context__).
    """

    def __init__(self, error: BaseException, still_closing: threading.Event):
        super().__init__(str(error) or error.__class__.__name__)
        self.still_closing = still_closing


class LaserOperationState(str, enum.Enum):
    PREPARED = "prepared"
    ARMED = "armed"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NidaqLaserOperation:
    """Generation-owned lifecycle for one finite NI output operation."""

    TERMINAL = {
        LaserOperationState.COMPLETED,
        LaserOperationState.FAILED,
        LaserOperationState.CANCELLED,
    }

    def __init__(self, *, resources, context=None, terminal_callback=None,
                 abort_task=None, before_abort=None):
        self.operation_id = str(uuid.uuid4())
        #: Called by cancel() before any abort: closes the pulse's shutter,
        #: as close() closes the shutters first. After an abort the output
        #: holds its amplitude until the cleanup's reset: 0 V came 24-40 ms
        #: after the cancel on the 6713 (H8a, H8c on c12189cd), the abort
        #: itself taking 12.6-32 ms. With the shutter closed the light is off
        #: for it.
        self._before_abort = before_abort
        #: Set once the pulse's waits have returned: it was delivered, and a
        #: cancel from then on changes nothing (_mark_finishing).
        self._finishing = False
        #: How cancel() ends a bound task that is running or armed: TASK_ABORT,
        #: given by the controller, as abort_task(task, name, running), the
        #: name the one the task was bound with. A stop() from another thread
        #: waits behind the operation's own wait on the task, until it ends or the
        #: wait times out (christielab10, H5a, 2026-09-29); an abort wakes the
        #: wait with -88709 in about 36 ms (H5b), and before a start it is
        #: harmless (H5c).
        self._abort_task = abort_task
        self.resources = tuple(sorted(set(resources)))
        self.context = dict(context or {})
        self._terminal_callbacks = (
            [] if terminal_callback is None else [terminal_callback]
        )
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._armed = threading.Event()
        self._start_requested = threading.Event()
        self._state = LaserOperationState.PREPARED
        self._observations = [(self._state.value, time.perf_counter(), "")]
        #: (name, task) pairs: the output, then its lines (_bind_tasks).
        self._tasks = ()
        #: The bound tasks started so far, by id: an abort of one of them can
        #: warn (200010), and says so (_mark_started).
        self._started = set()
        #: Set once the pulse's cleanup has taken its tasks, before it stops
        #: and closes them: a cancel's abort leaves them to it (_let_go_of_tasks).
        self._let_go = False
        self._error = None
        self._timing_status = {}
        self._thread = None

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def observations(self):
        with self._lock:
            return tuple(self._observations)

    def wait_until_finished(self, timeout=None) -> bool:
        """Wait for the operation to end, cleanup included; whether it has."""
        return self._done.wait(timeout)

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError(f"Laser operation {self.operation_id} did not finish")
        if self.error is not None:
            raise self.error
        return self.state

    def wait_until_armed(self, timeout=None):
        if not self._armed.wait(timeout):
            raise TimeoutError(f"Laser operation {self.operation_id} was not armed")
        if self.error is not None:
            raise self.error
        if self.state in {LaserOperationState.CANCELLED, LaserOperationState.FAILED}:
            raise RuntimeError(
                f"Laser operation reached {self.state.value} before it was armed"
            )
        return self.state

    def cancel(self):
        tasks = ()
        # The mark is made at once, under the lock and before any driver
        # call: a pulse opening its shutter looks again after it opens
        # (_open_shutters_unless_cancelled).
        with self._lock:
            if self._state in self.TERMINAL or self._finishing:
                return False
            self._transition_locked(LaserOperationState.CANCELLED, "cancel requested")
            tasks = self._tasks
        # The pulse's shutters first, as close() closes the shutters first:
        # the light is off from then on, whatever the output holds until the
        # cleanup's reset. The trade: this digital write comes before the
        # abort, which waits for it; about a millisecond through the
        # channel's own task, a transient task's making once close() has let
        # go of that one, and a driver hung in it holds the abort back too.
        # It writes the task the pulse opens its shutter through: where
        # DAQmx makes calls on one task wait for each other, it waits behind
        # a hung opening as well.
        if self._before_abort is not None:
            try:
                self._before_abort()
            except Exception:
                logger.exception("Failed to close the laser shutter before a cancel's abort")
        # Aborted, not stopped: the operation's own thread is in the task's
        # wait, which the abort ends at once. That thread, finding the
        # operation cancelled, ends it CANCELLED, not FAILED, and cleans up.
        # The output first: it is what holds the laser on, DAQmx keeping its
        # last sample until the cleanup's reset, and the lines run on its
        # clock. Its abort wakes that thread, whose cleanup takes every task
        # before it stops and closes them (H8a, H8c): a line it has taken is
        # its to stop, and is not aborted; one it takes during the abort is
        # not a failure.
        for name, task in tasks:
            if self._abort_task is None:
                break
            if self._tasks_let_go():
                logger.debug("A cancel left %s to its pulse's own cleanup", name)
                continue
            try:
                self._abort_task(task, name, self._is_started(task))
            except Exception as error:
                if self._tasks_let_go():
                    logger.debug("A cancel's abort of %s met its pulse's own "
                                 "cleanup: %s", name, error)
                else:
                    logger.debug("Laser task abort during cancellation failed", exc_info=True)
        # A deferred pulse is woken only now: woken before the aborts had
        # returned, its cleanup stopped a task still being aborted, -88710
        # (christielab10, H8b).
        self._start_requested.set()
        return True

    def trigger(self):
        with self._lock:
            if self._state is not LaserOperationState.ARMED:
                raise RuntimeError(
                    f"Laser operation must be armed, found {self._state.value}"
                )
            self._start_requested.set()
            return time.perf_counter()

    def to_record(self):
        with self._lock:
            return {
                "operation_id": self.operation_id,
                "state": self._state.value,
                "resources": list(self.resources),
                "context": dict(self.context),
                "observations": [
                    {"state": state, "perf_time": perf, "detail": detail}
                    for state, perf, detail in self._observations
                ],
                "error": (
                    None if self._error is None
                    else f"{type(self._error).__name__}: {self._error}"
                ),
                "timing_status": dict(self._timing_status),
            }

    def add_terminal_callback(self, callback):
        """Observe terminal cleanup without replacing controller ownership."""
        call_now = False
        with self._lock:
            if self._done.is_set():
                call_now = True
            else:
                self._terminal_callbacks.append(callback)
        if call_now:
            try:
                callback(self)
            except Exception:
                logger.exception("Laser terminal callback failed")

    def _set_timing_status(self, status):
        with self._lock:
            self._timing_status = dict(status)

    def _bind_tasks(self, tasks):
        """The tasks a cancel aborts, in the order it aborts them: (name, task).

        Named with the names the controller made them with: a task's own
        name is a driver query, which a cancel's abort must not make.
        """
        bound = tuple(tasks)
        with self._lock:
            self._tasks = bound

    def _mark_started(self, task):
        """`task` has been started: an abort of it from now on can warn."""
        with self._lock:
            self._started.add(id(task))

    def _is_started(self, task) -> bool:
        with self._lock:
            return id(task) in self._started

    def _let_go_of_tasks(self):
        """The pulse's cleanup has taken its tasks: a cancel leaves them to it."""
        with self._lock:
            self._let_go = True

    def _tasks_let_go(self) -> bool:
        with self._lock:
            return self._let_go

    def _open_shutters_unless_cancelled(self, open_shutters, close_shutters):
        """Open the pulse's shutters, by `open_shutters`, unless cancelled.

        A cancel closes them before its abort. Made just before the pulse
        opened them, it found them closed, and the pulse then opened them: a
        pulse that leaves its shutter open left it so. The pulse looks before
        it opens them and again after, however the opening ended: one that
        raised part-way can have opened some of them. Cancelled by the second
        look, it closes them itself, by `close_shutters`, and ends as the
        cancel; a cancel marked after it closes them after the opening.

        The cancel's mark does not wait on the opening. Its own shutter close
        does write the same tasks, and so does close()'s first loop: where
        DAQmx makes calls on one task wait for each other, those still wait
        behind a hung opening.
        """
        if self.state is LaserOperationState.CANCELLED:
            raise RuntimeError("Laser operation was cancelled before its shutter opened")
        try:
            open_shutters()
        finally:
            if self.state is LaserOperationState.CANCELLED:
                close_shutters()
                raise RuntimeError("Laser operation was cancelled as its shutter opened")

    def _mark_armed(self):
        with self._lock:
            if self._state is LaserOperationState.PREPARED:
                self._transition_locked(LaserOperationState.ARMED)
                self._armed.set()

    def _require_not_cancelled(self):
        if self.state is LaserOperationState.CANCELLED:
            raise RuntimeError("Laser operation was cancelled before arming")

    def _mark_finishing(self) -> bool:
        """The pulse's waits have returned; False when a cancel came first."""
        with self._lock:
            if self._state is LaserOperationState.CANCELLED:
                return False
            self._finishing = True
            return True

    def _mark_triggered(self, detail="waveform completed after trigger"):
        with self._lock:
            if self._state is LaserOperationState.ARMED:
                self._transition_locked(LaserOperationState.TRIGGERED, detail)

    def _complete(self):
        with self._lock:
            if self._state is LaserOperationState.ARMED:
                self._transition_locked(LaserOperationState.TRIGGERED)
            if self._state is LaserOperationState.TRIGGERED:
                self._transition_locked(LaserOperationState.COMPLETED)
        self._finish_terminal()

    def _fail(self, error):
        with self._lock:
            if self._state not in self.TERMINAL:
                self._error = error
                self._transition_locked(
                    LaserOperationState.FAILED,
                    f"{type(error).__name__}: {error}",
                )
        self._finish_terminal()

    def _transition_locked(self, state, detail=""):
        self._state = LaserOperationState(state)
        self._observations.append((self._state.value, time.perf_counter(), detail))

    def _finish_terminal(self):
        callbacks = ()
        with self._lock:
            if not self._done.is_set():
                self._done.set()
                self._armed.set()
                callbacks = tuple(self._terminal_callbacks)
                self._terminal_callbacks.clear()
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                logger.exception("Laser terminal callback failed")


@dataclasses.dataclass
class _NidaqLaserTasks:
    analog_output: Optional[object]
    diode_input: Optional[object]
    command_copy_input: Optional[object]
    shutter_output: object
    auxiliary_output: Optional[object]

    def close(self) -> None:
        errors = []
        for name, task in (
            ("shutter_output", self.shutter_output),
            ("auxiliary_output", self.auxiliary_output),
            ("analog_output", self.analog_output),
            ("command_copy_input", self.command_copy_input),
            ("diode_input", self.diode_input),
        ):
            if task is not None:
                try:
                    task.close()
                except Exception as exc:
                    errors.append((name, exc))
        if errors:
            names = ", ".join(name for name, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ laser task(s): {names}") from errors[0][1]


class NidaqLaserController:
    """NI-DAQmx-backed laser controller.

    Manual command/shutter paths use on-demand NI-DAQmx writes. When
    ``LaserSystemConfiguration.hardware_timed`` is enabled, finite pulse trains
    use transient sample-clocked AO tasks so manual command tasks do not reserve
    the same physical output channel.
    """

    def __init__(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
        timing_plan: Optional[NidaqTimingPlan] = None,
        analog_terminal_config: Optional[str] = None,
    ):
        if configuration.backend != "nidaq":
            raise ValueError("NidaqLaserController requires laser backend 'nidaq'")
        runtime_started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQmx runtime | consumer=laser")
        self._nidaqmx = _load_nidaqmx()
        _ignore_the_aborts_warning(self._nidaqmx)
        #: How every input this controller adds is referenced: the stream's
        #: analogTerminalConfig, given by its caller, as the stream maps it.
        #: None leaves each to DAQmx's default (_add_voltage_input).
        self._analog_terminal_config = resolve_analog_terminal_config(
            self._nidaqmx, analog_terminal_config)
        log_hardware_initialization(
            logger,
            "READY | NI-DAQmx runtime | consumer=laser elapsed=%.3fs",
            time.perf_counter() - runtime_started,
        )
        self._configuration = configuration
        self._feedback_reader = feedback_reader
        self._timing_plan = timing_plan
        self._last_timing_status = {
            "status": "independent",
            "reason": "No finite laser waveform has been executed",
        }
        self._tasks: Dict[LaserChannelId, _NidaqLaserTasks] = {}
        #: Backplane routes held open for this controller, source to
        #: destination, released in close(). Changed only under
        #: _operation_lock: a ramp releases its own routes from its thread
        #: while close() may be releasing all of them from another.
        self._trigger_routes: List[Tuple[str, str]] = []
        self._command_volts: Dict[LaserChannelId, float] = {}
        #: Bookkeeping only: nothing is called in the driver while it is held
        #: (a driver that hangs would hold close() at it, before it could mark
        #: the controller closed, with every cancel and abort behind it).
        self._operation_lock = threading.RLock()
        #: Routes being connected or released outside the lock, by the call
        #: doing it; see _shared_clock_for and _release_routes.
        self._pending_routes: Dict[Tuple[str, str], str] = {}
        self._routes_settled = threading.Condition(self._operation_lock)
        self._live_operations: Dict[str, NidaqLaserOperation] = {}
        #: The tasks of a calibration ramp in progress, for close() to abort
        #: from another thread, as (owner, name, task); see run_calibration_ramp.
        self._calibration_tasks: List[Tuple[str, str, object]] = []
        #: Those the ramp's finally has let go of, by id, under the lock:
        #: close()'s abort leaves them to its cleanup.
        self._released_calibration_tasks: set = set()
        #: The ramp's tasks started so far, by id, under the lock: close()'s
        #: abort of one of them can warn (200010), and says so.
        self._started_calibration_tasks: set = set()
        #: Those close() has aborted, by id, under the lock: the ramp's own
        #: stop of one after that is not noted as an early stop.
        self._aborted_calibration_tasks: set = set()
        #: Set by close(), under _operation_lock, before it calls the driver.
        #: A ramp or pulse train starts, and a route is kept, only while it is
        #: clear.
        self._closed = False
        #: Clear while a calibration ramp owns tasks; set by its finally once
        #: it has stopped and closed them. close() waits on it, bounded.
        self._calibration_released = threading.Event()
        self._calibration_released.set()
        #: What close() stopped waiting for: operations it cancelled that had
        #: not ended, and whether the ramp had not let go (work_left_running).
        self._given_up_operations: List[NidaqLaserOperation] = []
        self._given_up_on_ramp = False
        #: Lasers a CRITICAL said may be left driven, a pulse's or a ramp's
        #: own reset of the command refused, by id, under the lock; close()
        #: says so when its own reset of one succeeds (_note_reset_after_all).
        self._left_driven: set = set()
        try:
            for channel in configuration.channels:
                channel_started = time.perf_counter()
                log_hardware_initialization(
                    logger,
                    "START | NI-DAQ laser channel | id=%s AO=%s diode_AI=%s shutter_DO=%s auxiliary_DO=%s",
                    channel.channel_id.value,
                    channel.analog_output,
                    channel.diode_input,
                    channel.shutter_output,
                    channel.auxiliary_output,
                )
                self._tasks[channel.channel_id] = self._create_channel_tasks(channel)
                self._command_volts[channel.channel_id] = channel.minimum_command_volts
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
                self.set_shutter_open(channel.channel_id, False)
                if channel.auxiliary_output is not None:
                    self.set_auxiliary_output(channel.channel_id, False)
                self._connect_trigger_route(channel)
                log_hardware_initialization(
                    logger,
                    "READY | NI-DAQ laser channel | id=%s elapsed=%.3fs",
                    channel.channel_id.value,
                    time.perf_counter() - channel_started,
                )
        except Exception as error:
            still_closing = self._close_after_failed_open()
            if still_closing is not None:
                # For the caller, which never holds this controller: set when
                # that close ends, and laser work waits for it until then.
                raise LaserControllerStillClosing(error, still_closing) from error
            raise

    def _close_after_failed_open(self) -> Optional[threading.Event]:
        """Close what a failed open had opened, bounded; an event if it goes on.

        The close had no bound: a driver hung in it hung the Run start, or
        the calibration ramp, that was opening the controller. Past the bound
        it goes on inside the driver on a thread of its own, daemon so that a
        close that never returns does not hold the process open, and the
        event it returns is set when it ends. None when it ended in time.
        """
        closed = threading.Event()

        def close():
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close partially initialized NI-DAQ laser controller")
            finally:
                closed.set()

        threading.Thread(
            target=close, name="NidaqLaserFailedOpenClose", daemon=True).start()
        if closed.wait(_FAILED_OPEN_CLOSE_TIMEOUT_S):
            return None
        logger.error(
            "A NI-DAQ laser controller that failed to open had not closed "
            "what it had opened %.1f s later; its close goes on inside the "
            "driver", _FAILED_OPEN_CLOSE_TIMEOUT_S)
        return closed

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    @property
    def timing_status(self) -> dict:
        return dict(self._last_timing_status)

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        channel = self._configuration.get_channel(channel_id)
        applied = channel.clamp_command_voltage(volts)
        tasks = self._tasks[channel.channel_id]
        if tasks.analog_output is None:
            self._write_transient_analog_sample(channel, applied)
        else:
            tasks.analog_output.write(applied, auto_start=True)
        self._command_volts[channel.channel_id] = applied
        return applied

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        channel = self._configuration.get_channel(channel_id)
        self._tasks[channel.channel_id].shutter_output.write(bool(is_open), auto_start=True)

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        channel = self._configuration.get_channel(channel_id)
        auxiliary_output = self._tasks[channel.channel_id].auxiliary_output
        if auxiliary_output is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no auxiliary_output configured"
            )
        auxiliary_output.write(bool(enabled), auto_start=True)

    def read_diode_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
        if self._feedback_reader is not None:
            return float(self._feedback_reader(channel.diode_input))
        raw = self._tasks[channel.channel_id].diode_input.read()
        return float(raw) * channel.feedback_scale

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        channel = self._configuration.get_channel(channel_id)
        command_copy_volts = self._read_optional_command_copy_voltage(channel.channel_id)
        if command_copy_volts is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no AI command-copy input configured"
            )
        return command_copy_volts

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        normalized = normalize_laser_channel_id(channel_id)
        return LaserFeedbackSample(
            channel_id=normalized,
            command_volts=self._command_volts[normalized],
            diode_volts=self.read_diode_voltage(normalized),
            command_copy_volts=self._read_optional_command_copy_voltage(normalized),
        )

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        self.run_synchronized_pulse_train(
            LaserSynchronizedPulseTrain(
                pulse_trains=(pulse_train,),
                trigger_source=pulse_train.trigger_source,
                trigger_edge=pulse_train.trigger_edge,
                enable_pmt_shutter=pulse_train.enable_pmt_shutter,
                wait=pulse_train.wait,
                timeout_seconds=pulse_train.timeout_seconds,
            )
        )

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain):
        if not self._configuration.hardware_timed:
            raise LaserPulseRefused("Hardware-timed laser pulse trains require laser configuration hardware_timed=True")
        # Before the operation exists, as every refusal of a pulse that drives
        # nothing is. Made in the pulse's own thread, it came back as that
        # operation's failure, and the laser model kept the amplitude as what
        # the output may still hold (final re-review, affb7491).
        for item in pulse_train.pulse_trains:
            try:
                self._validate_command_voltage(
                    self._configuration.get_channel(item.channel_id), item.amplitude_volts)
            except ValueError as error:
                raise LaserPulseRefused(str(error)) from error
        # Both paths own their output through one operation, so close() can
        # cancel and wait for either. A waited-for train (Run Pulse) used to
        # run with none: close() could not stop it, and its command reset met
        # the train's task on the output at -50103.
        resources = tuple(
            self._configuration.get_channel(item.channel_id).analog_output
            for item in pulse_train.pulse_trains
        )
        with self._operation_lock:
            # close() cancels the operations it finds when it marks the
            # controller closed; one registered after that would drive the
            # laser after close() had reset it.
            if self._closed:
                raise LaserPulseRefused(
                    "the laser controller is closed; no pulse train is started")
            # By board, not by channel. A board has one analog output timing
            # engine, and its ao/SampleClock is that engine's; NI documents
            # one timed analog output task per board at a time. A second
            # pulse on another channel of the board went ahead, and borrowed
            # the first one's pulseClockLine route, which the first then
            # released under it. Run Pulse, which waits, is refused so while
            # a trial's pulse is armed. Counted until the operation is done,
            # not only until it is marked cancelled: its cleanup still holds
            # its tasks and routes then. Board names compare as DAQmx
            # compares them, without regard to case.
            boards = {_device_of(resource).lower() for resource in resources}
            conflicts = [
                operation
                for operation in self._live_operations.values()
                if boards & {_device_of(resource).lower() for resource in operation.resources}
                and not operation._done.is_set()
            ]
            if conflicts:
                raise LaserPulseRefused(self._board_refusal(pulse_train, conflicts, boards))
            lasers = [int(item.channel_id) for item in pulse_train.pulse_trains]
            owner = (f"laser{'s' if len(lasers) > 1 else ''} "
                     f"{', '.join(map(str, lasers))}")
            operation = NidaqLaserOperation(
                resources=resources,
                context=pulse_train.operation_context,
                terminal_callback=self._release_operation,
                abort_task=lambda task, name, running: self._abort_task(
                    task, owner, name, running=running),
                before_abort=lambda: self._close_pulse_shutters(pulse_train),
            )
            self._live_operations[operation.operation_id] = operation

        def execute():
            ended_by = None
            try:
                self._execute_synchronized_pulse_train(
                    pulse_train,
                    operation=operation,
                )
            except Exception as error:
                ended_by = error
            except BaseException as error:
                ended_by = error
                raise
            finally:
                # Terminal whatever ended it, a BaseException too: an operation
                # left live owns its output, and every pulse on it after that
                # is refused. Stored as an Exception: wait() raises the stored
                # error, and close(), waiting on the operation, catches only
                # Exception; given a KeyboardInterrupt it skipped the reset.
                # This thread still raises the original, above.
                if operation.state is LaserOperationState.CANCELLED:
                    operation._finish_terminal()
                elif ended_by is None:
                    operation._complete()
                elif isinstance(ended_by, Exception):
                    operation._fail(ended_by)
                else:
                    operation._fail(RuntimeError(
                        f"the pulse train was interrupted ({type(ended_by).__name__})"))

        if pulse_train.wait:
            # On the caller's thread, which it still blocks.
            operation._thread = threading.current_thread()
            execute()
            if operation.state is LaserOperationState.CANCELLED:
                # Not the DAQmx error the cancel's abort provoked (-88709).
                # A close says so; a cancel alone is only a cancel.
                raise RuntimeError(
                    f"Laser operation {operation.operation_id} was cancelled"
                    + (": the laser controller was closed while it ran"
                       if self._closed else ""))
            if operation.error is not None:
                raise operation.error
            return None
        operation._thread = threading.Thread(
            target=execute,
            name=f"NidaqLaser-{operation.operation_id[:8]}",
            daemon=True,
        )
        try:
            operation._thread.start()
        except Exception as error:
            # A thread that never started ran nothing, so there is nothing to
            # clean up; left PREPARED among the live operations it held the
            # whole board. One that did start (an error once it runs) is its
            # own thread's to end. A KeyboardInterrupt is not caught: it can
            # land while start() waits for a thread already running, whose
            # pulse is then still tracked, and ends by itself.
            if operation._thread.ident is None:
                operation._fail(RuntimeError(
                    f"the pulse train's thread did not start ({error})"))
            raise
        try:
            operation.wait_until_armed(timeout=5.0)
        except Exception:
            operation.cancel()
            # Ended before this returns, so a retry finds the board free.
            if not operation.wait_until_finished(CANCELLED_OPERATION_WAIT_S):
                logger.warning(
                    "The cancelled laser operation %s had not ended %.1f s later; "
                    "a pulse armed next on its board is refused until it does",
                    operation.operation_id, CANCELLED_OPERATION_WAIT_S)
            raise
        return operation

    def _close_pulse_shutters(self, pulse_train) -> None:
        """Close the shutter of each of the pulse's lasers, for a cancel.

        Each of them, whether or not the pulse opens it or would leave it
        open, and whoever opened it: a cancel leaves the laser safe. One that
        fails is logged, and the others are still closed.
        """
        for channel_pulse in pulse_train.pulse_trains:
            channel = self._configuration.get_channel(channel_pulse.channel_id)
            try:
                self._close_shutter(channel)
            except Exception:
                logger.exception(
                    "Failed to close the laser %s shutter before a cancel's abort",
                    channel.channel_id.value)

    def _close_shutter(self, channel) -> None:
        """Close `channel`'s shutter: by its own task, or, once close() has
        let go of that, by a transient one.

        Looked for once, as it is written: looked for first and used after,
        close() clearing the tasks in between raised KeyError.
        """
        try:
            self.set_shutter_open(channel.channel_id, False)
        except KeyError:
            self._write_transient_digital_line(
                channel.shutter_output, False,
                f"laser_{channel.channel_id.value}_shutter_reset")

    def _reset_command(self, channel) -> None:
        """`channel`'s command to its minimum, as _close_shutter closes it."""
        try:
            self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
        except KeyError:
            self._write_transient_analog_sample(channel, channel.minimum_command_volts)

    def _abort_task(self, task, owner: str, name: str, *, running: bool) -> None:
        """TASK_ABORT on a pulse's or a ramp's task, from any thread.

        Nothing else is asked of the driver: `name` is the one its owner made
        the task with. Task.name is a driver query, and the owner,
        woken by this abort, can have cleared the task before the abort
        returns (-200088, christielab10, H8c). Aborting a running task warns,
        DaqWarning 200010 (H5b, H8a), the abort doing what was asked: one
        filter, set as the driver loads, keeps that warning, by its text, off
        stderr (_ignore_the_aborts_warning), and this line says it can come.
        """
        if running:
            logger.debug("%s: aborting its running task %s (DAQmx may warn 200010)",
                         owner, name)
        else:
            logger.debug("%s: aborting its task %s, not started", owner, name)
        task.control(self._nidaqmx.constants.TaskMode.TASK_ABORT)

    def _board_refusal(self, pulse_train, conflicts, boards) -> str:
        """Why a pulse is refused while another holds its board, briefly first.

        Run Pulse's status line shows about 160 characters: the laser, what
        holds the board and what to do come first, the ids after.
        """
        requested = [int(item.channel_id) for item in pulse_train.pulse_trains]
        lasers = "Laser" if len(requested) == 1 else "Lasers"
        holders = " and ".join(
            self._describe_operation(operation) for operation in conflicts)
        # One already cancelled is only ending: nothing is left to cancel.
        remedy = (
            "it is finishing its cleanup; try again in a moment"
            if all(operation.state is LaserOperationState.CANCELLED
                   for operation in conflicts)
            else "wait for it to end, or cancel it")
        held = sorted({
            resource for operation in conflicts for resource in operation.resources
            if _device_of(resource).lower() in boards})
        held_boards = sorted({_device_of(resource) for resource in held})
        return (
            f"{lasers} {', '.join(map(str, requested))}: refused while {holders} "
            f"{'hold' if len(conflicts) > 1 else 'holds'} the analog output of "
            f"{', '.join(held_boards)}; {remedy}. A board "
            "runs one timed analog output at a time (laser operation "
            f"{', '.join(operation.operation_id for operation in conflicts)} on "
            f"{', '.join(held)})")

    def _describe_operation(self, operation) -> str:
        """An operation as an operator knows it: Test stim's, a trial's, a laser's.

        By its context: an `operation_label` its caller gave it (Test stim
        arms with a recipe whose trial number is 0), else its trial.
        """
        lasers = sorted(
            int(channel.channel_id) for channel in self._configuration.channels
            if channel.analog_output in operation.resources)
        on = (f"laser {', '.join(map(str, lasers))}" if lasers
              else ", ".join(operation.resources))
        label = operation.context.get("operation_label")
        if label:
            return f"{label}'s pulse on {on}"
        trial = operation.context.get("logical_trial_id")
        return f"trial {trial}'s pulse on {on}" if trial is not None else f"the pulse on {on}"

    def _release_operation(self, operation):
        with self._operation_lock:
            self._live_operations.pop(operation.operation_id, None)

    def _execute_synchronized_pulse_train(
        self,
        pulse_train: LaserSynchronizedPulseTrain,
        *,
        operation: NidaqLaserOperation,
    ) -> None:
        channels = [
            self._configuration.get_channel(channel_pulse.channel_id)
            for channel_pulse in pulse_train.pulse_trains
        ]
        # Every amplitude is in its laser's range: run_synchronized_pulse_train
        # refuses one that is not, before the operation is made.

        # Resolved before the waveform is built rather than after, because the
        # timing decides the rate it will be generated at and every sample
        # count below is computed from that rate.
        timing_kwargs, timing_status = self._resolve_pulse_timing(
            channels, pulse_train,
        )
        sample_rate_hz = self._require_sample_rate(timing_kwargs)
        waveforms = [
            self._build_pulse_train_waveform(channel, channel_pulse, sample_rate_hz)
            for channel, channel_pulse in zip(channels, pulse_train.pulse_trains)
        ]
        pmt_enabled = pulse_train.enable_pmt_shutter or any(
            channel_pulse.enable_pmt_shutter for channel_pulse in pulse_train.pulse_trains
        )
        pmt_open_delay_ms = max(
            (channel_pulse.pmt_shutter_open_delay_ms for channel_pulse in pulse_train.pulse_trains),
            default=0.0,
        ) if pmt_enabled else 0.0
        pmt_close_delay_ms = max(
            (channel_pulse.pmt_shutter_close_delay_ms for channel_pulse in pulse_train.pulse_trains),
            default=0.0,
        ) if pmt_enabled else 0.0
        pre_samples = _samples_from_ms(pmt_open_delay_ms, sample_rate_hz)
        post_samples = _samples_from_ms(pmt_close_delay_ms, sample_rate_hz)
        max_waveform_samples = max(len(waveform) for waveform in waveforms)
        timed_waveforms = []
        for channel, waveform in zip(channels, waveforms):
            minimum = channel.minimum_command_volts
            timed_waveforms.append(
                [minimum] * pre_samples
                + waveform
                + [minimum] * (max_waveform_samples - len(waveform) + post_samples)
            )
        total_samples = len(timed_waveforms[0])
        timeout_seconds = pulse_train.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = total_samples / sample_rate_hz + 5.0
        ao_name = "laser_sync_pulse_ao"
        ao_task = self._create_synchronized_analog_output_task(channels, ao_name)
        digital_tasks = []
        # The name each digital task is made with, in the same order.
        digital_names = []
        # The clock routes this pulse train adds for its digital outputs, by
        # name, which it releases when it ends, as the calibration ramp does.
        added_routes: List[Tuple[str, str]] = []
        run_error = None
        try:
            operation._set_timing_status(timing_status)
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=total_samples,
                **timing_kwargs,
            )
            self._configure_timing_reference(ao_task, timing_status)
            if pulse_train.trigger_source:
                ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    pulse_train.trigger_source,
                    trigger_edge=self._get_trigger_edge(pulse_train.trigger_edge),
                )
            ao_task.write(timed_waveforms[0] if len(timed_waveforms) == 1 else timed_waveforms, auto_start=False)
            sample_clock_source = self._analog_output_sample_clock_source(channels[0].analog_output)

            def digital_timing(physical_line):
                # The line's clock, and no start trigger (_pulse_digital_clock).
                return self._pulse_digital_clock(
                    physical_line, sample_clock_source, added_routes), None

            if pmt_enabled:
                pmt_line = self._require_pmt_shutter_output()
                digital_names.append("laser_pmt_shutter_do")
                digital_tasks.append(
                    self._create_finite_digital_output_task(
                        pmt_line,
                        digital_names[-1],
                        [True] * total_samples,
                        sample_rate_hz,
                        total_samples,
                        *digital_timing(pmt_line),
                        pulse_train.trigger_edge,
                    )
                )
            for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
                if channel_pulse.emit_trigger_output:
                    if channel.trigger_output is None:
                        raise RuntimeError(
                            f"laser channel {channel.channel_id.value} requested trigger output, "
                            "but trigger_output is not configured"
                        )
                    digital_names.append(f"laser_{channel.channel_id.value}_trigger_do")
                    digital_tasks.append(
                        self._create_finite_digital_output_task(
                            channel.trigger_output,
                            digital_names[-1],
                            self._build_digital_pulse_waveform(
                                total_samples,
                                channel_pulse.trigger_output_pulse_ms,
                                sample_rate_hz,
                            ),
                            sample_rate_hz,
                            total_samples,
                            *digital_timing(channel.trigger_output),
                            pulse_train.trigger_edge,
                        )
                    )
                if channel_pulse.emit_timing_trigger_output:
                    if channel.timing_trigger_output is None:
                        raise RuntimeError(
                            f"laser channel {channel.channel_id.value} requested timing trigger output, "
                            "but timing_trigger_output is not configured"
                        )
                    digital_names.append(
                        f"laser_{channel.channel_id.value}_timing_trigger_do")
                    digital_tasks.append(
                        self._create_finite_digital_output_task(
                            channel.timing_trigger_output,
                            digital_names[-1],
                            self._build_digital_pulse_waveform(
                                total_samples,
                                channel_pulse.timing_trigger_output_pulse_ms,
                                sample_rate_hz,
                            ),
                            sample_rate_hz,
                            total_samples,
                            *digital_timing(channel.timing_trigger_output),
                            pulse_train.trigger_edge,
                        )
                    )
            if digital_tasks:
                # Programmed before the lines start, so that whatever the AO
                # clock does while the board is programmed happens before a
                # line takes it: then the lines, then the AO.
                ao_task.control(self._nidaqmx.constants.TaskMode.TASK_COMMIT)

            def open_shutters():
                for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
                    if channel_pulse.open_shutter:
                        self.set_shutter_open(channel.channel_id, True)

            def close_shutters():
                # Each on its own: one that fails leaves the others to close.
                for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
                    if channel_pulse.open_shutter:
                        try:
                            self._close_shutter(channel)
                        except Exception:
                            logger.exception(
                                "Failed to close the laser %s shutter a cancel "
                                "found opening", channel.channel_id.value)

            operation._open_shutters_unless_cancelled(open_shutters, close_shutters)
            operation._bind_tasks(zip(
                (ao_name, *digital_names), (ao_task, *digital_tasks)))
            operation._require_not_cancelled()
            if pulse_train.defer_start:
                operation._mark_armed()
                if not operation._start_requested.wait(timeout_seconds):
                    raise TimeoutError("Deferred laser operation did not receive a start request")
                operation._require_not_cancelled()
            for task in (*digital_tasks, ao_task):
                task.start()
                operation._mark_started(task)
            if operation.state is LaserOperationState.CANCELLED:
                # A cancel between the look above and the starts aborted
                # tasks not yet started, which does nothing (H5c): the train
                # would run. Ended here instead, as a cancel.
                raise RuntimeError("Laser operation was cancelled as it started")
            if pulse_train.defer_start:
                operation._mark_triggered("NI software start accepted")
            else:
                operation._mark_armed()
            self._last_timing_status = timing_status
            ao_task.wait_until_done(timeout=timeout_seconds)
            for task in digital_tasks:
                task.wait_until_done(timeout=timeout_seconds)
            operation._mark_triggered()
            if not operation._mark_finishing():
                raise RuntimeError("Laser operation was cancelled as it ended")
        except Exception as exc:
            run_error = exc
            raise
        finally:
            # Before the first stop: a cancel from now on leaves them here.
            operation._let_go_of_tasks()
            self._cleanup_pulse_train(
                ao_task=ao_task,
                digital_tasks=digital_tasks,
                channels=channels,
                channel_pulses=pulse_train.pulse_trains,
                close_pmt=pmt_enabled,
                run_error=run_error,
                routes=added_routes,
            )

    def _resolve_pulse_timing(self, channels, pulse_train):
        plan = self._timing_plan
        output_devices = tuple(dict.fromkeys(
            channel.analog_output.strip("/").split("/", 1)[0]
            for channel in channels
        ))
        base = {
            "devices": output_devices,
            "sampleClockSource": None,
            "startTriggerSource": pulse_train.trigger_source,
            "referenceClockSource": None,
        }
        if pulse_train.defer_start:
            return {}, {
                **base,
                "status": "software_start",
                "reason": (
                    "Finite output is committed before a dedicated software "
                    "start request"
                ),
                "referenceClockSource": (
                    None
                    if plan is None or not plan.is_valid
                    else plan.reference_clock_source
                ),
            }
        if plan is None or not plan.is_valid:
            return {}, {
                **base,
                "status": "independent",
                "reason": "No valid shared NI timing plan was applied",
            }
        undeclared = tuple(
            device for device in output_devices
            if device not in plan.hardware_output_devices
        )
        if undeclared:
            return {}, {
                **base,
                "status": "unsupported",
                "reason": "Laser output device was not in the resolved timing topology",
            }
        if not pulse_train.trigger_source:
            return {}, {
                **base,
                "status": "declared_not_armed",
                "reason": (
                    "The acquisition clock is already running and this on-demand "
                    "waveform has no future hardware start trigger"
                ),
            }
        if not plan.sample_clock_source:
            return {}, {
                **base,
                "status": "unsupported",
                "reason": "Resolved timing topology has no shared sample clock",
            }
        return {"source": self._shared_clock_for(
            output_devices[0], plan.sample_clock_source,
            line=self._configuration.backplane_clock_line)}, {
            **base,
            "status": "hardware_synchronized",
            "reason": "Finite output armed for a future trigger on the shared sample clock",
            "sampleClockSource": plan.sample_clock_source,
            "referenceClockSource": plan.reference_clock_source,
        }

    def _pulse_digital_clock(
        self,
        physical_line: str,
        ao_clock: str,
        added: List[Tuple[str, str]],
    ) -> str:
        """The sample clock one of a pulse's clocked digital lines runs on.

        Always the pulse's own analog output clock, and never with a start
        trigger: that clock ticks only once the output has triggered, and the
        lines start before the output, so each samples on its edges. A line
        cannot wait for a trigger instead: an M Series board's clocked
        digital output takes none (christielab10's 6221 and 6713 report an
        empty do_trig_usage, and TASK_VERIFY refuses one with -200452,
        2026-09-25).

        On the output's own board the clock is named there, as it always was.
        On another board naming it is an implicit cross-board route, refused
        on christielab10's unidentified chassis (-89125), and a clocked line
        has to sit on the 6221 there: the clock is driven onto
        pulse_clock_line for the pulse and read on the line's board.
        backplane_clock_line is not used for it, since in a synchronized
        pulse it carries the input stream's clock to the output until close().
        """
        output_device = _device_of(physical_line)
        if output_device == _device_of(ao_clock):
            return ao_clock
        return self._shared_clock_for(
            output_device, ao_clock, added=added,
            line=self._configuration.pulse_clock_line)

    def _configure_timing_reference(self, task, timing_status) -> None:
        if timing_status.get("status") != "hardware_synchronized":
            return
        # This used to set the terminal straight from the plan. The plan's
        # default is spelled PXI_CLK10 and the PXI-6713 driving the laser has
        # no Clk10 terminal at all, so arming a triggered pulse died at
        # -200452 on laser_sync_pulse_ao - after the wiring was already
        # right, which made it look like a trigger fault.
        apply_reference_clock(self._nidaqmx, task, self._timing_plan)

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        if self._feedback_reader is not None:
            raise RuntimeError(
                "Laser calibration is unavailable while the shared NI-DAQ input "
                "stream owns the feedback channels"
            )
        if not self._configuration.hardware_timed:
            raise RuntimeError("Hardware-timed laser calibration ramps require laser configuration hardware_timed=True")
        channel = self._configuration.get_channel(ramp.channel_id)
        self._validate_command_voltage(channel, ramp.start_volts)
        self._validate_command_voltage(channel, ramp.stop_volts)
        if ramp.enable_pmt_shutter and (
            ramp.pmt_shutter_open_delay_ms > 0 or ramp.pmt_shutter_close_delay_ms > 0
        ):
            raise NotImplementedError("PMT shutter delays for calibration ramps are not implemented yet")
        sample_rate_hz = self._require_sample_rate()
        settle_samples = ramp.settle_sample_count(sample_rate_hz)
        waveform = self._build_calibration_ramp_waveform(ramp)
        timeout_seconds = ramp.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = len(waveform) / sample_rate_hz + 5.0
        ao_task = None
        ai_task = None
        digital_tasks = []
        run_error = None
        points = ()
        owner = f"laser {channel.channel_id.value}'s calibration ramp"
        # The ramp's tasks known to have finished, by id: one stopped before
        # it finished can warn (200010), which the filter hides, so the stop
        # says so (_cleanup_calibration_ramp).
        finished = set()
        # The clock routes this ramp adds, by name, which it releases when it
        # ends. Counted, as they were, the list was read and cut from two
        # threads: a close() in between emptied it, and the ramp's own route,
        # added after, was never released.
        added_routes: List[Tuple[str, str]] = []
        with self._operation_lock:
            if self._closed:
                raise RuntimeError(
                    "the laser controller was closed before the calibration "
                    "ramp started")
            # close() waits on this before it resets the laser.
            self._calibration_released.clear()
            self._released_calibration_tasks.clear()
            self._started_calibration_tasks.clear()
            self._aborted_calibration_tasks.clear()
        try:
            # Both inside the cleanup scope: created before it, an input task
            # that failed to create left the output task open on ao0.
            ao_name = f"laser_{channel.channel_id.value}_calibration_ao"
            ai_name = f"laser_{channel.channel_id.value}_calibration_ai"
            ao_task = self._create_analog_output_task(channel, ao_name)
            # Where close() can find them: reachAQ closing mid-ramp closes
            # this controller from its own thread, and this one may then be
            # inside DAQmx, waiting on these.
            self._hold_calibration_tasks(owner, ao_name, ao_task)
            ai_task = self._create_calibration_input_task(channel, ai_name)
            self._hold_calibration_tasks(owner, ai_name, ai_task)
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(waveform),
            )
            # The command's sample clock, as each input board can see it. On
            # christielab10 the command is on the 6713 and the diode on the
            # 6221: named across the boards, the clock needed a route DAQmx
            # reserves a backplane line for, and it refuses that on this
            # unidentified chassis (-89125). _shared_clock_for drives it onto
            # the backplane clock line by name, as a pulse train's shared
            # clock is, and leaves a single-board rig untouched.
            sample_clock_source = self._analog_output_sample_clock_source(channel.analog_output)
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=self._shared_clock_for(
                    _device_of(channel.diode_input), sample_clock_source,
                    added=added_routes, line=self._configuration.backplane_clock_line),
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(waveform),
            )
            if ramp.enable_pmt_shutter:
                pmt_line = self._require_pmt_shutter_output()
                pmt_name = "laser_pmt_shutter_calibration_do"
                digital_tasks.append(
                    self._create_finite_digital_output_task(
                        pmt_line,
                        pmt_name,
                        [True] * len(waveform),
                        sample_rate_hz,
                        len(waveform),
                        self._shared_clock_for(
                            _device_of(pmt_line), sample_clock_source,
                            added=added_routes,
                            line=self._configuration.backplane_clock_line),
                        None,
                        "rising",
                    )
                )
                self._hold_calibration_tasks(owner, pmt_name, digital_tasks[-1])
            ao_task.write(waveform, auto_start=False)
            # Only while the controller is open: a close that came first has
            # reset the outputs, and must not find the laser driven after it.
            # The starts themselves are made outside the lock, which close()
            # takes to mark the controller closed; they used to be made under
            # it, and a driver hung in one held close() there. A close that
            # comes while they are made aborts the tasks it finds, and waits
            # for this ramp to let go of them before it resets the laser;
            # the check after the starts ends the ramp at once.
            with self._operation_lock:
                if self._closed:
                    raise RuntimeError(
                        "the laser controller was closed before the calibration "
                        "ramp started")
            if ramp.open_shutter:
                self.set_shutter_open(channel.channel_id, True)
            for task in (*digital_tasks, ai_task, ao_task):
                task.start()
                self._mark_calibration_task_started(task)
            with self._operation_lock:
                if self._closed:
                    raise RuntimeError(
                        "the laser controller was closed as the calibration "
                        "ramp started")
            try:
                ao_task.wait_until_done(timeout=timeout_seconds)
                # The lines run on the output's clock, for as many samples.
                finished.update(id(task) for task in (ao_task, *digital_tasks))
                ai_task.wait_until_done(timeout=timeout_seconds)
                finished.add(id(ai_task))
                raw_samples = ai_task.read(
                    number_of_samples_per_channel=len(waveform), timeout=timeout_seconds)
            except Exception as error:
                with self._operation_lock:
                    closed = self._closed
                if closed:
                    # close() aborted the ramp's tasks, and the wait or read
                    # raised the abort's own error (-88709), which says
                    # nothing of the close. Told as a closed pulse is told;
                    # the driver's error is kept as the cause.
                    raise RuntimeError(
                        "the laser calibration ramp was stopped: the laser "
                        "controller was closed while it ran") from error
                raise
            points = self._build_calibration_points(channel, ramp, raw_samples, settle_samples)
        except Exception as exc:
            run_error = exc
            raise
        finally:
            try:
                with self._operation_lock:
                    # Let go of, before they are stopped and closed below: an
                    # abort from close() that meets one of them closing, or
                    # closed, is this cleanup overtaking it, not a failure.
                    self._released_calibration_tasks.update(
                        id(task) for _owner, _name, task in self._calibration_tasks)
                    unfinished = {
                        id(task): name
                        for _owner, name, task in self._calibration_tasks
                        if id(task) in self._started_calibration_tasks
                        and id(task) not in finished
                        and id(task) not in self._aborted_calibration_tasks}
                    self._calibration_tasks.clear()
                    self._started_calibration_tasks.clear()
                self._cleanup_calibration_ramp(
                    ao_task=ao_task,
                    ai_task=ai_task,
                    digital_tasks=digital_tasks,
                    channel=channel,
                    ramp=ramp,
                    run_error=run_error,
                    routes=added_routes,
                    owner=owner,
                    unfinished=unfinished,
                )
            finally:
                # After the tasks are stopped and closed, and after the
                # cleanup's own writes: close() resets the laser only then, so
                # its writes never meet a task of this ramp's on the lines.
                self._calibration_released.set()
        return points

    def _hold_calibration_tasks(self, owner: str, name: str, task) -> None:
        """Keep a ramp's task where close() can abort it, unless closed.

        With the name the ramp made it with: a task's own name is a driver
        query, which close()'s abort must not make (_abort_task).
        """
        with self._operation_lock:
            if self._closed:
                raise RuntimeError(
                    "the laser controller was closed before the calibration "
                    "ramp started")
            self._calibration_tasks.append((owner, name, task))

    def _mark_calibration_task_started(self, task) -> None:
        """A ramp's task has been started: close()'s abort of it can warn."""
        with self._operation_lock:
            self._started_calibration_tasks.add(id(task))

    def _validate_command_voltage(self, channel: LaserChannelConfiguration, volts: float) -> None:
        if not channel.minimum_command_volts <= volts <= channel.maximum_command_volts:
            raise ValueError(
                f"laser channel {channel.channel_id.value} command {volts} V is outside "
                f"the configured range {channel.minimum_command_volts}..{channel.maximum_command_volts} V"
            )

    def _cleanup_pulse_train(
        self,
        ao_task: object,
        digital_tasks: List[object],
        channels: List[LaserChannelConfiguration],
        channel_pulses: Tuple[LaserPulseTrain, ...],
        close_pmt: bool,
        run_error: Optional[BaseException],
        routes: Sequence[Tuple[str, str]] = (),
    ) -> None:
        errors = []
        self._stop_and_close_task("pulse analog output task", ao_task, errors)
        # The command straight after the output's own task lets go of it:
        # aborted mid-pulse, DAQmx leaves the output on the last sample it
        # wrote, which can be the pulse's high level, until this. Then the
        # shutters. The digital lines and routes come after.
        for channel, channel_pulse in zip(channels, channel_pulses):
            try:
                # Through a transient task once close() has let go of the
                # channel's tasks: past its wait for this pulse, its own reset
                # met this pulse's open task (-50103). Written here, the task
                # now closed, as the ramp's closed branch does.
                self._reset_command(channel)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} command reset", exc))
                self._log_command_left_driven(
                    channel, "its pulse train",
                    f"the pulse's amplitude, {channel_pulse.amplitude_volts:g} V")
        for channel, channel_pulse in zip(channels, channel_pulses):
            if channel_pulse.close_shutter:
                try:
                    self._close_shutter(channel)
                except Exception as exc:
                    errors.append((f"channel {channel.channel_id.value} shutter close", exc))
                    logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
        for index, task in enumerate(digital_tasks):
            self._stop_and_close_task(f"pulse digital output task {index}", task, errors)
        # After the tasks that use them, and only this pulse train's own: the
        # trigger routes and the shared clock's route stay until close().
        errors.extend(self._release_routes(routes))
        if close_pmt:
            try:
                self._write_transient_digital_line(
                    self._require_pmt_shutter_output(),
                    False,
                    "laser_pmt_shutter_reset",
                )
            except Exception as exc:
                errors.append(("PMT shutter close", exc))
                logger.exception("Failed to close NI-DAQ PMT shutter output")
        self._raise_or_log_cleanup_errors("NI-DAQ laser pulse train", errors, run_error)

    def _cleanup_calibration_ramp(
        self,
        ao_task: object,
        ai_task: object,
        digital_tasks: List[object],
        channel: LaserChannelConfiguration,
        ramp: LaserCalibrationRamp,
        run_error: Optional[BaseException],
        routes: Sequence[Tuple[str, str]] = (),
        owner: str = "",
        unfinished: Optional[Dict[int, str]] = None,
    ) -> None:
        errors = []

        def stop_and_close(label, task):
            # A finite task stopped before it finished warns (200010); the
            # filter keeps that off stderr, so this is its record.
            name = (unfinished or {}).get(id(task))
            if name is not None:
                logger.debug(
                    "%s: stopping its task %s before it finished (DAQmx may warn 200010)",
                    owner, name)
            self._stop_and_close_task(label, task, errors)

        # Either task is None when creating it is what failed.
        if ao_task is not None:
            stop_and_close("calibration analog output task", ao_task)
        if ai_task is not None:
            stop_and_close("calibration analog input task", ai_task)
        for index, task in enumerate(digital_tasks):
            stop_and_close(f"calibration digital output task {index}", task)
        # After the tasks that use them, and only the ramp's own: the
        # controller's trigger routes, connected before the ramp, stay until
        # close().
        errors.extend(self._release_routes(routes))
        if self._closed:
            # close() is waiting for this, and resets the laser after it with
            # tasks of its own. The command is also written back here, with
            # this ramp's output task now closed: were close() to have gone
            # on without it, past its wait, its reset met that task on ao0
            # (-50103) and the 6713 held the last ramp sample.
            try:
                self._write_transient_analog_sample(channel, channel.minimum_command_volts)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} command reset", exc))
                self._log_command_left_driven(
                    channel, "a closed calibration ramp", _ramp_holding(ramp))
            if ramp.enable_pmt_shutter:
                try:
                    self._write_transient_digital_line(
                        self._require_pmt_shutter_output(),
                        False,
                        "laser_pmt_shutter_calibration_reset",
                    )
                except Exception as exc:
                    errors.append(("PMT shutter close", exc))
                    logger.exception("Failed to close NI-DAQ PMT shutter output after calibration")
            self._raise_or_log_cleanup_errors("NI-DAQ laser calibration ramp", errors, run_error)
            return
        try:
            self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
        except Exception as exc:
            errors.append((f"channel {channel.channel_id.value} command reset", exc))
            self._log_command_left_driven(
                channel, "its calibration ramp", _ramp_holding(ramp))
        if ramp.close_shutter:
            try:
                self.set_shutter_open(channel.channel_id, False)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} shutter close", exc))
                logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
        if ramp.enable_pmt_shutter:
            try:
                self._write_transient_digital_line(
                    self._require_pmt_shutter_output(),
                    False,
                    "laser_pmt_shutter_calibration_reset",
                )
            except Exception as exc:
                errors.append(("PMT shutter close", exc))
                logger.exception("Failed to close NI-DAQ PMT shutter output after calibration")
        self._raise_or_log_cleanup_errors("NI-DAQ laser calibration ramp", errors, run_error)

    def _log_command_left_driven(
        self, channel: LaserChannelConfiguration, after: str, holding: str,
    ) -> None:
        """CRITICAL, within the except: `channel`'s output may stay driven.

        A pulse's or a ramp's own reset of the command, refused (-50103, for
        one), leaves the output on its last level: after a cancel or a failure
        that is the pulse's high level. Whatever else ended the run, as every
        output left driven is (controller ruling, final review). `holding`
        names that level: the pulse's amplitude, or up to the ramp's highest
        command, as the application names it for a ramp whose close hung. The laser
        is noted for close(), which says so if it resets it after all.
        """
        with self._operation_lock:
            self._left_driven.add(channel.channel_id)
        logger.critical(
            "Laser %s: its command on %s could not be put back to %g V after "
            "%s. The output may still hold %s: make the laser safe by hand.",
            channel.channel_id.value, channel.analog_output,
            channel.minimum_command_volts, after, holding, exc_info=True)

    def _note_reset_after_all(self, channel: LaserChannelConfiguration) -> None:
        """WARNING: close() has reset a laser a CRITICAL said may be driven.

        The CRITICAL told the operator to make the laser safe by hand;
        nothing said that close()'s own reset of it then succeeded.
        """
        with self._operation_lock:
            if channel.channel_id not in self._left_driven:
                return
            self._left_driven.discard(channel.channel_id)
        logger.warning(
            "Laser %s: close() put its command on %s back to %g V after all; "
            "the output an earlier CRITICAL said may still be driven was reset.",
            channel.channel_id.value, channel.analog_output,
            channel.minimum_command_volts)

    def _stop_and_close_task(self, name: str, task: object, errors: list) -> None:
        try:
            self._retry_while_aborting(task.stop, name)
        except Exception as exc:
            errors.append((f"{name} stop", exc))
            logger.exception("Failed to stop %s", name)
        try:
            self._retry_while_aborting(task.close, name)
        except Exception as exc:
            errors.append((f"{name} close", exc))
            logger.exception("Failed to close %s", name)

    @staticmethod
    def _retry_while_aborting(call, name: str):
        """`call`, tried again while DAQmx says an abort is still finishing.

        A stop made while a task never started is being aborted raises
        -88710 (H8b), by its status code, not its text; for a close the same
        is assumed, not measured. Anything else, or past the bound, is raised
        as it was.
        """
        deadline = time.monotonic() + _ABORT_FINISHING_S
        while True:
            try:
                return call()
            except Exception as error:
                if (getattr(error, "error_code", None) != _ABORT_IN_PROGRESS
                        or time.monotonic() >= deadline):
                    raise
                logger.debug("%s: an abort is still finishing (-88710); again", name)
                time.sleep(0.005)

    def _raise_or_log_cleanup_errors(
        self,
        context: str,
        errors: list,
        run_error: Optional[BaseException],
    ) -> None:
        if not errors:
            return
        locations = ", ".join(location for location, _ in errors)
        cleanup_error = RuntimeError(f"Failed to clean up {context}: {locations}")
        if run_error is None:
            raise cleanup_error from errors[0][1]
        logger.error("Failed to clean up %s after output error: %s", context, locations)

    def close_all_shutters(self) -> None:
        errors = []
        for channel in self._configuration.channels:
            try:
                self.set_shutter_open(channel.channel_id, False)
            except Exception as exc:
                errors.append((channel.channel_id, exc))
                logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
        if errors:
            channels = ", ".join(str(channel_id.value) for channel_id, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ laser shutter(s) for channel(s): {channels}") from errors[0][1]

    def close(self) -> None:
        """Close the shutters, stop what runs, reset the laser, let go of it all.

        Raises when any of it failed. What it stopped waiting for, a pulse
        train or a ramp that had not ended, can still act after this returns:
        work_left_running() names it until it ends.
        """
        errors = []
        # Marked closed, and what it must stop taken, before any driver call:
        # a cancel, an abort, or a ramp, pulse train or route checking for
        # closed then never waits on DAQmx behind this.
        calibration_tasks, operations = self._mark_closed()
        # The shutters first, before anything is waited for: nothing else
        # stops the light while a pulse or a ramp is being stopped. The
        # laser model closed them itself before this, before the controller
        # was marked closed; a driver hung in that write held the close there.
        # Only the channels opened, as the reset below: a controller that
        # failed part-way through opening is closed too.
        for channel in self._configuration.channels:
            if channel.channel_id not in self._tasks:
                continue
            try:
                self.set_shutter_open(channel.channel_id, False)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} shutter close", exc))
                logger.exception(
                    "Failed to close NI-DAQ laser shutter for channel %s during close",
                    channel.channel_id.value)
        errors.extend(self._abort_calibration_ramp(calibration_tasks))
        errors.extend(self._wait_for_calibration_release())
        for operation in operations:
            try:
                operation.cancel()
            except Exception as exc:
                errors.append((f"laser operation {operation.operation_id} cancel", exc))
                logger.exception("Failed to cancel active NI-DAQ laser operation")
            # Only one that has not ended is the close's failure. One that
            # failed by itself, before or during this close, keeps its own
            # error, where its caller reads it, and its cleanup reset its
            # laser: raised here as the close's, it made Stop say "make the
            # laser safe by hand".
            if not operation.wait_until_finished(_OPERATION_CANCEL_TIMEOUT_S):
                errors.append((
                    f"laser operation {operation.operation_id} cancel",
                    TimeoutError(f"Laser operation {operation.operation_id} did not finish")))
                logger.error(
                    "Failed to cancel active NI-DAQ laser operation %s: it had not "
                    "ended %.1f s later", operation.operation_id, _OPERATION_CANCEL_TIMEOUT_S)
                with self._operation_lock:
                    self._given_up_operations.append(operation)
        for channel in self._configuration.channels:
            if channel.channel_id not in self._tasks:
                continue
            try:
                self.set_shutter_open(channel.channel_id, False)
                if channel.auxiliary_output is not None:
                    self.set_auxiliary_output(channel.channel_id, False)
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
                self._note_reset_after_all(channel)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} reset", exc))
                logger.exception("Failed to reset NI-DAQ laser channel %s during close", channel.channel_id.value)
        for channel_id, tasks in self._tasks.items():
            try:
                tasks.close()
            except Exception as exc:
                errors.append((f"channel {channel_id.value} task close", exc))
                logger.exception("Failed to close NI-DAQ laser tasks for channel %s", channel_id.value)
        self._tasks.clear()
        errors.extend(self._disconnect_trigger_routes())
        if errors:
            locations = ", ".join(location for location, _ in errors)
            raise RuntimeError(f"Failed to close NI-DAQ laser controller cleanly: {locations}") from errors[0][1]

    def work_left_running(self) -> Tuple[str, ...]:
        """What close() stopped waiting for and has not ended yet.

        A pulse train whose cancel the driver had not acted on, or a ramp
        that had not let go of its tasks, when close() went on without it.
        Either can still act later: release a clock route, write the PMT or
        command line, after a new controller has opened on the same lines.
        """
        with self._operation_lock:
            operations = tuple(self._given_up_operations)
            ramp = self._given_up_on_ramp
        left = [
            f"laser operation {operation.operation_id}"
            for operation in operations if not operation._done.is_set()
        ]
        if ramp and not self._calibration_released.is_set():
            left.append("the calibration ramp")
        return tuple(left)

    def wait_for_work_left_running(self, timeout: Optional[float] = None) -> bool:
        """Wait for work_left_running() to end, up to `timeout`; whether it has."""
        with self._operation_lock:
            events = [operation._done for operation in self._given_up_operations]
            if self._given_up_on_ramp:
                events.append(self._calibration_released)
        deadline = None if timeout is None else time.monotonic() + timeout
        for event in events:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not event.wait(remaining):
                return False
        return True

    def _mark_closed(self):
        """Mark the controller closed; the ramp tasks and operations to stop.

        Bookkeeping under the lock and nothing else. What registers after
        this is refused: a ramp or pulse train checks for closed under the
        same lock before it starts. A caller waiting for a route to settle
        is woken, to find it closed.
        """
        with self._operation_lock:
            self._closed = True
            self._routes_settled.notify_all()
            return (
                tuple(self._calibration_tasks),
                tuple(self._live_operations.values()),
            )

    def _abort_calibration_ramp(self, tasks) -> List[Tuple[str, Exception]]:
        """Stop a calibration ramp running on another thread, for close().

        reachAQ closing mid-ramp closes this controller from the close path
        while the ramp's thread may be blocked inside DAQmx, in Wait Until
        Done. The ramp's tasks hold the analog output, so close() could not
        write the command back to its minimum (-50103), and the 6713 kept its
        last sample. Aborting them is the DAQmx operation meant for another
        thread: it returns a task to before it started, which releases its
        lines, and makes a wait blocked on it return with an error. The ramp
        thread still stops and clears them in its own finally; nothing here
        clears a task another thread may be using, and close() waits for it
        to have done so before resetting the laser, as an abort that does not
        release them would otherwise leave the reset refused
        (_wait_for_calibration_release). `tasks` are those _mark_closed took,
        under the lock the ramp checks for closed under, so a ramp not yet
        started never starts.
        """
        if not tasks:
            return []
        errors = []
        for owner, name, task in tasks:
            # One the ramp's own cleanup has taken is its to stop and close;
            # the first abort can be what let the ramp go to that cleanup.
            if self._calibration_task_released(task):
                continue
            # Before the abort: the ramp, woken by it, can reach its cleanup
            # before the abort returns (H8c), and must find it aborted then.
            with self._operation_lock:
                self._aborted_calibration_tasks.add(id(task))
            try:
                self._abort_task(
                    task, owner, name, running=self._calibration_task_started(task))
            except Exception as exc:
                if self._calibration_task_released(task):
                    # Taken by that cleanup while this aborted it: the abort
                    # met a task closing or closed, which is not a failure.
                    logger.debug(
                        "A NI-DAQ laser calibration task was let go of by its "
                        "ramp as close() aborted it: %s", exc)
                    continue
                errors.append(("calibration task abort", exc))
                logger.exception("Failed to abort a NI-DAQ laser calibration task")
        return errors

    def _calibration_task_released(self, task) -> bool:
        """Whether the ramp's finally has let go of `task`; bookkeeping."""
        with self._operation_lock:
            return id(task) in self._released_calibration_tasks

    def _calibration_task_started(self, task) -> bool:
        """Whether the ramp has started `task`; bookkeeping."""
        with self._operation_lock:
            return id(task) in self._started_calibration_tasks

    def _wait_for_calibration_release(self) -> List[Tuple[str, Exception]]:
        """Wait, bounded, for an aborted ramp to let go of its tasks; for close().

        As the pulse path waits for its operation's owner. close() reset the
        laser straight after the abort, and the ramp's own cleanup skipped
        its reset once it found the controller closed: were the abort not to
        release ao0, the reset failed at -50103 and the 6713 held the last
        ramp sample. Past the bound close() goes on without the ramp, whose
        cleanup still writes the command back when it ends.
        """
        released = self._calibration_released
        if released.is_set():
            return []
        errors = []
        if not released.wait(_CALIBRATION_RELEASE_TIMEOUT_S):
            # Not "resetting the laser": if the ramp's output task still holds
            # ao0, the reset below is refused (-50103) and close() raises.
            # An error of close()'s, not a log line alone: a ramp still in
            # the driver, in a start that stalled, can drive the output after
            # the reset below (work_left_running).
            with self._operation_lock:
                self._given_up_on_ramp = True
            errors.append(("calibration ramp release", TimeoutError(
                f"the calibration ramp had not let go of its tasks "
                f"{_CALIBRATION_RELEASE_TIMEOUT_S:.1f} s after close() aborted it")))
            logger.error(
                "A NI-DAQ laser calibration ramp had not ended %.1f s after "
                "close() aborted it; close() goes on without it, and its "
                "command reset is refused if the ramp's tasks still hold the "
                "output",
                _CALIBRATION_RELEASE_TIMEOUT_S,
            )
        # Held high for the whole ramp when it was asked for; nothing else
        # in close() knows the line. After the wait, so that the ramp's own
        # task on it has been closed.
        pmt_line = self._configuration.pmt_shutter_output
        if pmt_line:
            try:
                self._write_transient_digital_line(pmt_line, False, "laser_pmt_shutter_close")
            except Exception as exc:
                errors.append(("PMT shutter close", exc))
                logger.exception("Failed to close the NI-DAQ PMT shutter output during close")
        return errors

    def _connect_trigger_route(self, channel: LaserChannelConfiguration) -> None:
        """Drive this channel's trigger terminal from where the pulse arrives.

        A trigger that starts on one board and arms an output on another has
        to cross the PXI backplane. DAQmx will do that itself only by
        reserving a trigger line, and it refuses to reserve one when the
        chassis is unidentified: "no registered trigger lines could be found
        between the devices in the route", -89125, which is where every
        cross-board arm on this rig stopped. Driving a PXI_Trig line by name
        asks for no reservation and works - measured carrying the pellet
        board's STIM3 pulse from the PXI-6221 to the PXI-6713, which then ran
        its waveform on time.

        The route is a property of the system rather than of a task, so it is
        held for as long as the controller is open and released in close().
        """
        source = channel.trigger_route_source
        if not source or not channel.trigger_source:
            return
        # The route is made on the source board, not across the two: naming
        # the far board's terminal as the destination is asking DAQmx for the
        # cross-device route it refuses. Each board addresses the same
        # backplane line by its own name, so PFI0 is driven onto this board's
        # PXI_Trig0 and the other board arms on its own PXI_Trig0.
        line = channel.trigger_source.rsplit("/", 1)[-1]
        destination = f"{source.rsplit('/', 1)[0]}/{line}"
        # Channels sharing one stimulus line share one route. Connecting it
        # per channel registered it twice and would have released it twice on
        # close, the second call against a route that no longer existed.
        if (source, destination) in self._trigger_routes:
            return
        try:
            self._nidaqmx.system.System.local().connect_terms(
                source, destination)
        except Exception as error:
            raise RuntimeError(
                f"laser {channel.channel_id.value} could not route "
                f"{source} to {destination}: {error}"
            ) from error
        self._trigger_routes.append((source, destination))
        log_hardware_initialization(
            logger,
            "READY | NI-DAQ laser trigger route | id=%s %s -> %s",
            channel.channel_id.value,
            source,
            destination,
        )

    def _shared_clock_for(
        self,
        output_device: str,
        source: str,
        *,
        line: str,
        added: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """A sample clock as this output board can see it, over backplane `line`.

        The input stream's clock and the ramp's go on backplane_clock_line, a
        pulse's own clock for its digital lines on pulse_clock_line; the
        caller names which, as there is no default to fall back on.

        A clock produced on one board reaches an output on another the same
        way its trigger does, and runs into the same refusal: DAQmx will not
        route across an unidentified chassis. Driving it onto a backplane line
        and naming that line locally needs no reservation. A clock already on
        the output's own board is returned untouched, so a single-board rig
        never acquires a route it does not need.

        A route this call connects is appended to `added`, for a caller that
        releases its own routes when it ends; one already held is reused, not
        connected twice. Refused once the controller is closed: close() may
        already have released every route, and one added after it would
        outlive the controller. Refused too when something else already drives
        the line (_backplane_line_holder): in a synchronized pulse train it
        carries the 6221's clock to the 6713 until close(), and a second
        driver would corrupt both signals without DAQmx seeing it.
        """
        clock_device = source.strip("/").split("/", 1)[0]
        if not output_device or clock_device == output_device:
            return source
        destination = f"/{clock_device}/{line}"
        local = f"/{output_device}/{line}"
        route = (source, destination)
        # Chosen and recorded under the lock, connected outside it. Pending
        # meanwhile, so that nobody else connects or releases it, and close(),
        # which releases only the routes held, leaves it to this call.
        with self._operation_lock:
            give_up_at = time.monotonic() + _ROUTE_PENDING_GIVE_UP_S
            while True:
                if self._closed:
                    raise RuntimeError(
                        "the laser controller is closed; no clock route is "
                        f"made for {output_device}")
                if route in self._trigger_routes:
                    return local
                pending = self._pending_routes.get(route)
                if pending is None:
                    break
                if time.monotonic() >= give_up_at:
                    raise RuntimeError(
                        f"could not put {source} on {line} for {output_device}: "
                        f"the route {source} -> {destination} was still being "
                        f"{'connected' if pending == 'connect' else 'released'} by "
                        f"another call {_ROUTE_PENDING_GIVE_UP_S:.1f} s later")
                # Another caller is connecting or releasing this same route.
                self._wait_for_routes()
            holder = self._backplane_line_holder(line, source)
            if holder is not None:
                raise RuntimeError(
                    f"could not put {source} on {line} for "
                    f"{output_device}: {holder[1]} already carries "
                    f"{holder[0]}. Two signals driven onto one backplane "
                    "line corrupt each other, and DAQmx does not see it "
                    "across these boards")
            self._pending_routes[route] = "connect"
        try:
            self._nidaqmx.system.System.local().connect_terms(source, destination)
        except Exception as error:
            with self._operation_lock:
                self._settle_route(route)
            raise RuntimeError(
                f"could not put the sample clock {source} on "
                f"{destination} for {output_device}: {error}"
            ) from error
        with self._operation_lock:
            self._settle_route(route)
            closed = self._closed
            if not closed:
                self._trigger_routes.append(route)
                if added is not None:
                    added.append(route)
        if closed:
            # close() came while the driver made it, and released only the
            # routes it held: this one is this call's to undo.
            error = self._disconnect_route(route)
            raise RuntimeError(
                "the laser controller was closed while a clock route was made "
                f"for {output_device}; the route was "
                + ("released" if error is None else f"not released ({error})"))
        log_hardware_initialization(
            logger,
            "READY | NI-DAQ laser clock route | %s -> %s, read as %s",
            source,
            destination,
            local,
        )
        return local

    def _settle_route(self, route) -> None:
        """No longer pending; under the lock. Wakes whoever waits on it."""
        self._pending_routes.pop(route, None)
        self._routes_settled.notify_all()

    def _wait_for_routes(self) -> None:
        """Wait, under the lock, for a pending route to settle."""
        self._routes_settled.wait(_ROUTE_SETTLE_WAIT_S)

    def _disconnect_route(self, route) -> Optional[Exception]:
        """Disconnect one route in the driver, outside the lock; its error, if any."""
        source, destination = route
        try:
            self._nidaqmx.system.System.local().disconnect_terms(source, destination)
        except Exception as error:
            logger.exception("Failed to release NI-DAQ laser trigger route %s -> %s",
                             source, destination)
            return error
        return None

    def _backplane_line_holder(self, line: str, source: str) -> Optional[Tuple[str, str]]:
        """What already drives backplane `line` from another source, or None.

        The line is compared, not the terminal: every board names the same
        bussed line as its own. This controller's routes count, trigger and
        clock alike, and so do the input stream's exports in the timing plan,
        which its worker drives from its own process.
        """
        wanted = line.lower()
        plan = self._timing_plan
        exports = () if plan is None else (
            (plan.sample_clock_source, plan.sample_clock_export_terminal),
            (plan.start_trigger_source, plan.start_trigger_export_terminal),
        )
        # Pending ones too: being connected, or still being released.
        for held_source, held_destination in (
                *self._trigger_routes, *self._pending_routes, *exports):
            if not held_destination or held_source == source:
                continue
            if held_destination.rsplit("/", 1)[-1].lower() == wanted:
                return held_source, held_destination
        return None

    def _disconnect_trigger_routes(self) -> List[Tuple[str, Exception]]:
        with self._operation_lock:
            routes = tuple(self._trigger_routes)
        return self._release_routes(routes)

    def _release_routes(self, routes) -> List[Tuple[str, Exception]]:
        """Disconnect these routes, those this controller still holds.

        close() releases every route; an operation that connected its own,
        such as a calibration ramp's clock, releases just those when it ends,
        by name. A route close() has released already is not released again:
        each is taken off the held list under the lock by whichever caller
        gets it first, and disconnected by that caller, outside the lock,
        pending until it is gone so that nobody connects it meanwhile, and
        the collision check still counts it. One the driver would not
        disconnect may still be driven: it goes back on the held list, where
        the collision check keeps finding it and close() tries it again.
        """
        with self._operation_lock:
            owned = []
            for route in routes:
                if route in self._trigger_routes and route not in owned:
                    self._trigger_routes.remove(route)
                    self._pending_routes[route] = "release"
                    owned.append(route)
        errors = []
        for route in owned:
            error = self._disconnect_route(route)
            with self._operation_lock:
                self._settle_route(route)
                if error is not None and route not in self._trigger_routes:
                    self._trigger_routes.append(route)
            if error is not None:
                source, destination = route
                errors.append((f"trigger route {source} -> {destination}", error))
        return errors

    def _create_channel_tasks(self, channel: LaserChannelConfiguration) -> _NidaqLaserTasks:
        analog_output = None
        if not self._configuration.hardware_timed:
            analog_output = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_ao")
        diode_input = None
        if self._feedback_reader is None:
            diode_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_ai")
            self._add_voltage_input(diode_input, channel.diode_input)

        command_copy_input = None
        if channel.command_copy_input and self._feedback_reader is None:
            command_copy_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_command_copy_ai")
            self._add_voltage_input(command_copy_input, channel.command_copy_input)

        shutter_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_shutter")
        shutter_output.do_channels.add_do_chan(channel.shutter_output)

        auxiliary_output = None
        if channel.auxiliary_output is not None:
            auxiliary_output = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_aux")
            auxiliary_output.do_channels.add_do_chan(channel.auxiliary_output)

        return _NidaqLaserTasks(
            analog_output=analog_output,
            diode_input=diode_input,
            command_copy_input=command_copy_input,
            shutter_output=shutter_output,
            auxiliary_output=auxiliary_output,
        )

    def _add_voltage_input(self, task, physical_channel: str) -> None:
        """An input of this controller's, referenced as the stream's are.

        With no terminal configuration given, DAQmx picks per channel, which
        on christielab10's 6221 made ai3, ai4 and ai5 differential, paired
        with ai11-ai13, where the stream reads them single-ended (hardware
        check, phase 2 final).
        """
        terminal_config = self._analog_terminal_config
        if terminal_config is None:
            task.ai_channels.add_ai_voltage_chan(physical_channel)
        else:
            task.ai_channels.add_ai_voltage_chan(
                physical_channel, terminal_config=terminal_config)

    def _create_analog_output_task(self, channel: LaserChannelConfiguration, name: str):
        analog_output = self._nidaqmx.Task(name)
        analog_output.ao_channels.add_ao_voltage_chan(
            channel.analog_output,
            min_val=channel.minimum_command_volts,
            max_val=channel.maximum_command_volts,
        )
        return analog_output

    def _create_synchronized_analog_output_task(self, channels: List[LaserChannelConfiguration], name: str):
        analog_output = self._nidaqmx.Task(name)
        for channel in channels:
            analog_output.ao_channels.add_ao_voltage_chan(
                channel.analog_output,
                min_val=channel.minimum_command_volts,
                max_val=channel.maximum_command_volts,
            )
        return analog_output

    def _create_calibration_input_task(self, channel: LaserChannelConfiguration, name: str):
        analog_input = self._nidaqmx.Task(name)
        try:
            self._add_voltage_input(analog_input, channel.diode_input)
            if channel.command_copy_input is not None:
                self._add_voltage_input(analog_input, channel.command_copy_input)
        except Exception:
            # Not yet the ramp's to close: it never had it.
            analog_input.close()
            raise
        return analog_input

    def _create_finite_digital_output_task(
        self,
        physical_line: str,
        name: str,
        waveform: List[bool],
        sample_rate_hz: float,
        total_samples: int,
        sample_clock_source: str,
        trigger_source: Optional[str],
        trigger_edge: str,
    ):
        if len(waveform) != total_samples:
            raise RuntimeError(
                f"digital output waveform for {physical_line} has {len(waveform)} samples, "
                f"expected {total_samples}"
            )
        digital_output = self._nidaqmx.Task(name)
        try:
            digital_output.do_channels.add_do_chan(physical_line)
            digital_output.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                source=sample_clock_source,
                sample_mode=self._nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=total_samples,
            )
            if trigger_source:
                digital_output.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source,
                    trigger_edge=self._get_trigger_edge(trigger_edge),
                )
            digital_output.write(waveform, auto_start=False)
        except Exception:
            digital_output.close()
            raise
        return digital_output

    def _write_transient_analog_sample(self, channel: LaserChannelConfiguration, volts: float) -> None:
        task = self._create_analog_output_task(channel, f"laser_{channel.channel_id.value}_manual_ao")
        try:
            task.write(volts, auto_start=True)
        finally:
            task.close()

    def _write_transient_digital_line(self, physical_line: str, enabled: bool, name: str) -> None:
        task = self._nidaqmx.Task(name)
        try:
            task.do_channels.add_do_chan(physical_line)
            task.write(bool(enabled), auto_start=True)
        finally:
            task.close()

    def _read_optional_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> Optional[float]:
        channel = self._configuration.get_channel(channel_id)
        if channel.command_copy_input is None:
            return None
        if self._feedback_reader is not None:
            return float(self._feedback_reader(channel.command_copy_input))
        task = self._tasks[channel.channel_id].command_copy_input
        if task is None:
            return None
        return float(task.read()) * channel.command_copy_scale

    def _build_pulse_train_waveform(
        self,
        channel: LaserChannelConfiguration,
        pulse_train: LaserPulseTrain,
        sample_rate_hz: Optional[float] = None,
    ) -> list:
        # Taken from the caller when it knows better. Fetching it here made
        # the caller's choice moot: it resolved the shared clock's rate, and
        # the waveform was still laid out for the configured one.
        if sample_rate_hz is None:
            sample_rate_hz = self._require_sample_rate()
        baseline_samples = _samples_from_ms(pulse_train.baseline_ms, sample_rate_hz)
        high_samples = max(1, _samples_from_ms(pulse_train.duration_ms, sample_rate_hz))
        post_stim_samples = _samples_from_ms(pulse_train.post_stim_ms, sample_rate_hz)
        minimum = channel.minimum_command_volts
        amplitude = pulse_train.amplitude_volts
        waveform = [minimum] * baseline_samples
        if pulse_train.pulse_count == 1:
            waveform.extend([amplitude] * high_samples)
        else:
            period_samples = max(1, int(round(sample_rate_hz / pulse_train.frequency_hz)))
            if high_samples > period_samples:
                raise ValueError(
                    f"laser pulse duration {pulse_train.duration_ms} ms exceeds pulse period "
                    f"at {pulse_train.frequency_hz} Hz"
                )
            low_samples = period_samples - high_samples
            for pulse_index in range(pulse_train.pulse_count):
                waveform.extend([amplitude] * high_samples)
                if pulse_index < pulse_train.pulse_count - 1:
                    waveform.extend([minimum] * low_samples)
        waveform.extend([minimum] * post_stim_samples)
        if not waveform:
            raise ValueError("laser pulse train waveform is empty")
        return waveform

    def _build_digital_pulse_waveform(
        self,
        total_samples: int,
        pulse_ms: float,
        sample_rate_hz: float,
    ) -> List[bool]:
        pulse_samples = min(total_samples, max(1, _samples_from_ms(pulse_ms, sample_rate_hz)))
        return [True] * pulse_samples + [False] * (total_samples - pulse_samples)

    def _build_calibration_ramp_waveform(self, ramp: LaserCalibrationRamp) -> List[float]:
        waveform = []
        for index in range(ramp.steps):
            fraction = index / (ramp.steps - 1)
            command_volts = ramp.start_volts + fraction * (ramp.stop_volts - ramp.start_volts)
            waveform.extend([command_volts] * ramp.samples_per_step)
        if not waveform:
            raise ValueError("laser calibration ramp waveform is empty")
        return waveform

    def _build_calibration_points(
        self,
        channel: LaserChannelConfiguration,
        ramp: LaserCalibrationRamp,
        raw_samples,
        settle_samples: int,
    ) -> Tuple[LaserCalibrationPoint, ...]:
        channel_count = 2 if channel.command_copy_input is not None else 1
        samples = self._normalize_ai_samples(raw_samples, channel_count)
        points = []
        for index in range(ramp.steps):
            # After the step has settled: its first samples still read the
            # step before (LaserCalibrationRamp.settle_seconds).
            start = index * ramp.samples_per_step + settle_samples
            stop = (index + 1) * ramp.samples_per_step
            fraction = index / (ramp.steps - 1)
            command_volts = ramp.start_volts + fraction * (ramp.stop_volts - ramp.start_volts)
            diode_volts = _mean(samples[0][start:stop]) * channel.feedback_scale
            command_copy_volts = None
            if channel.command_copy_input is not None:
                command_copy_volts = _mean(samples[1][start:stop]) * channel.command_copy_scale
            points.append(
                LaserCalibrationPoint(
                    channel_id=channel.channel_id,
                    command_volts=command_volts,
                    diode_volts=diode_volts,
                    command_copy_volts=command_copy_volts,
                )
            )
        return tuple(points)

    def _normalize_ai_samples(self, raw_samples, channel_count: int) -> List[List[float]]:
        raw_samples = list(raw_samples)
        if channel_count == 1:
            if raw_samples and not isinstance(raw_samples[0], numbers.Number):
                return [list(raw_samples[0])]
            return [list(raw_samples)]
        if len(raw_samples) != channel_count:
            raise RuntimeError(f"expected {channel_count} analog input channels, received {len(raw_samples)}")
        return [list(channel_samples) for channel_samples in raw_samples]

    def _require_pmt_shutter_output(self) -> str:
        pmt_shutter_output = self._configuration.pmt_shutter_output
        if pmt_shutter_output is None:
            raise RuntimeError("PMT shutter output requested, but laser pmt_shutter_output is not configured")
        return pmt_shutter_output

    def _analog_output_sample_clock_source(self, physical_channel: str) -> str:
        parts = physical_channel.strip("/").split("/")
        if len(parts) < 2 or not parts[0]:
            raise RuntimeError(f"cannot infer NI-DAQ AO sample clock source from physical channel {physical_channel}")
        return f"/{parts[0]}/ao/SampleClock"

    def _require_sample_rate(self, timing_kwargs=None) -> float:
        """The rate the waveform will actually be generated at.

        A task clocked from a shared terminal ticks at that terminal's rate,
        not the one the laser was configured with, and nothing objects: the
        rate passed to cfg_samp_clk_timing beside an external source only
        sizes the buffer. Building for the wrong one silently stretches every
        pulse width and interval - built at 100 kHz and clocked at the
        stream's 10 kHz, a two-second train took twenty seconds and came out
        at 2 Hz rather than 20.
        """
        if timing_kwargs and timing_kwargs.get("source"):
            plan = self._timing_plan
            shared = None if plan is None else plan.sample_clock_rate_hz
            if shared:
                return shared
            raise RuntimeError(
                "hardware-synchronized laser output is clocked by "
                f"{timing_kwargs['source']} but the timing plan does not say "
                "at what rate, so the waveform cannot be built for it"
            )
        sample_rate_hz = self._configuration.sample_rate_hz
        if sample_rate_hz is None:
            raise RuntimeError("hardware-timed laser output requires sample_rate_hz")
        return sample_rate_hz

    def _get_trigger_edge(self, trigger_edge: str):
        if trigger_edge == "rising":
            return self._nidaqmx.constants.Edge.RISING
        if trigger_edge == "falling":
            return self._nidaqmx.constants.Edge.FALLING
        raise ValueError("trigger_edge must be 'rising' or 'falling'")


def _samples_from_ms(value_ms: float, sample_rate_hz: float) -> int:
    if value_ms <= 0:
        return 0
    return max(1, int(round(value_ms * sample_rate_hz / 1000.0)))


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise RuntimeError("cannot average an empty calibration sample segment")
    return float(sum(values)) / len(values)


def _ramp_holding(ramp: LaserCalibrationRamp) -> str:
    """What a ramp's output may hold when its reset is refused.

    Up to its highest command, whichever end that is: stopped part-way, the
    output holds a level between the two, and a falling ramp's stop, named
    alone, said 0 V where 5 V may be left.
    """
    return (f"up to {max(ramp.start_volts, ramp.stop_volts):g} V, the ramp's "
            f"highest command (a {ramp.start_volts:g} V to {ramp.stop_volts:g} V ramp)")


def _device_of(physical_channel: str) -> str:
    """The device a channel belongs to: PXI1Slot5 for PXI1Slot5/ai8."""
    return physical_channel.strip("/").split("/", 1)[0]


def _ignore_the_aborts_warning(nidaqmx) -> None:
    """Keep the warning an abort of a running task gives off stderr.

    One filter, by DaqWarning and its text, 200010 alone: every other DAQmx
    warning is still shown. Set each time the driver is loaded for a
    controller: a warnings.catch_warnings that ends puts back the filters it
    found, which pytest does after each test, and filterwarnings takes out
    the one already there and puts it first again, ahead of any catch-all
    set since. Filtering by recording them, around each abort, swapped the
    process's warnings state from the aborting thread for its duration.
    """
    category = getattr(getattr(nidaqmx, "errors", None), "DaqWarning", None)
    if not (isinstance(category, type) and issubclass(category, Warning)):
        return
    warnings.filterwarnings(
        "ignore", message=_ABORT_WARNING_PATTERN, category=category)


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NidaqLaserController. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
