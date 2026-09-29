from __future__ import annotations

import contextlib
import dataclasses
import enum
import logging
import numbers
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from autotrainer.core import NidaqTimingPlan
from autotrainer.core.logging import log_hardware_initialization

from .laser import (
    LaserCalibrationPoint,
    LaserCalibrationRamp,
    LaserChannelConfiguration,
    LaserChannelId,
    LaserFeedbackSample,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    LaserSystemConfiguration,
    normalize_laser_channel_id,
)


from autotrainer.device.nidaq_reference_clock import (
    apply_reference_clock,
)

logger = logging.getLogger(__name__)

#: How long close() waits for an aborted calibration ramp to close its own
#: tasks before resetting the laser; the pulse path waits as long for its
#: operation's owner.
_CALIBRATION_RELEASE_TIMEOUT_S = 5.0
#: How long close() waits for each pulse train it cancels to end.
_OPERATION_CANCEL_TIMEOUT_S = 5.0
#: How long a controller that failed to open waits for its own close, of
#: what it had opened, before it raises; as long as the application waits
#: for any laser close.
_FAILED_OPEN_CLOSE_TIMEOUT_S = 15.0
#: How long a caller waiting for another's route to settle sleeps between
#: looks; a settle, or close(), wakes it at once.
_ROUTE_SETTLE_WAIT_S = 1.0


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

    def __init__(self, *, resources, context=None, terminal_callback=None):
        self.operation_id = str(uuid.uuid4())
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
        self._tasks = ()
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
        with self._lock:
            if self._state in self.TERMINAL:
                return False
            self._transition_locked(LaserOperationState.CANCELLED, "cancel requested")
            tasks = self._tasks
            self._start_requested.set()
        for task in tasks:
            try:
                task.stop()
            except Exception:
                logger.debug("Laser task stop during cancellation failed", exc_info=True)
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
        with self._lock:
            self._tasks = tuple(task for task in tasks if task is not None)

    def _mark_armed(self):
        with self._lock:
            if self._state is LaserOperationState.PREPARED:
                self._transition_locked(LaserOperationState.ARMED)
                self._armed.set()

    def _require_not_cancelled(self):
        if self.state is LaserOperationState.CANCELLED:
            raise RuntimeError("Laser operation was cancelled before arming")

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
    ):
        if configuration.backend != "nidaq":
            raise ValueError("NidaqLaserController requires laser backend 'nidaq'")
        runtime_started = time.perf_counter()
        log_hardware_initialization(logger, "START | NI-DAQmx runtime | consumer=laser")
        self._nidaqmx = _load_nidaqmx()
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
        #: destination, released in close().
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
        #: from another thread; see run_calibration_ramp.
        self._calibration_tasks: List[object] = []
        #: Those the ramp's finally has let go of, by id, under the lock:
        #: close()'s abort leaves them to its cleanup.
        self._released_calibration_tasks: set = set()
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
                error.still_closing = still_closing
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
            raise RuntimeError("Hardware-timed laser pulse trains require laser configuration hardware_timed=True")
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
            if getattr(self, "_closed", False):
                raise RuntimeError(
                    "the laser controller is closed; no pulse train is started")
            conflicts = [
                operation
                for operation in self._live_operations.values()
                if set(operation.resources) & set(resources)
                and operation.state not in operation.TERMINAL
            ]
            if conflicts:
                # Waited for or not, a train holds its output until it ends;
                # DAQmx would refuse a second task on it (-50103), after this
                # one had made its own. Run Pulse, which waits, is refused so
                # while a trial's pulse is armed.
                shared = sorted({
                    resource for operation in conflicts
                    for resource in operation.resources if resource in resources})
                raise RuntimeError(
                    f"Laser output {', '.join(shared)} is in use by laser "
                    f"operation {', '.join(op.operation_id for op in conflicts)}, "
                    "armed or running; another pulse on it is refused until "
                    "that operation ends or is cancelled"
                )
            operation = NidaqLaserOperation(
                resources=resources,
                context=pulse_train.operation_context,
                terminal_callback=self._release_operation,
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
                # is refused.
                if operation.state is LaserOperationState.CANCELLED:
                    operation._finish_terminal()
                elif ended_by is None:
                    operation._complete()
                else:
                    operation._fail(ended_by)

        if pulse_train.wait:
            # On the caller's thread, which it still blocks.
            operation._thread = threading.current_thread()
            execute()
            if operation.state is LaserOperationState.CANCELLED:
                # Not the DAQmx error the cancel's stop provoked.
                raise RuntimeError(
                    f"Laser operation {operation.operation_id} was cancelled: "
                    "the laser controller was closed while it ran")
            if operation.error is not None:
                raise operation.error
            return None
        operation._thread = threading.Thread(
            target=execute,
            name=f"NidaqLaser-{operation.operation_id[:8]}",
            daemon=True,
        )
        operation._thread.start()
        try:
            operation.wait_until_armed(timeout=5.0)
        except Exception:
            operation.cancel()
            raise
        return operation

    def _release_operation(self, operation):
        with self._operation_lock:
            self._live_operations.pop(operation.operation_id, None)

    def _execute_synchronized_pulse_train(
        self,
        pulse_train: LaserSynchronizedPulseTrain,
        *,
        operation: Optional[NidaqLaserOperation] = None,
    ) -> None:
        channels = [
            self._configuration.get_channel(channel_pulse.channel_id)
            for channel_pulse in pulse_train.pulse_trains
        ]
        for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
            self._validate_command_voltage(channel, channel_pulse.amplitude_volts)
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
        ao_task = self._create_synchronized_analog_output_task(channels, "laser_sync_pulse_ao")
        digital_tasks = []
        # The clock routes this pulse train adds for its digital outputs, by
        # name, which it releases when it ends, as the calibration ramp does.
        added_routes: List[Tuple[str, str]] = []
        run_error = None
        try:
            if operation is not None:
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
                digital_tasks.append(
                    self._create_finite_digital_output_task(
                        pmt_line,
                        "laser_pmt_shutter_do",
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
                    digital_tasks.append(
                        self._create_finite_digital_output_task(
                            channel.trigger_output,
                            f"laser_{channel.channel_id.value}_trigger_do",
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
                    digital_tasks.append(
                        self._create_finite_digital_output_task(
                            channel.timing_trigger_output,
                            f"laser_{channel.channel_id.value}_timing_trigger_do",
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
            for channel, channel_pulse in zip(channels, pulse_train.pulse_trains):
                if channel_pulse.open_shutter:
                    self.set_shutter_open(channel.channel_id, True)
            if operation is not None:
                operation._bind_tasks((ao_task, *digital_tasks))
                operation._require_not_cancelled()
            if operation is not None and pulse_train.defer_start:
                operation._mark_armed()
                if not operation._start_requested.wait(timeout_seconds):
                    raise TimeoutError("Deferred laser operation did not receive a start request")
                operation._require_not_cancelled()
            for task in digital_tasks:
                task.start()
            ao_task.start()
            if operation is not None:
                if pulse_train.defer_start:
                    operation._mark_triggered("NI software start accepted")
                else:
                    operation._mark_armed()
            self._last_timing_status = timing_status
            ao_task.wait_until_done(timeout=timeout_seconds)
            for task in digital_tasks:
                task.wait_until_done(timeout=timeout_seconds)
            if operation is not None:
                operation._mark_triggered()
        except Exception as exc:
            run_error = exc
            raise
        finally:
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
        waveform = self._build_calibration_ramp_waveform(ramp)
        timeout_seconds = ramp.timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = len(waveform) / sample_rate_hz + 5.0
        ao_task = None
        ai_task = None
        digital_tasks = []
        run_error = None
        points = ()
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
        try:
            # Both inside the cleanup scope: created before it, an input task
            # that failed to create left the output task open on ao0.
            ao_task = self._create_analog_output_task(
                channel, f"laser_{channel.channel_id.value}_calibration_ao")
            # Where close() can find them: reachAQ closing mid-ramp closes
            # this controller from its own thread, and this one may then be
            # inside DAQmx, waiting on these.
            self._hold_calibration_tasks(ao_task)
            ai_task = self._create_calibration_input_task(channel)
            self._hold_calibration_tasks(ai_task)
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
                digital_tasks.append(
                    self._create_finite_digital_output_task(
                        pmt_line,
                        "laser_pmt_shutter_calibration_do",
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
                self._hold_calibration_tasks(digital_tasks[-1])
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
            for task in digital_tasks:
                task.start()
            ai_task.start()
            ao_task.start()
            with self._operation_lock:
                if self._closed:
                    raise RuntimeError(
                        "the laser controller was closed as the calibration "
                        "ramp started")
            ao_task.wait_until_done(timeout=timeout_seconds)
            ai_task.wait_until_done(timeout=timeout_seconds)
            raw_samples = ai_task.read(number_of_samples_per_channel=len(waveform), timeout=timeout_seconds)
            points = self._build_calibration_points(channel, ramp, raw_samples)
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
                        id(task) for task in self._calibration_tasks)
                    self._calibration_tasks.clear()
                self._cleanup_calibration_ramp(
                    ao_task=ao_task,
                    ai_task=ai_task,
                    digital_tasks=digital_tasks,
                    channel=channel,
                    ramp=ramp,
                    run_error=run_error,
                    routes=added_routes,
                )
            finally:
                # After the tasks are stopped and closed, and after the
                # cleanup's own writes: close() resets the laser only then, so
                # its writes never meet a task of this ramp's on the lines.
                self._calibration_released.set()
        return points

    def _hold_calibration_tasks(self, *tasks) -> None:
        """Keep a ramp's tasks where close() can abort them, unless closed."""
        with self._operation_lock:
            if self._closed:
                raise RuntimeError(
                    "the laser controller was closed before the calibration "
                    "ramp started")
            self._calibration_tasks.extend(tasks)

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
        for index, task in enumerate(digital_tasks):
            self._stop_and_close_task(f"pulse digital output task {index}", task, errors)
        # After the tasks that use them, and only this pulse train's own: the
        # trigger routes and the shared clock's route stay until close().
        errors.extend(self._release_routes(routes))
        for channel in channels:
            try:
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} command reset", exc))
                logger.exception("Failed to reset NI-DAQ laser command for channel %s", channel.channel_id.value)
        for channel, channel_pulse in zip(channels, channel_pulses):
            if channel_pulse.close_shutter:
                try:
                    self.set_shutter_open(channel.channel_id, False)
                except Exception as exc:
                    errors.append((f"channel {channel.channel_id.value} shutter close", exc))
                    logger.exception("Failed to close NI-DAQ laser shutter for channel %s", channel.channel_id.value)
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
    ) -> None:
        errors = []
        # Either task is None when creating it is what failed.
        if ao_task is not None:
            self._stop_and_close_task("calibration analog output task", ao_task, errors)
        if ai_task is not None:
            self._stop_and_close_task("calibration analog input task", ai_task, errors)
        for index, task in enumerate(digital_tasks):
            self._stop_and_close_task(f"calibration digital output task {index}", task, errors)
        # After the tasks that use them, and only the ramp's own: the
        # controller's trigger routes, connected before the ramp, stay until
        # close().
        errors.extend(self._release_routes(routes))
        if getattr(self, "_closed", False):
            # close() is waiting for this, and resets the laser after it with
            # tasks of its own. The command is also written back here, with
            # this ramp's output task now closed: were close() to have gone
            # on without it, past its wait, its reset met that task on ao0
            # (-50103) and the 6713 held the last ramp sample.
            try:
                self._write_transient_analog_sample(channel, channel.minimum_command_volts)
            except Exception as exc:
                errors.append((f"channel {channel.channel_id.value} command reset", exc))
                logger.exception(
                    "Failed to reset NI-DAQ laser command for channel %s after "
                    "a closed calibration ramp", channel.channel_id.value)
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
            logger.exception("Failed to reset NI-DAQ laser command for channel %s", channel.channel_id.value)
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

    def _stop_and_close_task(self, name: str, task: object, errors: list) -> None:
        try:
            task.stop()
        except Exception as exc:
            errors.append((f"{name} stop", exc))
            logger.exception("Failed to stop %s", name)
        try:
            task.close()
        except Exception as exc:
            errors.append((f"{name} close", exc))
            logger.exception("Failed to close %s", name)

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
                operation.wait(timeout=_OPERATION_CANCEL_TIMEOUT_S)
            except Exception as exc:
                errors.append((f"laser operation {operation.operation_id} cancel", exc))
                logger.exception("Failed to cancel active NI-DAQ laser operation")
            if not operation._done.is_set():
                with self._route_lock():
                    self._given_up_operations.append(operation)
        for channel in self._configuration.channels:
            if channel.channel_id not in self._tasks:
                continue
            try:
                self.set_shutter_open(channel.channel_id, False)
                if channel.auxiliary_output is not None:
                    self.set_auxiliary_output(channel.channel_id, False)
                self.set_command_voltage(channel.channel_id, channel.minimum_command_volts)
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
        with self._route_lock():
            operations = tuple(getattr(self, "_given_up_operations", ()))
            ramp = getattr(self, "_given_up_on_ramp", False)
        left = [
            f"laser operation {operation.operation_id}"
            for operation in operations if not operation._done.is_set()
        ]
        if ramp and not self._calibration_released.is_set():
            left.append("the calibration ramp")
        return tuple(left)

    def wait_for_work_left_running(self, timeout: Optional[float] = None) -> bool:
        """Wait for work_left_running() to end, up to `timeout`; whether it has."""
        with self._route_lock():
            events = [operation._done for operation in getattr(self, "_given_up_operations", ())]
            if getattr(self, "_given_up_on_ramp", False):
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
        with self._route_lock():
            self._closed = True
            settled = getattr(self, "_routes_settled", None)
            if settled is not None:
                settled.notify_all()
            return (
                tuple(getattr(self, "_calibration_tasks", ())),
                tuple(getattr(self, "_live_operations", {}).values()),
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
        for task in tasks:
            # One the ramp's own cleanup has taken is its to stop and close;
            # the first abort can be what let the ramp go to that cleanup.
            if self._calibration_task_released(task):
                continue
            try:
                task.control(self._nidaqmx.constants.TaskMode.TASK_ABORT)
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
        with self._route_lock():
            return id(task) in getattr(self, "_released_calibration_tasks", ())

    def _wait_for_calibration_release(self) -> List[Tuple[str, Exception]]:
        """Wait, bounded, for an aborted ramp to let go of its tasks; for close().

        As the pulse path waits for its operation's owner. close() reset the
        laser straight after the abort, and the ramp's own cleanup skipped
        its reset once it found the controller closed: were the abort not to
        release ao0, the reset failed at -50103 and the 6713 held the last
        ramp sample. Past the bound close() goes on without the ramp, whose
        cleanup still writes the command back when it ends.
        """
        released = getattr(self, "_calibration_released", None)
        if released is None or released.is_set():
            return []
        errors = []
        if not released.wait(_CALIBRATION_RELEASE_TIMEOUT_S):
            # Not "resetting the laser": if the ramp's output task still holds
            # ao0, the reset below is refused (-50103) and close() raises.
            # An error of close()'s, not a log line alone: a ramp still in
            # the driver, in a start that stalled, can drive the output after
            # the reset below (work_left_running).
            with self._route_lock():
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
        with self._route_lock():
            while True:
                if getattr(self, "_closed", False):
                    raise RuntimeError(
                        "the laser controller is closed; no clock route is "
                        f"made for {output_device}")
                if route in self._trigger_routes:
                    return local
                if route not in self._route_pending():
                    break
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
            self._route_pending()[route] = "connect"
        try:
            self._nidaqmx.system.System.local().connect_terms(source, destination)
        except Exception as error:
            with self._route_lock():
                self._settle_route(route)
            raise RuntimeError(
                f"could not put the sample clock {source} on "
                f"{destination} for {output_device}: {error}"
            ) from error
        with self._route_lock():
            self._settle_route(route)
            closed = getattr(self, "_closed", False)
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

    def _route_pending(self) -> Dict[Tuple[str, str], str]:
        """Routes a call is connecting or releasing outside the lock."""
        pending = getattr(self, "_pending_routes", None)
        if pending is None:
            # A controller built without __init__, as some tests build one.
            pending = self._pending_routes = {}
        return pending

    def _settle_route(self, route) -> None:
        """No longer pending; under the lock. Wakes whoever waits on it."""
        self._route_pending().pop(route, None)
        settled = getattr(self, "_routes_settled", None)
        if settled is not None:
            settled.notify_all()

    def _wait_for_routes(self) -> None:
        """Wait, under the lock, for a pending route to settle."""
        settled = getattr(self, "_routes_settled", None)
        if settled is None:
            raise RuntimeError("a route is pending on a controller with no lock")
        settled.wait(_ROUTE_SETTLE_WAIT_S)

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
        plan = getattr(self, "_timing_plan", None)
        exports = () if plan is None else (
            (plan.sample_clock_source, plan.sample_clock_export_terminal),
            (plan.start_trigger_source, plan.start_trigger_export_terminal),
        )
        # Pending ones too: being connected, or still being released.
        for held_source, held_destination in (
                *self._trigger_routes, *self._route_pending(), *exports):
            if not held_destination or held_source == source:
                continue
            if held_destination.rsplit("/", 1)[-1].lower() == wanted:
                return held_source, held_destination
        return None

    def _route_lock(self):
        """_operation_lock, which every change to _trigger_routes is made under.

        A ramp releases its own routes from its thread while close() may be
        releasing all of them from another. A controller built without
        __init__, as some tests build one, has no lock and one thread.
        """
        lock = getattr(self, "_operation_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def _disconnect_trigger_routes(self) -> List[Tuple[str, Exception]]:
        with self._route_lock():
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
        with self._route_lock():
            owned = []
            for route in routes:
                if route in self._trigger_routes and route not in owned:
                    self._trigger_routes.remove(route)
                    self._route_pending()[route] = "release"
                    owned.append(route)
        errors = []
        for route in owned:
            error = self._disconnect_route(route)
            with self._route_lock():
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
            diode_input.ai_channels.add_ai_voltage_chan(channel.diode_input)

        command_copy_input = None
        if channel.command_copy_input and self._feedback_reader is None:
            command_copy_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_command_copy_ai")
            command_copy_input.ai_channels.add_ai_voltage_chan(channel.command_copy_input)

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

    def _create_calibration_input_task(self, channel: LaserChannelConfiguration):
        analog_input = self._nidaqmx.Task(f"laser_{channel.channel_id.value}_calibration_ai")
        try:
            analog_input.ai_channels.add_ai_voltage_chan(channel.diode_input)
            if channel.command_copy_input is not None:
                analog_input.ai_channels.add_ai_voltage_chan(channel.command_copy_input)
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
    ) -> Tuple[LaserCalibrationPoint, ...]:
        channel_count = 2 if channel.command_copy_input is not None else 1
        samples = self._normalize_ai_samples(raw_samples, channel_count)
        points = []
        for index in range(ramp.steps):
            # After the step has settled: its first samples still read the
            # step before (LaserCalibrationRamp.settle_samples).
            start = index * ramp.samples_per_step + ramp.settle_samples
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


def _device_of(physical_channel: str) -> str:
    """The device a channel belongs to: PXI1Slot5 for PXI1Slot5/ai8."""
    return physical_channel.strip("/").split("/", 1)[0]


def _load_nidaqmx():
    try:
        import nidaqmx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nidaqmx is required for NidaqLaserController. Install NI-DAQmx and the nidaqmx Python package "
            "on the hardware runtime machine."
        ) from exc
    return nidaqmx
