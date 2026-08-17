from __future__ import annotations

import dataclasses
import json
import logging
import math
import queue
import threading
import time
from typing import Callable, Optional, Tuple, Union

from autotrainer.core import NidaqTimingPlan, ObservableObject
from autotrainer.core.logging import get_verbose_logger, log_hardware_initialization
from autotrainer.device import (
    LaserControllerProtocol,
    LaserCalibrationPoint,
    LaserCalibrationRamp,
    LaserChannelId,
    LaserDiodePowerCurve,
    LaserFeedbackSample,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    LaserSystemConfiguration,
    NidaqLaserController,
    NullLaserController,
)


logger = get_verbose_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LaserTraceBlock:
    channel_id: LaserChannelId
    source: str
    x_values: Tuple[float, ...] = tuple()
    command_volts: Tuple[float, ...] = tuple()
    diode_volts: Tuple[float, ...] = tuple()
    command_copy_volts: Tuple[float, ...] = tuple()
    output_name: str = ""
    output_value: Optional[float] = None
    replace: bool = False
    event: str = "trace"
    operation_id: str = ""
    context_json: str = "{}"
    timestamp_method: str = "laser_event_perf_counter"
    timing_confidence: str = "host_timestamp"
    origin_perf_time: Optional[float] = None
    origin_wall_time: Optional[float] = None


class LaserModelEvents:
    trace_received = Callable[[LaserTraceBlock], None]


class LaserModel(ObservableObject):
    CONFIGURATION = "configuration"
    IS_CONNECTED = "is_connected"
    LAST_FEEDBACK_SAMPLE = "last_feedback_sample"
    trace_received: LaserModelEvents.trace_received

    def __init__(self, controller: Optional[LaserControllerProtocol] = None):
        super().__init__(("trace_received",))
        self._controller: Optional[LaserControllerProtocol] = None
        self._configuration = LaserSystemConfiguration()
        self._last_feedback_sample: Optional[LaserFeedbackSample] = None
        self._prepared_profiles = {}
        self._prepared_profiles_lock = threading.RLock()
        self._direct_trigger_lock = threading.RLock()
        self._direct_trigger_queue = None
        self._direct_trigger_observer = None
        self._direct_trigger_stop = threading.Event()
        self._direct_trigger_thread = None
        if controller is not None:
            self.set_controller(controller)

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    @property
    def is_connected(self) -> bool:
        return self._controller is not None

    @property
    def last_feedback_sample(self) -> Optional[LaserFeedbackSample]:
        return self._last_feedback_sample

    @property
    def timing_status(self) -> dict:
        status = getattr(self._controller, "timing_status", None)
        return (
            {"status": "independent", "reason": "Laser controller is unavailable"}
            if status is None
            else dict(status)
        )

    def configure_null(self, configuration: LaserSystemConfiguration) -> None:
        self.set_controller(NullLaserController(configuration))

    def configure_nidaq(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
        timing_plan: Optional[NidaqTimingPlan] = None,
    ) -> None:
        self.set_controller(
            NidaqLaserController(
                configuration,
                feedback_reader=feedback_reader,
                timing_plan=timing_plan,
            )
        )

    def load_configuration(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
        persisted_configuration: Optional[LaserSystemConfiguration] = None,
        timing_plan: Optional[NidaqTimingPlan] = None,
    ) -> None:
        backend = configuration.backend
        if backend == "disabled":
            prev_config = self._configuration
            self.close()
            self._configuration = configuration
            self._on_property_changed(self.CONFIGURATION, configuration, prev_config)
            log_hardware_initialization(logger, "SKIP | laser controller | backend=disabled")
            return

        started = time.perf_counter()
        log_hardware_initialization(
            logger,
            "START | laser controller | backend=%s hardware_timed=%s channels=%s",
            backend,
            configuration.hardware_timed,
            tuple(
                (
                    channel.channel_id.value,
                    channel.analog_output,
                    channel.diode_input,
                    channel.shutter_output,
                )
                for channel in configuration.channels
            ),
        )
        try:
            if backend == "null":
                self.configure_null(configuration)
            elif backend == "nidaq":
                self.configure_nidaq(
                    configuration,
                    feedback_reader=feedback_reader,
                    timing_plan=timing_plan,
                )
            else:
                raise ValueError(f"Unsupported laser backend: {backend}")
        except Exception as exc:
            log_hardware_initialization(
                logger,
                "FAILED | laser controller | backend=%s elapsed=%.3fs error=%s",
                backend,
                time.perf_counter() - started,
                str(exc) or exc.__class__.__name__,
                level=logging.ERROR,
            )
            raise
        log_hardware_initialization(
            logger,
            "READY | laser controller | backend=%s elapsed=%.3fs",
            backend,
            time.perf_counter() - started,
        )
        if (
            persisted_configuration is not None
            and persisted_configuration != configuration
        ):
            runtime_configuration = self._configuration
            self._configuration = persisted_configuration
            self._on_property_changed(
                self.CONFIGURATION,
                persisted_configuration,
                runtime_configuration,
            )

    def set_configuration_offline(self, configuration: LaserSystemConfiguration) -> None:
        prev_config = self._configuration
        self.close()
        self._configuration = configuration
        self._on_property_changed(self.CONFIGURATION, configuration, prev_config)

    def save_configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    def set_controller(self, controller: LaserControllerProtocol) -> None:
        prev_config = self._configuration
        self.close()
        self._controller = controller
        self._configuration = controller.configuration
        self._on_property_changed(self.CONFIGURATION, self._configuration, prev_config)
        self._on_property_changed(self.IS_CONNECTED, True, False)

    def set_command_voltage(self, channel_id: Union[LaserChannelId, int], volts: float) -> float:
        applied = self._require_controller().set_command_voltage(channel_id, volts)
        self.trace_received(
            LaserTraceBlock(
                channel_id=LaserChannelId(int(channel_id)),
                source="manual command",
                x_values=(0.0,),
                command_volts=(applied,),
            )
        )
        return applied

    def set_shutter_open(self, channel_id: Union[LaserChannelId, int], is_open: bool) -> None:
        self._require_controller().set_shutter_open(channel_id, is_open)
        self.trace_received(
            LaserTraceBlock(
                channel_id=LaserChannelId(int(channel_id)),
                source="output state",
                output_name="shutter_open",
                output_value=float(bool(is_open)),
            )
        )

    def set_auxiliary_output(self, channel_id: Union[LaserChannelId, int], enabled: bool) -> None:
        self._require_controller().set_auxiliary_output(channel_id, enabled)
        self.trace_received(
            LaserTraceBlock(
                channel_id=LaserChannelId(int(channel_id)),
                source="output state",
                output_name="auxiliary_enabled",
                output_value=float(bool(enabled)),
            )
        )

    def read_feedback_sample(self, channel_id: Union[LaserChannelId, int]) -> LaserFeedbackSample:
        sample = self._require_controller().read_feedback_sample(channel_id)
        prev, self._last_feedback_sample = self._last_feedback_sample, sample
        self._on_property_changed(self.LAST_FEEDBACK_SAMPLE, sample, prev)
        self._emit_feedback_trace(sample)
        return sample

    def read_command_copy_voltage(self, channel_id: Union[LaserChannelId, int]) -> float:
        return self._require_controller().read_command_copy_voltage(channel_id)

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        self._require_controller().run_pulse_train(pulse_train)
        self.trace_received(self._make_pulse_trace(pulse_train))

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain):
        operation = self._require_controller().run_synchronized_pulse_train(pulse_train)
        for channel_pulse in pulse_train.pulse_trains:
            self.trace_received(self._make_pulse_trace(channel_pulse))
        return operation

    def prepare_pulse_profile(self, profile, recipe):
        """Resolve a frozen protocol profile into one pre-armed finite output."""
        route = getattr(profile.trigger_route, "value", profile.trigger_route)
        hardware_trigger = route == "hardware_stim3"
        direct_start = route == "direct_ni_software"
        if not hardware_trigger and not direct_start:
            raise ValueError(f"Unsupported protocol laser trigger route: {route}")
        pulse = LaserPulseTrain(
            channel_id=LaserChannelId(int(profile.channel_id)),
            amplitude_volts=float(profile.amplitude_volts),
            duration_ms=float(profile.pulse_duration_ms),
            pulse_count=int(profile.pulse_count),
            frequency_hz=profile.frequency_hz,
            baseline_ms=float(profile.baseline_ms),
            post_stim_ms=float(profile.post_stim_ms),
            pmt_shutter_open_delay_ms=float(profile.pmt_open_lead_ms),
            pmt_shutter_close_delay_ms=float(profile.pmt_close_lag_ms),
            enable_pmt_shutter=(
                profile.pmt_open_lead_ms > 0 or profile.pmt_close_lag_ms > 0
            ),
        )
        operation_context = {
            "session_id": recipe.session_id,
            "session_generation": recipe.session_generation,
            "protocol_id": recipe.protocol_id,
            "protocol_revision": recipe.protocol_revision,
            "logical_trial_id": recipe.logical_trial_id,
            "attempt_id": recipe.attempt_id,
            "trial_operation_id": recipe.operation_id,
            "profile_id": profile.profile_id,
            "profile_revision": profile.revision,
            "laser_channel_id": int(profile.channel_id),
            "trigger_route": route,
        }
        synchronized = LaserSynchronizedPulseTrain(
            pulse_trains=(pulse,),
            trigger_source=(profile.trigger_terminal if hardware_trigger else None),
            wait=False,
            defer_start=direct_start,
            operation_context=operation_context,
        )
        # Preparing an asynchronous task is not a physical output. Do not use
        # run_synchronized_pulse_train(), whose ordinary/manual trace represents
        # an executed waveform.
        operation = self._require_controller().run_synchronized_pulse_train(
            synchronized
        )
        if operation is None:
            raise RuntimeError("Protocol laser preparation did not return an operation")
        operation_record = operation.to_record()
        timing_status = operation_record.get("timing_status") or {}
        required_status = (
            "hardware_synchronized" if hardware_trigger else "software_start"
        )
        if timing_status.get("status") != required_status:
            operation.cancel()
            raise RuntimeError(
                "Protocol laser timing is not ready: expected "
                f"{required_status}, found "
                f"{timing_status.get('status', 'unknown')}; "
                + str(
                    timing_status.get("reason")
                    or "no timing reason was reported"
                )
            )
        self._emit_protocol_operation_event(
            operation,
            "prepared",
            timing_confidence="planned_only",
        )
        add_terminal_callback = getattr(operation, "add_terminal_callback", None)
        if add_terminal_callback is not None:
            add_terminal_callback(self._on_protocol_laser_terminal)
        if direct_start:
            with self._prepared_profiles_lock:
                self._prepared_profiles[recipe.operation_id] = {
                    "operation": operation,
                    "context": operation_context,
                    "nonce": None,
                    "enabled": False,
                }
        return operation

    def start_direct_trigger_receiver(self, trigger_queue, observer=None) -> None:
        with self._direct_trigger_lock:
            thread = self._direct_trigger_thread
            if thread is not None and thread.is_alive():
                if (
                    trigger_queue is not self._direct_trigger_queue
                    or observer != self._direct_trigger_observer
                ):
                    raise RuntimeError(
                        "Direct laser trigger receiver is already running with "
                        "different ownership"
                    )
                return
            self._direct_trigger_queue = trigger_queue
            self._direct_trigger_observer = observer
            self._direct_trigger_stop.clear()
            thread = threading.Thread(
                target=self._run_direct_trigger_receiver,
                name="StimToNidaqTrigger",
                daemon=True,
            )
            self._direct_trigger_thread = thread
            thread.start()

    def bind_direct_trigger_nonce(self, operation_id: str, nonce: str) -> None:
        with self._prepared_profiles_lock:
            prepared = self._prepared_profiles.get(str(operation_id))
            if prepared is None:
                return
            prepared["nonce"] = str(nonce)
            prepared["enabled"] = True

    def release_prepared_profile(self, operation) -> None:
        with self._prepared_profiles_lock:
            for operation_id, prepared in tuple(self._prepared_profiles.items()):
                if prepared["operation"] is operation:
                    self._prepared_profiles.pop(operation_id, None)

    def stop_direct_trigger_receiver(self) -> None:
        with self._direct_trigger_lock:
            self._direct_trigger_stop.set()
            thread = self._direct_trigger_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)
        with self._direct_trigger_lock:
            if self._direct_trigger_thread is thread:
                self._direct_trigger_thread = None
        with self._prepared_profiles_lock:
            self._prepared_profiles.clear()

    def _run_direct_trigger_receiver(self) -> None:
        while not self._direct_trigger_stop.is_set():
            try:
                message = self._direct_trigger_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            received = time.perf_counter()
            result = dict(message)
            result["ipc_receive_perf_time"] = received
            result["ipc_queue_delay_seconds"] = (
                received - float(message["ipc_send_perf_time"])
            )
            try:
                operation_id = str(message["operation_id"])
                with self._prepared_profiles_lock:
                    prepared = self._prepared_profiles.get(operation_id)
                    if prepared is None:
                        raise RuntimeError("direct trigger has no prepared NI operation")
                    context = prepared["context"]
                    expected = (
                        int(context["session_generation"]),
                        int(context["logical_trial_id"]),
                        int(context["attempt_id"]),
                        prepared["nonce"],
                    )
                    observed = (
                        int(message["session_generation"]),
                        int(message["logical_trial_id"]),
                        int(message["attempt_id"]),
                        str(message["nonce"]),
                    )
                    if observed != expected:
                        raise RuntimeError("stale or mismatched direct trigger context")
                    if not prepared["enabled"]:
                        raise RuntimeError("direct trigger arrived before pellet presentation")
                    operation = prepared["operation"]
                result["daqmx_start_entry_perf_time"] = time.perf_counter()
                operation.trigger()
                result["daqmx_start_return_perf_time"] = time.perf_counter()
                result["accepted"] = True
                result["timing_confidence"] = "software_start"
                self._emit_protocol_operation_event(
                    operation,
                    "triggered",
                    perf_time=result["daqmx_start_entry_perf_time"],
                    timing_confidence="software_start",
                )
            except Exception as error:
                result["accepted"] = False
                result["error"] = f"{type(error).__name__}: {error}"
                logger.exception("Direct stim-to-NI trigger rejected")
            observer = self._direct_trigger_observer
            if observer is not None:
                try:
                    observer(result)
                except Exception:
                    logger.exception("Direct trigger observer failed")

    def _on_protocol_laser_terminal(self, operation) -> None:
        state = getattr(operation.state, "value", operation.state)
        observations = tuple(getattr(operation, "observations", ()))
        perf_time = observations[-1][1] if observations else time.perf_counter()
        confidence = (
            "task_completion"
            if state == "completed"
            else "operation_failure"
            if state == "failed"
            else "operation_cancelled"
        )
        self._emit_protocol_operation_event(
            operation,
            str(state),
            perf_time=perf_time,
            timing_confidence=confidence,
        )

    def _emit_protocol_operation_event(
        self,
        operation,
        event,
        *,
        perf_time=None,
        timing_confidence,
    ) -> None:
        context = dict(getattr(operation, "context", {}) or {})
        channel_id = int(context.get("channel_id", 0) or 0)
        if channel_id <= 0:
            # The prepared profile currently contains one laser channel. Keep
            # the channel in the operation context for stable persistence.
            channel_id = int(context.get("laser_channel_id", 1))
        perf_time = time.perf_counter() if perf_time is None else float(perf_time)
        wall_time = time.time() - (time.perf_counter() - perf_time)
        self.trace_received(LaserTraceBlock(
            channel_id=LaserChannelId(channel_id),
            source="protocol operation",
            event=str(event),
            operation_id=str(getattr(operation, "operation_id", "")),
            context_json=json.dumps(context, sort_keys=True),
            timestamp_method="daqmx_operation_perf_counter",
            timing_confidence=str(timing_confidence),
            origin_perf_time=perf_time,
            origin_wall_time=wall_time,
        ))

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp) -> Tuple[LaserCalibrationPoint, ...]:
        self.trace_received(
            LaserTraceBlock(
                channel_id=ramp.channel_id,
                source="calibration",
                replace=True,
            )
        )
        points = self._require_controller().run_calibration_ramp(ramp)
        if points:
            prev, self._last_feedback_sample = self._last_feedback_sample, LaserFeedbackSample(
                channel_id=points[-1].channel_id,
                command_volts=points[-1].command_volts,
                diode_volts=points[-1].diode_volts,
                command_copy_volts=points[-1].command_copy_volts,
            )
            self._on_property_changed(self.LAST_FEEDBACK_SAMPLE, self._last_feedback_sample, prev)
            sample_rate_hz = self._configuration.sample_rate_hz or 1.0
            seconds_per_step = ramp.samples_per_step / sample_rate_hz
            self.trace_received(
                LaserTraceBlock(
                    channel_id=ramp.channel_id,
                    source="calibration",
                    x_values=tuple(index * seconds_per_step for index in range(len(points))),
                    command_volts=tuple(point.command_volts for point in points),
                    diode_volts=tuple(point.diode_volts for point in points),
                    command_copy_volts=tuple(
                        math.nan if point.command_copy_volts is None else point.command_copy_volts
                        for point in points
                    ),
                    replace=True,
                )
            )
        return points

    def make_diode_power_curve(self, points: Tuple[LaserCalibrationPoint, ...]) -> LaserDiodePowerCurve:
        if not points:
            raise ValueError("calibration points cannot be empty")
        return LaserDiodePowerCurve(points[0].channel_id, points)

    def close_all_shutters(self) -> None:
        controller = self._controller
        if controller is not None:
            controller.close_all_shutters()

    def close(self) -> None:
        controller = self._controller
        if controller is None:
            return
        was_connected = self.is_connected
        try:
            controller.close_all_shutters()
        finally:
            try:
                controller.close()
            finally:
                self._controller = None
                self._on_property_changed(self.IS_CONNECTED, False, was_connected)

    def _require_controller(self) -> LaserControllerProtocol:
        controller = self._controller
        if controller is None:
            raise RuntimeError("Laser controller is not configured")
        return controller

    def _emit_feedback_trace(self, sample: LaserFeedbackSample) -> None:
        self.trace_received(
            LaserTraceBlock(
                channel_id=sample.channel_id,
                source="feedback",
                x_values=(0.0,),
                command_volts=(sample.command_volts,),
                diode_volts=(sample.diode_volts,),
                command_copy_volts=(
                    math.nan if sample.command_copy_volts is None else sample.command_copy_volts,
                ),
            )
        )

    def _make_pulse_trace(self, pulse_train: LaserPulseTrain) -> LaserTraceBlock:
        channel = self._configuration.get_channel(pulse_train.channel_id)
        minimum = channel.minimum_command_volts
        amplitude = pulse_train.amplitude_volts
        duration_s = pulse_train.duration_ms / 1000.0
        baseline_s = pulse_train.baseline_ms / 1000.0
        post_stim_s = pulse_train.post_stim_ms / 1000.0
        period_s = (
            1.0 / pulse_train.frequency_hz
            if pulse_train.frequency_hz is not None
            else duration_s
        )
        x_values = [0.0]
        y_values = [minimum]
        current_t = 0.0

        def horizontal(to_t: float) -> None:
            nonlocal current_t
            if to_t <= current_t:
                return
            x_values.append(to_t)
            y_values.append(y_values[-1])
            current_t = to_t

        def transition(value: float) -> None:
            x_values.extend((current_t, current_t))
            y_values.extend((y_values[-1], value))

        horizontal(baseline_s)
        for pulse_index in range(pulse_train.pulse_count):
            pulse_start = baseline_s + pulse_index * period_s
            horizontal(pulse_start)
            transition(amplitude)
            horizontal(pulse_start + duration_s)
            transition(minimum)
        horizontal(current_t + post_stim_s)
        trigger_kind = "external" if pulse_train.trigger_source else "internal"
        return LaserTraceBlock(
            channel_id=pulse_train.channel_id,
            source=f"{trigger_kind} pulse",
            x_values=tuple(x_values),
            command_volts=tuple(y_values),
        )
