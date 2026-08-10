"""
Device interface for the CANbus protocol to the Alogus device.

Extends the Device class that defines a fixed API to access the device. This
class relies on the CanInterface class to send and receive data.

"""
import collections
import copy
import dataclasses
import logging
import math
import os
import queue
import threading
import time
from functools import partial
from typing import Callable, Iterable, Tuple, Union, SupportsInt, List, Optional, Any, cast, Dict

from autotrainer.core import Offset3DTuple, get_perf_now, Motor
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.message import SystemDataArgsKwargs

from autotrainer.core import SystemStatusMessageKind, SystemCommandKind, Offset3DTuple

from .motor_steps import MotorSteps
from .device import Device
from .emulation_interface import EmulationInterface
from .device_api import DeviceApi
from .can_interface import CanInterface, Target, target_of_motor
from .can_transport import CanTransportConfiguration, CanTransportKind
from .device_interface import (
    Acknowledge,
    AnalogOutput,
    AnalogOutputs,
    AudioData,
    DoorData,
    PressureReading,
    Motor,
    DigitalOutputs,
    Status,
    Tone,
    ColorLed,
    MagnetDigitalInputs,
    PelletDigitalInputs,
    ServoConfig,
    StepperConfig,
    SensorStatus,
    ServoStatus,
    StepperStatus,
    Version,
)


logger = get_verbose_logger(__name__)

_force_emulation = os.getenv("AUTOTRAINER_FORCE_CAN_EMULATION_IFACE", "") == "1"
if _force_emulation:
    HAVE_CAN_DEVICE = False
else:
    try:
        from pyjerrycan import JerryCAN, JerryCANMsg, JerryCANCfgMsg, JerryCANCmdType

        HAVE_CAN_DEVICE = True
    except ModuleNotFoundError:
        HAVE_CAN_DEVICE = False


# this is used to reduce the rate of messages sent to "clients/listeners" to the property changed callback:
_similar_data_refresh_delay = os.getenv("AUTOTRAINER_DEVICE_SIMILAR_DATA_REFRESH_DELAY") or 2.5
_similar_data_refresh_delay = float(_similar_data_refresh_delay)

# some sentinels object:

class _Sentinel:

    def __init__(self, role):
        self.role = role

    def __repr__(self):
        return self.role


# this is used from CAN reader thread to put to CAN writer thread message queue :
_uuid_ack = _Sentinel(role="uuid_ack")

# this is used by CAN writer thread to manage its handling of internal queue of received commands to be executed.
_next_compound = _Sentinel(role="next_compound")

# for eventual retry when uuid ack timeout:
_retry_compound = _Sentinel(role="retry_compound")
_retry_full = _Sentinel(role="retry_full")
_shutdown_requested = _Sentinel(role="shutdown_requested")


def _no_op():
    return True


def _no_op_handler(_):
    return True


def _to_tuple(value: Union[str, Any]):
    if not isinstance(value, str):
        return value
    if "," in value:
        parts = value.split(",")
        return float(parts[0].strip()), float(parts[1].strip())
    else:
        return float(value)


def unpack_data_arg(data):
    if isinstance(data, SystemDataArgsKwargs):
        return data.args, data.kwargs
    return (data,), {}


def apply_system_command_with_data_args(func, data):
    args, kwargs = unpack_data_arg(data)
    return func(*args, **kwargs)


MotorStatusCacheT = Dict[
    SystemStatusMessageKind,
    Tuple[float, float],  # data, perf_c
]


@dataclasses.dataclass
class _BoardPendingContext:
    uuid_ack_timeout_engaged_property_name: str  # but actually unused
    target: Optional[Target]
    ctx: Optional[str] = None  # current command context (token)
    kind: Optional[Any] = None  # "kind", can be different things
    uuid: Optional[int] = None # currently awaited uuid ack
    uuid_ack_perf_c: float = -math.inf
    skip_uuid_ack_perf_c: bool = False
    ack_perf_timeout: float = math.inf  # perf timeout for current uuid ack
    prev_command: Optional[Tuple[str, Any, str]] = None
    prev_command_relative: bool = False
    uuid_ack_timeout_engaged: bool = False
    repeated_command_count: int = 0
    compound_steps: Optional[List[Dict[str, Any]]] = None

    def is_available(self):
        return (
                self.ctx is None
            and self.uuid is None
            and self.prev_command is None
            and (self.compound_steps is None or len(self.compound_steps) == 0)
        )


class CanDevice(Device):

    default_command_write_failed_repeat_count: int = 3
    default_command_ack_timeout_duration: float = 3  # seconds

    default_command_ack_timeout_repeat_count: int = 3

    same_data_refresh_delay: float = _similar_data_refresh_delay
    """When > 0: if new data value is equal to previous value,
    and elapsed time since last one is smaller than this delay: skip data update.
    """

    default_board_status_timeout_delay: float = 15  # seconds

    _motor_to_status_kind = {
        Motor.PELLET_X_MOTOR: SystemStatusMessageKind.PELLET_MOTOR_X,
        Motor.PELLET_Y_MOTOR: SystemStatusMessageKind.PELLET_MOTOR_Y,
        Motor.PELLET_Z_MOTOR: SystemStatusMessageKind.PELLET_MOTOR_Z,
        Motor.PELLET_LOAD_SERVO: SystemStatusMessageKind.PELLET_LOAD,
        Motor.PELLET_COVER_SERVO: SystemStatusMessageKind.PELLET_COVER,
        Motor.TUNNEL_MAGNET_SERVO: SystemStatusMessageKind.HEAD_MAGNET,
        Motor.TUNNEL_GATE_SERVO: SystemStatusMessageKind.TUNNEL_GATE_SERVO,
        Motor.TUNNEL_FAN_SERVO: SystemStatusMessageKind.TUNNEL_FAN,
    }

    _motor_to_coordinate_char = {
        Motor.PELLET_X_MOTOR: "x",
        Motor.PELLET_Y_MOTOR: "y",
        Motor.PELLET_Z_MOTOR: "z",
    }
    _motor_to_coordinate_idx = {
        Motor.PELLET_X_MOTOR: 0,
        Motor.PELLET_Y_MOTOR: 1,
        Motor.PELLET_Z_MOTOR: 2,
    }

    def __init__(self, api: Optional[DeviceApi] = None, buffer_size: int = 50, force_emulation: bool = False,
                 *, can_transport: Optional[CanTransportConfiguration] = None,
                 required_targets: Optional[Iterable[Target]] = None,
                 shutdown_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the CANbus device interface.

        Args:
            api: The device API instance to use for communication
            buffer_size: Size of the measurement buffer
            force_emulation: Whether to force using emulation mode even if hardware is available
            can_transport: CAN backend selection. Defaults to the existing pyjerrycan path.
            required_targets: CAN boards that must be present for this runtime.
        """
        self._can_transport_configuration = can_transport or CanTransportConfiguration.from_environment()
        self._required_targets = tuple(required_targets or (Target.PELLET_DEVICE, Target.MAGNET_DEVICE))
        self._interface: Union[CanInterface, EmulationInterface] = self._make_device_interface(force_emulation)

        super().__init__(self._interface, api)

        self._want_exit = threading.Event()
        self._disconnect_lock = threading.Lock()
        self._command_execution_lock = threading.Lock()
        self._shutdown_callback = shutdown_callback

        self._init_default_move_configs()
        self._compound_movement: Optional[List[Dict[str, Any]]] = None

        self._last_limit_switch: Dict[Motor, Optional[bool]] = {
            Motor.PELLET_X_MOTOR: None,
            Motor.PELLET_Y_MOTOR: None,
            Motor.PELLET_Z_MOTOR: None,
        }
        self._last_pellet_pos = Offset3DTuple(math.nan, math.nan, math.nan)
        self._last_send_pos = Offset3DTuple(math.nan, math.nan, math.nan)

        if isinstance(self._interface, EmulationInterface):
            logger.warning(
                "Alogus hardware or hardware support not found. Using emulation interface.")

        self._motor_configs: Dict[Motor, Union[StepperConfig, ServoConfig]] = {}
        # ensure we have config for these steppers/servos, even if empty/default:
        for m in {Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR}:
            self._motor_configs[m] = StepperConfig()
        for m in {Motor.TUNNEL_GATE_SERVO, Motor.TUNNEL_FAN_SERVO, Motor.TUNNEL_MAGNET_SERVO,
                  Motor.PELLET_COVER_SERVO, Motor.PELLET_LOAD_SERVO}:
            self._motor_configs[m] = ServoConfig()
        # NB: these are the config possibly written/set to the motors.
        # Not the config reported by the motor themselves.

        self._init_handlers()

        self._prev_command_timeout: float = self.default_command_ack_timeout_duration
        self._prev_command: Optional[Tuple[object, Any, type(None)]] = None
        self._prev_command_is_relative = False

        self._commands_queue = queue.Queue()
        self._commands_handler_thread: Optional[threading.Thread] = None
        self._commands_handler_watchdog_perf_c = math.nan
        self._tunnel_pellet_status_check_thread: Optional[threading.Thread] = None
        # internal data cache:
        self._previous_stepper_status_pos_perf_c: MotorStatusCacheT = {}  # (None, -math.inf)
        self._previous_servo_status_pos_perf_c: MotorStatusCacheT = {}  # (None, -math.inf)
        self._prev_tunnel_gate_open_perf_c: Tuple[bool, float] = (None, -math.inf)

        self._boards_pending_ctx: Dict[Target, _BoardPendingContext] = {
            None: _BoardPendingContext(
                target=None, uuid_ack_timeout_engaged_property_name=""
            ),
            Target.PELLET_DEVICE: _BoardPendingContext(
                target=Target.PELLET_DEVICE,
                uuid_ack_timeout_engaged_property_name=self.PELLET_UUID_ACK_TIMEOUT_ENGAGED,
            ),
            Target.MAGNET_DEVICE: _BoardPendingContext(
                target=Target.MAGNET_DEVICE,
                uuid_ack_timeout_engaged_property_name=self.MAGNET_UUID_ACK_TIMEOUT_ENGAGED,
            ),
        }

    @property
    def can_transport_configuration(self) -> CanTransportConfiguration:
        return self._can_transport_configuration

    @property
    def required_targets(self) -> Tuple[Target, ...]:
        return self._required_targets

    def is_target_required(self, target: Target) -> bool:
        return target in self._required_targets

    def is_motor_required(self, motor: Motor) -> bool:
        if self._required_targets == (Target.PELLET_DEVICE,) and motor in {
            Motor.TUNNEL_MAGNET_SERVO,
            Motor.TUNNEL_GATE_SERVO,
            Motor.TUNNEL_FAN_SERVO,
        }:
            return False
        return self.is_target_required(target_of_motor(motor))

    def _make_device_interface(self, force_emulation: bool) -> Union[CanInterface, EmulationInterface]:
        transport = self._can_transport_configuration
        if force_emulation or transport.kind == CanTransportKind.EMULATION:
            return EmulationInterface()
        if transport.uses_linux_can_stack:
            return CanInterface(required_targets=self._required_targets, can_transport=transport)
        return CanInterface(required_targets=self._required_targets) if HAVE_CAN_DEVICE else EmulationInterface()

    def _init_default_move_configs(self):
        self._load_pellet = default_load_pellet()
        self._send_pellet = default_send_pellet()
        self._cover_pellet = default_cover_pellet()
        self._release_pellet = default_release_pellet()
        self._open_tunnel_gate = default_open_gate()
        self._close_tunnel_gate = default_close_gate()
        self._move_retract = default_move_retract()

    def _clear_caches(self):
        for cache in (
            self._previous_stepper_status_pos_perf_c,
            self._previous_servo_status_pos_perf_c,
        ):
            cache.clear()
        self._prev_tunnel_gate_open_perf_c = (None, -math.inf)

    def _init_handlers(self):

        def handle_servo_move(motor: Motor, position):
            steps = self._make_servo_move_steps(motor, position)
            return self._start_sequence(MotorSteps(f"move_servo_{motor.name}", steps))

        def handle_servo_sequence(motor: Motor, sequence: MotorSteps):
            step, steps = self._make_servo_steps(motor)
            for sub_step in sequence.steps:
                sub_step.update(step)  # eventual uuid_ack_timeout
            if len(steps) > 1:  # attach/detach
                step_idx = steps.index(step)
                steps[step_idx:-1] = sequence.steps
            else:
                steps = sequence.steps
            return self._start_sequence(MotorSteps(f"{motor.name}_{sequence.name}", steps))

        def set_load_pellet_proc(proc):
            if isinstance(proc, MotorSteps) and not proc.is_empty:
                self._load_pellet = proc
            else:
                logger.warning("set_load_pellet_proc: empty proc: %s", proc)
            return True

        def set_send_pellet_proc(proc):
            if isinstance(proc, MotorSteps) and not proc.is_empty:
                self._send_pellet = proc
            else:
                logger.warning("set_send_pellet_proc: empty proc: %s", proc)
            return True

        def set_cover_pellet_proc(proc):
            if isinstance(proc, MotorSteps) and not proc.is_empty:
                self._cover_pellet = proc
            else:
                logger.warning("set_cover_pellet_proc: empty proc: %s", proc)
            return True

        def set_release_pellet_proc(proc):
            if isinstance(proc, MotorSteps) and not proc.is_empty:
                self._release_pellet = proc
            else:
                logger.warning("set_release_pellet_proc: empty proc: %s", proc)
            return True

        def set_move_retract_proc(proc):
            prev, self._move_retract = self._move_retract, proc
            logger.verbose("replaced retract proc by %s (prev=%s)", proc, prev)
            return True

        def apply_set_or_move(func, motor, *args, **kwargs):
            has_relative = "relative" in kwargs
            is_relative = has_relative and kwargs["relative"]
            if motor is not None:
                cfg = self._motor_configs[motor]
                if cfg.uuid_ack_timeout is not None:
                    self._prev_command_timeout = cfg.uuid_ack_timeout
            self._prev_command_is_relative = is_relative
            return func(*args, **kwargs)

        # Initialize command handlers lookup table
        self._command_handlers = {
            SystemCommandKind.REQUEST_VERSION:
                lambda data: self._interface.request_version(),

            SystemCommandKind.BOARD_REBOOT: self._interface.board_reboot,

            SystemCommandKind.MOVE_MAGNET_SERVO: partial(handle_servo_move, Motor.TUNNEL_MAGNET_SERVO),

            SystemCommandKind.MOVE_LOAD_SERVO: partial(handle_servo_move, Motor.PELLET_LOAD_SERVO),

            SystemCommandKind.MOVE_COVER_SERVO: partial(handle_servo_move, Motor.PELLET_COVER_SERVO),

            SystemCommandKind.MOVE_GATE_SERVO: partial(handle_servo_move, Motor.TUNNEL_GATE_SERVO),

            SystemCommandKind.SET_X: partial(apply_set_or_move, self._interface.set_motor_x, None),

            SystemCommandKind.SET_Y: partial(apply_set_or_move, self._interface.set_motor_y, None),

            SystemCommandKind.SET_Z: partial(apply_set_or_move, self._interface.set_motor_z, None),

            SystemCommandKind.MOVE_X: partial(apply_set_or_move, self._interface.move_motor_x,
                                              Motor.PELLET_X_MOTOR),

            SystemCommandKind.MOVE_Y: partial(apply_set_or_move, self._interface.move_motor_y,
                                              Motor.PELLET_Y_MOTOR),

            SystemCommandKind.MOVE_Z: partial(apply_set_or_move, self._interface.move_motor_z,
                                              Motor.PELLET_Z_MOTOR),

            # NB: at the moment SEND_TO_LIMITS == SEND_HOME basically
            SystemCommandKind.SEND_TO_LIMITS:
                lambda data: self._start_sequence(MotorSteps(
                    "send_to_limits",
                    [{"home": d}
                     for d in (data if isinstance(data, (list, tuple)) else [data])
                     ]
                )),

            SystemCommandKind.SEND_HOME:
                lambda _: self._start_sequence(MotorSteps(
                    "send_home",
                    [{"home": m} for m in (Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR, Motor.PELLET_X_MOTOR)]
                )),

            # NB: only used by test and can_console, we should always use SEND_PELLET instead.
            SystemCommandKind.SEND_FIXED_XYZ: lambda _: self._interface.fixed_position(),

            # NB: the following sequences are using "predefined" move,
            # that are below automatically handled for possible servo attach/detach
            SystemCommandKind.LOAD_PELLET: lambda _: self._start_sequence(self._load_pellet),
            SystemCommandKind.SEND_PELLET: lambda _: self._start_sequence(self._send_pellet),

            SystemCommandKind.RELEASE_PELLET: lambda _: self._start_sequence(self._release_pellet),
            SystemCommandKind.COVER_PELLET: lambda _: self._start_sequence(self._cover_pellet),

            SystemCommandKind.SEND_RETRACT: lambda _: self._start_sequence(self._move_retract),

            # on the other side we don't have "predefined" for open/close gate:
            SystemCommandKind.OPEN_TUNNEL_GATE: lambda _: handle_servo_sequence(Motor.TUNNEL_GATE_SERVO, self._open_tunnel_gate),
            SystemCommandKind.CLOSE_TUNNEL_GATE: lambda _: handle_servo_sequence(Motor.TUNNEL_GATE_SERVO, self._close_tunnel_gate),

            SystemCommandKind.TUNNEL_FAN_ON: \
                lambda _: self._interface.set_digital_output(DigitalOutputs(1), True),
            SystemCommandKind.TUNNEL_FAN_OFF:
                lambda _: self._interface.set_digital_output(DigitalOutputs(1), False),

            # digital only allow 2 "position"
            # SystemCommandKind.TUNNEL_FAN_SET: partial(handle_servo_move, Motor.TUNNEL_FAN_SERVO),

            SystemCommandKind.DELAY: self._interface.delay,

            SystemCommandKind.READ_MOTOR_CONFIGURATION: self._interface.request_motor_config,

            SystemCommandKind.WRITE_MOTOR_CONFIGURATION: self._handle_write_motor_configuration,

            SystemCommandKind.SET_LOAD_PELLET_PROCEDURE: set_load_pellet_proc,

            SystemCommandKind.SET_SEND_PELLET_PROCEDURE: set_send_pellet_proc,

            SystemCommandKind.SET_COVER_PELLET_PROCEDURE: set_cover_pellet_proc,

            SystemCommandKind.SET_RELEASE_PELLET_PROCEDURE: set_release_pellet_proc,

            SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE: set_move_retract_proc,


            SystemCommandKind.SET_DIGITAL_OUTPUT:
                lambda data: self._interface.set_digital_output(DigitalOutputs(data[0]), data[1]),

            SystemCommandKind.SET_ANALOG_OUTPUT:
                lambda data: self._interface.set_analog_output(AnalogOutputs(data[0]), data[1]),

            SystemCommandKind.SET_RGB_LED:
                lambda data: self._interface.set_color_led(data[0], data[1], data[2]),

            SystemCommandKind.PLAY_TONE:
                lambda data: (
                    self._interface.emit_tone(data[0], data[1]) if isinstance(data, tuple)
                    else self._interface.emit_tone(data, 500)  # 500 millisecond
                ),

            SystemCommandKind.SET_MOTOR_DRIFT: self._interface.set_motors_drift,
            SystemCommandKind.SET_AUTO_CORRECT_DRIFT: self._interface.set_auto_correct_motor_drift,

            SystemCommandKind.SERVO_ATTACH: self._interface.servo_attach,
            SystemCommandKind.SERVO_DETACH: self._interface.servo_detach,

            # No-op handlers
            SystemCommandKind.STREAM_START: _no_op_handler,
            SystemCommandKind.STREAM_STOP: _no_op_handler,
        }

        # Initialize data / response handlers lookup table

        def handle_motor_config(m: Union[StepperConfig, ServoConfig]):
            # do we want to:
            #   self._motor_configs[m.motor] = m
            # as we do for written motor configs ? but probably better to not do it here.
            self._api.send_message(SystemStatusMessageKind.MOTOR_CONFIGURATION, m)

        previous_stimuli_data_perf_c = (None, -math.inf)
        def handle_stimuli_msg(m):
            nonlocal previous_stimuli_data_perf_c
            new_data = [m.stimulus_1, m.stimulus_2, m.stimulus_3, m.stimulus_4]
            prev_data, prev_perf_c = previous_stimuli_data_perf_c
            p_now = get_perf_now()
            if new_data != prev_data or p_now - prev_perf_c > self.same_data_refresh_delay:
                previous_stimuli_data_perf_c = (new_data, p_now)
                self._api.send_message(SystemStatusMessageKind.STIMULUS_INPUTS, new_data)

        previous_door_data_perf_c = (None, -math.inf)
        def handle_door_msg(m: DoorData):
            nonlocal previous_door_data_perf_c
            new_data = (m.door1, m.door2, m.door3, m.ext_button)
            p_now = get_perf_now()
            prev_data, prev_perf_c = previous_door_data_perf_c
            if new_data != prev_data or p_now - prev_perf_c > self.same_data_refresh_delay:
                previous_door_data_perf_c = (new_data, p_now)
                send_msg = self._api.send_message
                send_msg(SystemStatusMessageKind.FRONT_DOOR, m.door1 != 0),
                send_msg(SystemStatusMessageKind.DRAWER_DOOR, m.door2 != 0),
                send_msg(SystemStatusMessageKind.SPARE_DOOR, m.door3 != 0),
                send_msg(SystemStatusMessageKind.EXT_BUTTON, m.ext_button != 0)

        prev_color_led = (None, -math.inf)
        def handle_color_led(m: ColorLed):
            nonlocal prev_color_led
            new_data = (m.red, m.green, m.blue)
            p_now = get_perf_now()
            prev_data, prev_perf_c = prev_color_led
            if new_data != prev_data or p_now - prev_perf_c > self.same_data_refresh_delay:
                self._api.send_message(SystemStatusMessageKind.COLOR_LED, m)
                prev_color_led = (new_data, p_now)

        self._data_handlers = {
            Status: _no_op_handler,  # No-op for Status messages
            Tone: _no_op_handler,
            ColorLed: handle_color_led,
            AnalogOutput: _no_op_handler,

            PressureReading: _no_op_handler,
            SensorStatus: _no_op_handler,
            MagnetDigitalInputs: _no_op_handler,

            PelletDigitalInputs: handle_stimuli_msg,

            AudioData: _no_op_handler,

            StepperStatus: self._report_stepper_status,

            ServoStatus: self._report_servo_status,

            StepperConfig: handle_motor_config,
            ServoConfig: handle_motor_config,

            Version: lambda message: \
                self._api.send_message(SystemStatusMessageKind.FIRMWARE_VERSION, message.version),

            DoorData: handle_door_msg,

            Acknowledge: self._handle_ack,
        }

    @property
    def writer_watchdog_perf_c(self) -> float:
        return self._commands_handler_watchdog_perf_c

    def _put_to_cmd_queue(self, obj):
        if self._want_exit.is_set():
            logger.warning("Ignoring CAN command while device is shutting down")
            return
        cmd_thread = self._commands_handler_thread
        # cmd thread is started on connect()
        if cmd_thread is not None and not cmd_thread.is_alive():
            raise RuntimeError("CAN command handler thread not anymore alive: %s", cmd_thread)
        self._commands_queue.put(obj)

    def _execute_if_active(self, func, *args, **kwargs):
        with self._command_execution_lock:
            if self._want_exit.is_set():
                return _shutdown_requested
            return func(*args, **kwargs)

    def _check_tunnel_pellet_status_age(self):
        logger.verbose("running")
        while not self._want_exit.wait(1):  # no need check more often
            p_now = get_perf_now()
            boards_timeout = self.default_board_status_timeout_delay  # re-read
            pellet_age = p_now - self._interface.pellet_status_perf_c
            tunnel_age = p_now - self._interface.tunnel_status_perf_c
            check_pellet = self.is_target_required(Target.PELLET_DEVICE)
            check_tunnel = self.is_target_required(Target.MAGNET_DEVICE)
            ages = []
            if check_pellet:
                ages.append(pellet_age)
            if check_tunnel:
                ages.append(tunnel_age)
            if any(age > boards_timeout / 2 for age in ages):
                logger.verbose("pellet_status_age=%.1f tunnel_status_age=%.1f", pellet_age, tunnel_age)
            self.pellet_status_timeout_engaged = check_pellet and pellet_age > boards_timeout
            self.tunnel_status_timeout_engaged = check_tunnel and tunnel_age > boards_timeout
        logger.verbose("exiting")

    def _command_handler(self):
        try:
            self.__command_handler()
        except BaseException as err:
            logger.exception("command handler crashed: %s", err)
            callback = self._shutdown_callback
            if callback is not None:
                try:
                    callback(f"CAN command handler failure: {err}")
                except Exception:
                    logger.exception("Failed to request CAN safety shutdown")
            raise

    def __command_handler(self):
        cur_commands = []
        t_perf_last_command_with_uuid = None
        input_q = self._commands_queue
        has_read_from_queue = False
        boards_pending_ctx = self._boards_pending_ctx

        def boards_has_ack_timeout_engaged():
            for board in boards_pending_ctx.values():
                if board.uuid_ack_timeout_engaged:
                    return True
            return False

        def perform_next_compound(board: _BoardPendingContext, steps: Optional[List[Dict]]):
            if steps is None or len(steps) == 0:
                logger.warning("Got empty compound steps. board=%s kind=%s ctx=%s",
                               board.target, board.kind, board.ctx)
                return
            self._prev_command_timeout = self.default_command_ack_timeout_duration
            attempt_idx = 0
            while True:
                success = self._perform_next_compound_step(board, steps)
                if success:
                    break
                attempt_idx += 1
                if attempt_idx > self.default_command_write_failed_repeat_count:
                    break
            if not success:
                raise RuntimeError("too many failure trying _perform_next_compound_step")

        def sort_available_commands(r):
            k, d, c, perf_c = r  # kind data ctx perf
            t = self._find_command_next_board_target(k, d)
            b: _BoardPendingContext = self._boards_pending_ctx[t]
            return (0 if b.is_available() else 1, perf_c)

        p_before_loop = get_perf_now()
        while True:
            if self._want_exit.is_set():
                logger.verbose("CAN shutdown requested, exiting command loop")
                break
            if has_read_from_queue:
                input_q.task_done()
                has_read_from_queue = False

            p_now = get_perf_now()
            if __debug__ and os.getenv("_AUTOTRAINER_SIMULATE_CAN_WRITER_THREAD_CRASH") == "1":
                if p_now > p_before_loop + int(os.getenv("_AUTOTRAINER_SIMULATE_CAN_WRITER_THREAD_CRASH_TIMEOUT", "180")):
                    1 / 0

            # always update watchdog_perf_c:
            # NB: not using "p_now" but real time.perf_counter on purpose,
            # otherwise tests can fail because of patched/simulated one which is used.
            self._commands_handler_watchdog_perf_c = time.perf_counter()

            # don't loop too often, when nothing to do:
            if len(cur_commands) == 0 and all(board.is_available() for board in boards_pending_ctx.values()):
                timeout = 0.25
            else:
                timeout = 0.05  # there might be next-compound to execute, or wait for uuid-ack
            # what can anyway unblocks, is receiving anything, including _uuid_ack, in this input_q:
            try:
                raw = input_q.get(timeout=timeout)
            except queue.Empty:
                raw = None, None, None
                p_now = None
            else:
                has_read_from_queue = True
                p_now = time.perf_counter()
            if raw is None:
                input_q.task_done()
                logger.verbose("received exit sentinel, exiting main loop ..")
                break
            kind, data, ctx = raw
            raw = kind, data, ctx, p_now
            found_board_with_uuid_ack = None
            if kind is _uuid_ack:
                msg_uuid, msg_perf_c = data
                for target, board_ctx in boards_pending_ctx.items():
                    if board_ctx.uuid is not None and board_ctx.uuid == msg_uuid:
                        found_board_with_uuid_ack = board_ctx
                        ctx = board_ctx.ctx
                        cur_commands.insert(0, (_uuid_ack, data, ctx, -math.inf))
                        board_ctx.uuid = None
                        board_ctx.repeated_command_count = 0
                        board_ctx.prev_command = None
                        logger.debug("uuid_ack_perf_c=%.3f board=%s ctx=%s kind=%s",
                                     msg_perf_c, board_ctx.target, ctx, board_ctx.kind)
                        if board_ctx.skip_uuid_ack_perf_c:
                            board_ctx.skip_uuid_ack_perf_c = False
                        else:  # if board_ctx.ctx is not None and board_ctx.kind is not None:
                            board_ctx.uuid_ack_perf_c = msg_perf_c
                        if board_ctx.uuid_ack_timeout_engaged:
                            board_ctx.uuid_ack_timeout_engaged = False
                            if not boards_has_ack_timeout_engaged():
                                self.property_changed(
                                    self.UUID_ACK_TIMEOUT_ENGAGED,
                                    False,
                                    True,
                                )
                        break
                if found_board_with_uuid_ack is None:
                    logger.debug("skipping unknown CAN uuid: %s", data)
                    continue
            else:
                if kind is not None:
                    target = self._find_command_next_board_target(kind, data)
                    board = boards_pending_ctx[target]
                    commands_for_board_waiting = any(
                        self._find_command_next_board_target(k, d) == target
                        for (k, d, _, _) in cur_commands
                    )
                    if not commands_for_board_waiting and board.is_available():
                        cur_commands.insert(0, raw)
                    else:
                        cur_commands.append(raw)
            #
            has_compound_left = any(
                board_ctx.compound_steps is not None and len(board_ctx.compound_steps) > 0
                for board_ctx in boards_pending_ctx.values()
            )
            #
            p_now = get_perf_now()
            retrying_board = None
            if kind is _uuid_ack or kind is not None:
                # ensure we don't try to retry a command when we got an uuid ack
                search_retry_boards = {}
            else:
                search_retry_boards = boards_pending_ctx
            for target, board_ctx in sorted(search_retry_boards.items(), key=lambda t: t[1].ack_perf_timeout):
                if board_ctx.uuid is None or p_now < board_ctx.ack_perf_timeout:
                    continue
                logger.warning(
                    "timeout waiting ack previous command: %s ; context=%s ; pending_uuid=%s",
                    board_ctx.kind,
                    board_ctx.ctx,
                    board_ctx.uuid,
                )
                if not board_ctx.uuid_ack_timeout_engaged:
                    # note: checking the "before" value doesn't really matter,
                    # given "property_changed" always relays the value to listeners.
                    before = boards_has_ack_timeout_engaged()
                    board_ctx.uuid_ack_timeout_engaged = True
                    self.property_changed(self.UUID_ACK_TIMEOUT_ENGAGED, True, before)
                board_ctx.repeated_command_count += 1
                if board_ctx.repeated_command_count >= self.default_command_ack_timeout_repeat_count:
                    raise RuntimeError(
                        f"Reached default_command_ack_timeout_repeat_count {board_ctx.repeated_command_count} on board {target}"
                    )
                if board_ctx.prev_command_relative:
                    # TODO: should/could simply continue, probably, although surely only for retract command
                    raise RuntimeError(
                        f"Command {board_ctx.prev_command} uuid ack timed out ; refusing retry given relative."
                    )
                retrying_board = target
                cur_commands.insert(0, board_ctx.prev_command)
                board_ctx.prev_command = None
                board_ctx.uuid = None
                board_ctx.ctx = None  # it's also included in prev_command
                break  # only retry 1 board at a time
            # check for possible _next_compound to process:
            if retrying_board is None and has_compound_left and kind is not _uuid_ack:
                for board_ctx in boards_pending_ctx.values():
                    if board_ctx.uuid is None and board_ctx.compound_steps is not None:
                        cur_commands.insert(0, (_next_compound, (board_ctx.kind, board_ctx.compound_steps), board_ctx.ctx, -math.inf))
                        # don't forget detach (even if temporarily):
                        # or else below is_available() check will say no..
                        board_ctx.compound_steps = None
                        board_ctx.ctx = None
                        board_ctx.prev_command = None
                        break
            #
            if len(cur_commands) == 0:
                continue
            kind, data, ctx, perf_c = cur_commands[0]
            # check if need wait for another board:
            if kind is _uuid_ack:
                assert found_board_with_uuid_ack is not None
                steps = found_board_with_uuid_ack.compound_steps
                if steps:
                    target_board = boards_pending_ctx[self._find_steps_next_board_target(found_board_with_uuid_ack.kind, steps)]
                    if target_board is not found_board_with_uuid_ack and not target_board.is_available():
                        # target board has to finish some operation
                        cur_commands.pop(0)  # still pop it.
                        # but reinsert as _next_compound:
                        cur_commands.append((_next_compound, (found_board_with_uuid_ack.kind, steps), found_board_with_uuid_ack.ctx, perf_c))
                        found_board_with_uuid_ack.ctx = None
                        found_board_with_uuid_ack.kind = None
                        found_board_with_uuid_ack.compound_steps = None
                        continue
                else:
                    found_board_with_uuid_ack.compound_steps = None  # ensure None, always
                    target_board = found_board_with_uuid_ack
            else:
                # sort by availability and oldest first:
                # but only if not uuid_ack.
                cur_commands = sorted(cur_commands, key=sort_available_commands)
                kind, data, ctx, perf_c = cur_commands[0]
                target_board = boards_pending_ctx[self._find_command_next_board_target(kind, data)]
                if not target_board.is_available():
                    logger.spam("target %s not available yet", target_board.target)
                    continue
            # start processing of cur_commands[0]
            cur_commands.pop(0)  # (kind, data, ctx, perf_c) will be pushed back if command need to eventually retry
            if self._want_exit.is_set():
                logger.verbose("Discarding CAN command because shutdown was requested")
                break
            #
            # execute command
            logger.verbose("executing command kind: %s with ctx=%s ; target_board: ctx=%s",
                           kind, ctx, target_board.ctx)
            #
            self._prev_command_is_relative = False  # always before trying new command, it's used on ack timeout
            before_uuid = self._interface.uuid()  # to know if some command has used, or not, a new CAN uuid
            #
            if kind is _retry_compound:
                kind, step, steps = data
                logger.verbose("retrying perform next compound with %s", step)
                steps.insert(0, step)
                data = kind, steps
                kind = _next_compound
            #
            # preset the possible default command (uuid) timeout:
            self._prev_command_timeout = self.default_command_ack_timeout_duration
            #
            if kind is _next_compound:
                kind, steps = data
                target_board.kind = kind
                target_board.compound_steps = steps
                if self._execute_if_active(perform_next_compound, target_board, steps) is _shutdown_requested:
                    break

            elif kind is _uuid_ack:
                assert found_board_with_uuid_ack is not None
                assert target_board is not None
                logger.debug("executing ack perform next compound, board_target=%s",
                             found_board_with_uuid_ack.target)
                # detach current steps:
                kind = found_board_with_uuid_ack.kind
                steps = found_board_with_uuid_ack.compound_steps
                found_board_with_uuid_ack.compound_steps = None
                if steps is not None and len(steps) > 0:
                    # and attach to target:
                    assert target_board.compound_steps is None
                    target_board.compound_steps = steps
                    target_board.ctx = found_board_with_uuid_ack.ctx
                    target_board.kind = found_board_with_uuid_ack.kind
                    if found_board_with_uuid_ack is not target_board:
                        found_board_with_uuid_ack.ctx = None
                        found_board_with_uuid_ack.kind = None
                    if self._execute_if_active(perform_next_compound, target_board, steps) is _shutdown_requested:
                        break
                else:
                    assert target_board is found_board_with_uuid_ack

            else:
                if kind is _retry_full:
                    kind, data = data
                handler = self._command_handlers.get(kind)
                if handler is None:  # actually not anymore necessary,
                    # since we check target_board
                    logger.warning("unhandled command queue message: %s", kind)
                    continue
                success = False
                for _ in range(self.default_command_write_failed_repeat_count):
                    logger.debug("executing cmd %s with ctx %s", kind, ctx)
                    if isinstance(data, SystemDataArgsKwargs):
                        result = self._execute_if_active(handler, *data.args, **data.kwargs)
                    else:
                        result = self._execute_if_active(handler, data)
                    if result is _shutdown_requested:
                        break
                    success = result
                    if success:
                        break
                    logger.error("Failed sending %s to bus", kind)
                if self._want_exit.is_set():
                    break
                if not success:
                    raise RuntimeError(f"Failed writing too many consecutive times to the device/bus. kind={kind} ctx={ctx}")
                target_board.kind = kind  # only used for debug/log
            # end possible handling cases
            if self._want_exit.is_set():
                break
            #
            # get CAN uuid after, to distinguish both cases (with or without uuid used):
            after_uuid = self._interface.uuid()
            #
            if ctx is not None and target_board.ctx != ctx:
                logger.debug("attaching ctx %s to target_board %s", ctx, target_board.target)
                assert target_board.ctx is None or ctx == target_board.ctx
                target_board.ctx = ctx
            #
            compound = self._compound_movement
            if compound is not None and len(compound) > 0:  # on start_sequence commands
                assert target_board.compound_steps is None, f"{target_board.compound_steps=}"
                target_board.compound_steps = compound
                self._compound_movement = None
            #
            target_board.prev_command_relative = self._prev_command_is_relative
            self._prev_command_is_relative = False
            #
            if after_uuid != before_uuid:
                assert target_board.target is not None, (kind, ctx,)
                assert target_board.uuid is None
                #
                t_perf_last_command_with_uuid = get_perf_now()
                # for now we have this rule:
                if after_uuid != before_uuid + 1 and (before_uuid != 255 or after_uuid != 1):
                    # but this can eventually happens if/when we retry several time the same (write-) command
                    logger.warning("Unexpected uuid change count: before=%s after=%s", before_uuid, after_uuid)
                #
                prev_command = self._prev_command
                self._prev_command = None
                if prev_command is None:  # given compound step do set it itself
                    prev_command = (_retry_full, (kind, data), ctx, perf_c)
                else:
                    assert prev_command[0] is _retry_compound
                    prev_command[1][0] = kind
                    prev_command[2] = ctx  # ensure it keeps the context/token as well
                    prev_command[3] = perf_c  # and the perf_c as well.
                target_board.uuid = after_uuid
                command_timeout_delay = self._prev_command_timeout or self.default_command_ack_timeout_duration
                target_board.ack_perf_timeout = t_perf_last_command_with_uuid + command_timeout_delay
                self._prev_command_timeout = self.default_command_ack_timeout_duration
                target_board.prev_command = tuple(prev_command)
            else:
                # no uuid generated
                compound_steps = target_board.compound_steps
                has_compound_left = compound_steps is not None and len(compound_steps) > 0
                if not has_compound_left:
                    target_board.compound_steps = None
                    logger.success("finished executing %s ; target_board=%s ctx=%s board=%s perf_c=%.3f",
                                   kind, target_board.target, ctx, target_board.ctx, target_board.uuid_ack_perf_c)
                    if ctx is not None:
                        self._acknowledge_command(ctx, perf_c=target_board.uuid_ack_perf_c)
                        target_board.ctx = None
                    target_board.kind = None

    def _handle_ack(self, msg: Acknowledge):
        cur_can_uuid = self._interface.uuid()
        perf_c = msg.perf_c
        logger.debug("Received ack: target=%s - uuid=%s ; cur_can_uuid=%s ; perf_c=%.3f",
                     msg.target, msg.uuid, cur_can_uuid, perf_c)
        self._put_to_cmd_queue((_uuid_ack, (msg.uuid, perf_c), None))

    @property
    def api(self):
        """
        Get the current device API instance.

        Returns:
            The current DeviceApi instance
        """
        return self._api

    @api.setter
    def api(self, value: DeviceApi):
        """
        Set the device API instance.

        Args:
            value: The DeviceApi instance to use
        """
        self._api = value

    @property
    def connected(self) -> bool:
        iface = self._interface
        thread = self._commands_handler_thread
        return iface is not None and iface.is_open and thread is not None and thread.is_alive()

    def connect(self):
        # only start the command handler thread on connect,
        # which means we have already obtained the addr of desired devices.
        if self._commands_handler_thread is not None:
            logger.verbose("CAN command Handler thread already alive")
            self.disconnect()

        self._want_exit.clear()
        self._prev_command_timeout = self.default_command_ack_timeout_duration
        self._clear_caches()
        self._init_default_move_configs()

        logger.info("Starting CanCommandHandler thread handler")
        self._commands_handler_watchdog_perf_c = time.perf_counter()  # get_perf_now()
        thread = threading.Thread(
            target=self._command_handler, name="CanCommandHandler", daemon=True)
        thread.start()
        self._commands_handler_thread = thread  # only assign after start
        thread = self._tunnel_pellet_status_check_thread
        if thread is not None and thread.is_alive():
            logger.debug("TunnelPelletStatus check thread already alive")
        else:
            thread = threading.Thread(target=self._check_tunnel_pellet_status_age, name="CheckTunnelPelletStatus", daemon=True)
            thread.start()
            self._tunnel_pellet_status_check_thread = thread

    def _clear_pending_commands(self):
        cmd_queue = self._commands_queue
        while True:
            try:
                cmd_queue.get_nowait()
            except queue.Empty:
                break
            else:
                cmd_queue.task_done()

        self._compound_movement = None
        self._prev_command = None
        self._prev_command_is_relative = False
        self._prev_command_timeout = self.default_command_ack_timeout_duration
        for board in self._boards_pending_ctx.values():
            board.ctx = None
            board.kind = None
            board.uuid = None
            board.uuid_ack_perf_c = -math.inf
            board.skip_uuid_ack_perf_c = False
            board.ack_perf_timeout = math.inf
            board.prev_command = None
            board.prev_command_relative = False
            board.uuid_ack_timeout_engaged = False
            board.repeated_command_count = 0
            board.compound_steps = None

    def disconnect(self):
        with self._disconnect_lock:
            self._want_exit.set()
            with self._command_execution_lock:
                pass
            self._clear_pending_commands()

            cmd_thread, cmd_queue = self._commands_handler_thread, self._commands_queue
            if cmd_thread is not None:
                if cmd_thread.is_alive():
                    cmd_queue.put(None)
                if cmd_thread is not threading.current_thread():
                    cmd_thread.join(3)
                if cmd_thread.is_alive() and cmd_thread is not threading.current_thread():
                    logger.warning("CanCommand handler thread still alive: %s", cmd_thread)
                self._commands_handler_thread = None

            thread = self._tunnel_pellet_status_check_thread
            if thread is not None:
                if thread is not threading.current_thread():
                    thread.join(3)
                if thread.is_alive() and thread is not threading.current_thread():
                    logger.warning("PelletTunnel check thread still alive")
                self._tunnel_pellet_status_check_thread = None

            self._clear_pending_commands()

    def _start_sequence(self, movements: MotorSteps) -> bool:
        """
        Start a sequence of activities.

        Args:
            movements: The motor/device steps to execute
        """
        move_steps = movements.steps
        logger.notice("Starting sequence %s (%s steps): %s", movements.name, len(move_steps), move_steps)
        assert self._compound_movement is None or len(self._compound_movement) == 0
        self._compound_movement = move_steps
        tgt = self._find_steps_next_board_target("sequence", move_steps)
        board = self._boards_pending_ctx[tgt]
        return self._perform_next_compound_step(board, move_steps)

    def _find_step_board(self, step) -> Optional[Target]:
        if 'x' in step or 'x_rel' in step or 'send_x_rel' in step:
            motor = Motor.PELLET_X_MOTOR
        elif 'y' in step or 'y_rel' in step or 'send_y_rel' in step:
            motor = Motor.PELLET_Y_MOTOR
        elif 'z' in step or 'z_rel' in step or 'send_z_rel' in step:
            motor = Motor.PELLET_Z_MOTOR
        elif 'load_arm' in step:
            motor = Motor.PELLET_LOAD_SERVO
        elif 'barrier_arm' in step:
            motor = Motor.PELLET_COVER_SERVO
        elif 'gate' in step:
            motor = Motor.TUNNEL_GATE_SERVO
        elif 'magnet' in step:
            motor = Motor.TUNNEL_MAGNET_SERVO
        elif '_servo_move' in step:
            motor = step['_servo_move'][0]
        elif '_servo_max_pos' in step:
            motor = step['_servo_max_pos']
        elif '_servo_min_pos' in step:
            motor = step['_servo_min_pos']
        elif 'delay' in step:
            motor = Motor.DELAY
        elif 'tone' in step:
            motor = Motor.TONE
        elif 'servo_attach' in step:
            motor = step['servo_attach']
        elif 'servo_detach' in step:
            motor = step['servo_detach']
        elif 'home' in step:
            motor = step['home']
        elif '_internal_func_motor' in step:
            motor = step['_internal_func_motor']
        elif 'predefined' in step:
            predef = step['predefined']
            if predef in {'send', 'cover', 'release', 'retrieve', 'home', 'scoop'}:
                # single one where we don't use motor = ..
                # but they are all on PELLET board.
                return Target.PELLET_DEVICE
            return None
        else:
            return None
        return target_of_motor(motor)

    def _find_steps_next_board_target(self, kind, steps) -> Optional[Target]:
        if len(steps) == 0:
            logger.warning("find_steps_next_board: got empty steps ; kind=%s", kind)
            return None
        for step in steps:
            tgt = self._find_step_board(step)
            if tgt is not None:
                return tgt
        raise ValueError(f"Found no target board for kind={kind} steps: {steps}")

    def _find_command_next_board_target(self, kind, data) -> Optional[Target]:
        # NB: following is kind of fragile:
        # would need update if at least some of the devices change of board(target)
        if kind is _next_compound:
            kind, steps = data
            return self._find_steps_next_board_target(kind, steps)
        elif kind is _retry_compound:
            kind, step, steps = data
            return self._find_steps_next_board_target(kind, [step] + steps)
        elif kind is _retry_full:
            kind, data = data
            return self._find_command_next_board_target(kind, data)
        elif kind in {
            SystemCommandKind.SET_DIGITAL_OUTPUT,
            SystemCommandKind.SET_ANALOG_OUTPUT,
            SystemCommandKind.SET_RGB_LED,
            SystemCommandKind.MOVE_X,
            SystemCommandKind.MOVE_Y,
            SystemCommandKind.MOVE_Z,
            SystemCommandKind.SET_X,
            SystemCommandKind.SET_Y,
            SystemCommandKind.SET_Z,
            SystemCommandKind.COVER_PELLET,
            SystemCommandKind.RELEASE_PELLET,
            SystemCommandKind.LOAD_PELLET,
            SystemCommandKind.SEND_PELLET,
            SystemCommandKind.SEND_HOME,
            SystemCommandKind.MOVE_COVER_SERVO,
            SystemCommandKind.MOVE_LOAD_SERVO,
            SystemCommandKind.SEND_TO_LIMITS,
            SystemCommandKind.SEND_RETRACT,
            SystemCommandKind.SEND_FIXED_XYZ,
        }:
            return Target.PELLET_DEVICE
        elif kind == SystemCommandKind.MOVE_MAGNET_SERVO:
            motor = Motor.TUNNEL_MAGNET_SERVO
        elif kind in {
            SystemCommandKind.TUNNEL_FAN_ON,
            SystemCommandKind.TUNNEL_FAN_OFF,
        }:
            motor = Motor.TUNNEL_FAN_SERVO
        elif kind in {
            SystemCommandKind.MOVE_GATE_SERVO,
            SystemCommandKind.OPEN_TUNNEL_GATE,
            SystemCommandKind.CLOSE_TUNNEL_GATE,
        }:
            motor = Motor.TUNNEL_GATE_SERVO
        elif kind == SystemCommandKind.DELAY:
            motor = Motor.DELAY
        elif kind == SystemCommandKind.PLAY_TONE:
            motor = Motor.TONE
        elif kind == SystemCommandKind.WRITE_MOTOR_CONFIGURATION:
            motor = data[0]
        elif kind == SystemCommandKind.READ_MOTOR_CONFIGURATION:
            motor = data
        elif kind == SystemCommandKind.SERVO_ATTACH:
            motor = data
        elif kind == SystemCommandKind.SERVO_DETACH:
            motor = data
        elif kind == SystemCommandKind.BOARD_REBOOT:
            return data
        elif kind == SystemCommandKind.REQUEST_VERSION:
            # It does not use uuid, so there is no single board pending context to track.
            return None
        elif kind in {SystemCommandKind.STREAM_START, SystemCommandKind.STREAM_STOP}:
            # is no CAN operation
            return None
        elif kind in {
            SystemCommandKind.SET_LOAD_PELLET_PROCEDURE,
            SystemCommandKind.SET_SEND_PELLET_PROCEDURE,
            SystemCommandKind.SET_COVER_PELLET_PROCEDURE,
            SystemCommandKind.SET_RELEASE_PELLET_PROCEDURE,
            SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE,
        }:
            # is no CAN operation
            return None
        elif kind in {
            SystemCommandKind.SET_MOTOR_DRIFT, SystemCommandKind.SET_AUTO_CORRECT_DRIFT
        }:
            # no CAN operation, also currently unused.
            return None
        else:
            raise ValueError(f"Invalid kind for _find_command_next_board: {kind}")
        return target_of_motor(motor)

    def _handle_write_motor_configuration(self, data: Tuple[Motor, Union[StepperConfig, ServoConfig]]):
        """
        Handle writing motor configuration.

        Args:
            data: A tuple containing motor and config
        """
        # assert isinstance(data, Tuple)
        motor = data[0]
        config = data[1]
        # assert isinstance(motor, Motor)
        # assert isinstance(config, (ServoConfig, StepperConfig))
        success = self._interface.set_motor_configuration(motor, config)
        if success:
            logger.debug("setting internal motor %s cfg to %s", motor, config)
            self._motor_configs[motor] = copy.deepcopy(config)
        return success

    def notify_message(
        self,
        kind: SystemCommandKind,
        data: Any,  # Union[str, float, int, SupportsInt],
        context: object = None,
    ) -> None:
        """
        This method is called when a command to a target is requested. This method
        translates the application command to the appropriate call to the CanInterface
        instance.
        
        Args:
            kind: The kind of system command
            data: The data associated with the command
            context: The context object for this command
        """
        if self._interface is None:
            return

        self._put_to_cmd_queue((kind, data, context))
        return

    def notify_data(self, data: Any) -> None:
        """
        This method is called when data from the target is received. The data is
        forwarded to a DeviceAPI class.

        Args:
            data: The data received from the device
        """
        if self._api is None:
            return

        for message in data:
            # Get handler for the message type
            handler = self._data_handlers.get(type(message))
            if handler is not None:
                handler(message)
            else:
                logger.warning("Unhandled data type: %s", type(message))

    def _report_stepper_status(self, message: StepperStatus):
        """
        Report stepper status to the API.

        Args:
            message: StepperStatus
        """
        motor_idx = self._motor_to_coordinate_idx.get(message.motor, None)
        if motor_idx is not None:
            prev_limit_switch = self._last_limit_switch[message.motor]
            last_pos = list(self._last_pellet_pos)
            last_send_pos = list(self._last_send_pos)
            if message.is_at_limit != prev_limit_switch:
                logger.notice("%s: limit_switch: %s -> %s ; pos=%.02f (last=%.02f) send_pos=%.02f",
                              message.motor, prev_limit_switch, message.is_at_limit,
                              message.position, last_pos[motor_idx], message.send_position)
                self._last_limit_switch[message.motor] = message.is_at_limit
            last_pos[motor_idx] = message.position
            last_send_pos[motor_idx] = message.send_position
            self._last_pellet_pos = Offset3DTuple(last_pos)
            self._last_send_pos = Offset3DTuple(last_send_pos)
        kind = CanDevice._motor_to_status_kind.get(message.motor, None)
        api = self._api
        if api is not None and kind is not None:
            prev_data, prev_perf_c = self._previous_stepper_status_pos_perf_c.get(kind, (None, -math.inf))
            perf_now = get_perf_now()
            data = (message.position, message.send_position, message.is_at_limit, message.position_error)
            if data != prev_data or perf_now - prev_perf_c > self.same_data_refresh_delay:
                self._previous_stepper_status_pos_perf_c[kind] = (data, perf_now)
                api.send_message(kind, message)

    def _report_servo_status(self, message: ServoStatus):
        """
        Report servo status to the API.

        Args:
            motor: The motor that has reported its status
            position: The current position of the motor
        """
        motor = message.motor
        position = message.position
        kind = CanDevice._motor_to_status_kind.get(motor, None)
        if kind is None:
            return
        api = self._api
        # if self._api is not None and kind is not None:
        prev_data, prev_perf_c = self._previous_servo_status_pos_perf_c.get(kind, (None, -math.inf))
        perf_now = get_perf_now()
        if prev_data != position or perf_now - prev_perf_c > self.same_data_refresh_delay:
            self._previous_servo_status_pos_perf_c[kind] = (position, perf_now)
            api.send_message(kind, position)
        if motor == Motor.TUNNEL_GATE_SERVO:
            gate_cfg = self._motor_configs[motor]
            new_open = math.isclose(position, gate_cfg.minimum_position, abs_tol=0.5)
            prev_open, prev_perf_c = self._prev_tunnel_gate_open_perf_c
            if new_open != prev_open or perf_now - prev_perf_c > self.same_data_refresh_delay:
                self._prev_tunnel_gate_open_perf_c = (new_open, perf_now)
                api.send_message(SystemStatusMessageKind.TUNNEL_GATE_OPEN_STATUS, new_open)

    #

    def _make_servo_steps(self, motor: Motor) -> Tuple[Dict, List[Dict]]:
        cfg = self._motor_configs[motor]
        assert isinstance(cfg, ServoConfig)
        want_detach = cfg.detach
        step = {}
        if cfg.uuid_ack_timeout is not None:
            step['uuid_ack_timeout'] = cfg.uuid_ack_timeout
        if want_detach:
            return step, [{'servo_attach': motor}, step, {'servo_detach': motor}]
        return step, [step]

    def _make_servo_move_steps(self, motor, position):
        step, steps = self._make_servo_steps(motor)
        step['_servo_move'] = (motor, position)
        return steps

    def _make_servo_iface_func_steps(self, motor, func):
        step, steps = self._make_servo_steps(motor)
        step['_internal_func'] = func
        step['_internal_func_motor'] = motor
        return step, steps

    def _handle_servo_move_compound(self, compound_movements, motor, position):
        steps = self._make_servo_move_steps(motor, position)
        # replace current compound move by this new list of steps:
        # the [0] = : replace whatever previous step leaded to this _handle_servo_move_compound,
        # by one doing nothing now:
        compound_movements[0] = {
            '_internal_func': _no_op,
            '_internal_func_motor': motor,
        }  # noqa
        compound_movements[1:1] = steps
        logger.verbose("injected steps: new=%s", compound_movements)
        return True

    def _handle_servo_iface_cmd_compound(self, compound_movements, motor, iface_func):
        step, steps = self._make_servo_iface_func_steps(motor, iface_func)
        # replace current compound move by this new list of steps:
        # the [0] = : replace whatever previous step leaded to this _handle_servo_iface_cmd_compound,
        # by one doing nothing now:
        compound_movements[0] = {
            '_internal_func': _no_op,
            '_internal_func_motor': motor,
        }  # noqa
        compound_movements[1:1] = steps
        return True

    def _perform_send_rel_move(self, step: Dict[str, Any]):
        if 'send_x_rel' in step:
            rel_val = step['send_x_rel']
            motor = Motor.PELLET_X_MOTOR
            meth = self._interface.move_motor_x
            m_idx = 0
        elif 'send_y_rel' in step:
            rel_val = step['send_y_rel']
            motor = Motor.PELLET_Y_MOTOR
            meth = self._interface.move_motor_y
            m_idx = 1
        else:
            assert 'send_z_rel' in step
            rel_val = step['send_z_rel']
            motor = Motor.PELLET_Z_MOTOR
            meth = self._interface.move_motor_z
            m_idx = 2
        # calculate position:
        new_pos = self._last_send_pos[m_idx] + rel_val
        if not math.isfinite(new_pos):
            logger.warning(
                "skipping %s given last known send_position not finite (%s)",
                step, self._last_send_pos,
            )
            # or should we move to 0 instead ?
            return motor, True
        return motor, meth(new_pos)

    def _perform_next_compound_step(self, board: _BoardPendingContext, compound_movements: List[Dict[str, Any]]) -> bool:
        """
        Issue the next step in a multi-step motor sequence.
        """
        step = compound_movements[0]
        orig_step = step.copy()
        logger.debug("executing next compound step: %s (remains after=%s)",
                     step,
                     len(compound_movements) - 1,  # don't include current one
                     )

        assert isinstance(step, dict)
        save_as_fixed = step.get("save_as_fixed", False)

        motor = None
        before_uuid = self._interface.uuid()

        if 'x' in step:
            motor = Motor.PELLET_X_MOTOR
            location = _to_tuple(step['x'])
            success = self._interface.move_motor_x(location, save_as_fixed=save_as_fixed)

        elif 'y' in step:
            motor = Motor.PELLET_Y_MOTOR
            location = _to_tuple(step['y'])
            success = self._interface.move_motor_y(location, save_as_fixed=save_as_fixed)

        elif 'z' in step:
            motor = Motor.PELLET_Z_MOTOR
            location = _to_tuple(step['z'])
            success = self._interface.move_motor_z(location, save_as_fixed=save_as_fixed)

        elif '_servo_move' in step:  # should be internal only
            motor, position = step['_servo_move']  # noqa
            success = self._interface.move_servo_motor(motor, position)

        elif 'load_arm' in step:
            motor = Motor.PELLET_LOAD_SERVO
            location = _to_tuple(step['load_arm'])
            success = self._handle_servo_move_compound(compound_movements, motor, location)

        elif 'barrier_arm' in step:
            motor = Motor.PELLET_COVER_SERVO
            location = _to_tuple(step['barrier_arm'])
            success = self._handle_servo_move_compound(compound_movements, motor, location)

        elif 'magnet' in step:
            motor = Motor.TUNNEL_MAGNET_SERVO
            location = _to_tuple(step['magnet'])
            success = self._handle_servo_move_compound(compound_movements, motor, location)

        elif 'gate' in step:
            motor = Motor.TUNNEL_GATE_SERVO
            location = _to_tuple(step['gate'])
            success = self._handle_servo_move_compound(compound_movements, motor, location)

        # _servo_max_pos / _servo_max_pos are internal and should not be used from outside.
        elif '_servo_max_pos' in step:
            motor: Motor = step['_servo_max_pos']  # noqa
            cfg = self._motor_configs[motor]
            success = self._interface.move_servo_motor(motor, cfg.maximum_position)

        elif '_servo_min_pos' in step:
            motor: Motor = step['_servo_min_pos']  # noqa
            cfg = self._motor_configs[motor]
            success = self._interface.move_servo_motor(motor, cfg.minimum_position)

        elif 'delay' in step:
            motor = Motor.DELAY
            duration = step['delay']
            success = self._interface.delay(duration)

        elif 'tone' in step:
            motor = Motor.TONE
            freq, duration = step['tone'].split(',')  # noqa  # (hz), (sec)
            success = self._interface.emit_tone(int(freq), int(float(duration) * 1000))
            if success:
                board.skip_uuid_ack_perf_c = True

        elif 'predefined' in step:

            predefined = step['predefined']

            if predefined == 'send':
                motor = Motor.PELLET_X_MOTOR  # could choose Y or Z
                success = self._interface.fixed_position()

            elif predefined == 'cover':
                motor = Motor.PELLET_COVER_SERVO
                success = self._handle_servo_iface_cmd_compound(compound_movements, motor, self._interface.cover_pellet)

            elif predefined == 'release':
                motor = Motor.PELLET_COVER_SERVO
                success = self._handle_servo_iface_cmd_compound(compound_movements, motor, self._interface.release_pellet)

            elif predefined == 'retrieve':
                motor = Motor.PELLET_LOAD_SERVO
                success = self._handle_servo_iface_cmd_compound(compound_movements, motor, self._interface.retrieve_pellet)

            elif predefined == 'scoop':  # NB: unused
                motor = Motor.PELLET_LOAD_SERVO
                success = self._handle_servo_iface_cmd_compound(compound_movements, motor, self._interface.scoop_pellet)

            elif predefined == 'home':
                # replace by home steps (not predefined->home), 1 for each XYZ :
                motor = Motor.PELLET_Y_MOTOR
                step = {'home': motor}  # for possible retry on ack timeout
                compound_movements[0] = step
                # inject at index 1 the steps:
                compound_movements[1:1] = [
                    {'home': m}
                    for m in [Motor.PELLET_Z_MOTOR, Motor.PELLET_X_MOTOR]
                ]
                logger.debug("initiated home steps: %s", compound_movements)
                success = self._interface.stepper_home(motor)

            else:
                raise RuntimeError(f"Received unknown/unhandled compound step: {step}")

        elif 'servo_attach' in step:
            motor: Motor = step['servo_attach']  # noqa
            success = self._interface.servo_attach(motor)

        elif 'servo_detach' in step:
            motor: Motor = step['servo_detach']  # noqa
            success = self._interface.servo_detach(motor)

        elif 'home' in step:
            # NB: to not mix with predefined->home, which is 3 times this home but each with separate stepper.
            motor: Motor = step['home']  # noqa
            success = self._interface.stepper_home(motor)

        elif '_internal_func' in step:
            func = step['_internal_func']
            motor: Motor = step['_internal_func_motor']  # noqa
            success = func()  # noqa

        elif 'send_x_rel' in step or 'send_y_rel' in step or 'send_z_rel' in step:
            motor, success = self._perform_send_rel_move(step)

        elif 'x_rel' in step:
            motor = Motor.PELLET_X_MOTOR
            success = self._interface.move_motor_x(step['x_rel'], relative=True)

        elif 'y_rel' in step:
            motor = Motor.PELLET_Y_MOTOR
            success = self._interface.move_motor_y(step['y_rel'], relative=True)

        elif 'z_rel' in step:
            motor = Motor.PELLET_Z_MOTOR
            success = self._interface.move_motor_z(step['z_rel'], relative=True)

        else:
            raise RuntimeError(f"Received unknown/unhandled compound step: {step}")

        assert motor is not None, "all possible compound steps relate to a specific Motor"

        if success:
            if 'x_rel' in step or 'y_rel' in step or 'z_rel' in step:
                self._prev_command_is_relative = True

            after_uuid = self._interface.uuid()
            compound_movements.pop(0)  # remove the one that was just executed successfully
            logger.debug("executed %s write command ; uuid: before=%s after=%s",
                         orig_step, before_uuid, after_uuid)
            if after_uuid != before_uuid:
                # in case need for retry for ack timeout:
                self._prev_command = [_retry_compound, [None, step, compound_movements], None, None]
                # prefer eventual step.uuid_ack_timeout over motor_config.uuid_ack_timeout:
                cmd_ack_timeout = step.get('uuid_ack_timeout')
                if cmd_ack_timeout is None:
                    if motor is not None:
                        motor_cfg = self._motor_configs.get(motor)
                        if motor_cfg is not None:
                            cmd_ack_timeout = motor_cfg.uuid_ack_timeout
                if cmd_ack_timeout is not None:
                    self._prev_command_timeout = cmd_ack_timeout
        else:
            logger.error("%s write command failed", step)

        return success


def default_load_pellet() -> MotorSteps:
    """
    Create the default motor step sequence for loading a pellet.

    Returns:
        A MotorSteps object containing the load pellet sequence
    """
    return MotorSteps("load_pellet",
                      [
                          {'x': 0.0},
                          {'predefined': 'retrieve'},
                          {'delay': 1.0},  # in sec
                          {'z': 5.0},
                          {'predefined': 'scoop'},
                          {'delay': 1.0},
                          {'z': 0.0},  # in mm
                          {'delay': 1.0},
                      ]
                      )


def default_send_pellet() -> MotorSteps:
    """
    Create the default motor step sequence for sending a pellet.

    Returns:
        A MotorSteps object containing the send pellet sequence
    """
    return MotorSteps("send_pellet",
                      [
                          {'predefined': 'send'},
                      ]
                      )


def default_cover_pellet() -> MotorSteps:
    """
    Create the default motor step sequence for covering a pellet.

    Returns:
        A MotorSteps object containing the cover pellet sequence
    """
    return MotorSteps("cover_pellet",
                      [
                          {'predefined': 'cover'},
                      ]
                      )


def default_release_pellet() -> MotorSteps:
    """
    Create the default motor step sequence for releasing a pellet.

    Returns:
        A MotorSteps object containing the release pellet sequence
    """
    return MotorSteps("release_pellet",
                      [
                          {'predefined': 'release'},
                      ]
                      )


def default_open_gate() -> MotorSteps:
    """
    Create the default motor step sequence for open gate.

    Returns:
        A MotorSteps object containing the open gate sequence
    """
    return MotorSteps("open_gate", [{'_servo_min_pos': Motor.TUNNEL_GATE_SERVO}])


def default_close_gate() -> MotorSteps:
    """
    Create the default motor step sequence for close gate.

    Returns:
        A MotorSteps object containing the close gate sequence
    """
    return MotorSteps("close_gate", [{'_servo_max_pos': Motor.TUNNEL_GATE_SERVO}])


def default_move_retract() -> MotorSteps:
    """
    Create the default motor step sequence for move retract.

    Returns:
        A MotorSteps object containing the move retract sequence
    """
    return MotorSteps("move_retract", [
        # NB: need to have each one, even if 0 relative,
        # to make the corresponding axis move to the desired send axis position.
            {"send_z_rel": 0},  # move up/down first
            {"send_x_rel": 0},  # then left/right
            {"send_y_rel": -15},  # then forward/backward
        ],
    )
