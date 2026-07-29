"""
Interface classes that define the capabilities of the mouse gym hardware in a way
that hardware can be swapped or emulated.

There is a set of enumerations that define entities of the hardware:
+ Target - Either the Pellet or Magnet module
+ Motor - Motor that can be controlled
+ DigitalOutputs - Set of available digital outputs
+ AnalogOutputs - Set of available analog outputs

There is a set of data classes that contain state and status of the gym hardware.

There is a main interface class (DeviceInterface) that defines the API for access to the abstracted
hardware.
"""

import dataclasses
import math
import os
import typing
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Union, Optional, Dict, Any, Tuple

from autotrainer.core import Offset3DTuple, get_verbose_logger
from autotrainer.core.message import Motor
from autotrainer.core.message.system_status_message import StepperStatusMessage

logger = get_verbose_logger(__name__)


_map_idx_motors = {
    0: Motor.PELLET_X_MOTOR,
    1: Motor.PELLET_Y_MOTOR,
    2: Motor.PELLET_Z_MOTOR,
}


class Target(IntEnum):
    """Target is also known as board"""
    PELLET_DEVICE = 0
    MAGNET_DEVICE = 1


class DigitalOutputs(IntEnum):
    STIMULUS_1 = 1
    STIMULUS_2 = 2
    STIMULUS_3 = 3
    STIMULUS_4 = 4


class AnalogOutputs(IntEnum):
    STATUS_OUT = 1


@dataclass
class Source:
    """
    Base class of any data set received by the device
    """
    target: Target = None
    timestamp_ns: int = dataclasses.field(init=False, default=0)  # realtime unix timestamp but in integer nanoseconds
    index: int = dataclasses.field(init=False, default=0)  # perf counter in integer nanosecond
    # init=False: preserve the original behavior/semantic of constructor with position args
    # for subclasses adding other fields.

    @property
    def perf_c(self) -> float:
        return self.index / 1e9


@dataclass
class MotorSource(Source):
    motor: Motor = Motor.NONE


@dataclass
class Heartbeat(Source):
    unused: bool = False


@dataclass
class MagnetDigitalInputs(Source):
    continuity_0: bool = False
    continuity_1: bool = False


@dataclass
class PelletDigitalInputs(Source):
    stimulus_1: bool = False
    stimulus_2: bool = False
    stimulus_3: bool = False
    stimulus_4: bool = False


@dataclass
class ServoConfig(MotorSource):
    _minimum_position: float = 0  # (deg)
    _maximum_position: float = 120  # (deg)
    _minimum_pwm_duration: float = 1000  # (us)
    _maximum_pwm_duration: float = 2000  # (us)
    _maximum_velocity: float = 200  # (deg/sec)
    _maximum_acceleration: float = 100.0  # (deg/sec^2)
    _detach: bool = False
    uuid_ack_timeout: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        config = ServoConfig()
        if "min_pos" in data:
            config.minimum_position = data["min_pos"]
        if "max_pos" in data:
            config.maximum_position = data["max_pos"]
        if "min_pwm" in data:
            config.minimum_pwm_duration = data["min_pwm"]
        if "max_pwm" in data:
            config.maximum_pwm_duration = data["max_pwm"]
        if "max_vel" in data:
            config.maximum_velocity = data["max_vel"]
        if "max_acc" in data:
            config.maximum_acceleration = data["max_acc"]
        if "detach" in data:
            config.detach = data["detach"]
        config.uuid_ack_timeout = data.get('uuid_ack_timeout')
        return config

    @property
    def maximum_velocity(self) -> float:
        return self._maximum_velocity

    @maximum_velocity.setter
    def maximum_velocity(self, value: float):
        self._maximum_velocity = value

    @property
    def maximum_acceleration(self) -> float:
        return self._maximum_acceleration

    @maximum_acceleration.setter
    def maximum_acceleration(self, value: float):
        self._maximum_acceleration = value

    @property
    def minimum_position(self) -> float:
        return self._minimum_position

    @minimum_position.setter
    def minimum_position(self, value: float):
        self._minimum_position = value

    @property
    def maximum_position(self) -> float:
        return self._maximum_position

    @maximum_position.setter
    def maximum_position(self, value: float):
        self._maximum_position = value

    @property
    def minimum_pwm_duration(self) -> float:
        return self._minimum_pwm_duration

    @minimum_pwm_duration.setter
    def minimum_pwm_duration(self, value: float):
        self._minimum_pwm_duration = value

    @property
    def maximum_pwm_duration(self) -> float:
        return self._maximum_pwm_duration

    @maximum_pwm_duration.setter
    def maximum_pwm_duration(self, value: float):
        self._maximum_pwm_duration = value

    @property
    def detach(self) -> bool:
        return self._detach

    @detach.setter
    def detach(self, value):
        self._detach = value


@dataclass
class ServoStatus(Source):
    motor: Motor = Motor.NONE
    position: float = 0  # (deg)

    def __init__(self, target: Target, motor: Motor, position: float):
        super().__init__(target)
        self.motor = motor
        self.position = position

    @property
    def location(self) -> float:
        """Current servo position in degrees."""
        return self.position

    @property
    def status(self) -> int:
        """Current servo status value."""
        return 0


@dataclass
class StepperConfig(MotorSource):
    _microsteps: int = 64
    _steps_per_revolution: float = 48.0
    _maximum_velocity: float = 61  # mm/sec
    _maximum_acceleration: float = 244  # mm/sec^2
    _flip_limit_orientation: bool = False
    _homing_velocity: float = 60  # mm/sec
    uuid_ack_timeout: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict):
        config = StepperConfig()

        if "microsteps" in data:
            config.microsteps = data["microsteps"]
        if "steps_per_revolution" in data:
            config.steps_per_revolution = data["steps_per_revolution"]
        if "max_vel" in data:
            config.maximum_velocity = data["max_vel"]
        if "max_acc" in data:
            config.maximum_acceleration = data["max_acc"]
        if "home_vel" in data:
            config.homing_velocity = data["home_vel"]
        if "flip_limit_orientation" in data:
            config.flip_limit_orientation = data["flip_limit_orientation"] == 1
        config.uuid_ack_timeout = data.get("uuid_ack_timeout")
        return config

    @property
    def motor(self) -> Motor:
        return self._motor

    @motor.setter
    def motor(self, value: Motor):
        self._motor = value

    @property
    def maximum_velocity(self) -> float:
        return self._maximum_velocity

    @maximum_velocity.setter
    def maximum_velocity(self, value: float):
        self._maximum_velocity = value

    @property
    def maximum_acceleration(self) -> float:
        return self._maximum_acceleration

    @maximum_acceleration.setter
    def maximum_acceleration(self, value: float):
        self._maximum_acceleration = value

    @property
    def homing_velocity(self) -> float:
        return self._homing_velocity

    @homing_velocity.setter
    def homing_velocity(self, value: float):
        self._homing_velocity = value

    @property
    def microsteps(self) -> int:
        return self._microsteps

    @microsteps.setter
    def microsteps(self, value: int):
        self._microsteps = value

    @property
    def steps_per_revolution(self) -> float:
        return self._steps_per_revolution

    @steps_per_revolution.setter
    def steps_per_revolution(self, value: float):
        self._steps_per_revolution = value

    @property
    def flip_limit_orientation(self) -> bool:
        return self._flip_limit_orientation

    @flip_limit_orientation.setter
    def flip_limit_orientation(self, value: bool):
        self._flip_limit_orientation = value


@dataclass
class StepperStatus(Source, StepperStatusMessage):
    _motor: Motor = Motor.NONE
    _position: float = 0  # (mm)
    _send_position: float = 0 # (mm)
    _limit_switch: bool = False
    position_error: bool = False

    def __init__(
        self,
        target: Target,
        motor: Motor,
        position: float,
        send_position: float,
        limit_switch: bool,
        *,
        position_error: bool = False,
    ):
        super().__init__(target)

        self._motor = motor
        self._position = position
        self._send_position = send_position
        self._limit_switch = limit_switch
        self.position_error = position_error

    @property
    def motor(self) -> Motor:
        return self._motor

    @property
    def position(self) -> float:
        """Current motor position in mm."""
        return self._position
    
    @property
    def send_position(self) -> float:
        """Current send motor position in mm."""
        return self._send_position

    @property
    def is_at_limit(self) -> bool:
        return self._limit_switch


@dataclass
class Tone(Source):
    time_remaining_ms: int = 0
    frequency_hz: int = 0


@dataclass
class AnalogOutput(Source):
    status_out_mv: int = 0


@dataclass
class PressureReading(Source):
    pressure: float = 0


@dataclass
class ColorLed(Source):
    """As integer % [0, 100] value"""

    red: int = 0  # [0, 100]
    green: int = 0  # [0, 100]
    blue: int = 0  # [0, 100]


@dataclass
class AudioData(Source):
    packet_id: int = 0
    when: float = 0  # real-time unix timestamp, also in Source (Source.timestamp_ns, so as ns actually)
    # but keeping here for now, this could become a property, eventually with setter.
    index: int = 0  # also in Source now, keeping also for now, as it allows to pass in constructor/init
    magnitudes: List[float] = dataclasses.field(default_factory=list)


@dataclass
class DoorData(Source):
    door1: bool = False
    door2: bool = False
    door3: bool = False
    ext_button: bool = False


@dataclass
class Status(Source):
    unused: bool = True


@dataclass
class SensorStatus(Source):
    temperature_c: float = 0
    humidity_percent: float = 0


@dataclass
class Version(Source):
    version: str = "Unknown"


@dataclass
class Acknowledge(Source):
    uuid: int = 0


_zero_position = Offset3DTuple(0, 0, 0)


_device_float_precision = os.getenv("AUTOTRAINER_DEVICE_FLOAT_PRECISION", "2")
if _device_float_precision is not None and len(_device_float_precision) > 0:
    _device_float_precision = int(_device_float_precision)
else:
    _device_float_precision = None


class DeviceInterface:
    """Base class that provides low-level communication with a device, such as serial, or CAN"""

    float_precision: Optional[int] = _device_float_precision

    _motors_prev_warn_error = {
        m: (False, False) for m in list(Motor)
    }

    def __init__(self):
        super().__init__()
        self._auto_correct_motor_drift = False
        self._motors_drift = _zero_position
        self._active_motors_drift = _zero_position
        self._max_motor_drift_error_threshold = 2  # mm
        self._motors_drift_error = [False, False, False]
        self._tunnel_status_perf_c = -math.inf
        self._pellet_status_perf_c = -math.inf

    @classmethod
    def uuid(cls) -> int:
        raise NotImplementedError

    @classmethod
    def next_uuid(cls) -> int:
        raise NotImplementedError

    @property
    def motors_position(self) -> Offset3DTuple:
        raise NotImplementedError

    def round_float(self, value: float) -> float:
        return value if self.float_precision is None else round(value, self.float_precision)

    def open(self) -> bool:
        """ Opens the interface

        This method should not raise an exception and return False instead.  Exception details may be logged.

        :return: True if successfully opened
        """
        return False

    def close(self):
        pass

    @property
    def is_open(self) -> bool:
        return False

    def are_addresses_valid(self):
        raise NotImplementedError

    def can_read(self) -> bool:
        pass

    def read(self, max_count: int = 1, *, collect_ms: int = 0) -> typing.Any:
        """ Reads the available number of values from the interface up to max_count

        :param max_count: maximum number of values to read
        :param collect_ms: maximum duration to read until, if <= 0 then only 1 attempt is made.
        :returns typing.Any: the data
        :rtype: typing.Any
        """
        pass

    def write(self, value: typing.Any) -> int:
        """ Writes the content value(s) to the interface

        :param value: The content to be written
        :return: the number of values written"""
        pass

    def write_str(self, value: str) -> int:
        """ Writes the string to the interface

        :param value: The string to be written
        :return: the number of bytes written"""
        pass

    def get_motor_configuration(self, motor: Motor) -> Union[ServoConfig, StepperConfig]:
        """Return current motor config"""
        raise NotImplementedError

    def set_auto_correct_motor_drift(self, value):
        """Set the auto correct motor drift"""
        prev = self._auto_correct_motor_drift
        no_drift = Offset3DTuple(0, 0, 0)
        # if not value and prev:
        #     self.set_motors_drift(no_drift)
        if value != prev:
            logger.verbose("auto_correct_motor_drift: %s -> %s", prev, value)
            self._auto_correct_motor_drift = value
            if not value:
                self._motors_drift = no_drift
        return True

    def move_motor_x(
        self,
        position: Union[float, Tuple[float, float]],
        save_as_fixed: bool = False,
        *,
        relative: bool = False,
    ) -> bool:
        """
         Move the X-direction motor

        Args:
            position: Either a position (float) or a (position, rate (%)) pair
            save_as_fixed: Save the position as a new fixed location for this motor
                If True then the position is only saved-as-fixed, the motor is not moved.
            relative: Relative movement or absolute, default absolute.

         Returns:
             bool: True if successful else False
         """
        raise NotImplementedError

    def move_motor_y(
        self,
        position: Union[float, Tuple[float, float]],
        save_as_fixed: bool = False,
        *,
        relative: bool = False,
    ) -> bool:
        """
         Move the Y-direction motor

        Args:
            position: Either a position (float) or a (position, rate (%)) pair
            save_as_fixed: Save the position as a new fixed location for this motor
                If True then the position is only saved-as-fixed, the motor is not moved.
            relative: Relative movement or absolute, default absolute.

         Returns:
             bool: True if successful else False
         """
        raise NotImplementedError

    def move_motor_z(
        self,
        position: Union[float, Tuple[float, float]],
        save_as_fixed: bool = False,
        *,
        relative: bool = False,
    ) -> bool:
        """
         Move the Z-direction motor

        Args:
            position: Either a position (float) or a (position, rate (%)) pair
            save_as_fixed: Save the position as a new fixed location for this motor
                If True then the position is only saved-as-fixed, the motor is not moved.
            relative: Relative movement or absolute, default absolute.

         Returns:
             bool: True if successful else False
         """
        raise NotImplementedError

    def set_motors_drift(self, drifts: Offset3DTuple):
        prev_drifts = self._motors_drift
        logger.debug("Received new motors drift: %s ; prev=%s",
                     drifts.humanize(n_digits=3), prev_drifts.humanize(n_digits=3))
        for motor_axis_idx, motor in enumerate((Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)):
            axis_drift = drifts[motor_axis_idx]
            if abs(axis_drift) > self._max_motor_drift_error_threshold:
                logger.critical("%s drift over error threshold: %.3f", motor, axis_drift)
                self._motors_drift_error[motor_axis_idx] = True
                # hopefully keep same drift direction:
                if axis_drift < 0:
                    axis_drift = -self._max_motor_drift_error_threshold
                else:
                    axis_drift = self._max_motor_drift_error_threshold
                drifts = drifts.replace(**{"xyz"[motor_axis_idx]: axis_drift})
            else:
                if self._motors_drift_error[motor_axis_idx]:
                    logger.notice("%s recovered from axis position error ; new drift = %.3f prev = %.3f",
                                  motor, drifts[motor_axis_idx], prev_drifts[motor_axis_idx])
                    self._motors_drift_error[motor_axis_idx] = False
            # Must be called via SystemCommandKind message:
            # save-as-fixed with 0 relative,
            # this will make the current saved-as-fixed to be auto-corrected:
            # self.move_motor(motor, 0, relative=True, save_as_fixed=True)
        # end motors loop
        # previous loop could have modified drifts, so assign after the loop:
        self._motors_drift = drifts
        return True

    def servo_attach(self, motor: Motor):
        raise NotImplementedError

    def servo_detach(self, motor: Motor):
        raise NotImplementedError

    def set_tunnel_fan_on(self) -> bool:
        raise NotImplementedError

    def set_tunnel_fan_off(self) -> bool:
        raise NotImplementedError

    @property
    def pellet_status_perf_c(self) -> float:
        return self._pellet_status_perf_c

    @pellet_status_perf_c.setter
    def pellet_status_perf_c(self, value):
        self._pellet_status_perf_c = value

    @property
    def tunnel_status_perf_c(self) -> float:
        return self._tunnel_status_perf_c

    @tunnel_status_perf_c.setter
    def tunnel_status_perf_c(self, value):
        self._tunnel_status_perf_c = value
