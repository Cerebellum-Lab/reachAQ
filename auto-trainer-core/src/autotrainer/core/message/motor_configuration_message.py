from typing import Protocol, Tuple, Optional
from enum import IntEnum


class Motor(IntEnum):
    """
    Identifiers for the different motor types
    """

    NONE = 0
    PELLET_X_MOTOR = 2
    PELLET_Y_MOTOR = 3
    PELLET_Z_MOTOR = 4
    PELLET_COVER_SERVO = 5
    PELLET_LOAD_SERVO = 6
    DELAY = 9
    TONE = 10


class StepperConfigMessage(Protocol):
    """
    Protocol for the interface for obtaining stepper motor configuration entities
    """

    @property
    def motor(self) -> Motor:
        """Motor associated with the configuration"""

    @property
    def maximum_velocity(self) -> float:
        """Units: turns/sec"""

    @property
    def maximum_acceleration(self) -> float:
        """Units: turns/sec^2"""

    @property
    def minimum_step_inverted(self) -> int:
        """Units: 1/min step"""

    @property
    def steps_per_revolution(self) -> float:
        """Units: steps/rev"""

    @property
    def flip_limit_orientation(self) -> int:
        """
        0 - Do no invert motor direction
        1 - Invert motor direction
        """

    uuid_ack_timeout: Optional[float]
    """Optional custom uuid ack timeout for the motor"""


class ServoConfigMessage(Protocol):
    """
    Protocol for the interface for obtaining servo motor configuration entities
    """

    @property
    def motor(self) -> Motor:
        """Motor associated with the configuration"""

    @property
    def maximum_velocity(self) -> float:
        """Units: deg/sec"""

    @property
    def maximum_acceleration(self) -> float:
        """Units: deg/sec^2"""

    @property
    def minimum_position(self) -> float:
        """Units: deg"""

    @property
    def maximum_position(self) -> float:
        """Units: deg"""

    @property
    def min_pwm_duration(self) -> float:
        """Units: µs"""

    @property
    def max_pwm_duration(self) -> float:
        """Units: µs"""

    @property
    def detach(self) -> bool:
        """Is attach/detach used"""

    uuid_ack_timeout: Optional[float]
    """Optional custom uuid ack timeout for the motor"""


class MotorConfigurations(Protocol):
    """
    Protocol for access to a full set of motor configurations
    """

    @property
    def load_config(self) -> Tuple[Motor, ServoConfigMessage]:
        """The pellet-load config"""

    @property
    def cover_config(self) -> Tuple[Motor, ServoConfigMessage]:
        """The pellet-cover config"""

    @property
    def x_config(self) -> Tuple[Motor, StepperConfigMessage]:
        """The pellet stepper X motor & config"""

    @property
    def y_config(self) -> Tuple[Motor, StepperConfigMessage]:
        """The pellet stepper Y motor & config"""

    @property
    def z_config(self) -> Tuple[Motor, StepperConfigMessage]:
        """The pellet stepper Z motor & config"""
