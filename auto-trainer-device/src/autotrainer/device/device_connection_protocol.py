from typing import Protocol, Any, Optional


from autotrainer.core import MotorConfigurations, Offset3DTuple, Motor
from autotrainer.core.logging import get_verbose_logger
from autotrainer.device import MotorConfigurationFile, CompoundMovements, Device
from autotrainer.device.motor_steps import CompoundMovementDataSet

logger = get_verbose_logger(__name__)


class DeviceConnectionProtocol(Protocol):

    @property
    def watchdog_perf_c(self) -> float:
        """The alive/watchdog perf-counter of the related device"""
        return min(self.watchdog_reader_perf_c, self.watchdog_writer_perf_c)

    @property
    def watchdog_reader_perf_c(self) -> float:
        """The last alive/watchdog perf-counter of the related device reader thread"""

    @property
    def watchdog_writer_perf_c(self) -> float:
        """The last alive/watchdog perf-counter of the related device writer thread"""

    @property
    def device(self) -> Device:
        """Get the associated physical device"""
        raise NotImplementedError

    @property
    def read_limit(self) -> int:
        """Must return the read limit"""
        raise NotImplementedError

    @read_limit.setter
    def read_limit(self, value: int):
        raise NotImplementedError

    def request_connect(self):
        """Request the connection to the physical device to be established"""
        raise NotImplementedError

    def request_disconnect(self):
        """Request disconnection from the physical device"""
        raise NotImplementedError

    def join(self):
        """Join any of the thread/resources used during the connection"""

    def send_message(self, kind: int, data: Optional[Any] = None, context: Optional[Any] = None):
        """Send a message/command to the command writer handler thread"""

    def load_default_motor_config(self) -> MotorConfigurations:
        default_motors_cfg_file = MotorConfigurationFile.DEFAULT_LOCATION.expanduser()
        if default_motors_cfg_file.exists():
            logger.notice("Reading and applying default motors config: %s", default_motors_cfg_file)
            motors_cfg = MotorConfigurationFile.from_file(default_motors_cfg_file)
        else:
            logger.warning(
                "Default motor config file %s not present, empty motor config auto-applied, this might be critical",
                default_motors_cfg_file)
            motors_cfg = MotorConfigurationFile()
        self.use_motor_configurations(motors_cfg)
        return motors_cfg

    def use_motor_configurations(self, data: MotorConfigurations):
        """Apply the given motor configuration"""
        raise NotImplementedError

    def _motor_configuration_is_required(self, motor: Motor) -> bool:
        """Return whether the connected device requires the motor's CAN target."""
        is_motor_required = getattr(self.device, "is_motor_required", None)
        if callable(is_motor_required):
            required = is_motor_required(motor)
            if not required:
                logger.info("Skipping unused %s configuration", motor)
            return required

        required_targets = getattr(self.device, "required_targets", None)
        if required_targets is None:
            return True

        # Import locally to avoid adding a package initialization cycle.
        from .can_interface import target_of_motor

        target = target_of_motor(motor)
        if target in required_targets:
            return True
        logger.info(
            "Skipping %s configuration because CAN target %s is not required",
            motor,
            target,
        )
        return False

    def load_default_move_config(self):
        default_move_cfg_file = CompoundMovements.DEFAULT_LOCATION.expanduser()
        if default_move_cfg_file.exists():
            logger.notice("Reading and applying default move config: %s", default_move_cfg_file)
            move_cfg = CompoundMovements.from_file(default_move_cfg_file)
            self.use_compound_movements(move_cfg)
        else:
            logger.warning(
                "Default move config file %s not present, no move config auto-applied, this might be critical",
                default_move_cfg_file)

    def use_compound_movements(self, data: CompoundMovementDataSet):
        """Apply the given motor compound movement"""
        raise NotImplementedError
