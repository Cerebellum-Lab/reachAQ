"""
Class to manage compound motor configuration YAML file or YAML dictionary.

The YAML base key is "pellet", with five subgroups:
* "load" - for load servo configuration
* "cover" - for cover servo configuration
* "x" - for x stepper configuration
* "y" - for y stepper configuration
* "z" - for z stepper configuration
"""
from functools import partial

import yaml
from pathlib import Path
from typing import Tuple, Union, Dict, Optional, Type, TypeVar

from typing_extensions import Self

from autotrainer.core import MotorConfigurations
from autotrainer.core.logging import get_verbose_logger

from .device_interface import ServoConfig, StepperConfig, Motor

logger = get_verbose_logger(__name__)


T_MotorConfig = TypeVar("T_MotorConfig", ServoConfig, StepperConfig)


class MotorConfigurationFile(MotorConfigurations):
    """
    Implement MotorConfigurations Protocol
    """

    DEFAULT_LOCATION = Path("~/Autotrainer/motor_config.yaml")  # you shall use .expanduser() when you use it

    _load_config: ServoConfig
    _cover_config: ServoConfig
    _x_config: StepperConfig
    _y_config: StepperConfig
    _z_config: StepperConfig

    def __init__(self):
        """
        Define the set of configuration data sets as the defaults of their respective
        motor configuration types.
        """
        # NB: the following _convert() is actually the main "initializer" of any instance,
        # it ensures we actually set the .motor on each one, or any other needed extra convert:
        self._convert({}, source=None)

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> Self:
        """
        Import configurations from a file.

        Args:
            filename (str or Path): Filename to load from

        Returns:
            MotorConfigurationFile: populated with file contents
        """
        inst = cls()
        inst._load(filename)
        return inst

    @classmethod
    def from_yaml_dict(cls, yaml_dict, *, source: Optional[str]="NA") -> Self:
        """
        Import configurations from a dictionary.

        Args:
            yaml_dict (dict): Dictionary of data
            source (str): Eventual source of the data

        Returns:
            MotorConfigurationFile: populated with file contents
        """
        inst = cls()
        inst._convert(yaml_dict, source=source)
        return inst

    def _load(self, filename: Union[str, Path]):
        """
        Load configurations from a file.

        Args:
            filename (str or Path): Filename to load from
        """
        filename: Path = Path(filename)
        if filename.exists():
            try:
                with filename.open("r") as fh:
                    loaded = self._convert(yaml.safe_load(fh), source=filename.as_posix())
            except Exception as e:
                logger.error(f"Alogus motor configuration file {filename}: {e}")
                raise
            else:
                logger.notice("Config %s, loaded: %s", filename, loaded)
        else:
            logger.error(f"Alogus motor configuration file {filename}: No such file")

    def _convert(self, yaml_dict, *, source: Optional[str]="NA"):
        """
        Load configurations from a dictionary.

        Args:
            yaml_dict (dict): Dictionary of data
        """
        items_loaded = []

        pellet_dct: Dict = yaml_dict.get("pellet", {})

        def do_load(parent_dct: Dict, section: str, item: str, motor: Motor, config_cls: Type[T_MotorConfig]) -> T_MotorConfig:
            dct: Optional[Dict] = parent_dct.get(section)
            if dct is None:
                cfg = config_cls()
            else:
                cfg = config_cls.from_dict(dct)
                if source is not None:
                    logger.info("%s configuration: %s", item, cfg)
                items_loaded.append(item)
            cfg.motor = motor
            return cfg
        #
        do_load_pellet = partial(do_load, pellet_dct)
        self._load_config = do_load_pellet("load", "pellet-load", Motor.PELLET_LOAD_SERVO, ServoConfig)
        self._cover_config = do_load_pellet("barrier", "pellet-barrier", Motor.PELLET_COVER_SERVO, ServoConfig)
        #
        self._x_config = do_load_pellet("x", "pellet-x", Motor.PELLET_X_MOTOR, StepperConfig)
        self._y_config = do_load_pellet("y", "pellet-y", Motor.PELLET_Y_MOTOR, StepperConfig)
        self._z_config = do_load_pellet("z", "pellet-z", Motor.PELLET_Z_MOTOR, StepperConfig)
        #
        if len(items_loaded) != 5:
            if source is not None:
                logger.warning(
                    "Expected 5 pellet sections loaded from source %r but got %s: loaded=%s",
                    source,
                    len(items_loaded),
                    items_loaded,
                )

        return items_loaded

    @property
    def load_config(self) -> Tuple[Motor, ServoConfig]:
        return Motor.PELLET_LOAD_SERVO, self._load_config

    @property
    def cover_config(self) -> Tuple[Motor, ServoConfig]:
        return Motor.PELLET_COVER_SERVO, self._cover_config

    @property
    def x_config(self) -> Tuple[Motor, StepperConfig]:
        return Motor.PELLET_X_MOTOR, self._x_config

    @property
    def y_config(self) -> Tuple[Motor, StepperConfig]:
        return Motor.PELLET_Y_MOTOR, self._y_config

    @property
    def z_config(self) -> Tuple[Motor, StepperConfig]:
        return Motor.PELLET_Z_MOTOR, self._z_config
