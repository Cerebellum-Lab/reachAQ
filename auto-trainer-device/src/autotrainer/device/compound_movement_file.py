"""
Class to manage compound movement configuration YAML file or YAML dictionary.

In either case, the YAML key for contents is "actions".
It can have 5 subgroups:
* "load_pellet" - a sequence of commands for loading a pellet
* "send_pellet" - a sequence of commands for sending the pellet to the cage
* "cover_pellet" - a sequence of commands for covering the pellet
* "release_pellet" - a sequence of commands for uncovering the pellet
* "move_retract" - a sequence of commands for retracting the pellet
"""
import enum
from pathlib import Path
import yaml

from autotrainer.core.config_directory import DEFAULT_CONFIG_DIR
from autotrainer.core.logging import get_verbose_logger

from .motor_steps import MotorSteps, CompoundMovementDataSet


logger = get_verbose_logger(__name__)


class CompoundMovementKind(str, enum.Enum):
    LOAD_PELLET = "load_pellet"
    SEND_PELLET = "send_pellet"
    COVER_PELLET = "cover_pellet"
    RELEASE_PELLET = "release_pellet"
    MOVE_RETRACT = "move_retract"


CompoundMovementByStringValue = {
    kind.value: kind
    for kind in CompoundMovementKind
}


class CompoundMovements(CompoundMovementDataSet):
    """
    Class that loads compound movements (MotorSteps) from either a file or
    a YAML-type dictionary.

    Presents the data in the form of a CompoundMovementDataSet Protocol
    """

    DEFAULT_LOCATION = DEFAULT_CONFIG_DIR / "move_config.yaml"  # you shall use .expanduser() when you use it

    def __init__(self):
        """
        Set the default movements to an empty sequence
        """
        self._movements = {
            kind: MotorSteps()
            for kind in CompoundMovementKind
        }

    @classmethod
    def from_file(cls, filename):
        """
        Import sequences from a file.

        Args:
            filename (str or Path): Filename to load from

        Returns:
            CompoundMovements: populated with file contents
        """
        inst = cls()
        inst._load(filename)
        return inst

    @classmethod
    def from_yaml_dict(cls, yaml_dict):
        """
        Import sequences from a dictionary.

        Args:
            yaml_dict (dict): Dictionary of data

        Returns:
            CompoundMovements: populated with file contents
        """
        inst = cls()
        inst._convert(yaml_dict)
        return inst

    def _load(self, file_path):
        """
        Load sequences from a file.

        Args:
            file_path (str or Path): Filename to load from
        """
        file_path = Path(file_path)
        with file_path.expanduser().open() as fh:
            self._convert(yaml.safe_load(fh))

    def _convert(self, yaml_dict):
        """
        Load sequences from a dictionary.

        Args:
            yaml_dict (dict): Dictionary of data
        """
        if not isinstance(yaml_dict, dict):
            raise TypeError(f"Expected dict for main compound move content, but got {type(yaml_dict)}")
        try:
            actions = yaml_dict["actions"]
        except KeyError:
            raise ValueError("Expected an 'actions' key in main compound move content") from None
        if not isinstance(actions, dict):
            raise TypeError(f"Expected dict for 'actions' in main compound move content, but got {type(actions)}")

        for name, value in actions.items():
            kind = CompoundMovementByStringValue.get(name, None)
            if kind is None:
                logger.warning("Unhandled %r in compound movement definition. value=%s", name, value)
                continue
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Expected dict for action {name!r}, got {value!r}")
            steps = MotorSteps.from_raw(name, list(value))
            self._movements[kind] = steps
            logger.debug("loaded compound move %r: %s", name, steps)

    '''
    Meet the CompoundMovementDataSet Protocol
    '''

    @property
    def load_pellet(self) -> MotorSteps:
        return self._movements[CompoundMovementKind.LOAD_PELLET]

    @property
    def send_pellet(self) -> MotorSteps:
        return self._movements[CompoundMovementKind.SEND_PELLET]

    @property
    def cover_pellet(self) -> MotorSteps:
        return self._movements[CompoundMovementKind.COVER_PELLET]

    @property
    def release_pellet(self) -> MotorSteps:
        return self._movements[CompoundMovementKind.RELEASE_PELLET]

    @property
    def move_retract(self) -> MotorSteps:
        return self._movements[CompoundMovementKind.MOVE_RETRACT]
