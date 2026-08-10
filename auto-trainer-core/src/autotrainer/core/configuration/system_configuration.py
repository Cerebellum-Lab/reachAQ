from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Union, ClassVar, TextIO, Type, Any
from typing_extensions import Self
import yaml

from autotrainer.core.logging import get_verbose_logger
from . import GenericSafeLoader, SystemConfigurationLoader, SystemConfigurationDumper
from .watchdog_config import WatchdogConfig
from .. import make_camelize_representer, make_decamelize_constructor
from .behavior_configuration import BehaviorConfiguration, add_behavior_configuration_representers, \
    add_behavior_configuration_constructors
from .camera_configuration import CameraConfiguration, CameraId
from .hardware_configuration import HardwareConfiguration
from .inference_configuration import InferenceConfiguration
from .laser_configuration import LaserChannelConfiguration, LaserSystemConfiguration
from .nidaq_port_configuration import NidaqPortConfiguration
from .nidaq_stream_configuration import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
from .persistence_configuration import PersistenceConfiguration

logger = get_verbose_logger(__name__)

#


@dataclass
class SystemConfiguration:
    """
    The user-visible and editable elements of the system configuration are evolving.  This class is largely intended
    manage transitions and maintain a consistent interface to applications.
    """

    DEFAULT_CONFIG_DIR: ClassVar[Path] = Path("~/Autotrainer")  # caller/user must expanduser() on it
    DEFAULT_NAME: ClassVar[str] = "system_configuration"

    DEFAULT_PATH: ClassVar[Path] = DEFAULT_CONFIG_DIR.joinpath(f"{DEFAULT_NAME}.yaml")  # caller/user must expanduser() on it

    version: int = 57

    cameras: List[CameraConfiguration] = field(default_factory=list)
    hardware: HardwareConfiguration = field(default_factory=HardwareConfiguration)
    inference: InferenceConfiguration = field(default_factory=InferenceConfiguration)
    laser: LaserSystemConfiguration = field(default_factory=LaserSystemConfiguration)
    nidaq_ports: NidaqPortConfiguration = field(default_factory=NidaqPortConfiguration)
    nidaq_stream: NidaqSignalStreamConfiguration = field(default_factory=NidaqSignalStreamConfiguration)
    behavior: BehaviorConfiguration = field(default_factory=BehaviorConfiguration)
    persistence: PersistenceConfiguration = field(default_factory=PersistenceConfiguration)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)

    _camera_map = None  # do not include in fields

    def __post_init__(self):
        self._camera_map = {}

    @classmethod
    def load_yaml(cls: Type[Self], data: TextIO, *, file_path: Optional[Path] = None) -> Self:
        raw_content: Dict[str, Any] = yaml.load(data, GenericSafeLoader)
        data.seek(0)
        version = raw_content.get("version", 0)
        if version != SystemConfiguration.version:
            raise ValueError(
                f"Unsupported system configuration version {version}; "
                f"reachAQ requires version {SystemConfiguration.version}"
            )
        return yaml.load(data, SystemConfigurationLoader)

    @classmethod
    def load_yaml_file(cls: Type[Self], path: Union[Path, str], *, save_backup: bool = False) -> Self:
        path: Path = Path(path)
        logger.debug("loading configuration from %r", path)
        with path.open() as fh:
            return cls.load_yaml(fh, file_path=path if save_backup else None)

    @classmethod
    def make_default_yaml_config_path(cls, dir_path: Path) -> Path:
        return dir_path.joinpath(f"{SystemConfiguration.DEFAULT_NAME}.yaml")

    @classmethod
    def load_default(cls: Type[Self], location: Union[str, Path], *, save_backup: bool = False) -> Optional[Self]:
        path = cls.make_default_yaml_config_path(Path(location))
        if path.is_file():
            return cls.load_yaml_file(path, save_backup=save_backup)
        logger.debug("cannot load default from %s ; not a file", path)
        return None

    def save_default(self, dir_path: Union[Path, str]):
        path = self.make_default_yaml_config_path(Path(dir_path))
        save_path: Path = path.with_suffix("")  # noqa
        self.save_file(save_path, as_yaml=True)

    def dump_yaml(self) -> str:
        return yaml.dump(self, Dumper=SystemConfigurationDumper,
                         # we sort/iter by dataclasses.fields() order in our representer function
                         sort_keys=False)

    def save_file(self, path: Union[Path, str], as_yaml: bool = False, as_json: bool = False):
        if not (as_json or as_yaml):
            raise ValueError("Missing one of as_json or as_yaml")
        path: Path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if as_json:
            p = path.with_suffix(".json")
            logger.notice("Writing to %r as json", p.as_posix())
            content = json.dumps(asdict(self))
            # dump before writing to file, to prevent empty file on dump issue
            with p.open("w") as file:
                file.write(content)
        if as_yaml:
            p = path.with_suffix(".yaml")
            logger.notice("Writing to %r as yaml", p.as_posix())
            content = self.dump_yaml()
            # dump before writing to file, to prevent empty file on dump issue
            with p.open("w") as file:
                file.write(content)

    def get_camera(self, camera_id: CameraId) -> Optional[CameraConfiguration]:
        if len(self._camera_map) == 0 or camera_id not in self._camera_map:
            for camera in self.cameras:
                self._camera_map[camera.id] = camera

        return self._camera_map.get(camera_id, None)

#

system_configuration_representer = make_camelize_representer("!SystemConfiguration")

_tag_2_cls = dict(
    SystemConfiguration=SystemConfiguration,
    HardwareConfiguration=HardwareConfiguration,
    CameraConfiguration=CameraConfiguration,
    InferenceConfiguration=InferenceConfiguration,
    LaserChannelConfiguration=LaserChannelConfiguration,
    LaserSystemConfiguration=LaserSystemConfiguration,
    NidaqPortConfiguration=NidaqPortConfiguration,
    NidaqSignalChannelConfiguration=NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration=NidaqSignalStreamConfiguration,
    WatchdogConfig=WatchdogConfig,
    PersistenceConfiguration=PersistenceConfiguration,
)

def add_repr(_tag, _cls):
    SystemConfigurationDumper.add_representer(_cls, make_camelize_representer(f"!{_tag}"))


for tag, cls in _tag_2_cls.items():
    add_repr(tag, cls)


add_behavior_configuration_representers(SystemConfigurationDumper)


for cls in SystemConfigurationLoader, :
    # SystemConfigurationSafeLoader:
    # no need also add on SystemConfigurationSafeLoader given it subclass SystemConfigurationLoader

    for tag, tag_cls in _tag_2_cls.items():
        cls.add_constructor(f"!{tag}", make_decamelize_constructor(tag_cls))

    add_behavior_configuration_constructors(cls)
