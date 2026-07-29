from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Union, ClassVar, TextIO, Type, Any
from typing_extensions import Self
import yaml

import humps

from autotrainer.core.logging import get_verbose_logger
from . import GenericSafeLoader, SystemConfigurationLoader, SystemConfigurationDumper, SystemConfigurationSafeLoader
from .alarm_detector import AlarmDetectorConfig
from .autoclamp_evasion_config import AnimalEvasionAlarmConfig, AutoClampEvasionDetectorConfig
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
from ..project.project_info import DATE_FORMAT, TIME_FORMAT

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
        if version == SystemConfiguration.version:
            # easy case
            configuration: Self = yaml.load(data, SystemConfigurationLoader)
        elif version < SystemConfiguration.version:
            content = humps.decamelize(raw_content)
            configuration = cls()
            if version == 0:
                configuration._deserialize_version_zero(content)
            elif version == 1:
                configuration._deserialize_version_one(content)
            else:
                configuration: Self = yaml.load(data, SystemConfigurationSafeLoader)
        else:
            assert version > SystemConfiguration.version
            logger.warning("Loading configuration version %s while SystemConfiguration.version == %s, "
                           "only considering known config attributes/properties.",
                           version, SystemConfiguration.version)
            configuration: Self = yaml.load(data, SystemConfigurationSafeLoader)

        if version != SystemConfiguration.version and file_path is not None:
            now = datetime.now()
            now_str = now.strftime(f"{DATE_FORMAT}_{TIME_FORMAT}")
            new_p = file_path.parent.joinpath(
                f"{file_path.stem}_v{version}_{now_str}{file_path.suffix}")
            logger.notice("Detected config version change/mismatch, saving old config to %s,"
                          " and replacing with new after.", new_p)
            shutil.copy2(file_path, new_p)
            configuration.version = SystemConfiguration.version
            # and save new one over previous:
            configuration.save_file(file_path.with_suffix(""), as_yaml=True)

        return configuration

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

    def _deserialize_version_zero(self, content: Dict):
        self.cameras.clear()

        self._try_append_version_zero_camera("camera1", content)
        self._try_append_version_zero_camera("camera2", content)
        self._try_append_version_zero_camera("camera3", content)

        self.hardware = HardwareConfiguration.from_version_zero(content)

        # Typo from earlier version of the file.
        self.inference = InferenceConfiguration.from_version_zero(content.get("pellet", {}))

        self.behavior = BehaviorConfiguration.from_version_zero(content)
        self.persistence = PersistenceConfiguration.from_version_zero(content)

    def _try_append_version_zero_camera(self, entry: str, content: Dict) -> bool:
        if entry in content:
            camera = CameraConfiguration.from_version_zero(entry, content[entry])
            if camera is not None:
                self.cameras.append(camera)
                return True

        return False

    def _deserialize_version_one(self, content: Dict):
        self.cameras = [
            CameraConfiguration(**kw)
            for kw in content.get("cameras", [])
        ]
        self.hardware = HardwareConfiguration(**content.get("hardware", {}))
        self.inference = InferenceConfiguration(**content.get("inference", {}))
        self.laser = LaserSystemConfiguration(**self._deserialize_laser_configuration(content.get("laser", {})))
        self.nidaq_ports = NidaqPortConfiguration(**content.get("nidaq_ports", {}))
        self.behavior = BehaviorConfiguration.from_version_one(content.get("behavior", {}))
        self.persistence = PersistenceConfiguration(**content.get("persistence", {}))

    @staticmethod
    def _deserialize_laser_configuration(content: Dict) -> Dict:
        laser = dict(content or {})
        channels = []
        for channel in laser.get("channels", []):
            if isinstance(channel, LaserChannelConfiguration):
                channels.append(channel)
                continue
            channel = dict(channel)
            if "command_copy_input" not in channel and "command_monitor_input" in channel:
                channel["command_copy_input"] = channel.pop("command_monitor_input")
            else:
                channel.pop("command_monitor_input", None)
            if "command_copy_scale" not in channel and "command_monitor_scale" in channel:
                channel["command_copy_scale"] = channel.pop("command_monitor_scale")
            else:
                channel.pop("command_monitor_scale", None)
            channel.pop("command_copy_output", None)
            channels.append(LaserChannelConfiguration(**channel))
        laser["channels"] = tuple(channels)
        return laser


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
    AlarmDetectorConfig=AlarmDetectorConfig,
    AnimalEvasionAlarmConfig=AnimalEvasionAlarmConfig,
    AutoClampEvasionDetectorConfig=AutoClampEvasionDetectorConfig,
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
