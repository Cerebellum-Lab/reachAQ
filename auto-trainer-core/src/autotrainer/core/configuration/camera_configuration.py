import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Any, Tuple

from typing_extensions import Self

import yaml

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core import make_camelize_representer, make_decamelize_constructor
from autotrainer.core.configuration import SystemConfigurationDumper


logger = get_verbose_logger(__name__)


class CameraId(IntEnum):
    Left = 0
    Right = 1
    Web = 2
    Camera3 = 3
    Camera4 = 4
    Camera5 = 5
    Camera6 = 6

    def __str__(self) -> str:
        # Used as part of video file and related naming conventions.
        if self == CameraId.Left:
            return "left"
        elif self == CameraId.Right:
            return "right"
        elif self == CameraId.Web:
            return "web"
        elif self in (CameraId.Camera3, CameraId.Camera4, CameraId.Camera5, CameraId.Camera6):
            return f"camera{int(self)}"
        else:
            raise ValueError(f"Invalid camera id: {self}")

    @classmethod
    def supported_count(cls) -> int:
        return 6

    @classmethod
    def reach_camera_ids(cls) -> Tuple["CameraId", ...]:
        return (cls.Left, cls.Right, cls.Camera3, cls.Camera4, cls.Camera5, cls.Camera6)


@dataclass
class CameraConfiguration:
    id: CameraId = CameraId.Left
    name: str = "(unnamed)"
    is_enabled: bool = False
    is_record_enabled: bool = False
    record_mode: int = 0
    record_prebuffer_duration: float = 1  # seconds, delay of pre-buffer to use/record when start-recording is executed
    is_still_image_capture_enabled: bool = False
    still_image_capture_interval: float = 5.0
    scheme: str = ""
    host: str = ""
    port: int = 0
    path: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, CameraId):
            self.id = CameraId(self.id)
        if "%" in self.path:
            # this could possibly be removed later when ~all present config files would be already fixed.
            self.path = urllib.parse.unquote(self.path)
            logger.info("Replaced %%-encoded path from configuration with unquoted one: %r", self.path)
        self.path = re.sub(r"/+", "/", self.path)  # sanitize

    @classmethod
    def from_version_zero(cls, name: str, content: dict) -> Self:
        if name == "camera1":
            camera_id = CameraId.Left
        elif name == "camera2":
            camera_id = CameraId.Right
        else:
            camera_id = CameraId.Web

        if "camera" in content:
            camera = content["camera"]
        else:
            return None

        known_params = ["scheme", "host", "path", "port"]

        params = dict()

        for key in camera:
            if key not in known_params:
                params[key] = camera[key]

        return cls(
            id=camera_id,
            name=content.get("name", "(unnamed)"),
            is_enabled=content.get("is_enabled", False),
            is_record_enabled=content.get("is_record_enabled", False),
            record_mode=content.get("record_mode", 0),
            is_still_image_capture_enabled=content.get("is_still_image_capture_enabled", False),
            still_image_capture_interval=content.get("still_image_capture_interval", 0.0),
            scheme=camera.get("scheme", ""),
            host=camera.get("host", ""),
            port=camera.get("port", 0),
            path=camera.get("path", ""),
            params=params
        )


def camera_id_representer(dumper: yaml.SafeDumper, obj):
    return dumper.represent_data(int(obj))

SystemConfigurationDumper.add_representer(CameraId, camera_id_representer)
