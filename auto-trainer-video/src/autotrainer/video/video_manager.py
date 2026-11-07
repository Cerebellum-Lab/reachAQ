import copy
import sys
from enum import Enum
from typing import Optional, Dict, List, Tuple, Type
from urllib.parse import urlparse, ParseResult

from autotrainer.core.logging import get_verbose_logger

from .camera.camera_base import CameraBase
from .camera.random_cam import RandomCam
from .camera.playback_cam import PlaybackCam

logger = get_verbose_logger(__name__)


_spincam_cls = None

def _get_spincam_cls():
    global _spincam_cls
    if _spincam_cls is not None:
        return _spincam_cls
    # if sys.version_info.major == 3 and sys.version_info.minor == 8 and not sys.platform.startswith("darwin"):
    try:
        from .camera.spinnaker_cam import SpinCam
    except ModuleNotFoundError:
        logger.warning("SpinCam module not available. SpinCam disabled.")
        raise
    except Exception as err:
        logger.exception("Cannot import SpinCam: %s. SpinCam disabled.", err)
        raise
    else:
        _spincam_cls = SpinCam
        return SpinCam


class CameraKind(str, Enum):
    Random = "random"
    Playback = "playback"
    Spinnaker = "spinnaker"


class VideoManager:

    @classmethod
    def list_spin_cameras(cls) -> List[str]:
        spincam_cls = _get_spincam_cls()
        return spincam_cls.list()

    @classmethod
    def get_spin_camera(cls, serial_number: str, name: str = "") -> Optional[CameraBase]:
        spincam_cls = _get_spincam_cls()
        return spincam_cls.create(serial_number, name)

    @staticmethod
    def get_cam_class(cam_kind: CameraKind) -> Type[CameraBase]:
        if cam_kind == CameraKind.Spinnaker:
            return _get_spincam_cls()
        cam_cls = {
            CameraKind.Random: RandomCam,
            CameraKind.Playback: PlaybackCam,
        }.get(cam_kind, None)
        if cam_cls is None:
            raise ValueError(f"Unhandled camera kind: {cam_kind!r}")
        return cam_cls

    @classmethod
    def parse_params(cls, camera_url: str) -> Tuple[ParseResult, Dict[str, str]]:
        parsed = urlparse(camera_url)
        kind = CameraKind(parsed.scheme)
        cam_cls = cls.get_cam_class(kind)
        parameters = copy.deepcopy(cam_cls.default_params)
        params = parsed.query.split("&")
        for param in params:
            values = param.split("=")
            if len(values) == 2:
                parameters[values[0].lower()] = values[1]
        return parsed, parameters

    @classmethod
    def create_camera(cls, camera_url: str, name: str = "") -> Optional[CameraBase]:
        parsed, parameters = cls.parse_params(camera_url)
        if parsed.scheme == CameraKind.Random:
            camera = RandomCam(name)
        elif parsed.scheme == CameraKind.Spinnaker:
            camera = cls.get_spin_camera(parsed.hostname, name)
        elif parsed.scheme == CameraKind.Playback:
            camera = PlaybackCam(parsed.path, name)
        else:
            logger.warning("No such cam scheme: %s", parsed.scheme)
            return None

        if camera is None:
            logger.error("Cannot create %s-cam ; is library installed ?", parsed.scheme)
            return None

        # set cam params before init:
        for name, value in parameters.items():
            camera.set_property(name, value)

        # init cam after set of properties here before:
        camera.init()

        return camera
