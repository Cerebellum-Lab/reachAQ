from __future__ import annotations

import datetime
import yaml

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core import Offset3DTuple

logger = get_verbose_logger(__name__)


DEFAULT_3D_CALIB_DIR_NAME = "4mm_6r_8c_4x"


def generic_constructor(loader, tag, node):
    cls_name = node.__class__.__name__
    if cls_name == "SequenceNode":
        return loader.construct_sequence(node)
    elif cls_name == "MappingNode":
        return loader.construct_mapping(node)
    else:
        return loader.construct_scalar(node)


class GenericSafeLoader(yaml.SafeLoader):
    pass


GenericSafeLoader.add_multi_constructor("", generic_constructor)

#

class SystemConfigurationLoader(yaml.SafeLoader):
    """Dedicated yaml loader for SystemConfiguration"""
    safe_load: bool = False


class SystemConfigurationDumper(yaml.SafeDumper):
    """Dedicated yaml dumper for SystemConfiguration"""


class SystemConfigurationSafeLoader(SystemConfigurationLoader):

    safe_load = True

    def ignore_unknown(self, suffix, node):
        logger.warning("Skipping/ignoring tag/section %s", suffix)
        return None


def _ignore_unknown(loader, suffix, node):
    return loader.ignore_unknown(suffix, node)


# allow to ignore unknown tags:
SystemConfigurationSafeLoader.add_multi_constructor('', _ignore_unknown)
SystemConfigurationSafeLoader.add_multi_constructor('!', _ignore_unknown)


#
_offset3dtag = "!Offset3DTuple"
_shift_xyz_tag = "!ShiftXYZTarget"

def construct_offset3d_tuple(loader: yaml.Loader, node: yaml.nodes.MappingNode):
    content = loader.construct_mapping(node)
    return Offset3DTuple(**content)


SystemConfigurationLoader.add_constructor(_offset3dtag, construct_offset3d_tuple)
SystemConfigurationLoader.add_constructor(_shift_xyz_tag, construct_offset3d_tuple)

#

_time_tag = "!Time"

def time_representer(dumper, data):
    """Converts datetime.time into an ISO format string for YAML."""
    return dumper.represent_scalar(_time_tag, data.isoformat())


def time_constructor(loader, node):
    """Parses a !Time scalar specifically into a datetime.time object."""
    value = loader.construct_scalar(node)
    # Handle cases where microsecond is or isn't present
    try:
        return datetime.time.fromisoformat(value)
    except ValueError:
        # Fallback for alternative or truncated ISO formats
        return datetime.datetime.strptime(value, "%H:%M:%S").time()


SystemConfigurationLoader.add_constructor(_time_tag, time_constructor)
SystemConfigurationDumper.add_representer(datetime.time, time_representer)

#


def repr_offset3d_tuple(dumper: SystemConfigurationDumper, obj: Offset3DTuple):
    x, y, z = obj
    return dumper.represent_mapping(_offset3dtag, dict(x=x, y=y, z=z))


SystemConfigurationDumper.add_representer(Offset3DTuple, repr_offset3d_tuple)

# above must be before below


from .behavior_configuration import (
    BehaviorConfiguration,
    SessionControlConfiguration,
    ShiftXYZTarget,
)
from .camera_configuration import CameraConfiguration, CameraId
from .hardware_configuration import HardwareConfiguration
from .inference_configuration import InferenceConfiguration
from .laser_configuration import LaserChannelConfiguration, LaserChannelId, LaserSystemConfiguration, \
    normalize_laser_channel_id
from .nidaq_port_configuration import (
    NidaqDeviceIdentity,
    NidaqPortConfiguration,
    NidaqTimingConfiguration,
    NidaqTimingPlan,
    NidaqTimingRoute,
)
from .nidaq_stream_configuration import NidaqSignalChannelConfiguration, NidaqSignalStreamConfiguration
from .persistence_configuration import PersistenceConfiguration
from .system_configuration import SystemConfiguration
