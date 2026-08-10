import io

import pytest
import yaml

from autotrainer.core import (
    SystemConfiguration,
    CameraId,
    CameraConfiguration,
    Offset3DTuple,
)
from autotrainer.core.configuration import (
    NidaqSignalStreamConfiguration,
    SystemConfigurationDumper,
    SystemConfigurationLoader,
)
from autotrainer.core.configuration.behavior_configuration import (
    PelletDeliveryConfiguration,
    ShiftXYZBufferHandlerConfig,
)


@pytest.mark.parametrize(
    ("constructor", "kwargs", "retired_name"),
    (
        (
            NidaqSignalStreamConfiguration,
            {"record_to_acquisition": True},
            "record_to_acquisition",
        ),
        (
            ShiftXYZBufferHandlerConfig,
            {"target_x": 1.0},
            "target_x",
        ),
    ),
)
def test_current_configuration_rejects_retired_fields(
    constructor,
    kwargs,
    retired_name,
):
    with pytest.raises(TypeError, match=retired_name):
        constructor(**kwargs)

def test_same_version_unknown_attribute_raise():
    config_text = f"""
!SystemConfiguration
version: {SystemConfiguration.version}
unknown_attribute: 42
"""
    with pytest.raises(TypeError, match="unknown_attribute"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_older_version_is_rejected():
    config_text = """
!SystemConfiguration
version: 56
behavior: !BehaviorConfiguration
  pelletDelivery: !PelletDeliveryConfiguration
    isEnabled: true
    maxPelletsPerSession: 20
    maxPelletsPerDay: 100
"""

    with pytest.raises(ValueError, match="requires version"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_higher_version_is_rejected():
    config_text = f"""
!SystemConfiguration
version: {SystemConfiguration.version + 1}
unknown_attribute: 42
persistence: !PersistenceConfiguration
  outputLocation: /output_location_path
  another_unknown_attribute: foobar
"""
    with pytest.raises(ValueError, match="requires version"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_unknown_tags_in_unsupported_version_are_rejected_by_version():
    config_text = f"""
    !SystemConfiguration
    version: {SystemConfiguration.version + 1}
    unknown_attribute: 42
    bar: !unknown_tag
    baz:
    - !second_unknown_tag
      param1: anything
    """
    with pytest.raises(ValueError, match="requires version"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_save_file_without_specify_save_type_fails():
    cfg = SystemConfiguration()
    with pytest.raises(ValueError, match="Missing one of as_json or as_yaml"):
        cfg.save_file("foobar.baz")


def test_offset3d_yaml():
    o = Offset3DTuple(1, 2, 3.5)
    data = yaml.dump(o, Dumper=SystemConfigurationDumper)
    o2 = yaml.load(data, Loader=SystemConfigurationLoader)
    assert isinstance(o2, Offset3DTuple)
    assert o2 == o


