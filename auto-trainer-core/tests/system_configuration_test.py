import dataclasses
import io
from pathlib import Path

import pytest
import yaml

from autotrainer.core import (
    SystemConfiguration,
    CameraId,
    CameraConfiguration,
    Offset3DTuple,
)
from autotrainer.core.configuration import (
    SystemConfigurationDumper,
    SystemConfigurationLoader,
)
from autotrainer.core.configuration.behavior_configuration import PelletDeliveryConfiguration

def test_same_version_unknown_attribute_raise():
    config_text = f"""
!SystemConfiguration
version: {SystemConfiguration.version}
unknown_attribute: 42
"""
    with pytest.raises(TypeError, match="unknown_attribute"):
        SystemConfiguration.load_yaml(io.StringIO(config_text))


def test_version_56_drops_removed_pellet_limit_fields():
    config_text = """
!SystemConfiguration
version: 56
behavior: !BehaviorConfiguration
  pelletDelivery: !PelletDeliveryConfiguration
    isEnabled: true
    maxPelletsPerSession: 20
    maxPelletsPerDay: 100
"""

    cfg = SystemConfiguration.load_yaml(io.StringIO(config_text))

    assert cfg.version == 56
    assert cfg.behavior.pellet_delivery.is_enabled is True
    assert not hasattr(cfg.behavior.pellet_delivery, "max_pellets_per_session")
    assert not hasattr(cfg.behavior.pellet_delivery, "max_pellets_per_day")


def test_higher_version_drop_unknown_config_items():
    config_text = f"""
!SystemConfiguration
version: {SystemConfiguration.version + 1}
unknown_attribute: 42
persistence: !PersistenceConfiguration
  outputLocation: /output_location_path
  another_unknown_attribute: foobar
"""
    cfg = SystemConfiguration.load_yaml(io.StringIO(config_text))
    assert isinstance(cfg, SystemConfiguration)
    expected_result = dataclasses.asdict(SystemConfiguration())
    # apart the version and persistence.output_location, these are all the defaults values
    expected_result["version"] = SystemConfiguration.version + 1
    expected_result["persistence"]["output_location"] = "/output_location_path"
    assert dataclasses.asdict(cfg) == expected_result


def test_safe_loader_ignore_unknown_tags():
    config_text = f"""
    !SystemConfiguration
    version: {SystemConfiguration.version + 1}
    unknown_attribute: 42
    bar: !unknown_tag
    baz:
    - !second_unknown_tag
      param1: anything
    """
    cfg = SystemConfiguration.load_yaml(io.StringIO(config_text))
    assert isinstance(cfg, SystemConfiguration)
    expected_result = dataclasses.asdict(SystemConfiguration())
    # apart the version, these are all the defaults values
    expected_result["version"] = SystemConfiguration.version + 1
    assert dataclasses.asdict(cfg) == expected_result


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


