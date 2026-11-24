import pytest

from reachaq.core.pose_elements import SceneElement, _BaseSceneElement


def test_scene_element_base():
    assert SceneElement.Diamond == "Diamond"


@pytest.mark.parametrize('base_cls', [_BaseSceneElement, SceneElement])
def test_scene_element_singleton(base_cls):
    assert base_cls('Diamond') is SceneElement.Diamond
