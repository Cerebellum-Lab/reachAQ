from types import SimpleNamespace

import pytest

from tools.acquisition.model.app_model import _camera_exposure_seconds


class _Camera:
    def __init__(self, name, params, fail=False):
        self.name, self._params, self._fail = name, params, fail

    def save_configuration(self, *, runtime_mode=False):
        assert runtime_mode
        if self._fail:
            raise RuntimeError("no url")
        return SimpleNamespace(params=self._params)


def test_exposures_are_the_configured_microseconds_in_seconds_or_none():
    cameras = [_Camera("left", {"exposure": 175}), _Camera("right", {"exposure": "250"}),
               _Camera("cam3", {}), _Camera("bad", {}, fail=True),
               _Camera("odd", {"exposure": "nan"})]
    assert _camera_exposure_seconds(cameras) == {
        "left": pytest.approx(175e-6), "right": pytest.approx(250e-6), "cam3": None, "bad": None,
        "odd": None,  # alignment.json is written with allow_nan=False
    }
