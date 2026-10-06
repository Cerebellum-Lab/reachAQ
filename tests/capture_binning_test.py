import pytest

from tools.acquisition.model.capture_binning import (
    apply_capture_binning,
    base_binning,
    capture_binning_available,
    capture_factor,
    common_capture_binning,
    runtime_capture_params,
)

BASE = {"hbin": "4", "vbin": "4", "width": "256", "height": "256",
        "offsetx": "52", "offsety": "6", "gain": "1"}


def test_without_a_preset_nothing_changes():
    assert runtime_capture_params(BASE) is None
    assert capture_factor(BASE) == 1
    assert base_binning(BASE) == 4


@pytest.mark.parametrize("binning,k,size,offx,offy,gain", [
    (4, 1, "256", "52", "6", "1.0000"),
    (2, 2, "512", "104", "12", "13.0412"),
    (1, 4, "1024", "208", "24", "25.0824"),
])
def test_a_preset_keeps_the_field_of_view_and_the_brightness(binning, k, size, offx, offy, gain):
    params = {**BASE, "capture_binning": str(binning)}
    assert capture_factor(params) == k
    assert runtime_capture_params(params) == {
        "hbin": str(binning), "vbin": str(binning), "width": size, "height": size,
        "offsetx": offx, "offsety": offy, "gain": gain,
    }


def test_a_gain_of_none_counts_as_zero_db():
    params = {**BASE, "gain": "None", "capture_binning": "2"}
    assert runtime_capture_params(params)["gain"] == "12.0412"


def test_the_offset_alias_is_the_base_offset_when_it_is_set():
    # SpinCam applies offset_x after the offsetx default, so offset_x is what the camera used.
    params = {**BASE, "offset_x": "100", "offset_y": "10", "capture_binning": "2"}
    resolved = runtime_capture_params(params)
    assert (resolved["offsetx"], resolved["offsety"]) == ("200", "20")


@pytest.mark.parametrize("key,value,read", [
    ("hbin", "4x", base_binning),
    ("capture_binning", "two", capture_factor),
    ("width", "wide", runtime_capture_params),
])
def test_a_malformed_value_is_refused_with_the_key_it_came_from(key, value, read):
    params = {**BASE, "capture_binning": "2", key: value}
    with pytest.raises(ValueError, match=f"{key}='{value}' is not a number"):
        read(params)


@pytest.mark.parametrize("params,message", [
    ({**BASE, "capture_binning": "3"}, "whole factor"),
    ({**BASE, "capture_binning": "8"}, "whole factor"),
    ({**BASE, "vbin": "2", "capture_binning": "2"}, "equal base binning"),
    ({"width": "256", "height": "256", "capture_binning": "2"}, "needs the base hbin"),
])
def test_an_impossible_preset_is_refused(params, message):
    with pytest.raises(ValueError, match=message):
        runtime_capture_params(params)


class _Camera:
    def __init__(self, base=4, preset=None):
        self.base_binning = base
        self.capture_binning = preset

    @property
    def effective_capture_binning(self):
        return self.base_binning if self.capture_binning is None else self.capture_binning

    def set_capture_binning(self, value):
        self.capture_binning = value


def test_the_common_binning_is_none_when_the_cameras_disagree():
    assert common_capture_binning([_Camera(), _Camera()]) == 4
    assert common_capture_binning([_Camera(preset=2), _Camera(preset=2)]) == 2
    assert common_capture_binning([_Camera(preset=2), _Camera()]) is None
    assert common_capture_binning([]) is None


def test_a_preset_is_available_only_when_it_divides_every_base():
    cameras = [_Camera(4), _Camera(4)]
    assert [b for b in (4, 2, 1) if capture_binning_available(cameras, b)] == [4, 2, 1]
    assert not capture_binning_available([_Camera(4), _Camera(None)], 2)
    assert not capture_binning_available([], 2)


def test_applying_the_base_binning_clears_the_preset():
    cameras = [_Camera(preset=2), _Camera(preset=2)]
    apply_capture_binning(cameras, 1)
    assert [c.capture_binning for c in cameras] == [1, 1]
    apply_capture_binning(cameras, 4)
    assert [c.capture_binning for c in cameras] == [None, None]
    with pytest.raises(ValueError, match="does not divide"):
        apply_capture_binning(cameras, 3)
