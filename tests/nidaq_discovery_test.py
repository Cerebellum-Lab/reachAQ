import pytest

from tools.acquisition.model.nidaq_discovery import _optional_device_property


class _DaqError(Exception):
    def __init__(self, error_code):
        super().__init__(f"DAQ error {error_code}")
        self.error_code = error_code


class _Device:
    def __init__(self, error_code):
        self._error_code = error_code

    @property
    def capability(self):
        raise _DaqError(self._error_code)


def test_optional_device_property_ignores_unsupported_capability():
    assert _optional_device_property(_Device(-200197), "capability", _DaqError) is None


def test_optional_device_property_propagates_other_daq_errors():
    with pytest.raises(_DaqError, match="DAQ error -1"):
        _optional_device_property(_Device(-1), "capability", _DaqError)
