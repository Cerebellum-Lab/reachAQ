"""The host's tone request, which the board takes in whole milliseconds."""

from unittest import mock

import pytest

from autotrainer.core import SystemCommandKind

from tools.acquisition.model.hardware_model import HardwareModel


# A protocol tone passes profile.duration_ms / 1000.0, and 1.001 * 1000 is
# 1000.999..., which truncation queued as 1000.
@pytest.mark.parametrize("duration_ms", [1001, 2002, 32763])
def test_a_tone_is_queued_with_the_milliseconds_its_seconds_were_built_from(duration_ms):
    sent = []
    hardware = object.__new__(HardwareModel)
    hardware._device_conn = mock.sentinel.device_conn
    hardware._send_with_token = (
        lambda device, kind, data=None: sent.append((kind, data)) or "token"
    )

    assert hardware.play_tone(6000, duration_ms / 1000.0) == "token"
    assert sent == [(SystemCommandKind.PLAY_TONE, (6000, duration_ms))]
