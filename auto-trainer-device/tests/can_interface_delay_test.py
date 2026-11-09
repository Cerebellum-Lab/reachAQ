"""The board Delay command, which takes whole milliseconds."""

import pytest

from autotrainer.device.can_interface import CanInterface


class FakeJerryCan:
    def __init__(self):
        self.delays_ms = []

    def Delay(self, dst_id, delay, uuid):
        self.delays_ms.append(delay)
        return 0


@pytest.fixture
def interface():
    jerrycan = FakeJerryCan()
    api = CanInterface.__new__(CanInterface)
    api._jc = jerrycan
    api._tgt2addr = lambda target: 0
    return api, jerrycan


# Callers build the seconds from whole milliseconds, e.g. the pre-reveal step's
# delay_us / 1e6. 1001 / 1000 is 1.000999..., and truncating that sent 1000.
@pytest.mark.parametrize("delay_sec, delay_ms", [
    (1001 / 1000, 1001),
    (1_001_000 / 1_000_000.0, 1001),
    (1003 / 1000, 1003),
    (32760 / 1000, 32760),
])
def test_the_board_is_sent_the_milliseconds_the_seconds_were_built_from(interface, delay_sec, delay_ms):
    api, jerrycan = interface

    assert api.delay(delay_sec) is True
    assert jerrycan.delays_ms == [delay_ms]


def test_no_pre_reveal_delay_loses_a_millisecond(interface):
    api, jerrycan = interface
    delays_ms = range(1, 60_001)

    for delay_ms in delays_ms:
        api.delay(delay_ms * 1000 / 1_000_000.0)

    assert [wanted for sent, wanted in zip(jerrycan.delays_ms, delays_ms) if sent != wanted] == []
