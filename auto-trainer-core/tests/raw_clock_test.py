import math
import sys

from autotrainer.core.latency import RAW_CLOCK_AVAILABLE, monotonic_raw_time, raw_clock


def test_the_raw_clock_is_finite_and_increasing_where_the_platform_has_it():
    if not RAW_CLOCK_AVAILABLE:
        assert math.isnan(monotonic_raw_time())
        return
    first = monotonic_raw_time()
    second = monotonic_raw_time()
    assert math.isfinite(first) and second >= first


def test_linux_has_the_clock():
    assert RAW_CLOCK_AVAILABLE or not sys.platform.startswith("linux")


def test_without_the_clock_the_stamp_is_nan(monkeypatch):
    monkeypatch.setattr(raw_clock, "_RAW_CLOCK_ID", None)
    assert math.isnan(raw_clock.monotonic_raw_time())
