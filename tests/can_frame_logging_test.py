"""python-can's per-frame logging is off unless REACHAQ_CAN_FRAME_LOG=1.

python-can logs every frame it receives (level 9 on "can.bus") and every frame
it sends (the socketcan "tx" logger, at DEBUG). On christielab10 that was about
270 records a second, each formatted and pickled into the log queue on the CAN
reader thread. The level has to be set before setup_logging(): with
multiprocess logging, setup replaces Logger.setLevel with a relay to the
listener process, so a later call changes nothing in the GUI process.
"""

import inspect
import json
import logging
import os
import subprocess
import sys
import textwrap

import pytest

from tools.acquisition import gui, headless
from tools.acquisition.can_frame_logging import (
    CAN_FRAME_LOG_VARIABLE,
    limit_can_frame_logging,
)

# python-can's RECV_LOGGING_LEVEL: the per-frame receive line.
_RECEIVE_FRAME_LEVEL = 9


@pytest.fixture
def can_logger(monkeypatch):
    monkeypatch.delenv(CAN_FRAME_LOG_VARIABLE, raising=False)
    logger = logging.getLogger("can")
    root = logging.getLogger()
    previous_levels = logger.level, root.level
    logger.setLevel(logging.NOTSET)
    # The application's root logger passes everything (setup_logging's
    # root_level is NOTSET); pytest's is WARNING, which would hide the frames
    # whether or not the helper ran.
    root.setLevel(logging.NOTSET)
    yield logger
    logger.setLevel(previous_levels[0])
    root.setLevel(previous_levels[1])


def test_per_frame_logging_is_off_by_default(can_logger):
    assert limit_can_frame_logging() is False

    bus = logging.getLogger("can.bus")
    assert not bus.isEnabledFor(_RECEIVE_FRAME_LEVEL)
    assert not logging.getLogger("can.interfaces.socketcan.socketcan.tx").isEnabledFor(
        logging.DEBUG
    )
    # python-can's own warnings and errors still get through.
    assert bus.isEnabledFor(logging.WARNING)
    assert bus.isEnabledFor(logging.ERROR)


def test_the_switch_leaves_the_can_logger_alone(can_logger, monkeypatch):
    monkeypatch.setenv(CAN_FRAME_LOG_VARIABLE, "1")

    assert limit_can_frame_logging() is True

    assert can_logger.level == logging.NOTSET


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_one_turns_the_switch_on(can_logger, monkeypatch, value):
    monkeypatch.setenv(CAN_FRAME_LOG_VARIABLE, value)

    assert limit_can_frame_logging() is False

    assert can_logger.level == logging.WARNING


def _levels_after_setup_logging(multiprocess_enabled):
    """Run the helper and then setup_logging() in a fresh interpreter.

    setup_logging() installs handlers, hooks and, for multiprocess logging, a
    listener process, so it never runs in the test process.
    """
    script = textwrap.dedent(f"""
        import json, logging

        if __name__ == "__main__":
            from tools.acquisition.can_frame_logging import limit_can_frame_logging
            limit_can_frame_logging()

            from autotrainer.core.logging import setup_logging, stop_multiproc_logging
            setup_logging(
                "autotrainer",
                time_precision=6,
                multiprocess_enabled={multiprocess_enabled!r},
            )
            try:
                bus = logging.getLogger("can.bus")
                print("LEVELS " + json.dumps({{
                    "can": logging.getLogger("can").level,
                    "receive_frame": bus.isEnabledFor({_RECEIVE_FRAME_LEVEL}),
                    "warning": bus.isEnabledFor(logging.WARNING),
                }}), flush=True)
            finally:
                stop_multiproc_logging()
    """)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    environment.pop(CAN_FRAME_LOG_VARIABLE, None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    (line,) = [l for l in result.stdout.splitlines() if l.startswith("LEVELS ")]
    return json.loads(line[len("LEVELS "):])


@pytest.mark.parametrize("multiprocess_enabled", [False, True])
def test_setup_logging_leaves_the_can_logger_at_warning(multiprocess_enabled):
    levels = _levels_after_setup_logging(multiprocess_enabled)

    assert levels == {"can": logging.WARNING, "receive_frame": False, "warning": True}


@pytest.mark.parametrize("entry_point", [gui.main, headless.main], ids=["gui", "headless"])
def test_both_entry_points_limit_the_logger_before_they_set_up_logging(entry_point):
    # After setup_logging(multiprocess_enabled=True) a setLevel here would
    # reach only the listener process, so the order is the whole fix.
    source = inspect.getsource(entry_point)

    assert 0 <= source.index("limit_can_frame_logging()") < source.index("setup_logging(")
    assert source.index("setup_logging(") < source.index("log_can_frame_logging(")
