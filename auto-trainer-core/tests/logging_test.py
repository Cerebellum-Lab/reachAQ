import io
import logging

from autotrainer.core.logging import (
    ConsoleHandler,
    _notify_fatal_exception,
    get_verbose_logger,
    log_hardware_initialization,
    register_fatal_exception_callback,
    unregister_fatal_exception_callback,
)


def test_hardware_initialization_bypasses_only_console_level():
    stream = io.StringIO()
    handler = ConsoleHandler(stream=stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.setLevel(logging.WARNING)
    detailed_stream = io.StringIO()
    detailed_handler = logging.StreamHandler(stream=detailed_stream)
    detailed_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    detailed_handler.setLevel(logging.DEBUG)

    test_logger = get_verbose_logger("test.hardware_initialization_console")
    previous_handlers = list(test_logger.handlers)
    previous_propagate = test_logger.propagate
    previous_level = test_logger.level
    test_logger.handlers = [handler, detailed_handler]
    test_logger.propagate = False
    test_logger.level = logging.DEBUG
    try:
        test_logger.info("ordinary detail")
        log_hardware_initialization(test_logger, "START | camera discovery")
        test_logger.warning("ordinary warning")
    finally:
        test_logger.handlers = previous_handlers
        test_logger.propagate = previous_propagate
        test_logger.level = previous_level
        handler.close()
        detailed_handler.close()

    output = stream.getvalue()
    assert "ordinary detail" not in output
    assert "INFO HARDWARE INIT | START | camera discovery" in output
    assert "WARNING ordinary warning" in output
    assert "INFO ordinary detail" in detailed_stream.getvalue()
    assert "INFO HARDWARE INIT | START | camera discovery" in detailed_stream.getvalue()


def test_fatal_exception_callbacks_can_be_registered_once_and_removed():
    calls = []

    def callback(source, exception):
        calls.append((source, exception))

    register_fatal_exception_callback(callback)
    register_fatal_exception_callback(callback)
    try:
        error = RuntimeError("fatal")
        _notify_fatal_exception("test", error)
        assert calls == [("test", error)]
    finally:
        unregister_fatal_exception_callback(callback)

    _notify_fatal_exception("after removal", RuntimeError("ignored"))
    assert len(calls) == 1
