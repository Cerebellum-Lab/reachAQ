import io
import logging

from autotrainer.core.logging import (
    ConsoleHandler,
    get_verbose_logger,
    log_hardware_initialization,
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
