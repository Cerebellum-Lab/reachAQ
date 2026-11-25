import logging
from typing import Optional

from ..logging import get_verbose_logger
from ..project import ProjectInfo
from .event_info import EventInfo

from .event_manager_plugin import EventManagerPlugin

logger = get_verbose_logger(__name__)


class LoggerEventPlugin(EventManagerPlugin):
    """
    An EventManager plugin that sends event to the default logger facility.
    """

    def __init__(self, level: int = logging.DEBUG):
        self._level = level

    @property
    def level(self) -> int:
        return self._level

    @level.setter
    def level(self, value: int) -> None:
        self._level = value

    def set_project(self, project: Optional[ProjectInfo]) -> None:
        pass

    def set_enable(self, enable: bool) -> None:
        pass

    def process_event(self, info: EventInfo, repeat_count: int) -> None:
        output = f"{info.when}, {info.index}, {info.kind}, {str(info.kind)}, {str(info.context)}, {repeat_count}"
        logger.log(self._level, output)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
