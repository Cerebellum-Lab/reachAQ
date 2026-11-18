from typing import Optional

from autotrainer.core.api import ApiOptions, RpcService, create_api_service

from ..logging import get_verbose_logger
from ..project import ProjectInfo
from .event_info import EventInfo

from .event_manager_plugin import EventManagerPlugin

logger = get_verbose_logger(__name__)


class ApiEventPlugin(EventManagerPlugin):
    """
    An EventManager plugin that publishes events on the reachAQ local API
    (`autotrainer.core.api`) and owns that service's lifetime.
    """

    def __init__(self, options: ApiOptions):
        self._options = options
        self._service: Optional[RpcService] = None

    @property
    def service(self) -> Optional[RpcService]:
        return self._service

    def set_project(self, project: Optional[ProjectInfo]) -> None:
        pass

    def set_enable(self, enable: bool) -> None:
        svc = self._service
        if enable and svc is None:
            svc = self._service = create_api_service(self._options)
            if svc is not None:
                svc.start()
        elif not enable and svc is not None:
            svc.stop()
            self._service = None

    def process_event(self, info: EventInfo, repeat_count: int) -> None:
        svc = self._service
        if svc is not None:
            svc.send_event(info)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        svc = self._service
        if svc is not None:
            svc.stop()
            self._service = None
