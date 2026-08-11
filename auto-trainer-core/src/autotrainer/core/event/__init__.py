
from typing import Optional, Any

from autotrainer.api import ApiEventKind
from autotrainer.api.api_options import create_default_api_options

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.event.api_event_plugin import ApiEventPlugin

from .event_manager import EventManager


logger = get_verbose_logger(__name__)


__all__ = [
    "ApiEventKind",
    "try_register_api_event_plugin",
    "post_api_event_content",
    "post_api_detector_event_content",
]


def try_register_api_event_plugin() -> Optional[ApiEventPlugin]:
    """
    Attempt to register the Autotrainer External API plugin with the default instance of the event manager.

    Returns:
        bool: True if the plugin was successfully registered, False otherwise.
    """
    try:
        # NB: keeping import here,
        # given otherwise it gives app crash at start if it's imported before the main qt app instance is created

        plugin = ApiEventPlugin(create_default_api_options())
        plugin.set_enable(True)
        EventManager.default().register_plugin(plugin)
        return plugin
    except Exception as err:
        logger.exception("API plugin creation or registration failed: %s", err)

    return None


def post_api_event_content(kind: int, data: Any):
    EventManager.default().post_event_content(kind, data=data)


def post_api_detector_event_content(manager: EventManager, detector_id: int, active: bool, enabled: bool):
    manager.post_event_content(ApiEventKind.detectorChanged, data=dict(
        detector_id=detector_id,
        is_active=active,
        is_enabled=enabled,
    ))
