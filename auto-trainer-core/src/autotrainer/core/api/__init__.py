"""reachAQ's published event kinds, event record and local API.

Persisted event values (ids, names and reprs written to event files) are kept
identical to what reachAQ rigs have recorded; see ``kinds``.
"""

from .kinds import (
    ApiApplicationMode,
    ApiDetectorKind,
    ApiEventKind,
    ApiPelletShiftSource,
    ApiTrainingMode,
)
from .event import ApiEvent
from .commands import (
    ApiCommand,
    ApiCommandRequest,
    ApiCommandRequestErrorKind,
    ApiCommandRequestResponse,
    ApiCommandRequestResult,
    ApiSystemConfiguration,
)
from .options import ApiOptions, create_default_api_options
from .service import RpcService, create_api_service

__all__ = [
    "ApiApplicationMode",
    "ApiCommand",
    "ApiCommandRequest",
    "ApiCommandRequestErrorKind",
    "ApiCommandRequestResponse",
    "ApiCommandRequestResult",
    "ApiDetectorKind",
    "ApiEvent",
    "ApiEventKind",
    "ApiOptions",
    "ApiPelletShiftSource",
    "ApiSystemConfiguration",
    "ApiTrainingMode",
    "RpcService",
    "create_api_service",
    "create_default_api_options",
]
