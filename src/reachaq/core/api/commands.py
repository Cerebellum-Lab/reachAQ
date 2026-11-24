"""Commands accepted by the reachAQ local API and their responses."""

import enum
from dataclasses import dataclass
from typing import Any, Optional

from .kinds import _ApiIntEnum


class ApiCommand(_ApiIntEnum):
    NONE = 0
    START_ACQUISITION = 1
    STOP_ACQUISITION = 2
    GET_STATUS = 3
    GET_CONFIGURATION = 4
    USER_DEFINED = 100


class ApiCommandRequestResult(str, enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    EXCEPTION = "exception"
    UNRECOGNIZED = "unrecognized"
    # Accepted; the outcome follows later as a separate response with the same nonce.
    PENDING_WITH_NOTIFICATION = "pending_with_notification"


class ApiCommandRequestErrorKind(str, enum.Enum):
    NONE = "none"
    COMMAND_ERROR = "command_error"
    SYSTEM_ERROR = "system_error"


@dataclass(frozen=True)
class ApiCommandRequest:
    command: ApiCommand
    nonce: Optional[str] = None
    custom_command: Optional[str] = None
    data: Optional[Any] = None


@dataclass(frozen=True)
class ApiCommandRequestResponse:
    nonce: Optional[str]
    command: ApiCommand
    result: ApiCommandRequestResult
    data: Optional[Any] = None
    error_code: ApiCommandRequestErrorKind = ApiCommandRequestErrorKind.NONE
    error_message: Optional[str] = None


@dataclass(frozen=True)
class ApiSystemConfiguration:
    application_version: Optional[str]
    device_id: Optional[str]
    configuration_location: Optional[str]
    data_location: Optional[str]
    animal_location: Optional[str]
    log_location: Optional[str]
    inference_model: Optional[str]
