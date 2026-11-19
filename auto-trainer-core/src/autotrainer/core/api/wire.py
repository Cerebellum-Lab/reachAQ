"""JSON encoding of local API messages.

One message per line, UTF-8 JSON objects with a ``type`` of ``request``,
``response`` or ``event``. See docs/acquisition/local-api.md.
"""

import dataclasses
import enum
import json
import math
from datetime import date, datetime
from pathlib import PurePath
from typing import Any, Dict, Optional

from .commands import (
    ApiCommand,
    ApiCommandRequest,
    ApiCommandRequestErrorKind,
    ApiCommandRequestResponse,
    ApiCommandRequestResult,
)
from .event import ApiEvent
from .kinds import _ApiIntEnum

MAX_LINE_BYTES = 1 << 20


def to_jsonable(value: Any) -> Any:
    """Plain JSON types for a payload: id enums by name, string enums by value,
    dataclasses as objects, times as ISO 8601, anything else unknown as str()."""
    if isinstance(value, _ApiIntEnum):
        return value.name
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {_key(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return value.as_posix()
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        try:
            return to_jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _key(key: Any) -> str:
    if isinstance(key, _ApiIntEnum):
        return key.name
    if isinstance(key, enum.Enum):
        return str(key.value)
    return str(key)


def dumps_line(message: Dict[str, Any]) -> bytes:
    return (json.dumps(message, allow_nan=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")


def _finite(value: Any) -> Any:
    # JSON has no NaN or infinity; they go out as null.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def encode_event(info: ApiEvent) -> bytes:
    kind = info.kind
    return dumps_line({
        "type": "event",
        "kind": int(kind),
        "name": kind.name if isinstance(kind, enum.Enum) else None,
        "when": None if info.when is None else info.when.isoformat(),
        "index": info.index,
        "context": _finite(to_jsonable(info.context)),
    })


def encode_response(response: ApiCommandRequestResponse) -> bytes:
    command = response.command
    return dumps_line({
        "type": "response",
        "nonce": response.nonce,
        "command": command.name if isinstance(command, ApiCommand) else to_jsonable(command),
        "result": ApiCommandRequestResult(response.result).value,
        "errorCode": ApiCommandRequestErrorKind(response.error_code).value,
        "errorMessage": response.error_message,
        "data": _finite(to_jsonable(response.data)),
    })


class RequestDecodeError(ValueError):
    def __init__(self, message: str, nonce: Optional[str] = None, command: Any = None):
        super().__init__(message)
        self.nonce = nonce
        self.command = command


def decode_request(line: bytes) -> ApiCommandRequest:
    """A request line as an ``ApiCommandRequest``. ``command`` is a member name
    (``"GET_STATUS"``) or its integer id."""
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise RequestDecodeError(f"invalid JSON: {err}") from None
    if not isinstance(message, dict):
        raise RequestDecodeError("a message must be a JSON object")
    nonce = message.get("nonce")
    nonce = None if nonce is None else str(nonce)
    raw = message.get("command")
    if message.get("type", "request") != "request":
        raise RequestDecodeError(f"unsupported message type: {message.get('type')!r}", nonce, raw)
    try:
        if isinstance(raw, str):
            command = ApiCommand[raw]
        elif isinstance(raw, int) and not isinstance(raw, bool):
            command = ApiCommand(raw)
        else:
            raise KeyError(raw)
    except (KeyError, ValueError):
        raise RequestDecodeError(f"unknown command: {raw!r}", nonce, raw) from None
    custom = message.get("customCommand")
    return ApiCommandRequest(
        command=command,
        nonce=nonce,
        custom_command=None if custom is None else str(custom),
        data=message.get("data"),
    )
