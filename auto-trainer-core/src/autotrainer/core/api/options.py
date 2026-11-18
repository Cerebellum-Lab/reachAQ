import os
from dataclasses import dataclass
from typing import Mapping, Optional


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 47611


@dataclass(frozen=True)
class ApiOptions:
    """Where the local API listens. Loopback by default: it accepts commands
    such as START_ACQUISITION from any process that can reach the port."""
    enabled: bool = True
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT


def create_default_api_options(environ: Optional[Mapping[str, str]] = None) -> ApiOptions:
    """Options from ``REACHAQ_API_ENABLED``, ``REACHAQ_API_HOST`` and ``REACHAQ_API_PORT``.

    The service is registered before any system configuration is loaded, so it
    is configured from the environment rather than from the configuration file.
    """
    env = os.environ if environ is None else environ
    enabled = env.get("REACHAQ_API_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
    host = env.get("REACHAQ_API_HOST", "").strip() or DEFAULT_API_HOST
    port_text = env.get("REACHAQ_API_PORT", "").strip()
    port = int(port_text) if port_text else DEFAULT_API_PORT
    if not 0 <= port <= 65535:
        raise ValueError(f"REACHAQ_API_PORT out of range: {port}")
    return ApiOptions(enabled=enabled, host=host, port=port)
