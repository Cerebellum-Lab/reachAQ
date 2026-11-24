import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


DEFAULT_API_HOST = "127.0.0.1"
#: Port 0: the OS picks a free port, so two instances on one host never collide.
#: Each instance publishes the port it got in a discovery file (``discovery``).
DEFAULT_API_PORT = 0


@dataclass(frozen=True)
class ApiOptions:
    """Where the local API listens. Loopback by default: it accepts commands
    such as START_ACQUISITION from any process that can reach the port.

    ``discovery_dir`` is where the running service publishes its record; None
    publishes nothing. ``simulated`` is recorded there so a client can tell a
    ``--simulate`` instance from the rig's.
    """
    enabled: bool = True
    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    simulated: bool = False
    discovery_dir: Optional[Path] = None


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def create_default_api_options(environ: Optional[Mapping[str, str]] = None) -> ApiOptions:
    """Options from ``REACHAQ_API_ENABLED``, ``REACHAQ_API_HOST``, ``REACHAQ_API_PORT``,
    ``REACHAQ_API_DISCOVERY_DIR`` and ``REACHAQ_SIMULATION``.

    The service is registered before any system configuration is loaded, so it
    is configured from the environment rather than from the configuration file.
    """
    from .discovery import default_discovery_dir

    env = os.environ if environ is None else environ
    enabled = env.get("REACHAQ_API_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
    host = env.get("REACHAQ_API_HOST", "").strip() or DEFAULT_API_HOST
    port_text = env.get("REACHAQ_API_PORT", "").strip()
    port = int(port_text) if port_text else DEFAULT_API_PORT
    if not 0 <= port <= 65535:
        raise ValueError(f"REACHAQ_API_PORT out of range: {port}")
    directory_text = env.get("REACHAQ_API_DISCOVERY_DIR", "").strip()
    discovery_dir = Path(directory_text) if directory_text else default_discovery_dir(env)
    return ApiOptions(
        enabled=enabled, host=host, port=port,
        simulated=_flag(env, "REACHAQ_SIMULATION"),
        discovery_dir=discovery_dir,
    )
