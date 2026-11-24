"""Best-effort Linux CAN state captured at the point of failure."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict


def capture_can_diagnostics(channel: str) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "channel": channel,
        "captured_wall_time": time.time(),
    }
    try:
        completed = subprocess.run(
            ["ip", "-details", "-statistics", "-json", "link", "show", "dev", channel],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        diagnostics["capture_error"] = str(exc) or exc.__class__.__name__
        return diagnostics

    diagnostics["ip_exit_code"] = completed.returncode
    if completed.returncode != 0:
        diagnostics["ip_error"] = completed.stderr.strip() or completed.stdout.strip()
        return diagnostics
    try:
        links = json.loads(completed.stdout)
    except (TypeError, ValueError) as exc:
        diagnostics["capture_error"] = f"invalid ip JSON: {exc}"
        return diagnostics
    if not links:
        diagnostics["capture_error"] = "CAN interface not found"
        return diagnostics

    link = links[0]
    diagnostics.update({
        "interface_state": link.get("operstate"),
        "flags": link.get("flags", ()),
        "mtu": link.get("mtu"),
        "link_info": link.get("linkinfo", {}),
        "stats": link.get("stats64", link.get("stats", {})),
    })
    return diagnostics

