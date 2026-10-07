"""Rebuild a session's latency analysis from its raw latency streams.

    python -m tools.latency.rebuild <session_dir> [--primary-camera NAME]

Writes streams/latency.rebuilt-<UTC>.h5 beside the original. It never rewrites
latency.h5: that file is listed in stream_manifest.json, and changing it after
publication fails session validation (session.manifest).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .finalize import finalize_session_latency


def _primary_from_alignment(session_dir: Path) -> str:
    path = session_dir / "streams" / "alignment.json"
    try:
        with path.open("r", encoding="utf-8") as stream:
            return str(json.load(stream)["canonicalBoundary"].get("primaryCamera") or "")
    except (OSError, KeyError, ValueError):
        return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--primary-camera", default=None)
    args = parser.parse_args(argv)
    primary = args.primary_camera or _primary_from_alignment(args.session_dir)
    name = f"latency.rebuilt-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.h5"
    status = finalize_session_latency(args.session_dir, primary_camera=primary, output_name=name)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
