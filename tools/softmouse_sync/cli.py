"""The command the SoftMouse publisher service has always run.

It now runs the animal metadata CLI: ``--setup`` sets up the SoftMouse
source, and a plain run publishes from the configured source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.acquisition.model.animal_sources.cli import main as animal_metadata_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish this computer's animal metadata source (SoftMouse service)"
    )
    parser.add_argument("--setup", action="store_true", help="set up the SoftMouse source")
    parser.add_argument("--destination", type=Path, help="publication directory")
    parser.add_argument("--require-mount", type=Path, help="refuse unless this is a mount point")
    args = parser.parse_args(argv)
    if args.setup:
        return animal_metadata_main(["setup", "--kind", "softmouse"])
    forwarded = ["publish"]
    if args.destination:
        forwarded += ["--destination", str(args.destination)]
    if args.require_mount:
        forwarded += ["--require-mount", str(args.require_mount)]
    return animal_metadata_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
