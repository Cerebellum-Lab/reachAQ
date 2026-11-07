from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.acquisition.model.atomic_session_io import atomic_write_json

from .engine import validate_session
from .model import ValidationProfile, ValidationStatus


def _parser():
    parser = argparse.ArgumentParser(description="Validate a published reachAQ session read-only")
    parser.add_argument("session", type=Path)
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--quick", action="store_true")
    profiles.add_argument("--fast", action="store_true")
    profiles.add_argument("--full", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_stdout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-warnings", action="store_true")
    parser.add_argument("--rule", action="append", default=[])
    parser.add_argument("--skip-rule", action="append", default=[])
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    profile = (
        ValidationProfile.QUICK if args.quick
        else ValidationProfile.FULL if args.full
        else ValidationProfile.FAST
    )
    report = validate_session(
        args.session,
        profile=profile,
        selected_rules=args.rule,
        skipped_rules=args.skip_rule,
    )
    record = report.to_record()
    if args.output:
        atomic_write_json(args.output, record)
    if args.json_stdout:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(
            f"reachAQ session validation: {profile.value} | "
            f"{report.session_path} | {report.counts}"
        )
        for item in report.results:
            print(f"[{item.status.value.upper():14}] {item.rule_id}: {item.message}")
    exit_code = report.exit_code
    if args.strict_warnings and any(
        item.status is ValidationStatus.WARNING for item in report.results
    ):
        exit_code = max(1, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
