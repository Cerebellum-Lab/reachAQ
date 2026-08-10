from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.acquisition.model.softmouse_spreadsheet_source import (
    ImportGuardrails,
    SoftMouseMappingProfile,
    SoftMouseSpreadsheetSource,
)

from .https_source import (
    SoftMouseCredentials,
    SoftMouseHttpsConfiguration,
    SoftMouseHttpsSource,
)
from .publisher import SoftMouseExportPublisher


def _configuration(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    mapping = SoftMouseMappingProfile(**value.get("mapping", {}))
    guardrails = ImportGuardrails(**value.get("guardrails", {}))
    https = SoftMouseHttpsConfiguration(**value["https"])
    return value, SoftMouseSpreadsheetSource(mapping, guardrails), https


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish a validated, complete SoftMouse export to shared storage"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--keyring-service", default="reachAQ-softmouse-publisher")
    args = parser.parse_args(argv)

    value, spreadsheet, https_config = _configuration(args.config)
    credentials = SoftMouseCredentials.from_keyring(args.keyring_service, args.username)
    publisher = SoftMouseExportPublisher(
        export_source=SoftMouseHttpsSource(https_config, credentials),
        destination_directory=Path(value["publicationDirectory"]),
        spreadsheet_source=spreadsheet,
        lock_path=(
            None if value.get("lockPath") is None else Path(value["lockPath"])
        ),
    )
    result = publisher.publish()
    print(
        f"Published {result.total_source_rows} source rows "
        f"({result.tagged_rows} tagged), SHA-256 {result.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
