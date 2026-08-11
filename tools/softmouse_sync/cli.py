from __future__ import annotations

import argparse
import getpass
import json
import os
import tempfile
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


DEFAULT_KEYRING_SERVICE = "reachAQ-softmouse-publisher"


def _read_configuration(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Publisher configuration must be a JSON object")
    return value


def _configuration(path: Path, *, value=None):
    value = _read_configuration(path) if value is None else value
    mapping = SoftMouseMappingProfile(**value.get("mapping", {}))
    guardrails = ImportGuardrails(**value.get("guardrails", {}))
    https = SoftMouseHttpsConfiguration(**value["https"])
    return value, SoftMouseSpreadsheetSource(mapping, guardrails), https


def _credential_location(
    value,
    *,
    username_override=None,
    service_override=None,
):
    configured = value.get("credentials", {})
    if not isinstance(configured, dict):
        raise ValueError("credentials must be a JSON object")
    username = (username_override or configured.get("username") or "").strip()
    service = (
        service_override
        or configured.get("keyringService")
        or DEFAULT_KEYRING_SERVICE
    ).strip()
    if not username:
        raise ValueError(
            "No SoftMouse username is configured; run with "
            "--configure-credentials once"
        )
    if not service:
        raise ValueError("The keyring service name cannot be empty")
    return service, username


def _write_private_configuration(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _configure_credentials(
    path: Path,
    value,
    *,
    username=None,
    keyring_service=None,
    username_prompt=None,
    password_prompt=None,
    keyring_backend=None,
):
    configured = value.get("credentials", {})
    if not isinstance(configured, dict):
        raise ValueError("credentials must be a JSON object")
    existing_username = str(configured.get("username") or "").strip()
    if username is None:
        username_prompt = username_prompt or input
        prompt = "SoftMouse username"
        if existing_username:
            prompt += f" [{existing_username}]"
        entered = username_prompt(prompt + ": ").strip()
        username = entered or existing_username
    username = str(username or "").strip()
    if not username:
        raise ValueError("SoftMouse username cannot be empty")

    service = str(
        keyring_service
        or configured.get("keyringService")
        or DEFAULT_KEYRING_SERVICE
    ).strip()
    if not service:
        raise ValueError("The keyring service name cannot be empty")

    password_prompt = password_prompt or getpass.getpass
    password = password_prompt("SoftMouse password: ")
    if not password:
        raise ValueError("SoftMouse password cannot be empty")
    if keyring_backend is None:
        try:
            import keyring as keyring_backend
        except ImportError as exc:
            raise RuntimeError(
                "keyring is required to store SoftMouse credentials"
            ) from exc
    keyring_backend.set_password(service, username, password)
    if keyring_backend.get_password(service, username) != password:
        raise RuntimeError("The OS keyring did not return the stored password")

    value["credentials"] = {
        "username": username,
        "keyringService": service,
    }
    _write_private_configuration(path, value)
    return service, username


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish a validated, complete SoftMouse export to shared storage"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--configure-credentials",
        action="store_true",
        help="prompt once for the username/password and store them securely",
    )
    parser.add_argument(
        "--username",
        help="temporary username override (normally read from the config)",
    )
    parser.add_argument(
        "--keyring-service",
        help="temporary keyring-service override",
    )
    args = parser.parse_args(argv)

    value = _read_configuration(args.config)
    if args.configure_credentials:
        service, username = _configure_credentials(
            args.config,
            value,
            username=args.username,
            keyring_service=args.keyring_service,
        )
        print(
            f"Credentials stored for {username!r} in OS keyring service "
            f"{service!r}; the password was not written to the config file."
        )
        return 0

    service, username = _credential_location(
        value,
        username_override=args.username,
        service_override=args.keyring_service,
    )
    value, spreadsheet, https_config = _configuration(args.config, value=value)
    credentials = SoftMouseCredentials.from_keyring(service, username)
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
