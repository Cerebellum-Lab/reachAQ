from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

from tools.acquisition.model.softmouse_spreadsheet_source import (
    SoftMouseSpreadsheetSource,
)

from .https_source import (
    SoftMouseCredentials,
    SoftMouseHttpsConfiguration,
    SoftMouseHttpsSource,
)
from .publisher import SoftMouseExportPublisher


DEFAULT_KEYRING_SERVICE = "reachAQ-softmouse-publisher"
KEYRING_USERNAME_ACCOUNT = "__username__"


logger = logging.getLogger(__name__)


def _preferred_publication() -> tuple[Path | None, Path | None]:
    """The directory and mount the app's SoftMouse preferences name.

    The directory is the one holding softmouse/manifest_path, so the publisher
    writes where the app reads; the mount is softmouse/publication_mount.
    Either is None when its preference is empty.
    """
    from tools.acquisition.model.user_preferences import UserPreferences

    preferences = UserPreferences()
    manifest = preferences.softmouse_manifest_path
    mount = preferences.softmouse_publication_mount
    return (Path(manifest).parent if manifest else None), (Path(mount) if mount else None)


def _ensure_publication_directory(
    destination: Path,
    *,
    mount_root: Path | None = None,
    is_mount=os.path.ismount,
) -> Path:
    destination = Path(destination)
    if mount_root is not None and not is_mount(Path(mount_root)):
        raise RuntimeError(
            f"The publication mount {mount_root} is not mounted; publication stopped"
        )
    if not destination.parent.is_dir():
        raise RuntimeError(
            f"Publication parent does not exist: {destination.parent}"
        )
    destination.mkdir(exist_ok=True)
    return destination


def _stored_username(keyring_backend, service: str = DEFAULT_KEYRING_SERVICE):
    username = keyring_backend.get_password(service, KEYRING_USERNAME_ACCOUNT)
    if username:
        return username.strip()
    # Migrate credentials stored by the earlier config-based setup without
    # requiring the user to enter them again.
    get_credential = getattr(keyring_backend, "get_credential", None)
    if get_credential is not None:
        credential = get_credential(service, None)
        if credential is not None and credential.username != KEYRING_USERNAME_ACCOUNT:
            username = credential.username.strip()
            if username:
                keyring_backend.set_password(
                    service,
                    KEYRING_USERNAME_ACCOUNT,
                    username,
                )
                return username
    return None


def _configure_credentials(
    *,
    username=None,
    username_prompt=None,
    password_prompt=None,
    keyring_backend=None,
):
    if keyring_backend is None:
        try:
            import keyring as keyring_backend
        except ImportError as exc:
            raise RuntimeError(
                "keyring is required to store SoftMouse credentials"
            ) from exc
    existing_username = _stored_username(keyring_backend)
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

    password_prompt = password_prompt or getpass.getpass
    password = password_prompt("SoftMouse password: ")
    if not password:
        raise ValueError("SoftMouse password cannot be empty")
    keyring_backend.set_password(DEFAULT_KEYRING_SERVICE, username, password)
    keyring_backend.set_password(
        DEFAULT_KEYRING_SERVICE,
        KEYRING_USERNAME_ACCOUNT,
        username,
    )
    if keyring_backend.get_password(DEFAULT_KEYRING_SERVICE, username) != password:
        raise RuntimeError("The OS keyring did not return the stored password")
    if (
        keyring_backend.get_password(
            DEFAULT_KEYRING_SERVICE, KEYRING_USERNAME_ACCOUNT
        )
        != username
    ):
        raise RuntimeError("The OS keyring did not return the stored username")
    return username


def _credentials_from_keyring(keyring_backend=None) -> SoftMouseCredentials:
    if keyring_backend is None:
        try:
            import keyring as keyring_backend
        except ImportError as exc:
            raise RuntimeError(
                "keyring is required to read SoftMouse credentials"
            ) from exc
    username = _stored_username(keyring_backend)
    if not username:
        raise RuntimeError(
            "No SoftMouse credentials are stored; run "
            "`python -m tools.softmouse_sync.cli --setup` once"
        )
    password = keyring_backend.get_password(DEFAULT_KEYRING_SERVICE, username)
    if not password:
        raise RuntimeError(
            "The stored SoftMouse username has no password; run "
            "`python -m tools.softmouse_sync.cli --setup` again"
        )
    return SoftMouseCredentials(username, password)


def main(argv=None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    parser = argparse.ArgumentParser(
        description="Publish the Christie SoftMouse active-animal export to the shared publication directory"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="prompt once for the SoftMouse username and password",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="publication directory (default: the folder of the app's "
             "softmouse/manifest_path preference)",
    )
    parser.add_argument(
        "--require-mount",
        type=Path,
        help="refuse to publish unless this path is a mount point (default: "
             "the softmouse/publication_mount preference; none when empty)",
    )
    parser.add_argument(
        "--configure-credentials",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    try:
        if args.setup or args.configure_credentials:
            username = _configure_credentials()
            logger.info(
                "SoftMouse credentials stored in OS keyring: service=%s username=%s",
                DEFAULT_KEYRING_SERVICE,
                username,
            )
            print(
                f"SoftMouse credentials stored securely for {username!r}. "
                "No configuration file is needed."
            )
            return 0

        credentials = _credentials_from_keyring()
        destination, mount_root = args.destination, args.require_mount
        if destination is None or mount_root is None:
            preferred_destination, preferred_mount = _preferred_publication()
            destination = destination or preferred_destination
            mount_root = mount_root or preferred_mount
        if destination is None:
            raise RuntimeError(
                "No SoftMouse publication directory is configured; set the "
                "Shared publication preference or pass --destination"
            )
        destination = _ensure_publication_directory(destination, mount_root=mount_root)
        logger.info(
            "SoftMouse publication requested: colony=%s destination=%s",
            SoftMouseHttpsConfiguration().colony_name,
            destination,
        )
        publisher = SoftMouseExportPublisher(
            export_source=SoftMouseHttpsSource(
                SoftMouseHttpsConfiguration(), credentials
            ),
            destination_directory=destination,
            spreadsheet_source=SoftMouseSpreadsheetSource(),
        )
        result = publisher.publish()
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        logger.exception("SoftMouse publication stopped")
        print(f"SoftMouse publication stopped: {exc}", file=sys.stderr)
        return 1
    logger.info(
        "SoftMouse publication complete: rows=%d tagged=%d sha256=%s export=%s",
        result.total_source_rows,
        result.tagged_rows,
        result.sha256,
        result.export_path,
    )
    print(
        f"Published {result.total_source_rows} Christie active-animal rows "
        f"({result.tagged_rows} tagged), SHA-256 {result.sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
