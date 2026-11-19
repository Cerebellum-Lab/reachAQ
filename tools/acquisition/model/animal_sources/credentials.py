"""Source passwords and tokens, kept out of the preferences and the repository.

The OS keyring is used when it works. A computer without one (a headless
publisher with no Secret Service, say) falls back to a JSON file only its
user can read, in the user's own config directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from autotrainer.core.logging import get_verbose_logger


logger = get_verbose_logger(__name__)

KEYRING_SERVICE = "reachAQ-animal-metadata"
FALLBACK_FILE_NAME = "animal-metadata-credentials.json"


def user_config_directory() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "reachAQ"


def _usable_keyring(backend) -> bool:
    if backend is None:
        return False
    try:
        priority = float(getattr(backend, "priority", 1))
    except (TypeError, ValueError):
        priority = 1.0
    if priority <= 0:
        return False
    name = f"{type(backend).__module__}.{type(backend).__name__}"
    return not name.startswith("keyring.backends.fail")


class CredentialStore:
    """Read and write source secrets by key.

    ``keyring_backend`` is the keyring module or a backend object; None loads
    the ``keyring`` package, and ``False`` skips the keyring entirely.
    """

    def __init__(self, *, keyring_backend=None, fallback_path: Optional[Path] = None):
        if keyring_backend is None:
            try:
                import keyring

                keyring_backend = keyring.get_keyring()
            except Exception:
                keyring_backend = False
        if keyring_backend is not False and hasattr(keyring_backend, "get_keyring"):
            keyring_backend = keyring_backend.get_keyring()
        self._keyring = keyring_backend if _usable_keyring(keyring_backend or None) else None
        self.fallback_path = Path(
            fallback_path or (user_config_directory() / FALLBACK_FILE_NAME)
        )

    @property
    def uses_keyring(self) -> bool:
        return self._keyring is not None

    def get(self, key: str) -> Optional[str]:
        if self._keyring is not None:
            try:
                value = self._keyring.get_password(KEYRING_SERVICE, key)
            except Exception as exc:
                logger.warning("OS keyring read failed; trying the credential file: %s", exc)
            else:
                if value is not None:
                    return value
        return self._read_file().get(key)

    def legacy_keyring_password(self, service: str, account: str) -> Optional[str]:
        """A secret stored under an older service name, for migration."""
        if self._keyring is None:
            return None
        try:
            return self._keyring.get_password(service, account)
        except Exception:
            return None

    def set(self, key: str, value: str) -> str:
        """Store a secret; returns where it went, for the operator to see."""
        if not value:
            raise ValueError("A stored secret cannot be empty")
        if self._keyring is not None:
            try:
                self._keyring.set_password(KEYRING_SERVICE, key, value)
                if self._keyring.get_password(KEYRING_SERVICE, key) != value:
                    raise RuntimeError("the OS keyring did not return the stored value")
            except Exception as exc:
                logger.warning(
                    "OS keyring write failed; using the user-only credential file: %s", exc
                )
            else:
                self._remove_from_file(key)
                return "the OS keyring"
        values = self._read_file()
        values[key] = value
        self._write_file(values)
        logger.warning(
            "Animal metadata secret stored in a user-only file, not an OS keyring: %s",
            self.fallback_path,
        )
        return str(self.fallback_path)

    def delete(self, key: str) -> None:
        if self._keyring is not None:
            try:
                self._keyring.delete_password(KEYRING_SERVICE, key)
            except Exception:
                pass
        self._remove_from_file(key)

    def _remove_from_file(self, key: str) -> None:
        values = self._read_file()
        if key in values:
            del values[key]
            self._write_file(values)

    def _read_file(self) -> Dict[str, str]:
        path = self.fallback_path
        if not path.is_file():
            return {}
        if os.name == "posix" and path.stat().st_mode & 0o077:
            raise RuntimeError(
                f"{path} can be read by other users; run `chmod 600 {path}` "
                "or set the source up again"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"{path} is not a credential file")
        return {str(key): str(item) for key, item in value.items()}

    def _write_file(self, values: Dict[str, str]) -> None:
        path = self.fallback_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(path.parent, 0o700)
        temporary = path.with_name(path.name + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(values, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if os.name == "posix":
            os.chmod(temporary, 0o600)
        os.replace(temporary, path)
