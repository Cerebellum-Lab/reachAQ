from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

from .base import AnimalMetadataSource, SetupField
from .mapping import ColumnMapping, SourceTable, read_table_file, softmouse_mapping


#: Where the SoftMouse publisher kept its password before sources existed.
LEGACY_KEYRING_SERVICE = "reachAQ-softmouse-publisher"


class SoftMouseSource(AnimalMetadataSource):
    """The active animals of one SoftMouse colony, downloaded with the user's login."""

    kind = "softmouse"
    label = "SoftMouse"
    provider = "softmouse"

    @classmethod
    def setup_fields(cls) -> Tuple[SetupField, ...]:
        return (
            SetupField("username", "SoftMouse username"),
            SetupField("password", "SoftMouse password", kind="secret"),
            SetupField(
                "colony",
                "Colony",
                kind="choice",
                dynamic_choices=True,
                help="Listed after logging in with the username and password.",
            ),
        )

    @classmethod
    def default_mapping(cls) -> ColumnMapping:
        return softmouse_mapping()

    def describe(self) -> str:
        return f"SoftMouse colony {self.settings['colony']!r} as {self.settings['username']}"

    def secret(self, field_name: str):
        value = super().secret(field_name)
        if value is None and field_name == "password":
            value = self.credentials.legacy_keyring_password(
                LEGACY_KEYRING_SERVICE, self.settings.get("username", "")
            )
        return value

    def _credentials(self):
        from tools.softmouse_sync.https_source import SoftMouseCredentials

        username = self.settings["username"]
        password = self.secret("password")
        if not password:
            raise RuntimeError(
                f"No SoftMouse password is stored for {username!r}; set the source up again"
            )
        return SoftMouseCredentials(username, password)

    def _client(self, colony: str, session=None):
        from tools.softmouse_sync.https_source import (
            SoftMouseHttpsConfiguration,
            SoftMouseHttpsSource,
        )

        configuration = SoftMouseHttpsConfiguration(
            **({"base_url": self.settings["base_url"]} if self.settings.get("base_url") else {}),
            colony_name=colony,
        )
        return SoftMouseHttpsSource(
            configuration,
            self._credentials(),
            **({} if session is None else {"session": session}),
        )

    #: Tests replace this with a fake requests session.
    session = None

    def choices(self, field_name: str) -> Sequence[str]:
        if field_name != "colony":
            return super().choices(field_name)
        return self._client("", self.session).list_colonies()

    def fetch(self, work_directory: Path) -> SourceTable:
        if not self.settings.get("colony"):
            raise ValueError("SoftMouse setup is missing: Colony")
        downloaded = self._client(self.settings["colony"], self.session).download(
            Path(work_directory)
        )
        return read_table_file(downloaded)
