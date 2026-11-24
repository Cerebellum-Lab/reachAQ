from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Tuple
from urllib.parse import quote

from .base import AnimalMetadataSource, SetupField
from .mapping import SourceTable, table_sha256


_READ_ONLY_QUERY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class DatabaseSource(AnimalMetadataSource):
    """Rows returned by one read-only query.

    A SQLite file needs nothing extra. Any other database is reached through
    an SQLAlchemy URL, which needs SQLAlchemy and that database's driver
    installed; the password is kept as a secret, never in the URL.
    """

    kind = "database"
    label = "Database (SQLite or SQL URL)"
    provider = "database"

    @classmethod
    def setup_fields(cls) -> Tuple[SetupField, ...]:
        return (
            SetupField(
                "url",
                "Database",
                help=(
                    "A SQLite file path, or an SQLAlchemy URL without the "
                    "password, e.g. postgresql+psycopg://host/colony"
                ),
            ),
            SetupField("username", "Username", required=False,
                       help="Only for a server database."),
            SetupField("password", "Password", kind="secret", required=False),
            SetupField(
                "query",
                "Query",
                kind="multiline",
                help="One SELECT returning a row per animal; its column names are mapped next.",
            ),
        )

    def __init__(self, settings, credentials):
        super().__init__(settings, credentials)
        if not _READ_ONLY_QUERY.match(self.settings["query"]):
            raise ValueError("The database query must be a single SELECT (or WITH … SELECT)")

    @property
    def is_sqlite_file(self) -> bool:
        url = self.settings["url"]
        return "://" not in url or url.startswith("sqlite:///")

    def _sqlite_path(self) -> Path:
        url = self.settings["url"]
        return Path(url[len("sqlite:///"):] if url.startswith("sqlite:///") else url).expanduser()

    def describe(self) -> str:
        if self.is_sqlite_file:
            return f"SQLite database {self._sqlite_path()}"
        return f"Database {self._redacted_url()}"

    def _redacted_url(self) -> str:
        return re.sub(r"//[^/@]*@", "//", self.settings["url"])

    def fetch(self, work_directory: Path) -> SourceTable:
        headers, rows = (
            self._fetch_sqlite() if self.is_sqlite_file else self._fetch_sqlalchemy()
        )
        if len(set(headers)) != len(headers):
            raise ValueError("The database query returns two columns with the same name")
        return SourceTable(
            headers=headers,
            rows=rows,
            content_sha256=table_sha256(headers, rows),
            source_name=self.describe(),
            exact_numbers=True,
        )

    def _fetch_sqlite(self):
        path = self._sqlite_path()
        if not path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        uri = f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        try:
            cursor = connection.execute(self.settings["query"])
            headers = tuple(str(item[0]) for item in cursor.description or ())
            rows = tuple(tuple(row) for row in cursor.fetchall())
        finally:
            connection.close()
        return headers, rows

    def _fetch_sqlalchemy(self):
        try:
            import sqlalchemy
        except ImportError as exc:
            raise RuntimeError(
                "This database URL needs SQLAlchemy and the database's driver: "
                "pip install sqlalchemy <driver>"
            ) from exc
        url = sqlalchemy.engine.make_url(self.settings["url"])
        if url.password is not None:
            raise ValueError("Keep the password out of the URL; enter it as the Password")
        username = self.settings.get("username")
        if username:
            url = url.set(username=username)
        password = self.secret("password")
        if password:
            url = url.set(password=password)
        engine = sqlalchemy.create_engine(url)
        try:
            with engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(self.settings["query"]))
                headers = tuple(str(name) for name in result.keys())
                rows = tuple(tuple(row) for row in result.fetchall())
                connection.rollback()
        finally:
            engine.dispose()
        return headers, rows
