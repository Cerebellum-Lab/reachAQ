"""The fake-mice set through every built-in source kind, end to end.

Each kind is set up with the same CLI steps an operator follows (physical
tag, name and RFID columns), refreshed into a scratch cache, and must end with
the same animals, identifiers and duplicate handling. Everything lives in
``tmp_path``: scratch preferences, settings directory and credentials.
"""

import shutil
import sqlite3
from pathlib import Path

import pytest

from reachaq.app.model.animal_registry import AnimalRegistry
from reachaq.app.model.animal_sources import cli
from reachaq.app.model.animal_sources.credentials import CredentialStore
from reachaq.app.model.user_preferences import UserPreferences

FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED = {
    "FAKE-001": ("Fake RFID Mouse 1", "999000000000001"),
    "FAKE-002": ("Fake RFID Mouse 2", "999000000000002"),
    "FAKE-003": ("Fake Barcode Mouse 3", "BC-0003-A"),
    "FAKE-004": ("Fake Barcode Mouse 4", "bc-0004-b"),
    "FAKE-005": ("Fake Untagged Mouse 5", None),
    "FAKE-006": ("Fake Duplicate Mouse 6", None),
    "FAKE-007": ("Fake Duplicate Mouse 7", None),
}
DUPLICATE_WARNING = "RFID 'DUP-0006' is shared by FAKE-006, FAKE-007"
SQLITE_QUERY = "SELECT * FROM mice ORDER BY physical_tag"


class Scratch:
    def __init__(self, root: Path, monkeypatch):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.prefs = root / "prefs.ini"
        self.config = root / "config"
        preferences = UserPreferences(settings_file_path=self.prefs)
        preferences.configuration_location = str(self.config)
        preferences.save()
        self.credentials = CredentialStore(
            keyring_backend=False, fallback_path=root / "credentials.json"
        )
        # cli.main builds its own store; keep it in the scratch directory too.
        monkeypatch.setattr(cli, "CredentialStore", lambda: self.credentials)

    def setup(self, kind, *answers):
        prompt_answers = list(answers)

        def prompt(text):
            assert prompt_answers, f"unexpected prompt {text!r}"
            return prompt_answers.pop(0)

        cli.run_setup(
            UserPreferences(settings_file_path=self.prefs),
            self.credentials,
            kind_name=kind,
            prompt=prompt,
            secret_prompt=lambda _text: "",
        )
        assert prompt_answers == []

    def refresh(self, capsys):
        capsys.readouterr()
        code = cli.main(["--preferences", str(self.prefs), "refresh"])
        return code, capsys.readouterr()

    def animals(self):
        registry = AnimalRegistry(self.config / "softmouse" / "animal-registry.sqlite3")
        return {
            record.identity.subject_id: (record.new_animal_name_candidate, record.physical_rfid)
            for record in registry.list_current_records()
        }


# Column prompts after the kind's own settings: physical tag, name and RFID
# (the suggestions), "skip animals without an RFID" (no).
COLUMNS = ("", "", "", "n")


def assert_fake_mice(scratch, code, output):
    assert code == 0, output.err
    assert "Imported 7 of 7 rows (4 with RFID)" in output.out
    assert DUPLICATE_WARNING in output.out
    assert scratch.animals() == EXPECTED


def test_fixtures_match_the_csv_master(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_fake_mice", FIXTURES / "make_fake_mice.py"
    )
    make_fake_mice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_fake_mice)

    rebuilt = make_fake_mice.build_sqlite(tmp_path / "rebuilt.sqlite")
    with sqlite3.connect(rebuilt) as fresh, sqlite3.connect(FIXTURES / "fake-mice.sqlite") as kept:
        assert fresh.execute(SQLITE_QUERY).fetchall() == kept.execute(SQLITE_QUERY).fetchall()
    openpyxl = pytest.importorskip("openpyxl")
    rebuilt = make_fake_mice.build_xlsx(tmp_path / "rebuilt.xlsx")
    for name in ("mice", "numeric-id"):
        fresh = list(openpyxl.load_workbook(rebuilt)[name].values)
        kept = list(openpyxl.load_workbook(FIXTURES / "fake-mice.xlsx")[name].values)
        assert fresh == kept


def test_csv_spreadsheet(tmp_path, monkeypatch, capsys):
    scratch = Scratch(tmp_path / "csv", monkeypatch)
    scratch.setup("spreadsheet", str(FIXTURES / "fake-mice.csv"), "", *COLUMNS, "")

    assert_fake_mice(scratch, *scratch.refresh(capsys))


def test_xlsx_spreadsheet_with_text_cells(tmp_path, monkeypatch, capsys):
    pytest.importorskip("openpyxl")
    scratch = Scratch(tmp_path / "xlsx", monkeypatch)
    scratch.setup("spreadsheet", str(FIXTURES / "fake-mice.xlsx"), "mice", *COLUMNS, "")

    assert_fake_mice(scratch, *scratch.refresh(capsys))


def test_xlsx_numeric_rfid_cell_is_refused_and_keeps_the_last_cache(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("openpyxl")
    scratch = Scratch(tmp_path / "xlsx-numeric", monkeypatch)
    workbook = tmp_path / "animals.xlsx"
    shutil.copyfile(FIXTURES / "fake-mice.xlsx", workbook)
    scratch.setup("spreadsheet", str(workbook), "mice", *COLUMNS, "")
    assert_fake_mice(scratch, *scratch.refresh(capsys))

    scratch.setup("spreadsheet", str(workbook), "numeric-id", *COLUMNS, "")
    code, output = scratch.refresh(capsys)

    assert code != 0
    assert "numeric cell" in output.err
    assert scratch.animals() == EXPECTED


def test_sqlite_database(tmp_path, monkeypatch, capsys):
    scratch = Scratch(tmp_path / "sqlite", monkeypatch)
    # Database, username (none), password (secret prompt), query.
    scratch.setup(
        "database", str(FIXTURES / "fake-mice.sqlite"), "", SQLITE_QUERY, *COLUMNS, ""
    )

    assert_fake_mice(scratch, *scratch.refresh(capsys))


def test_sqlalchemy_url_with_the_real_library(tmp_path, monkeypatch, capsys):
    # A non-"sqlite:///" URL goes through SQLAlchemy, not the stdlib reader.
    pytest.importorskip("sqlalchemy")
    scratch = Scratch(tmp_path / "sqlalchemy", monkeypatch)
    url = "sqlite+pysqlite:///" + str(FIXTURES / "fake-mice.sqlite")
    scratch.setup("database", url, "", SQLITE_QUERY, *COLUMNS, "")

    assert_fake_mice(scratch, *scratch.refresh(capsys))


def test_shared_publication_round_trip(tmp_path, monkeypatch, capsys):
    share = tmp_path / "share"
    publisher = Scratch(tmp_path / "publisher", monkeypatch)
    publisher.setup(
        "spreadsheet", str(FIXTURES / "fake-mice.csv"), "", *COLUMNS, str(share), ""
    )
    assert_fake_mice(publisher, *publisher.refresh(capsys))
    manifest = share / "spreadsheet-AnimalList-current.manifest.json"
    assert manifest.is_file()

    reader = Scratch(tmp_path / "reader", monkeypatch)
    reader.setup("publication", str(manifest), *COLUMNS)

    assert_fake_mice(reader, *reader.refresh(capsys))
