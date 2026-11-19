"""Animal metadata sources: the mapping, the spreadsheet and database kinds,
credentials, the stored setup, lab plug-ins and the shared setup steps."""

import csv
import datetime as dt
import json
import os
import sqlite3
import stat
import sys
import types

import openpyxl
import pytest

from tools.acquisition.model.animal_sources import base
from tools.acquisition.model.animal_sources.base import (
    AnimalMetadataSource,
    SetupField,
    available_source_kinds,
)
from tools.acquisition.model.animal_sources.config import (
    SourceConfiguration,
    build_source,
    load_source_configuration,
    save_source_configuration,
)
from tools.acquisition.model.animal_sources.credentials import KEYRING_SERVICE, CredentialStore
from tools.acquisition.model.animal_sources.database import DatabaseSource
from tools.acquisition.model.animal_sources.mapping import (
    AnimalTableMapper,
    ColumnMapping,
    SourceTable,
    read_table_file,
    softmouse_mapping,
    table_sha256,
)
from tools.acquisition.model.animal_sources.source_setup import (
    read_columns,
    save_setup,
    suggest_mapping,
)
from tools.acquisition.model.animal_sources.spreadsheet import SpreadsheetSource


TAG_A = "360002353933099"
TAG_B = "360002353933101"
HEADERS = ("Mouse ID", "RFID", "Status", "Sex", "DOB", "Line", "Cage")
LAB_MAPPING = ColumnMapping(
    external_identity_column="Mouse ID",
    new_animal_name_column="Mouse ID",
    rfid_column="RFID",
    state_column="Status",
    sex_column="Sex",
    date_of_birth_column="DOB",
    strain_column="Line",
    cage_column="Cage",
    ended_states=("dead", "transferred"),
)
ROWS = (
    ("M-1", TAG_A, "alive", "F", "2026-01-02", "C57", "C-1"),
    ("M-2", "", "alive", "M", "2026-01-03", "C57", "C-2"),
    ("M-3", TAG_B, "dead", "M", "2026-01-04", "C57", "C-3"),
)


class FakeKeyring:
    priority = 1

    def __init__(self):
        self.values = {}

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[(service, account)] = value

    def delete_password(self, service, account):
        self.values.pop((service, account), None)


class Preferences:
    animal_metadata_source = ""
    softmouse_manifest_path = ""
    softmouse_name_column = "Physical Tag"

    def save(self):
        pass


def table(rows=ROWS, headers=HEADERS, **kwargs):
    return SourceTable(
        headers=headers,
        rows=rows,
        content_sha256=table_sha256(headers, rows),
        source_name="lab.csv",
        **kwargs,
    )


def write_csv(path, rows=ROWS, headers=HEADERS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def no_keyring(tmp_path):
    return CredentialStore(keyring_backend=False, fallback_path=tmp_path / "secrets.json")


# Mapping ---------------------------------------------------------------------


def test_without_require_rfid_animals_with_no_rfid_are_imported():
    batch = AnimalTableMapper(LAB_MAPPING, provider="spreadsheet").preview_table(table()).batch

    assert [r.identity.subject_id for r in batch.records] == ["M-1", "M-2"]
    assert [r.physical_rfid for r in batch.records] == [TAG_A, None]
    assert batch.ignored_missing_rfid_rows == 0
    assert batch.ignored_ended_rows == 1
    first = batch.records[0]
    assert first.identity.provider == "spreadsheet"
    assert first.identity.subject_id_kind == "subject_id"
    assert (first.sex, first.date_of_birth, first.strain, first.cage) == (
        "F", "2026-01-02", "C57", "C-1",
    )


def test_require_rfid_skips_and_counts_animals_without_one():
    mapping = ColumnMapping(**{**LAB_MAPPING.to_dict(), "require_rfid": True})

    batch = AnimalTableMapper(mapping, provider="spreadsheet").preview_table(table()).batch

    assert [r.identity.subject_id for r in batch.records] == ["M-1"]
    assert batch.ignored_missing_rfid_rows == 1


def test_a_mapping_needs_only_an_id_and_a_name():
    mapping = ColumnMapping(external_identity_column="Mouse ID", new_animal_name_column="Mouse ID")

    batch = AnimalTableMapper(mapping, provider="spreadsheet").preview_table(table()).batch

    assert len(batch.records) == 3
    assert {r.state for r in batch.records} == {"unknown"}


def test_a_mapping_is_validated_and_round_trips():
    with pytest.raises(ValueError, match="external ID"):
        ColumnMapping(external_identity_column=" ", new_animal_name_column="Name")
    with pytest.raises(ValueError, match="needs an RFID column"):
        ColumnMapping(external_identity_column="ID", new_animal_name_column="ID", require_rfid=True)
    with pytest.raises(ValueError, match="Unknown column mapping keys"):
        ColumnMapping.from_dict({**LAB_MAPPING.to_dict(), "colour": "red"})

    assert ColumnMapping.from_dict(json.loads(json.dumps(LAB_MAPPING.to_dict()))) == LAB_MAPPING


def test_a_missing_required_column_and_a_missing_id_are_errors():
    with pytest.raises(ValueError, match="Missing required source column 'Animal'"):
        AnimalTableMapper(
            ColumnMapping(external_identity_column="Animal", new_animal_name_column="Mouse ID"),
            provider="spreadsheet",
        ).preview_table(table())
    with pytest.raises(ValueError, match="row 2 has no 'Mouse ID'"):
        AnimalTableMapper(LAB_MAPPING, provider="spreadsheet").preview_table(
            table(rows=(("", TAG_A, "alive", "F", "", "", ""),))
        )


def test_database_integers_are_exact_but_spreadsheet_numbers_are_refused():
    rows = (("M-1", int(TAG_A), "alive", "F", None, None, None),)
    mapper = AnimalTableMapper(LAB_MAPPING, provider="database")

    assert mapper.preview_table(table(rows, exact_numbers=True)).batch.records[0].physical_rfid == TAG_A
    with pytest.raises(ValueError, match="numeric cell"):
        mapper.preview_table(table(rows))


def test_the_softmouse_mapping_keeps_its_historical_shape():
    mapping = softmouse_mapping("Cage Tag")

    assert mapping.require_rfid
    assert mapping.rfid_column == "Plate ID"
    assert mapping.external_identity_column == "Physical Tag"
    assert mapping.new_animal_name_column == "Cage Tag"
    assert (mapping.profile_id, mapping.version, mapping.subject_id_kind) == (
        "softmouse-default", 2, "physical_tag",
    )


# Spreadsheet -----------------------------------------------------------------


def test_spreadsheet_source_reads_a_csv_through_a_local_copy(tmp_path):
    path = write_csv(tmp_path / "animals.csv")
    source = SpreadsheetSource({"path": str(path)}, no_keyring(tmp_path))
    work = tmp_path / "work"
    work.mkdir()

    fetched = source.fetch(work)

    assert fetched.headers == HEADERS
    assert len(fetched.rows) == 3
    assert fetched.source_name == "animals.csv"
    assert fetched.path.parent == work
    assert fetched.content_sha256 == source.current_version()


def test_spreadsheet_source_reads_a_named_or_the_first_xlsx_sheet(tmp_path):
    path = tmp_path / "animals.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Notes"
    workbook.active.append(["not", "animals"])
    sheet = workbook.create_sheet("Colony")
    sheet.append(list(HEADERS))
    sheet.append(["M-1", TAG_A, "alive", "F", dt.datetime(2026, 1, 2), "C57", "C-1"])
    workbook.save(path)
    workbook.close()

    named = SpreadsheetSource({"path": str(path), "sheet": "Colony"}, no_keyring(tmp_path))
    fetched = named.fetch(tmp_path)
    batch = AnimalTableMapper(LAB_MAPPING, provider="spreadsheet").preview_table(fetched).batch

    assert fetched.sheet == "Colony"
    assert batch.records[0].date_of_birth == "2026-01-02"
    assert read_table_file(path).sheet == "Notes"
    with pytest.raises(ValueError, match="Missing worksheet 'Absent'"):
        SpreadsheetSource({"path": str(path), "sheet": "Absent"}, no_keyring(tmp_path)).fetch(tmp_path)


def test_spreadsheet_source_needs_a_path_and_a_known_format(tmp_path):
    with pytest.raises(ValueError, match="missing: Spreadsheet file"):
        SpreadsheetSource({}, no_keyring(tmp_path))
    other = tmp_path / "animals.ods"
    other.write_text("x")
    with pytest.raises(ValueError, match=".xlsx or .csv"):
        SpreadsheetSource({"path": str(other)}, no_keyring(tmp_path)).fetch(tmp_path)


# Database --------------------------------------------------------------------


def colony_database(path):
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE mice (mouse_id TEXT, rfid INTEGER, status TEXT, sex TEXT, born TEXT)"
        )
        db.executemany(
            "INSERT INTO mice VALUES (?, ?, ?, ?, ?)",
            [("M-1", int(TAG_A), "alive", "F", "2026-01-02"), ("M-2", None, "alive", "M", None)],
        )
    return path


def test_sqlite_database_source_reads_a_query_read_only(tmp_path):
    path = colony_database(tmp_path / "colony.sqlite")
    source = DatabaseSource(
        {"url": str(path), "query": "SELECT mouse_id, rfid, status FROM mice ORDER BY mouse_id"},
        no_keyring(tmp_path),
    )

    fetched = source.fetch(tmp_path)
    mapping = ColumnMapping(
        external_identity_column="mouse_id",
        new_animal_name_column="mouse_id",
        rfid_column="rfid",
        state_column="status",
    )
    batch = AnimalTableMapper(mapping, provider="database").preview_table(fetched).batch

    assert fetched.headers == ("mouse_id", "rfid", "status")
    assert fetched.exact_numbers
    assert [r.physical_rfid for r in batch.records] == [TAG_A, None]
    assert source.describe() == f"SQLite database {path}"
    # The same rows hash the same, so an unchanged database is an unchanged import.
    assert source.fetch(tmp_path).content_sha256 == fetched.content_sha256


def test_the_database_query_cannot_write(tmp_path):
    path = colony_database(tmp_path / "colony.sqlite")
    with pytest.raises(ValueError, match="single SELECT"):
        DatabaseSource({"url": str(path), "query": "DELETE FROM mice"}, no_keyring(tmp_path))
    sneaky = DatabaseSource(
        {"url": f"sqlite:///{path}", "query": "WITH x AS (SELECT 1) INSERT INTO mice SELECT * FROM mice"},
        no_keyring(tmp_path),
    )
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        sneaky.fetch(tmp_path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM mice").fetchone()[0] == 2


def test_a_server_url_without_sqlalchemy_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    source = DatabaseSource(
        {"url": "postgresql+psycopg://db.example/colony", "query": "SELECT 1"},
        no_keyring(tmp_path),
    )

    with pytest.raises(RuntimeError, match="needs SQLAlchemy"):
        source.fetch(tmp_path)


def test_a_server_url_gets_its_password_from_the_credential_store(tmp_path, monkeypatch):
    seen = {}

    class Url:
        def __init__(self, text, **parts):
            self.text, self.parts = text, parts
            self.password = parts.get("password")

        def set(self, **parts):
            return Url(self.text, **{**self.parts, **parts})

    class Result:
        def keys(self):
            return ["mouse_id"]

        def fetchall(self):
            return [("M-1",)]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement):
            seen["query"] = statement
            return Result()

        def rollback(self):
            seen["rolled_back"] = True

    class Engine:
        def connect(self):
            return Connection()

        def dispose(self):
            seen["disposed"] = True

    def create_engine(url):
        seen["url"] = url
        return Engine()

    fake = types.SimpleNamespace(
        engine=types.SimpleNamespace(make_url=lambda text: Url(text)),
        create_engine=create_engine,
        text=lambda query: f"text({query})",
    )
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake)
    credentials = no_keyring(tmp_path)
    settings = {"url": "postgresql+psycopg://db.example/colony", "username": "lab", "query": "SELECT mouse_id FROM mice"}
    source = DatabaseSource(settings, credentials)
    credentials.set(source.secret_key("password"), "s3cret")

    fetched = source.fetch(tmp_path)

    assert seen["url"].parts == {"username": "lab", "password": "s3cret"}
    assert seen["query"] == "text(SELECT mouse_id FROM mice)"
    assert seen["rolled_back"] and seen["disposed"]
    assert fetched.rows == (("M-1",),)
    assert "s3cret" not in source.describe()

    in_url = DatabaseSource({**settings, "url": "postgresql://lab:pw@db.example/colony"}, credentials)
    fake.engine.make_url = lambda text: Url(text, password="pw")
    with pytest.raises(ValueError, match="password out of the URL"):
        in_url.fetch(tmp_path)
    assert "pw" not in in_url.describe()


# Credentials -----------------------------------------------------------------


def test_secrets_go_to_the_keyring_when_it_works(tmp_path):
    keyring = FakeKeyring()
    store = CredentialStore(keyring_backend=keyring, fallback_path=tmp_path / "secrets.json")

    assert store.set("softmouse/password/lab", "pw") == "the OS keyring"
    assert keyring.values == {(KEYRING_SERVICE, "softmouse/password/lab"): "pw"}
    assert store.get("softmouse/password/lab") == "pw"
    assert not (tmp_path / "secrets.json").exists()


def test_without_a_keyring_secrets_go_to_a_user_only_file(tmp_path):
    store = no_keyring(tmp_path)

    where = store.set("database/password/lab", "pw")

    assert where == str(tmp_path / "secrets.json")
    assert CredentialStore(keyring_backend=False, fallback_path=tmp_path / "secrets.json").get(
        "database/password/lab"
    ) == "pw"
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "secrets.json").stat().st_mode) == 0o600


def test_a_failing_keyring_falls_back_to_the_file(tmp_path):
    class Broken(FakeKeyring):
        def set_password(self, *_):
            raise RuntimeError("locked")

    store = CredentialStore(keyring_backend=Broken(), fallback_path=tmp_path / "secrets.json")

    assert store.set("k", "v") == str(tmp_path / "secrets.json")
    assert store.get("k") == "v"


def test_an_unusable_keyring_backend_is_not_used(tmp_path):
    class Fail(FakeKeyring):
        priority = 0

    assert not CredentialStore(keyring_backend=Fail(), fallback_path=tmp_path / "s.json").uses_keyring


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
def test_a_credential_file_others_can_read_is_refused(tmp_path):
    store = no_keyring(tmp_path)
    store.set("k", "v")
    os.chmod(tmp_path / "secrets.json", 0o644)

    with pytest.raises(RuntimeError, match="chmod 600"):
        store.get("k")


# Stored setup ----------------------------------------------------------------


def test_the_setup_round_trips_through_the_preferences():
    preferences = Preferences()
    configuration = SourceConfiguration(
        kind="spreadsheet",
        settings={"path": "/srv/lab/animals.xlsx"},
        mapping=LAB_MAPPING,
        publish_directory="/srv/share/reachaq",
    )

    save_source_configuration(preferences, configuration)

    assert load_source_configuration(preferences) == configuration
    assert "secret" not in preferences.animal_metadata_source


def test_a_computer_set_up_before_sources_reads_its_publication():
    preferences = Preferences()
    preferences.softmouse_manifest_path = "/srv/share/SoftMouse-AnimalList-current.manifest.json"
    preferences.softmouse_name_column = "Cage Tag"

    configuration = load_source_configuration(preferences)

    assert configuration.kind == "publication"
    assert configuration.settings == {"manifest": preferences.softmouse_manifest_path}
    assert configuration.mapping == softmouse_mapping("Cage Tag")
    assert load_source_configuration(Preferences()) is None


def test_build_source_reports_an_uninstalled_kind_and_a_republished_publication(tmp_path):
    with pytest.raises(ValueError, match="'labdb' is not installed"):
        build_source(SourceConfiguration("labdb", {}, None), credentials=no_keyring(tmp_path))
    with pytest.raises(ValueError, match="cannot also be published"):
        build_source(
            SourceConfiguration(
                "publication", {"manifest": "/m.json"}, None, publish_directory="/srv"
            ),
            credentials=no_keyring(tmp_path),
        )


# Lab plug-ins ----------------------------------------------------------------


class LabSource(AnimalMetadataSource):
    kind = "labdb"
    label = "Lab database"
    provider = "labdb"

    @classmethod
    def setup_fields(cls):
        return (SetupField("server", "Server"), SetupField("token", "Token", kind="secret"))

    def fetch(self, work_directory):
        return table()


class EntryPoint:
    def __init__(self, name, value):
        self.name, self._value = name, value

    def load(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def test_a_lab_source_is_found_through_its_entry_point():
    kinds = available_source_kinds([EntryPoint("labdb", LabSource)])

    assert kinds["labdb"] is LabSource
    assert {"softmouse", "spreadsheet", "database", "publication"} <= set(kinds)


def test_bad_plug_ins_are_left_out_without_stopping_the_app(caplog):
    class Misnamed(LabSource):
        kind = "other"

    kinds = available_source_kinds(
        [
            EntryPoint("spreadsheet", LabSource),
            EntryPoint("broken", ImportError("no module")),
            EntryPoint("notasource", dict),
            EntryPoint("wrong", Misnamed),
        ]
    )

    assert kinds["spreadsheet"] is SpreadsheetSource
    assert not {"broken", "notasource", "wrong"} & set(kinds)
    assert "would replace a built-in" in caplog.text


def test_the_installed_entry_point_group_is_read(monkeypatch):
    import importlib.metadata as metadata

    monkeypatch.setattr(
        metadata, "entry_points",
        lambda **kwargs: [EntryPoint("labdb", LabSource)] if kwargs.get("group") == base.ENTRY_POINT_GROUP else [],
    )

    assert available_source_kinds()["labdb"] is LabSource


# Shared setup steps ----------------------------------------------------------


def test_setup_saves_settings_in_preferences_and_secrets_in_the_store(tmp_path):
    preferences = Preferences()
    credentials = no_keyring(tmp_path)

    saved = save_setup(
        preferences, credentials, LabSource, {"server": "db.lab", "token": "t0ken"}, LAB_MAPPING,
    )

    assert saved.configuration.settings == {"server": "db.lab"}
    assert "t0ken" not in preferences.animal_metadata_source
    assert credentials.get("labdb/token/") == "t0ken"
    assert saved.secret_locations == (str(tmp_path / "secrets.json"),)


def test_setup_refuses_a_missing_secret_but_keeps_a_stored_one(tmp_path):
    credentials = no_keyring(tmp_path)
    with pytest.raises(ValueError, match="missing: Token"):
        save_setup(Preferences(), credentials, LabSource, {"server": "db.lab"}, LAB_MAPPING)

    credentials.set("labdb/token/", "old")
    saved = save_setup(Preferences(), credentials, LabSource, {"server": "db.lab", "token": ""}, LAB_MAPPING)
    assert saved.secret_locations == ()


def test_read_columns_uses_an_entered_secret_before_it_is_saved(tmp_path):
    seen = []

    class TokenSource(LabSource):
        def fetch(self, work_directory):
            seen.append(self.secret("token"))
            return table()

    columns = read_columns(TokenSource, {"server": "db.lab", "token": "fresh"}, no_keyring(tmp_path))

    assert seen == ["fresh"]
    assert columns.headers == HEADERS
    assert no_keyring(tmp_path).get("labdb/token/") is None


def test_suggested_mapping_prefers_the_kinds_own_then_header_names():
    from tools.acquisition.model.animal_sources.softmouse import SoftMouseSource

    softmouse_headers = ("SoftMouse ID", "Physical Tag", "Plate ID", "State", "Sex")
    assert suggest_mapping(SoftMouseSource, softmouse_headers)["rfid_column"] == "Plate ID"

    guess = suggest_mapping(SpreadsheetSource, HEADERS)
    assert guess["external_identity_column"] == "Mouse ID"
    assert guess["rfid_column"] == "RFID"
    assert guess["state_column"] == "Status"
    assert guess["date_of_birth_column"] == "DOB"
    assert guess["genotype_column"] is None
