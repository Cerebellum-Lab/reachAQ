"""The SoftMouse source against a recorded-shape fake of the site: no real login."""

import pytest

from softmouse_https_source_test import COLONY, Session  # noqa: E402

from reachaq.app.model.animal_metadata_sync import AnimalMetadataSyncService
from reachaq.app.model.animal_registry import AnimalRegistry
from reachaq.app.model.animal_sources.config import SourceConfiguration, build_source
from reachaq.app.model.animal_sources.credentials import KEYRING_SERVICE, CredentialStore
from reachaq.app.model.animal_sources.softmouse import (
    LEGACY_KEYRING_SERVICE,
    SoftMouseSource,
)
from reachaq.app.model.animal_sources.source_setup import list_choices, read_columns


class FakeKeyring:
    priority = 1

    def __init__(self, values=None):
        self.values = dict(values or {})

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[(service, account)] = value


@pytest.fixture
def session(monkeypatch):
    fake = Session()
    monkeypatch.setattr(SoftMouseSource, "session", fake)
    return fake


def store(tmp_path, values=None):
    return CredentialStore(keyring_backend=FakeKeyring(values), fallback_path=tmp_path / "s.json")


def test_setup_lists_the_colonies_with_the_entered_login(tmp_path, session):
    colonies = list_choices(
        SoftMouseSource, {"username": "ben", "password": "secret"}, "colony", store(tmp_path)
    )

    assert colonies == [COLONY, "Other Colony"]


def test_setup_reads_the_export_columns_for_the_chosen_colony(tmp_path, session):
    table = read_columns(
        SoftMouseSource,
        {"username": "ben", "password": "secret", "colony": COLONY},
        store(tmp_path),
    )

    assert table.headers[:3] == ("SoftMouse ID", "Animal SID", "Physical Tag")
    assert len(table.rows) == 2


def test_the_softmouse_source_refreshes_the_cache_with_its_stored_password(tmp_path, session):
    credentials = store(tmp_path, {(KEYRING_SERVICE, "softmouse/password/ben"): "secret"})
    configured = build_source(
        SourceConfiguration(
            "softmouse",
            {"username": "ben", "colony": COLONY},
            SoftMouseSource.default_mapping(),
        ),
        credentials=credentials,
    )
    service = AnimalMetadataSyncService(
        configured=configured,
        local_staging_directory=tmp_path / "local",
        registry=AnimalRegistry(tmp_path / "registry.sqlite3"),
        can_refresh=lambda: True,
    )

    result = service.refresh_now()

    assert [r.identity.subject_id for r in result.preview.batch.records] == ["PT-1", "PT-2"]
    assert service.registry.resolve_rfid("360002353933099").identity.provider == "softmouse"
    assert service.refresh_due()  # SoftMouse has no cheap version, so it is always fetched
    assert configured.source.describe() == f"SoftMouse colony {COLONY!r} as ben"


def test_a_password_from_the_old_publisher_service_still_works(tmp_path, session):
    credentials = store(tmp_path, {(LEGACY_KEYRING_SERVICE, "ben"): "secret"})
    source = SoftMouseSource({"username": "ben", "colony": COLONY}, credentials)

    assert len(source.fetch(tmp_path).rows) == 2


def test_setup_can_keep_the_old_publishers_password(tmp_path, session):
    from reachaq.app.model.animal_sources.source_setup import save_setup

    class Preferences:
        animal_metadata_source = ""

    credentials = store(tmp_path, {(LEGACY_KEYRING_SERVICE, "ben"): "secret"})
    saved = save_setup(
        Preferences(),
        credentials,
        SoftMouseSource,
        {"username": "ben", "password": "", "colony": COLONY},
        SoftMouseSource.default_mapping(),
    )

    assert saved.secret_locations == ()


def test_no_password_and_no_colony_are_reported(tmp_path, session):
    with pytest.raises(RuntimeError, match="No SoftMouse password is stored for 'ben'"):
        SoftMouseSource({"username": "ben", "colony": COLONY}, store(tmp_path)).fetch(tmp_path)
    credentials = store(tmp_path, {(KEYRING_SERVICE, "softmouse/password/ben"): "secret"})
    with pytest.raises(ValueError, match="missing: Colony"):
        SoftMouseSource({"username": "ben"}, credentials).fetch(tmp_path)
