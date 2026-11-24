"""`reachaq-animal-metadata`: interactive setup, refresh, status and publish."""

import pytest

from softmouse_https_source_test import COLONY, Session  # noqa: E402

from reachaq.app.model.animal_sources import cli
from reachaq.app.model.animal_sources.config import load_source_configuration
from reachaq.app.model.animal_sources.credentials import KEYRING_SERVICE, CredentialStore
from reachaq.app.model.animal_sources.softmouse import SoftMouseSource
from reachaq.app.model.user_preferences import UserPreferences


TAG = "360002353933099"


class FakeKeyring:
    priority = 1

    def __init__(self):
        self.values = {}

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[(service, account)] = value


def answers(*values):
    remaining = list(values)

    def prompt(text):
        assert remaining, f"unexpected prompt {text!r}"
        return remaining.pop(0)

    prompt.remaining = remaining
    return prompt


@pytest.fixture
def preferences(tmp_path):
    preferences = UserPreferences(settings_file_path=tmp_path / "prefs.ini")
    preferences.configuration_location = str(tmp_path / "config")
    return preferences


def test_softmouse_setup_logs_in_lists_colonies_and_keeps_the_password_out_of_prefs(
    tmp_path, monkeypatch, preferences
):
    monkeypatch.setattr(SoftMouseSource, "session", Session())
    keyring = FakeKeyring()
    store = CredentialStore(keyring_backend=keyring, fallback_path=tmp_path / "s.json")
    prompt = answers(
        "ben",        # username
        "1",          # colony: the first one listed
        "",           # physical tag column: the suggested Physical Tag
        "",           # name column: the suggested Physical Tag
        "",           # RFID column: the suggested Plate ID
        "",           # keep "skip animals without RFID" (SoftMouse: yes)
        str(tmp_path / "share"),  # publish to
        "",           # no mount check
    )

    kind = cli.run_setup(
        preferences, store, kind_name="softmouse", prompt=prompt, secret_prompt=lambda _t: "secret"
    )

    assert kind == "softmouse"
    assert prompt.remaining == []
    configuration = load_source_configuration(
        UserPreferences(settings_file_path=tmp_path / "prefs.ini")
    )
    assert configuration.settings == {"username": "ben", "colony": COLONY}
    assert configuration.mapping == SoftMouseSource.default_mapping()
    assert configuration.publish_directory == str(tmp_path / "share")
    assert keyring.values == {(KEYRING_SERVICE, "softmouse/password/ben"): "secret"}
    assert "secret" not in (tmp_path / "prefs.ini").read_text()


def test_spreadsheet_setup_maps_columns_then_refresh_and_status_use_it(
    tmp_path, preferences, capsys, monkeypatch
):
    sheet = tmp_path / "animals.csv"
    sheet.write_text(f"Mouse,Chip,Sex\nM-1,{TAG},F\nM-2,,M\n", encoding="utf-8")
    store = CredentialStore(keyring_backend=False, fallback_path=tmp_path / "s.json")
    prompt = answers(
        str(sheet), "",               # path, no sheet
        "Alt. ID",                    # not a column in the file: asked again
        "Mouse",                      # physical tag column
        "",                           # name column: defaults to the tag
        "Chip",                       # RFID column
        "n",                          # keep animals without RFID
        "",                           # no publication
    )

    cli.run_setup(preferences, store, kind_name="spreadsheet", prompt=prompt, secret_prompt=None)
    assert prompt.remaining == []
    configuration = load_source_configuration(preferences)
    assert configuration.mapping.external_identity_column == "Mouse"
    assert configuration.mapping.new_animal_name_column == "Mouse"
    assert configuration.mapping.rfid_column == "Chip"
    # Filled in by name, not asked.
    assert configuration.mapping.sex_column == "Sex"
    assert configuration.mapping.state_column is None
    assert not configuration.mapping.require_rfid

    monkeypatch.setattr(cli, "_preferences", lambda _path: preferences)
    monkeypatch.setattr(cli, "CredentialStore", lambda: store)
    assert cli.main(["refresh"]) == 0
    assert "Imported 2 of 2 rows" in capsys.readouterr().out
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert f"Source: Spreadsheet {sheet}" in out
    assert "Cache: 2 animals from 2 rows" in out


def test_publish_without_a_source_explains_setup(tmp_path, preferences, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_preferences", lambda _path: preferences)

    assert cli.main(["publish"]) == 1
    assert "reachaq-animal-metadata setup" in capsys.readouterr().err


def test_the_old_softmouse_command_forwards_setup_and_publish(monkeypatch):
    from reachaq.softmouse_sync import cli as old_cli

    calls = []
    monkeypatch.setattr(old_cli, "animal_metadata_main", lambda argv: calls.append(argv) or 0)

    assert old_cli.main(["--setup"]) == 0
    assert old_cli.main([]) == 0
    assert old_cli.main(["--destination", "/srv/pub", "--require-mount", "/srv"]) == 0
    assert calls == [
        ["setup", "--kind", "softmouse"],
        ["publish"],
        ["publish", "--destination", "/srv/pub", "--require-mount", "/srv"],
    ]
