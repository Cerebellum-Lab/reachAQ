from dataclasses import dataclass

import pytest

from tools.softmouse_sync.cli import (
    DEFAULT_KEYRING_SERVICE,
    KEYRING_USERNAME_ACCOUNT,
    _configure_credentials,
    _credentials_from_keyring,
    _ensure_publication_directory,
    main,
)


@dataclass
class Credential:
    username: str
    password: str


class MemoryKeyring:
    def __init__(self):
        self.values = {}

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def get_password(self, service, username):
        return self.values.get((service, username))

    def get_credential(self, service, _username):
        for (stored_service, username), password in self.values.items():
            if stored_service == service:
                return Credential(username, password)
        return None


def test_guided_setup_prompts_once_and_stores_both_credentials_in_keyring():
    keyring = MemoryKeyring()
    username_prompts = []
    password_prompts = []

    username = _configure_credentials(
        username_prompt=lambda prompt: username_prompts.append(prompt) or "breynolds",
        password_prompt=lambda prompt: password_prompts.append(prompt) or "secret",
        keyring_backend=keyring,
    )

    assert username == "breynolds"
    assert keyring.get_password(DEFAULT_KEYRING_SERVICE, "breynolds") == "secret"
    assert (
        keyring.get_password(DEFAULT_KEYRING_SERVICE, KEYRING_USERNAME_ACCOUNT)
        == "breynolds"
    )
    assert len(username_prompts) == len(password_prompts) == 1


def test_existing_username_can_be_reused_when_rotating_password():
    keyring = MemoryKeyring()
    keyring.set_password(DEFAULT_KEYRING_SERVICE, "breynolds", "old")
    keyring.set_password(
        DEFAULT_KEYRING_SERVICE, KEYRING_USERNAME_ACCOUNT, "breynolds"
    )

    _configure_credentials(
        username_prompt=lambda _prompt: "",
        password_prompt=lambda _prompt: "replacement",
        keyring_backend=keyring,
    )

    assert keyring.get_password(DEFAULT_KEYRING_SERVICE, "breynolds") == "replacement"


def test_credentials_are_read_without_a_configuration_file():
    keyring = MemoryKeyring()
    keyring.set_password(DEFAULT_KEYRING_SERVICE, "breynolds", "secret")
    keyring.set_password(
        DEFAULT_KEYRING_SERVICE, KEYRING_USERNAME_ACCOUNT, "breynolds"
    )

    credentials = _credentials_from_keyring(keyring)

    assert credentials.username == "breynolds"
    assert credentials.password == "secret"


def test_earlier_keyring_entry_is_migrated_without_prompting_again():
    keyring = MemoryKeyring()
    keyring.set_password(DEFAULT_KEYRING_SERVICE, "breynolds", "secret")

    credentials = _credentials_from_keyring(keyring)

    assert credentials.username == "breynolds"
    assert (
        keyring.get_password(DEFAULT_KEYRING_SERVICE, KEYRING_USERNAME_ACCOUNT)
        == "breynolds"
    )


def test_missing_credentials_explain_the_single_setup_command():
    with pytest.raises(RuntimeError, match="--setup"):
        _credentials_from_keyring(MemoryKeyring())


def test_publication_directory_is_created_only_on_verified_mount(tmp_path):
    mount = tmp_path / "isilon"
    destination = mount / "Data" / "ReachingData" / "SoftMouse"
    destination.parent.mkdir(parents=True)

    assert (
        _ensure_publication_directory(
            destination,
            mount_root=mount,
            is_mount=lambda path: path == mount,
        )
        == destination
    )
    assert destination.is_dir()


def test_publication_stops_when_isilon_is_not_mounted(tmp_path):
    with pytest.raises(RuntimeError, match="not mounted"):
        _ensure_publication_directory(
            tmp_path / "isilon" / "Data" / "ReachingData" / "SoftMouse",
            mount_root=tmp_path / "isilon",
            is_mount=lambda _path: False,
        )


def test_cli_reports_expected_setup_errors_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        "tools.softmouse_sync.cli._credentials_from_keyring",
        lambda: (_ for _ in ()).throw(RuntimeError("credential unavailable")),
    )

    assert main([]) == 1
    assert capsys.readouterr().err.strip() == (
        "SoftMouse publication stopped: credential unavailable"
    )
