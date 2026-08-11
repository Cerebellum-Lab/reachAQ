import json
import stat

from tools.softmouse_sync.cli import (
    DEFAULT_KEYRING_SERVICE,
    _configure_credentials,
    _credential_location,
)


class MemoryKeyring:
    def __init__(self):
        self.values = {}

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def get_password(self, service, username):
        return self.values.get((service, username))


def test_guided_setup_prompts_once_and_saves_no_password(tmp_path):
    path = tmp_path / "publisher.json"
    value = {"publicationDirectory": "/shared", "https": {}}
    path.write_text(json.dumps(value), encoding="utf-8")
    keyring = MemoryKeyring()
    username_prompts = []
    password_prompts = []

    service, username = _configure_credentials(
        path,
        value,
        username_prompt=lambda prompt: username_prompts.append(prompt) or "breynolds",
        password_prompt=lambda prompt: password_prompts.append(prompt) or "secret",
        keyring_backend=keyring,
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert (service, username) == (DEFAULT_KEYRING_SERVICE, "breynolds")
    assert saved["credentials"] == {
        "username": "breynolds",
        "keyringService": DEFAULT_KEYRING_SERVICE,
    }
    assert "secret" not in path.read_text(encoding="utf-8")
    assert keyring.get_password(service, username) == "secret"
    assert len(username_prompts) == len(password_prompts) == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_username_can_be_reused_when_rotating_password(tmp_path):
    path = tmp_path / "publisher.json"
    value = {
        "credentials": {
            "username": "breynolds",
            "keyringService": "lab-softmouse",
        }
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    keyring = MemoryKeyring()

    _configure_credentials(
        path,
        value,
        username_prompt=lambda _prompt: "",
        password_prompt=lambda _prompt: "replacement",
        keyring_backend=keyring,
    )

    assert keyring.get_password("lab-softmouse", "breynolds") == "replacement"


def test_publication_reads_credential_location_from_config():
    value = {
        "credentials": {
            "username": "breynolds",
            "keyringService": "lab-softmouse",
        }
    }

    assert _credential_location(value) == ("lab-softmouse", "breynolds")
    assert _credential_location(
        value,
        username_override="temporary-user",
        service_override="temporary-service",
    ) == ("temporary-service", "temporary-user")
